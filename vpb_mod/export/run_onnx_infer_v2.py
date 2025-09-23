import json
import torch
import librosa
import numpy as np
import onnxruntime
import nemo.collections.asr as nemo_asr
from pathlib import Path
from tqdm import tqdm
from nemo.collections.asr.metrics.wer import word_error_rate as wer
import argparse # Thư viện để đọc tham số dòng lệnh

# --- 0. Đọc tham số từ dòng lệnh ---
parser = argparse.ArgumentParser(description="Evaluate a NeMo ASR model using a 3-file ONNX pipeline.")
parser.add_argument(
    "--device", 
    type=str, 
    default="cpu", 
    choices=["gpu", "cpu"], 
    help="Device to run inference on: 'gpu' or 'cpu'. Defaults to 'gpu'."
)
args = parser.parse_args()


# --- 1. Cấu hình ---
# CHỈNH SỬA CÁC ĐƯỜNG DẪN BÊN DƯỚI CHO PHÙ HỢP
nemo_file_path = "/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo"
encoder_onnx_path = "/home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy/onnx/encoder.onnx"
predictor_onnx_path = "/home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy/onnx/predictor.onnx"
joint_onnx_path = "/home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy/onnx/joint.onnx"
test_manifest = "/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl" 

# --- 2. Khởi tạo ---
print("🚀 Bước 1: Khởi tạo - Tải model và các thành phần phụ trợ...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(nemo_file_path, map_location=torch.device('cpu'))
asr_model.eval()

tokenizer = asr_model.decoding.tokenizer
preprocessor = asr_model.preprocessor
blank_id = tokenizer.tokenizer.vocab_size
hidden_size = asr_model.decoder.pred_hidden
num_layers = asr_model.decoder.pred_rnn_layers

print(f"   - Tokenizer: Vocab size = {blank_id + 1}, Blank ID = {blank_id}")
print(f"   - Predictor LSTM: {num_layers} layer(s), Hidden size = {hidden_size}")

# ===================== THAY ĐỔI TẠI ĐÂY =====================
# Chọn provider cho ONNX Runtime dựa trên tham số --device
if args.device == "gpu":
    print("   - Chế độ thực thi: GPU (CUDAExecutionProvider)")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
    print("   - Chế độ thực thi: CPU (CPUExecutionProvider)")
    providers = ['CPUExecutionProvider']
# ==========================================================

print("⚡ Tải 3 model ONNX vào Inference Session...")
encoder_session = onnxruntime.InferenceSession(encoder_onnx_path, providers=providers)
predictor_session = onnxruntime.InferenceSession(predictor_onnx_path, providers=providers)
joint_session = onnxruntime.InferenceSession(joint_onnx_path, providers=providers)
print("✅ Tải xong model và các thành phần.")


# --- 3. Hàm thực thi Inference cho một file Audio ---
def transcribe_audio_onnx(audio_path: str) -> str:
    """
    Hàm này nhận đường dẫn file audio và trả về văn bản dự đoán
    sử dụng pipeline 3 file ONNX: encoder, predictor, và joint.
    """
    try:
        # a. Tiền xử lý audio -> Mel Spectrogram
        audio_signal, _ = librosa.load(audio_path, sr=16000)
        processed_signal, processed_length = preprocessor(
            input_signal=torch.from_numpy(audio_signal).unsqueeze(0),
            length=torch.tensor([len(audio_signal)]).long(),
        )
        processed_signal_np = processed_signal.cpu().numpy()
        processed_length_np = processed_length.cpu().numpy()

        # b. Chạy Encoder ONNX
        encoder_inputs = {
            'processed_signal': processed_signal_np,
            'processed_signal_length': processed_length_np
        }
        encoder_outputs, _ = encoder_session.run(None, encoder_inputs)

        # c. Vòng lặp giải mã Greedy Search (step-by-step)
        hypotheses_ids = []
        h0 = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        c0 = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        states_hc = np.stack([h0, c0])
        prev_token = np.array([[blank_id]], dtype=np.int64)

        num_frames = encoder_outputs.shape[1]
        for i in range(num_frames):
            enc_frame = encoder_outputs[:, i:i+1, :] 
            
            # -- Chạy Predictor ONNX --
            predictor_inputs = {
                'targets': prev_token,
                'states_hc': states_hc
            }
            pred_out, next_states_hc = predictor_session.run(None, predictor_inputs)
            
            # -- Chạy Joint ONNX --
            joint_inputs = {
                'enc': enc_frame,
                'pred': pred_out
            }
            logits = joint_session.run(None, joint_inputs)[0]

            pred_token = np.argmax(logits[0, 0, 0, :]).item()
            
            if pred_token != blank_id:
                hypotheses_ids.append(pred_token)
                prev_token = np.array([[pred_token]], dtype=np.int64)
            
            states_hc = next_states_hc

        # d. Hậu xử lý: Chuyển ID thành văn bản
        predicted_text = tokenizer.tokenizer.decode(hypotheses_ids)
        return predicted_text

    except Exception as e:
        print(f"❌ Lỗi khi xử lý file {audio_path}: {e}")
        import traceback
        traceback.print_exc() 
        return "" 

# --- 4. Vòng lặp chính: Đọc Manifest và tính toán WER ---
print(f"\n🚀 Bước 2: Bắt đầu đánh giá trên file manifest: {test_manifest}")
all_predictions = []
all_references = []
print("🔍 Running ONNX transcription and WER calculation...")
with open(test_manifest, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(tqdm(lines, desc="Processing manifest")):
        item = json.loads(line)
        audio_path = item['audio_filepath']
        reference_text = item['text']

        predicted_text = transcribe_audio_onnx(audio_path)

        all_predictions.append(predicted_text.lower())
        all_references.append(reference_text.lower())

        if (i + 1) % 1 == 0:
            print(f"\n--- Processed {i + 1}/{len(lines)} samples. ---")
            print(f"Sample:")
            print(f"  REFERENCE: {reference_text}")
            print(f"  PREDICTED: {predicted_text}")

# --- 5. Tính toán và báo cáo kết quả cuối cùng ---
print("\n✅ Finished testing.")
if not all_predictions:
    print("⚠️ Không có mẫu nào được xử lý. Không thể tính WER.")
else:
    wer_score = wer(hypotheses=all_predictions, references=all_references)
    print("=" * 100)
    print(f"✨ Final WER for the test set: {wer_score:.4f}")
    print(f"(Calculated on {len(all_predictions)} samples)")
    print("=" * 100)