
# =============================


import json
import torch
import librosa
import numpy as np
import onnxruntime
import nemo.collections.asr as nemo_asr
from pathlib import Path
from tqdm import tqdm
from nemo.collections.asr.metrics.wer import word_error_rate as wer

# --- 1. Cấu hình ---
nemo_file_path = "/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo"
encoder_onnx_path = "./encoder-vpb_fastconformer.onnx"
decoder_joint_onnx_path = "./decoder_joint-vpb_fastconformer.onnx"
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

print("⚡ Tải các model ONNX vào Inference Session...")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
encoder_session = onnxruntime.InferenceSession(encoder_onnx_path, providers=providers)
decoder_joint_session = onnxruntime.InferenceSession(decoder_joint_onnx_path, providers=providers)

# ===================== SỬA LỖI TẠI ĐÂY (PHẦN 1) =====================
# Lấy tên input và output tự động từ model ONNX
decoder_input_names = [inp.name for inp in decoder_joint_session.get_inputs()]
decoder_output_names = [out.name for out in decoder_joint_session.get_outputs()]

print(f"✅ Tải xong model. Tên input của Decoder ONNX: {decoder_input_names}")
# ====================================================================

# --- 3. Hàm thực thi Inference cho một file Audio ---
def transcribe_audio_onnx(audio_path: str) -> str:
    try:
        # a. Tiền xử lý audio
        audio_signal, _ = librosa.load(audio_path, sr=16000)
        processed_signal, processed_length = preprocessor(
            input_signal=torch.from_numpy(audio_signal).unsqueeze(0),
            length=torch.tensor([len(audio_signal)]).long(),
        )
        processed_signal_np = processed_signal.cpu().numpy()
        processed_length_np = processed_length.cpu().numpy()

        # b. Chạy Encoder
        encoder_outputs, _ = encoder_session.run(
            None, {'audio_signal': processed_signal_np, 'length': processed_length_np}
        )

        # c. Chạy vòng lặp Decoding (Greedy Search)
        hypotheses_ids = []
        hidden = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        cell = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        prev_token = np.array([blank_id], dtype=np.int64)

        num_frames = encoder_outputs.shape[1]
        for i in range(num_frames):
            frame = encoder_outputs[:, i, :]

            # ===================== SỬA LỖI TẠI ĐÂY (PHẦN 2) =====================
            # Tạo dictionary input bằng cách sử dụng các tên đã lấy tự động
            # Giả định thứ tự là: encoder_output, prev_token, hidden_state, cell_state
            decoder_inputs = {
                decoder_input_names[0]: frame,
                decoder_input_names[1]: prev_token,
                decoder_input_names[2]: hidden,
                decoder_input_names[3]: cell,
            }

            # Chạy session với tên input/output đã được lấy tự động
            outputs = decoder_joint_session.run(decoder_output_names, decoder_inputs)
            # ====================================================================

            logits, hidden, cell = outputs[0], outputs[1], outputs[2]
            pred_token = np.argmax(logits, axis=-1)[0]

            if pred_token != blank_id:
                hypotheses_ids.append(pred_token.item())
                prev_token = np.array([pred_token], dtype=np.int64)

        # d. Hậu xử lý: Chuyển ID thành văn bản
        predicted_text = tokenizer.tokenizer.decode(hypotheses_ids)
        return predicted_text

    except Exception as e:
        print(f"❌ Lỗi khi xử lý file {audio_path}: {e}")
        return "" 

# --- 4. Vòng lặp chính: Đọc Manifest và tính toán WER ---
# (Phần này giữ nguyên, không cần thay đổi)
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
# (Phần này giữ nguyên, không cần thay đổi)
print("\n✅ Finished testing.")

if not all_predictions:
    print("⚠️ Không có mẫu nào được xử lý. Không thể tính WER.")
else:
    wer_score = wer(hypotheses=all_predictions, references=all_references)
    print("=" * 100)
    print(f"✨ Final WER for the test set: {wer_score:.4f}")
    print(f"(Calculated on {len(all_predictions)} samples)")
    print("=" * 100)

