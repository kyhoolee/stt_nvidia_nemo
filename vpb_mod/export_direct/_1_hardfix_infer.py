import onnxruntime as ort
import numpy as np
import librosa

# --- Cấu hình ---
# 1. Đường dẫn đến file ONNX đã export
onnx_model_path = "vpb_fastconformer.onnx"

# 2. Đường dẫn đến file âm thanh bạn muốn test
# !!! THAY ĐỔI ĐƯỜNG DẪN NÀY !!!
audio_file_path = "/home/ubuntu/work/clean_dataset_vpb/audio/archive_2/wavs/E_huongds_D_2025-06-02_H_091735_844_CLID_0813494717___000040066___right___000040926.wav"
# -----------------

# --- 1. Khởi tạo ONNX Runtime Session ---
print("▶️ Khởi tạo ONNX Runtime session...")
# Sử dụng 'CUDAExecutionProvider' nếu có GPU, nếu không nó sẽ tự chuyển về 'CPUExecutionProvider'
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(onnx_model_path, providers=providers)
print(f"✅ Session đã được tạo với provider: {session.get_providers()}")


# --- 2. Chuẩn bị dữ liệu đầu vào (âm thanh) ---
print(f"🔊 Đang xử lý file âm thanh: {audio_file_path}")
# NeMo thường làm việc với sample rate 16000 Hz
waveform, sample_rate = librosa.load(audio_file_path, sr=16000)

# Input của model ONNX thường là (batch_size, num_samples)
# Ở đây batch_size = 1
input_signal = np.asarray(waveform, dtype=np.float32)[np.newaxis, :]
input_signal_length = np.asarray([input_signal.shape[1]], dtype=np.int64)

# Lấy tên input của model (thường là 'audio_signal' và 'length')
input_names = [inp.name for inp in session.get_inputs()]
print(f"🏷️ Tên các input của model ONNX: {input_names}")

# Tạo dictionary cho input feed
input_feed = {
    input_names[0]: input_signal,
    input_names[1]: input_signal_length
}


# --- 3. Chạy Inference ---
print("🧠 Đang thực hiện nhận dạng...")
# Chạy session.run(), output sẽ là log probabilities (logits)
# Lấy tên output của model
output_names = [out.name for out in session.get_outputs()]
print(f"🏷️ Tên các output của model ONNX: {output_names}")

outputs = session.run(output_names, input_feed=input_feed)
logits = outputs[0]
print(f"✅ Nhận dạng hoàn tất! Kích thước của logits output: {logits.shape}")


# --- 4. Giải mã (Decode) output để ra văn bản ---
# Để giải mã logits ra văn bản, cách đơn giản nhất là dùng lại chính
# đối tượng `asr_model` của NeMo đã tải ở Phần 1, vì nó chứa sẵn tokenizer.
print("📝 Đang giải mã logits ra văn bản...")

# `asr_model` phải được tải sẵn từ Phần 1
if 'asr_model' in locals():
    # Sử dụng phương thức post-processing của model NeMo để giải mã
    # outputs[0] là logits, outputs[2] là độ dài đã được encode
    hypotheses = asr_model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=outputs[0],
        encoded_lengths=outputs[2],
        return_hypotheses=False,
    )
    # hypotheses[0] thường là list các text được giải mã
    if hypotheses and len(hypotheses) > 0:
        transcription = hypotheses[0]
        print("\n--- KẾT QUẢ NHẬN DẠNG ---")
        print(f"Text: {transcription}")
        print("--------------------------\n")
    else:
        print("Không thể giải mã được văn bản.")
else:
    print("Vui lòng chạy lại Phần 1 để tải model NeMo và thực hiện giải mã.")