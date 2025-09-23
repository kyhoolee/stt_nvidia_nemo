import nemo.collections.asr as nemo_asr
from pathlib import Path

# --- Cấu hình ---
# 1. Đường dẫn đến file .nemo của bạn
nemo_file_path = Path("/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo")

# 2. Tên file ONNX đầu ra
onnx_output_path = "./vpb_fastconformer.onnx"
# -----------------

print(f"🚀 Bắt đầu quá trình export model...")
print(f"🧠 Đang tải model NeMo từ: {nemo_file_path}")

try:
    # Tải model từ file .nemo
    # Đối với model của bạn là EncDecRNNTBPEModel
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=str(nemo_file_path))
    asr_model.eval()

    print("==================================================")
    print(asr_model)
    print("==================================================")
    print(dir(asr_model))
    print("==================================================")
    print(vars(asr_model))
    print("==================================================")

    # Giả sử bạn đã tải model vào biến asr_model
    # asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(...)

    # 1. In ra chính đối tượng 'decoding'
    print("Đối tượng decoding:")
    print(asr_model.decoding)

    print("==================================================")


    # 2. In ra tokenizer bên trong đối tượng 'decoding'
    print("\nTokenizer bên trong decoding:")
    print(asr_model.decoding.tokenizer)

    print("==================================================")


    # Thực hiện export
    # print(f"📦 Đang export sang ONNX:: {onnx_output_path}")
    # asr_model.export(onnx_output_path)

    # print(f"✅ Export thành công! File ONNX đã được lưu tại: {onnx_output_path}")

except Exception as e:
    print(f"❌ Đã xảy ra lỗi: {e}")