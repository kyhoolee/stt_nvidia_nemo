import nemo.collections.asr as nemo_asr
from pathlib import Path
import json
import shutil

# --- Cấu hình ---
nemo_file_path = Path("/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo")
deploy_dir = Path("./vpb_fastconformer_deployed_corrected")
# -----------------

# Tạo thư mục
onnx_dir = deploy_dir / "onnx"
tokenizer_dir = deploy_dir / "tokenizer"
onnx_dir.mkdir(parents=True, exist_ok=True)
tokenizer_dir.mkdir(parents=True, exist_ok=True)

print(f"🚀 Bắt đầu quá trình export model sang: {deploy_dir}")
asr_model = nemo.collections.asr.models.EncDecRNNTBPEModel.restore_from(str(nemo_file_path))
asr_model.eval()

# --- Export các thành phần ra ONNX (ĐÃ SỬA LỖI) ---
print("📦 Đang export Encoder, Predictor, Joint...")
asr_model.encoder.export(str(onnx_dir / "encoder.onnx"))

# === SỬA LỖI TẠI ĐÂY ===
# Export prediction network từ bên trong decoder
asr_model.decoder.prediction.export(str(onnx_dir / "predictor.onnx"))

asr_model.joint.export(str(onnx_dir / "joint.onnx"))

# --- Sao chép tokenizer ---
tokenizer_path = asr_model.tokenizer.tokenizer.model_path
if tokenizer_path:
    shutil.copy(tokenizer_path, tokenizer_dir / "tokenizer.model")
    print(f"☑️ Đã sao chép tokenizer vào: {tokenizer_dir}")

# --- Tạo file config tối giản (ĐÃ SỬA LỖI) ---
config = {
    "sample_rate": asr_model.cfg.preprocessor.sample_rate,
    "blank_id": asr_model.decoder.blank_idx,
    "vocab_size": asr_model.decoder.vocab_size,
    # === SỬA LỖI TẠI ĐÂY ===
    # Lấy thông số từ đúng vị trí trong cấu trúc model
    "pred_num_layers": asr_model.decoder.prediction.dec_rnn.lstm.num_layers,
    "pred_hidden": asr_model.decoder.prediction.dec_rnn.lstm.hidden_size,
    "preprocessor": asr_model.cfg.preprocessor
}
config_path = deploy_dir / "config_minimal.json"
with config_path.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print(f"☑️ Đã tạo file config tại: {config_path}")

print("\n✅ Hoàn tất! Cấu trúc thư mục deploy đã sẵn sàng.")