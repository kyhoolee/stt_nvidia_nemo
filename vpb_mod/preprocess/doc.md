Chuẩn rồi 👍 — bạn đọc doc chính thức thì thấy DeepFilterNet đã có hỗ trợ **inference only** khá gọn. Mình tổng hợp lại các bước **cài đặt & chạy nhanh để chỉ inference** (không cần training) cho đúng tình huống của bạn (callbot, chỉ cần `.wav` denoise → input cho STT).

---

# 1. Cài đặt gói Python

DeepFilterNet đã có sẵn trên PyPI, cài đơn giản:

```bash
# Cài đặt Pytorch trước (CPU hoặc CUDA tuỳ máy bạn)
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Cài DeepFilterNet (chỉ inference, không cần option train)
pip install deepfilternet
```

---

# 2. Tải model pretrained

Khi cài `deepfilternet`, bạn có thể chỉ định model name (`DeepFilterNet`, `DeepFilterNet2`, hoặc `DeepFilterNet3`).

* Mặc định: sẽ load model `DeepFilterNet2`.
* Nếu bạn muốn **DeepFilterNet3 (khuyến nghị)** thì truyền tham số `-m DeepFilterNet3`.
  → Không cần tự tay tải `.onnx`, vì gói này sẽ **tự fetch model pretrained** về cache `~/.cache/deepFilterNet/`.

---

# 3. Denoise 1 file WAV (CLI có sẵn)

Ví dụ, bạn có `sample.wav` (mono, 16kHz hoặc 8kHz).
Chạy lệnh:

```bash
python -m df.enhance -m DeepFilterNet3 -o out_dir sample.wav
```

* `-m DeepFilterNet3` → chọn model v3 pretrained.
* `-o out_dir` → thư mục chứa file đã denoise.
* Output sẽ được lưu tại `out_dir/sample_enhanced.wav`.

---

# 4. Dùng trực tiếp trong Python script

Bạn có thể nhúng thẳng vào pipeline:

```python
from df import enhance, init_df
import soundfile as sf

# Load model (DeepFilterNet3)
model, df_state, _ = init_df(model_base_dir="DeepFilterNet3")

# Đọc file input
x, sr = sf.read("sample.wav")
if sr != 16000:
    import librosa
    x = librosa.resample(x, sr, 16000)
    sr = 16000

# Enhance
y = enhance(model, df_state, x)

# Ghi ra file
sf.write("sample_denoised.wav", y, sr)
```

---

# 5. Cách tích hợp vào pipeline STT

* Bạn chọn 1 file từ tập callbot → chạy qua `df.enhance` hoặc hàm Python như trên.
* Output `.wav` này chính là input chuẩn cho pipeline STT (`--test-manifest` có thể trỏ tới file đã denoise).
* Khi test quick-win: bạn chỉ cần chạy inference STT **2 lần** (original vs denoised) để so sánh WER.

---

# 6. Ưu/nhược điểm của cách “inference only”

✅ Rất nhanh, không cần build lại model, không cần training.
✅ Tương thích ngay với file WAV hiện có (chỉ cần resample 16kHz).
❌ Model gốc được train cho full-band 48kHz → khi chạy 16kHz có thể chưa tối ưu tuyệt đối cho thoại telco, nhưng vẫn giảm noise rõ rệt.
❌ Không tinh chỉnh được cho domain callbot (sẽ cần training riêng nếu muốn thêm vài % cải thiện WER).

---

👉 Với tình huống của bạn (muốn **quick win** để A/B test WER):

* Dùng luôn `pip install deepfilternet`
* Chạy `python -m df.enhance -m DeepFilterNet3 file.wav`
* Lấy output làm input STT → so sánh WER.

---

Bạn muốn mình viết luôn script nhỏ `batch_denoise_manifest.py` dựa trên **cách load model từ PyPI** (thay vì .onnx thủ công) để áp dụng trực tiếp lên manifest test của bạn không?



================================


# Cài phụ thuộc
pip install deepfilternet soundfile librosa

# Chạy

python -m vpb_mod.preprocess.dfn_quickwin._1_denoise_filternet \
  --jsonl vpb_mod/preprocess/dfn_quickwin/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v1_hard.jsonl \
  --out-dir vpb_mod/preprocess/dfn_quickwin/data/denoised_in \
  --model DeepFilterNet3 \
  --copy-only 


python -m vpb_mod.preprocess.dfn_quickwin._1_denoise_filternet \
  --jsonl vpb_mod/preprocess/dfn_quickwin/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v1_hard.jsonl \
  --out-dir vpb_mod/preprocess/dfn_quickwin/data/denoised_out \
  --model DeepFilterNet3 \
  --limit 80 \
  --device cpu
  





