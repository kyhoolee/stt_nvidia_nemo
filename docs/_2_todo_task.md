Dưới đây là **tổng hợp đầy đủ các bước** để bạn có thể **train + test mô hình ASR tiếng Việt nhỏ (VIVOS)** bằng **NVIDIA NeMo**, cho cả hai mô hình:

* ✅ `SimpleCNN` (dựa trên `ConvASREncoder`)
* ✅ `FastConformer` (dùng trong `asr_transducer`)

---

# ✅ Tổng quan các bước cần làm

## Bước 0: Chuẩn bị môi trường

```bash
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
pip install -e .
```

> ⚠️ Nên tạo virtualenv/conda, cài theo `requirements/asr.txt`.

---

## Bước 1: Chuẩn bị dữ liệu VIVOS

### 1.1 Tải và chuẩn hóa:

* Tải từ: [https://ailab.hcmus.edu.vn/vivos](https://ailab.hcmus.edu.vn/vivos)
* Tổ chức lại thành:

```
data/vivos/
├── train/
│   ├── xxx.wav
│   └── ...
├── dev/
│   ├── xxx.wav
│   └── ...
└── test/
    ├── xxx.wav
```

### 1.2 Sinh manifest:

Chạy script tạo manifest (`train.json`, `dev.json`, `test.json`) theo format chuẩn NeMo:

```json
{"audio_filepath": "data/vivos/train/xxx.wav", "text": "xin chao cac ban", "duration": 1.23}
```

> Mỗi dòng là 1 JSON object. Dùng Python hoặc tool sẵn trong `NeMo` (`nemo_text_processing` nếu cần).

---

## Bước 2: Chuẩn bị tokenizer

```bash
python scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest=data/vivos/train.json \
  --tokenizer_dir=data/tokenizer \
  --vocab_size=128 \
  --spe_type=bpe
```

> Kết quả: thư mục `data/tokenizer/` sẽ chứa tokenizer để dùng cho cả training & inference.

---

## Bước 3: Chạy train

### ✳️ Option 1: SimpleCNN (ConvASREncoder + CTC Loss)

#### 3.1 Dùng config mẫu (viết mới hoặc chỉnh từ example):

```bash
cp examples/asr/conf/asr_ctc/conformer_ctc_bpe.yaml configs/vivos_cnn_ctc.yaml
```

#### 3.2 Sửa `configs/vivos_cnn_ctc.yaml`:

```yaml
model:
  sample_rate: 16000
  encoder:
    _target_: nemo.collections.asr.modules.ConvASREncoder
    feat_in: 80
    ...
  decoder:
    ...
  tokenizer:
    dir: data/tokenizer
    type: bpe
  train_ds:
    manifest_filepath: data/vivos/train.json
  validation_ds:
    manifest_filepath: data/vivos/dev.json
```

#### 3.3 Chạy train:

```bash
python examples/asr/asr_ctc/speech_to_text_ctc.py \
  model.tokenizer.dir=data/tokenizer \
  model.train_ds.manifest_filepath=data/vivos/train.json \
  model.validation_ds.manifest_filepath=data/vivos/dev.json \
  trainer.devices=1 \
  trainer.max_epochs=30 \
  exp_manager.exp_dir=outputs/vivos_cnn_ctc
```

---

### ✳️ Option 2: FastConformer (RNNT Loss)

#### 3.1 Sử dụng config:

```bash
cp tutorials/asr/configs/vpb_fast-conformer_transducer_bpe.yaml configs/vivos_fastconformer.yaml
```

#### 3.2 Sửa cho phù hợp:

```yaml
model:
  sample_rate: 16000
  tokenizer:
    dir: data/tokenizer
  train_ds:
    manifest_filepath: data/vivos/train.json
  validation_ds:
    manifest_filepath: data/vivos/dev.json
```

#### 3.3 Chạy train:

```bash
python examples/asr/asr_transducer/speech_to_text_rnnt.py \
  model.tokenizer.dir=data/tokenizer \
  model.train_ds.manifest_filepath=data/vivos/train.json \
  model.validation_ds.manifest_filepath=data/vivos/dev.json \
  trainer.devices=1 \
  trainer.max_epochs=30 \
  exp_manager.exp_dir=outputs/vivos_fastconformer
```

---

## Bước 4: Test mô hình

### 4.1 Tạo manifest test nếu chưa có:

```json
{"audio_filepath": "data/vivos/test/abc.wav", "text": "chao ban", "duration": 1.2}
```

### 4.2 Chạy test:

```bash
python scripts/evaluate_asr_model.py \
  model_path=<your_ckpt_or_nemo> \
  dataset_manifest=data/vivos/test.json \
  tokenizer_dir=data/tokenizer \
  output_path=results/test_output.json
```

> ⚠️ Nếu là FastConformer, phải dùng script `asr_transducer_infer.py`.

---

## Bước 5: (Tuỳ chọn) Convert `.ckpt` → `.nemo` để inference dễ

```bash
python scripts/checkpoint_to_nemo.py \
  --checkpoint_dir=outputs/vivos_cnn_ctc/checkpoints \
  --output_path=models/vivos_cnn_ctc.nemo
```

---

## ✅ Kết luận checklist:

| Việc cần làm                              | Trạng thái |
| ----------------------------------------- | ---------- |
| Chuẩn bị VIVOS + manifest                 | ✅          |
| Tạo tokenizer                             | ✅          |
| Sửa config SimpleCNN (CTC)                | ✅          |
| Sửa config FastConformer (RNNT)           | ✅          |
| Chạy train                                | ✅          |
| Chạy test                                 | ✅          |
| Convert model `.ckpt` → `.nemo` (nếu cần) | ✅          |

---

Nếu bạn muốn mình cung cấp sẵn:

* 📄 `vivos_cnn_ctc.yaml` tối giản
* 📄 `vivos_fastconformer.yaml` đã chỉnh
* 🐍 Script tạo manifest từ `.txt + .wav`

Hãy nói nhé, mình gửi ngay.
