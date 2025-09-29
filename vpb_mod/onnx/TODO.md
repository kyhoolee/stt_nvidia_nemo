Để chạy được trọn vẹn service + client theo hướng A, bạn cần các package sau (mình chia thành 2 phần: **server** và **client/test**):

---

## 🔹 1. Trên server (chạy Triton + model.py)

Bạn chạy Triton qua container NVIDIA thì bản thân image `nvcr.io/nvidia/tritonserver:24.xx-py3` đã có sẵn:

* **Python backend runtime**
* **onnxruntime**
* **PyTorch (torch)**

Nhưng vì `model.py` của mình dùng thêm NeMo, bạn cần mount/cài bổ sung:

```bash
pip install nemo_toolkit['all']==1.23.0
pip install onnxruntime
pip install soundfile
```

Tuỳ môi trường bạn có GPU hay CPU mà có thể cần:

```bash
pip install torch==2.1.2 torchaudio==2.1.2
```

---

## 🔹 2. Trên client (script `client_infer.py`)

```bash
pip install tritonclient[grpc]
pip install soundfile
pip install librosa
```

(thêm `numpy` và `argparse` đã có sẵn trong Python chuẩn)

---

## 🔹 3. Tổng hợp nhanh

### Server

* torch
* nemo\_toolkit\[all]
* onnxruntime
* soundfile

### Client

* tritonclient\[grpc]
* soundfile
* librosa
* numpy

---

👉 Nếu bạn dùng Docker Triton thì cách tiện nhất:

* Chạy Triton server trong container,
* Cài thêm `nemo_toolkit`, `torch`, `onnxruntime`, `soundfile` bằng pip ngay trong container hoặc mount môi trường ảo vào.

Bạn có muốn mình viết sẵn Dockerfile mở rộng từ `tritonserver:24.06-py3` có cài đủ các package cần thiết cho bạn không?


==============================================



cd vpb_mod/onnx/server

# Build image
docker build -t my_triton_rnnt:latest .

# Run CPU:
docker run --rm --net=host \
  -v $PWD:/models \
  my_triton_rnnt:latest \
  tritonserver --model-repository=/models

# (Nếu có GPU)
# docker run --rm --net=host --gpus=all \
#   -v $PWD:/models \
#   my_triton_rnnt:latest \
#   tritonserver --model-repository=/models


docker run --rm --net=host \
  -v "$PWD:/models" \
  my_triton_rnnt:cpu \
  tritonserver --model-repository=/models

docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD:/models" \
  my_triton_rnnt:cpu \
  tritonserver --model-repository=/models


docker run --shm-size=8g --rm --net=host   -e DEVICE=cpu   -v "$PWD:/models"   my_triton_rnnt:cpu




=================================


pip install tritonclient[grpc] soundfile librosa numpy



python vpb_mod/onnx/client/client_infer.py \
  --server localhost:8001 \
  --model rnnt_greedy \
  --manifest /path/to/your/test_manifest.jsonl \
  --limit 1


========================


# Health
curl -s http://localhost:8000/v2/health/ready
curl -s http://localhost:8000/v2/health/live

# Model index
curl -s http://localhost:8000/v2/models

# Metadata model rnnt_greedy (đổi tên model nếu khác)
curl -s http://localhost:8000/v2/models/rnnt_greedy


=========================


python client_infer.py \
  --server localhost:8001 \
  --model rnnt_greedy \
  --manifest /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl \
  --limit 1 --warmup 2



python client_batch_infer.py \
  --server localhost:8001 \
  --model rnnt_greedy \
  --manifest /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl \
  --limit 8 \
  --batch-size 8 \
  --warmup 1


python client_batch_infer.py \
  --server localhost:8001 \
  --model rnnt_greedy \
  --manifest /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl \
  --limit 64 \
  --batch-size 8 \
  --concurrency 8 \
  --warmup 1

=====
manifest_vpb_right_2/valid_meta_nemo.jsonl

python client_batch_infer.py --server localhost:8001 --model rnnt_greedy --model-version 1 --manifest /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_nemo.jsonl --limit 64 --batch-size 8 --concurrency 1 --warmup 1

== Summary ==
Batches: 4, Total utt: 29
Avg time / batch: 14853.11 ms
Avg time / utt  : 2048.71 ms

== Summary ==
Batches: 8, Total utt: 64
Avg time / batch: 398.11 ms
Avg time / utt  : 49.76 ms

=====

python client_batch_infer.py --server localhost:8001 --model rnnt_greedy --model-version 2 --manifest /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_nemo.jsonl --limit 64 --batch-size 8 --concurrency 1 --warmup 1

== Summary ==
Batches: 8, Total utt: 64
Avg time / batch: 366.28 ms
Avg time / utt  : 45.78 ms

== Summary ==
Batches: 4, Total utt: 29
Avg time / batch: 14186.25 ms
Avg time / utt  : 1956.72 ms


======


docker run --shm-size=8g --rm --net=host   -e DEVICE=cpu   -v "$PWD:/models"   my_triton_rnnt:cpu


/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_nemo.jsonl

/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.jsonl
/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.jsonl
/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.jsonl


tmux new -f vpb_client