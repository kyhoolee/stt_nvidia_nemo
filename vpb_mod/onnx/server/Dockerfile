# Triton 24.06 (Python 3.10), đã có Python backend
FROM nvcr.io/nvidia/tritonserver:24.06-py3

# Locale + libsndfile cho soundfile (một số thành phần của NeMo cần)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libsndfile1 locales && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Mặc định chạy CPU; đổi thành "cuda" nếu bật GPU và đã cài torch/cu phù hợp
    DEVICE=cpu \
    # Hạn chế tqdm/log gây noise trong container
    TQDM_DISABLE=1

# Tránh xung đột numpy>=2 (một số phiên bản NeMo yêu cầu <2)
# Cài torch CPU trước, rồi NeMo + onnxruntime
# (giữ nguyên numpy 1.26.4 nếu bạn muốn ổn định với NeMo)
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "numpy==1.26.4" --no-cache-dir && \
    # Nâng PyTorch lên 2.5.1 CPU (đủ có torch.nn.attention)
    python3 -m pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.5.1" "torchaudio==2.5.1" --no-cache-dir && \
    python3 -m pip install \
        "Cython>=0.29.36" \
        "onnxruntime==1.18.0" \
        "pytorch-lightning==2.3.3" \
        "torchmetrics==1.4.0" \
        "omegaconf==2.3.0" \
        --no-cache-dir && \
    python3 -m pip install "nemo_toolkit[asr]==2.3.2" --no-cache-dir


# Tạo thư mục cho model repository (sẽ mount từ host vào /models)
RUN mkdir -p /models

# Cổng mặc định HTTP=8000, gRPC=8001, metrics=8002
EXPOSE 8000 8001 8002

# Mặc định chỉ chạy server; mount repo ở runtime
ENTRYPOINT ["tritonserver", "--model-repository=/models"]
