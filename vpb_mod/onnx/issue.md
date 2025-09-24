 # server/Dockerfile
FROM nvcr.io/nvidia/tritonserver:24.06-py3

# 1) System deps (soundfile + build toolchain cho youtokentome)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*

# 2) Pip toolchain + Cython
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel Cython

# 3) Torch CPU wheels (tránh xung đột CUDA trong image Triton)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.1.2" "torchaudio==2.1.2"

# 4) Cài sẵn YTTM (tránh lỗi build metadata), rồi NeMo ASR + ONNXRuntime + SoundFile
<!-- RUN pip install --no-cache-dir \
    "youtokentome==1.0.6.post1" \
 &&  -->
pip install --no-cache-dir \
    "nemo_toolkit[asr]==1.23.0" \
    "onnxruntime" \
    "soundfile"

 
 =========================
 
 > [5/5] RUN pip install --no-cache-dir     "youtokentome==1.0.6.post1"  && pip install --no-cache-dir     "nemo_toolkit[asr]==1.23.0"     "onnxruntime"     "soundfile":                                       
0.587 ERROR: Could not find a version that satisfies the requirement youtokentome==1.0.6.post1 (from versions: 1.0.0, 1.0.1, 1.0.2, 1.0.3rc1, 1.0.3, 1.0.4, 1.0.5, 1.0.6)                                       
0.626 ERROR: No matching distribution found for youtokentome==1.0.6.post1
------
Dockerfile:18
--------------------
  17 |     # 4) Cài sẵn YTTM (tránh lỗi build metadata), rồi NeMo ASR + ONNXRuntime + SoundFile
  18 | >>> RUN pip install --no-cache-dir \
  19 | >>>     "youtokentome==1.0.6.post1" \
  20 | >>>  && pip install --no-cache-dir \
  21 | >>>     "nemo_toolkit[asr]==1.23.0" \
  22 | >>>     "onnxruntime" \
  23 | >>>     "soundfile"
  24 |     
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c pip install --no-cache-dir     \"youtokentome==1.0.6.post1\"  && pip install --no-cache-dir     \"nemo_toolkit[asr]==1.23.0\"     \"onnxruntime\"     \"soundfile\"" did not complete successfully: exit code: 1

==============+


(base) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo/vpb_mod/onnx$ tree 
.
├── TODO.md
├── client
│   ├── __init__.py
│   └── client_infer.py
├── issue.md
└── server
    ├── Dockerfile
    ├── __init__.py
    ├── requirements_server.txt
    └── rnnt_greedy
        ├── 1
        │   ├── decoder_joint-vpb_fastconformer.onnx
        │   ├── encoder-vpb_fastconformer.onnx
        │   ├── model.py
        │   └── rnnt_asr.nemo
        ├── __init__.py
        └── config.pbtxt

4 directories, 13 files