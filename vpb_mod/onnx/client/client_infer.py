#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, time, os
import numpy as np
import soundfile as sf
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

import tritonclient.grpc as grpcclient


def load_from_manifest(manifest_path: str, limit: int = 1):
    paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            paths.append(item["audio_filepath"])
            if len(paths) >= limit:
                break
    if not paths:
        raise RuntimeError("Manifest rỗng hoặc không đọc được audio_filepath.")
    return paths


def load_audio_pcm_f32(path: str, target_sr: int = 16000) -> np.ndarray:
    """Đọc audio => mono float32, resample về 16k nếu cần."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1).astype(np.float32)
    if sr != target_sr:
        if not HAVE_LIBROSA:
            raise RuntimeError(f"Audio SR={sr} khác {target_sr}, cần librosa để resample (chưa cài).")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best").astype(np.float32)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="localhost:8001", help="host:port Triton gRPC")
    ap.add_argument("--model", default="rnnt_greedy", help="Triton model name")
    ap.add_argument("--manifest", required=True, help="Đường dẫn manifest JSONL")
    ap.add_argument("--limit", type=int, default=1, help="Số mẫu lấy từ manifest (mặc định 1)")
    ap.add_argument("--warmup", type=int, default=2, help="Số lượt warmup trước khi đo")
    args = ap.parse_args()

    audio_paths = load_from_manifest(args.manifest, limit=args.limit)
    triton = grpcclient.InferenceServerClient(url=args.server, verbose=False)

    for idx, apath in enumerate(audio_paths):
        pcm = load_audio_pcm_f32(apath, target_sr=16000)
        # === IMPORTANT === add batch dim -> (1, T)
        pcm_batched = pcm.reshape(1, -1).astype(np.float32)
        length = np.array([[pcm.shape[0]]], dtype=np.int32)  # INT32 & shape (1,1)

        inp_signal = grpcclient.InferInput("AUDIO_SIGNAL", pcm_batched.shape, "FP32")
        inp_len    = grpcclient.InferInput("AUDIO_LENGTH", length.shape, "INT32")
        inp_signal.set_data_from_numpy(pcm_batched)
        inp_len.set_data_from_numpy(length)

        out = grpcclient.InferRequestedOutput("TRANSCRIPT")

        # warmup
        for _ in range(args.warmup):
            _ = triton.infer(args.model, [inp_signal, inp_len], outputs=[out])

        # measure
        t0 = time.perf_counter()
        resp = triton.infer(args.model, [inp_signal, inp_len], outputs=[out])
        dt = (time.perf_counter() - t0) * 1000.0  # ms

        texts = resp.as_numpy("TRANSCRIPT")
        # TYPE_BYTES -> bytes; TYPE_STRING -> str
        v = texts[0]
        text = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)

        print(f"[{idx}] file: {apath}")
        print(f" -> transcript: {text}")
        print(f" -> latency: {dt:.2f} ms")


if __name__ == "__main__":
    main()
