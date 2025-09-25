#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, time, os
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

import tritonclient.grpc as grpcclient


def load_from_manifest(manifest_path: str, limit: int = None):
    paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            paths.append(item["audio_filepath"])
            if limit is not None and len(paths) >= limit:
                break
    if not paths:
        raise RuntimeError("Manifest rỗng hoặc không đọc được audio_filepath.")
    return paths


def load_audio_pcm_f32(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1).astype(np.float32)
    if sr != target_sr:
        if not HAVE_LIBROSA:
            raise RuntimeError(f"Audio SR={sr} khác {target_sr}, cần librosa để resample (chưa cài).")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best").astype(np.float32)
        sr = target_sr
    return wav, sr


def collate_batch(wavs: list[np.ndarray]):
    """Pad list wavs -> (B,Tmax) FP32 và lengths -> (B,1) INT32"""
    B = len(wavs)
    lengths = np.array([w.shape[0] for w in wavs], dtype=np.int32)
    Tmax = int(lengths.max()) if B > 0 else 0
    sig = np.zeros((B, Tmax), dtype=np.float32)
    for i, w in enumerate(wavs):
        Ti = w.shape[0]
        if Ti > 0:
            sig[i, :Ti] = w
    return sig, lengths.reshape(-1, 1), lengths, Tmax


def infer_batch(triton: grpcclient.InferenceServerClient, model: str, batch_paths: list[str],
                warmup: int = 0, sr: int = 16000):
    # 1) load
    wavs, lens = [], []
    for p in batch_paths:
        w, _ = load_audio_pcm_f32(p, sr)
        wavs.append(w)
        lens.append(w.shape[0])

    # 2) pre-stats
    B = len(batch_paths)
    lengths = np.array(lens, dtype=np.int32)
    secs = lengths / sr
    total_sec = float(secs.sum())
    Tmax = int(lengths.max()) if B > 0 else 0
    pad_eff = float(lengths.sum()) / float(B * Tmax) if (B > 0 and Tmax > 0) else 0.0  # 0..1

    print(f"[BATCH] files: {B} | Tmax: {Tmax/sr:.2f}s | total_real: {total_sec:.2f}s | pad_eff: {pad_eff*100:.1f}%")
    for j, (p, s) in enumerate(zip(batch_paths, secs)):
        print(f"  - [{j}] {os.path.basename(p)} | dur={s:.2f}s")

    # 3) collate & send
    sig, lengths2d, _, _ = collate_batch(wavs)
    inp_signal = grpcclient.InferInput("AUDIO_SIGNAL", sig.shape, "FP32")
    inp_len    = grpcclient.InferInput("AUDIO_LENGTH", lengths2d.shape, "INT32")
    inp_signal.set_data_from_numpy(sig)
    inp_len.set_data_from_numpy(lengths2d)
    out = grpcclient.InferRequestedOutput("TRANSCRIPT")

    # warmup
    for _ in range(warmup):
        _ = triton.infer(model, [inp_signal, inp_len], outputs=[out])

    # 4) measure
    t0 = time.perf_counter()
    resp = triton.infer(model, [inp_signal, inp_len], outputs=[out])
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # 5) decode
    texts = resp.as_numpy("TRANSCRIPT")  # (B,)
    outs = [(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)) for v in texts]

    # 6) normalized time metrics
    ms_per_sec = (dt_ms / total_sec) if total_sec > 0 else float("nan")   # ms cho mỗi 1 giây audio thật (toàn batch)
    rtf = (dt_ms / 1000.0) / max(total_sec, 1e-9)                         # real-time factor
    print(f"[BATCH] latency: {dt_ms:.2f} ms | avg/utt: {dt_ms/B:.2f} ms | ms_per_sec: {ms_per_sec:.2f} | RTF: {rtf:.3f}")

    # 7) per-file estimated time by length share
    #    (ước lượng công bằng theo độ dài: thời gian ~ tỉ lệ với dur_i)
    for j, (p, s, hyp) in enumerate(zip(batch_paths, secs, outs)):
        est_ms = ms_per_sec * s
        print(f"[{j}] {os.path.basename(p)} | dur={s:.2f}s | est_time={est_ms:.2f} ms")
        print(f" -> {hyp}")

    print()  # dòng trống giữa các batch
    return outs, dt_ms



def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="localhost:8001", help="host:port Triton gRPC")
    ap.add_argument("--model", default="rnnt_greedy", help="Triton model name")
    ap.add_argument("--manifest", required=True, help="Đường dẫn manifest JSONL")
    ap.add_argument("--limit", type=int, default=32, help="Tổng số mẫu dùng để test")
    ap.add_argument("--batch-size", type=int, default=8, help="Số mẫu mỗi request")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup mỗi batch")
    ap.add_argument("--concurrency", type=int, default=1, help="Số luồng gửi đồng thời")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate model (mặc định 16k)")
    args = ap.parse_args()

    paths = load_from_manifest(args.manifest, limit=args.limit)
    batches = list(chunks(paths, args.batch_size))
    triton = grpcclient.InferenceServerClient(url=args.server, verbose=False)

    total_ms = 0.0
    total_samples = 0

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            fut2info = {}
            for i, b in enumerate(batches):
                fut = ex.submit(infer_batch, triton, args.model, b, args.warmup, args.sr)
                fut2info[fut] = (i, b)
            for fut in as_completed(fut2info):
                i, b = fut2info[fut]
                try:
                    outs, dt = fut.result()
                except Exception as e:
                    print(f"[BATCH {i}] ERROR: {e}")
                    continue
                total_ms += dt
                total_samples += len(b)
                for j, (p, t) in enumerate(zip(b, outs)):
                    print(f"[{i}:{j}] {os.path.basename(p)}")
                    print(f" -> {t}")
                print(f"[BATCH {i}] latency: {dt:.2f} ms for {len(b)} utt")
    else:
        for i, b in enumerate(batches):
            outs, dt = infer_batch(triton, args.model, b, args.warmup, args.sr)
            total_ms += dt
            total_samples += len(b)
            for j, (p, t) in enumerate(zip(b, outs)):
                print(f"[{i}:{j}] {os.path.basename(p)}")
                print(f" -> {t}")
            print(f"[BATCH {i}] latency: {dt:.2f} ms for {len(b)} utt")

    if total_samples:
        print(f"\n== Summary ==")
        print(f"Batches: {len(batches)}, Total utt: {total_samples}")
        print(f"Avg time / batch: {total_ms/len(batches):.2f} ms")
        print(f"Avg time / utt  : {total_ms/total_samples:.2f} ms")


if __name__ == "__main__":
    main()
