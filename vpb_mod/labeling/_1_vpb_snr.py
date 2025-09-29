#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute VAD-based SNR (dB) cho các manifest .jsonl kiểu NeMo.

- Tách voice bằng Energy-VAD (librosa.effects.split) hoặc WebRTC VAD (tùy chọn).
- Fallback ước lượng SNR bằng percentile STE khi thiếu non-voice.
- Song song cực cơ bản: chia list records thành K phần, mỗi process xử lý 1 phần.
- Ghi *.with_snr.jsonl cho từng manifest + summary CSV tổng hợp (nếu chỉ định).
- NEW: Debug progress per-process (processed/total, elapsed, ETA).

Ví dụ:
python vpb_snr_vad_manifest.py \
  --vad auto --top-db 22 --aggr 2 --min-gap-ms 60 --min-len-ms 100 --jobs 8 \
  --log-every 50 --log-interval 2.0 \
  --summary-csv /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/snr_summary.csv \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


# ======================== Helpers chung ========================

def merge_intervals(ints: List[Tuple[int, int]], min_gap: int = 0, min_len: int = 0) -> List[Tuple[int, int]]:
    """Gộp các interval gần nhau (<min_gap) & loại interval quá ngắn (<min_len)."""
    if not ints:
        return []
    ints = sorted(ints)
    merged = [list(ints[0])]
    for s, e in ints[1:]:
        if s - merged[-1][1] <= min_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged if (e - s) >= min_len]
    return merged


def frame_energy(x: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Short-time energy (mean square) theo frame."""
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    pad = (-(len(x) - frame_length)) % hop_length
    if pad:
        x = np.pad(x, (0, pad))
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
    eng = np.mean(frames.astype(np.float64) ** 2, axis=0)
    return eng


# ======================== VAD ========================

def vad_energy(
    x: np.ndarray,
    sr: int,
    top_db: int = 22,
    frame_length: int = 1024,
    hop_length: int = 256,
    min_gap_ms: int = 60,
    min_len_ms: int = 100,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Energy-VAD bằng librosa.effects.split + hậu xử lý."""
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float64)

    intervals = librosa.effects.split(x, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    min_gap = int(sr * min_gap_ms / 1000)
    min_len = int(sr * min_len_ms / 1000)
    intervals = merge_intervals([tuple(seg) for seg in intervals], min_gap, min_len)

    mask = np.zeros(len(x), dtype=bool)
    for s, e in intervals:
        mask[s:e] = True
    return intervals, mask


def _frame_generator_int16(x: np.ndarray, sr: int, frame_ms: int):
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = np.clip(x, -1.0, 1.0)
    pcm16 = (x * 32767.0).astype(np.int16)
    frame_len = int(sr * frame_ms / 1000)
    for i in range(0, len(pcm16) - frame_len + 1, frame_len):
        chunk = pcm16[i : i + frame_len]
        yield i, chunk.tobytes(), frame_len


def vad_webrtc(
    x: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    min_gap_ms: int = 60,
    min_len_ms: int = 100,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """WebRTC VAD, cần pip install webrtcvad."""
    import webrtcvad  # import ở đây để không bắt buộc khi không dùng

    vad = webrtcvad.Vad(aggressiveness)
    voiced_frames, frame_idxs = [], []
    for start_idx, bytes16, frame_len in _frame_generator_int16(x, sr, frame_ms):
        is_voiced = vad.is_speech(bytes16, sample_rate=sr)
        voiced_frames.append(1 if is_voiced else 0)
        frame_idxs.append((start_idx, start_idx + frame_len))

    intervals: List[Tuple[int, int]] = []
    in_seg, seg_start = False, None
    for (s, e), v in zip(frame_idxs, voiced_frames):
        if v and not in_seg:
            in_seg, seg_start = True, s
        elif not v and in_seg:
            intervals.append((seg_start, e))
            in_seg = False
    if in_seg:
        intervals.append((seg_start, frame_idxs[-1][1]))

    min_gap = int(sr * min_gap_ms / 1000)
    min_len = int(sr * min_len_ms / 1000)
    intervals = merge_intervals(intervals, min_gap, min_len)

    mask = np.zeros(len(x), dtype=bool)
    for s, e in intervals:
        mask[s:e] = True
    return intervals, mask


def choose_vad_and_intervals(
    x: np.ndarray,
    sr: int,
    mode: str = "auto",
    top_db: int = 22,
    frame_length: int = 1024,
    hop_length: int = 256,
    aggr: int = 2,
    frame_ms: int = 30,
    min_gap_ms: int = 60,
    min_len_ms: int = 100,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Chọn VAD: energy | webrtc | auto."""
    def _energy():
        return vad_energy(
            x,
            sr,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
            min_gap_ms=min_gap_ms,
            min_len_ms=min_len_ms,
        )

    if mode == "energy":
        return _energy()
    elif mode == "webrtc":
        try:
            return vad_webrtc(
                x,
                sr,
                aggressiveness=aggr,
                frame_ms=frame_ms,
                min_gap_ms=min_gap_ms,
                min_len_ms=min_len_ms,
            )
        except Exception:
            return _energy()
    else:  # auto
        intervals, mask = _energy()
        nonvoice_len = len(x) - mask.sum()
        if nonvoice_len < int(0.1 * sr) or mask.sum() > int(0.95 * len(x)):
            try:
                return vad_webrtc(
                    x,
                    sr,
                    aggressiveness=aggr,
                    frame_ms=frame_ms,
                    min_gap_ms=min_gap_ms,
                    min_len_ms=min_len_ms,
                )
            except Exception:
                pass
        return intervals, mask


# ======================== SNR ========================

def snr_from_percentiles(
    x: np.ndarray,
    frame_length: int = 1024,
    hop_length: int = 256,
    low_q: float = 10.0,
    high_q: float = 90.0,
) -> Optional[float]:
    """Fallback SNR: noise ≈ median của 10% frame năng lượng thấp nhất, speech ≈ median của 10% cao nhất."""
    eng = frame_energy(x, frame_length=frame_length, hop_length=hop_length)
    if len(eng) < 8:
        return None
    low_th = np.percentile(eng, low_q)
    high_th = np.percentile(eng, high_q)
    low_bins = eng[eng <= low_th]
    high_bins = eng[eng >= high_th]
    if len(low_bins) == 0 or len(high_bins) == 0:
        return None
    En = float(np.median(low_bins))
    Es = float(np.median(high_bins))
    if En <= 0 or Es <= 0:
        return None
    return 10.0 * math.log10(Es / En)


def compute_snr_array(
    x: np.ndarray,
    sr: int,
    vad_mode: str = "auto",
    top_db: int = 22,
    aggr: int = 2,
    min_gap_ms: int = 60,
    min_len_ms: int = 100,
) -> Optional[float]:
    """Tính SNR (dB) cho mảng audio mono hoặc stereo."""
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float64)
    x = x - np.mean(x)
    x = np.clip(x, -1.0, 1.0)

    intervals, mask = choose_vad_and_intervals(
        x,
        sr,
        mode=vad_mode,
        top_db=top_db,
        aggr=aggr,
        min_gap_ms=min_gap_ms,
        min_len_ms=min_len_ms,
    )

    # Nếu thiếu non-voice → fallback
    if len(intervals) == 0 or (len(x) - mask.sum()) < int(0.1 * sr):
        return snr_from_percentiles(x)

    speech = x[mask]
    noise = x[~mask]
    if len(speech) == 0 or len(noise) < int(0.05 * sr):
        return snr_from_percentiles(x)

    Es = float(np.median(speech ** 2))
    En = float(np.median(noise ** 2))
    if En <= 0 or Es <= 0:
        return snr_from_percentiles(x)
    return 10.0 * math.log10(Es / En)


def compute_snr_wav(
    wav_path: Path,
    vad_mode: str = "auto",
    top_db: int = 22,
    aggr: int = 2,
    min_gap_ms: int = 60,
    min_len_ms: int = 100,
) -> Optional[float]:
    """Tính SNR (dB) cho file WAV."""
    try:
        x, sr = sf.read(str(wav_path), always_2d=False)
    except Exception:
        x, sr = librosa.load(str(wav_path), sr=None, mono=False)
    return compute_snr_array(
        x,
        sr,
        vad_mode=vad_mode,
        top_db=top_db,
        aggr=aggr,
        min_gap_ms=min_gap_ms,
        min_len_ms=min_len_ms,
    )


# ======================== Multiprocessing (K-part) ========================

@dataclass
class SNRConfig:
    vad_mode: str = "auto"
    top_db: int = 22
    aggr: int = 2
    min_gap_ms: int = 60
    min_len_ms: int = 100
    log_every: int = 50
    log_interval: float = 2.0
    quiet: bool = False


def chunkify(lst: List[Any], k: int) -> List[List[Any]]:
    """Chia lst thành k mảnh gần bằng nhau theo thứ tự."""
    n = len(lst)
    if k <= 1 or n == 0:
        return [lst]
    base = n // k
    rem = n % k
    chunks: List[List[Any]] = []
    start = 0
    for i in range(k):
        extra = 1 if i < rem else 0
        end = start + base + extra
        if start < end:
            chunks.append(lst[start:end])
        start = end
    return chunks


def _fmt_eta(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _process_batch(
    batch_records: List[dict],
    cfg: SNRConfig,
    part_idx: int,
    num_parts: int,
) -> List[dict]:
    """Một process xử lý một batch records, trả về list record kèm snr_db, có in debug."""
    outs: List[dict] = []
    total = len(batch_records)
    if total == 0:
        return outs

    start_t = time.time()
    last_log_t = start_t
    errors = 0

    if not cfg.quiet:
        print(f"[P{part_idx+1}/{num_parts}] START  | {total} items | pid={os.getpid()}", flush=True)

    for i, rec in enumerate(batch_records, 1):
        out = dict(rec)
        try:
            wav = Path(rec["audio_filepath"])
            snr = compute_snr_wav(
                wav,
                vad_mode=cfg.vad_mode,
                top_db=cfg.top_db,
                aggr=cfg.aggr,
                min_gap_ms=cfg.min_gap_ms,
                min_len_ms=cfg.min_len_ms,
            )
            out["snr_db"] = round(float(snr), 3) if (snr is not None and np.isfinite(snr)) else None
        except Exception:
            out["snr_db"] = None
            errors += 1
        outs.append(out)

        # logging
        now = time.time()
        need_log = (i == 1) or (i == total) or (i % cfg.log_every == 0) or ((now - last_log_t) >= cfg.log_interval)
        if (not cfg.quiet) and need_log:
            elapsed = now - start_t
            rate = i / elapsed if elapsed > 0 else 0.0
            remain = (total - i) / rate if rate > 0 else None
            tail = str(rec.get("audio_filepath", ""))[-48:]
            print(
                f"[P{part_idx+1}/{num_parts}] {i:>6}/{total:<6} "
                f"| elapsed={_fmt_eta(elapsed)} "
                f"| eta={_fmt_eta(remain)} "
                f"| rate={rate:5.2f} it/s "
                f"| errs={errors} "
                f"| last=…{tail}",
                flush=True,
            )
            last_log_t = now

    if not cfg.quiet:
        total_elapsed = time.time() - start_t
        print(
            f"[P{part_idx+1}/{num_parts}] DONE   | {total} items | "
            f"elapsed={_fmt_eta(total_elapsed)} | errs={errors}",
            flush=True,
        )

    return outs


def process_manifest(
    manifest_path: Path,
    out_path: Optional[Path],
    vad_mode: str = "auto",
    top_db: int = 22,
    aggr: int = 2,
    min_gap_ms: int = 60,
    min_len_ms: int = 100,
    n_jobs: int = 0,
    log_every: int = 50,
    log_interval: float = 2.0,
    quiet: bool = False,
) -> dict:
    """Phiên bản chia K-part: mỗi process xử lý một batch, kèm debug progress."""
    if out_path is None:
        out_path = manifest_path.with_suffix(".with_snr.jsonl")

    # Đọc records
    records: List[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    cfg = SNRConfig(
        vad_mode=vad_mode,
        top_db=top_db,
        aggr=aggr,
        min_gap_ms=min_gap_ms,
        min_len_ms=min_len_ms,
        log_every=log_every,
        log_interval=log_interval,
        quiet=quiet,
    )

    # Không song song
    if not n_jobs or n_jobs <= 1 or len(records) == 0:
        outputs = _process_batch(records, cfg, part_idx=0, num_parts=1)
    else:
        parts = chunkify(records, n_jobs)
        ctx = get_context("spawn")  # an toàn, tránh lỗi pickling/fork
        args = [(p, cfg, idx, len(parts)) for idx, p in enumerate(parts)]
        with ctx.Pool(processes=len(parts)) as pool:
            results: List[List[dict]] = pool.starmap(_process_batch, args)
        outputs = [o for part_out in results for o in part_out]  # bảo toàn thứ tự theo part

    # Ghi file output
    with open(out_path, "w", encoding="utf-8") as fo:
        for o in outputs:
            fo.write(json.dumps(o, ensure_ascii=False) + "\n")

    # Summary
    vals = [o["snr_db"] for o in outputs if (o.get("snr_db") is not None)]
    summary = {
        "manifest": str(manifest_path),
        "out_file": str(out_path),
        "total": len(records),
        "snr_computed": len(vals),
        "snr_coverage": (len(vals) / len(records)) if records else 0.0,
        "snr_mean_db": float(np.mean(vals)) if vals else None,
        "snr_median_db": float(np.median(vals)) if vals else None,
        "snr_p10_db": float(np.percentile(vals, 10)) if vals else None,
        "snr_p90_db": float(np.percentile(vals, 90)) if vals else None,
    }
    return summary


def write_summary_csv(summaries: List[dict], csv_path: Path):
    if not summaries:
        return
    fields = [
        "manifest",
        "out_file",
        "total",
        "snr_computed",
        "snr_coverage",
        "snr_mean_db",
        "snr_median_db",
        "snr_p10_db",
        "snr_p90_db",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)


# ======================== CLI ========================

def main():
    ap = argparse.ArgumentParser(description="Compute VAD-based SNR for NeMo manifests (K-part multiprocessing, with progress).")
    ap.add_argument("--vad", type=str, default="auto", choices=["auto", "energy", "webrtc"], help="Chọn VAD")
    ap.add_argument("--top-db", type=int, default=22, help="Ngưỡng dB cho energy-VAD")
    ap.add_argument("--aggr", type=int, default=2, help="WebRTC VAD aggressiveness (0..3)")
    ap.add_argument("--min-gap-ms", type=int, default=60, help="Gộp các gap < X ms")
    ap.add_argument("--min-len-ms", type=int, default=100, help="Bỏ đoạn voice < X ms")
    ap.add_argument("--jobs", type=int, default=0, help="Số process song song (0 = single)")
    ap.add_argument("--summary-csv", type=str, default="", help="Đường dẫn xuất CSV tổng hợp cho nhiều manifest")

    # Debug progress
    ap.add_argument("--log-every", type=int, default=50, help="In progress mỗi N bản ghi (mỗi process)")
    ap.add_argument("--log-interval", type=float, default=2.0, help="Tối thiểu X giây giữa 2 lần in (mỗi process)")
    ap.add_argument("--quiet", action="store_true", help="Tắt log tiến trình")

    ap.add_argument("manifests", nargs="+", help="Một hoặc nhiều file manifest .jsonl")
    args = ap.parse_args()

    results = []
    for m in args.manifests:
        mpath = Path(m)
        summ = process_manifest(
            manifest_path=mpath,
            out_path=None,
            vad_mode=args.vad,
            top_db=args.top_db,
            aggr=args.aggr,
            min_gap_ms=args.min_gap_ms,
            min_len_ms=args.min_len_ms,
            n_jobs=args.jobs,
            log_every=args.log_every,
            log_interval=args.log_interval,
            quiet=args.quiet,
        )
        print(json.dumps(summ, ensure_ascii=False, indent=2))
        results.append(summ)

    if args.summary_csv:
        write_summary_csv(results, Path(args.summary_csv))


if __name__ == "__main__":
    main()
