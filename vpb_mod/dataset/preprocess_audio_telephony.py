#!/usr/bin/env python3
"""
Downsample audio to 8kHz (telephony simulation), then upsample back to 16kHz for model input.
Updates the NeMo JSONL manifest with new audio paths and sample_rate=16000. Duration is recomputed.

Usage example:
  python preprocess_audio_telephony.py \
      --manifest ~/work/public_datasets/vi_small/nemo_manifests/vivos/vivos_test.jsonl \
      --output-audio-root ~/work/public_datasets/vi_small/audio_telephony_sim \
      --output-manifest   ~/work/public_datasets/vi_small/nemo_manifests_processed/vivos/vivos_test.jsonl \
      --num-workers 8 --chunksize 16 --skip-existing
"""

import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pool
from typing import Tuple, Dict, Any

import torchaudio
from tqdm import tqdm

TARGET_LOW_SR = 8000
FINAL_SR = 16000


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _derive_rel_path(in_path: Path) -> Path:
    """
    Mirror cấu trúc dưới anchor 'vi_small' nếu có, ví dụ:
      /.../vi_small/vivos/test/audio/utt.wav -> vivos/test/audio/utt.wav
    Ngược lại chỉ dùng tên file.
    """
    parts = list(in_path.resolve().parts)
    if "vi_small" in parts:
        anchor = parts.index("vi_small")
        return Path(*parts[anchor + 1 :])
    return Path(in_path.name)


def _process_one(job) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns:
      (True, new_item)  if success
      (False, {"error": "...", "index": i, "source": "...", "dest": "..."}) if failed
    """
    i, item, out_root_str, force_mono = job
    out_root = Path(os.path.expanduser(out_root_str))

    try:
        in_path = Path(os.path.expanduser(item["audio_filepath"]))
        if not in_path.exists():
            return False, {
                "index": i,
                "error": f"Input not found: {in_path}",
                "source": str(in_path),
            }

        rel = _derive_rel_path(in_path)
        out_wav_path = out_root / rel
        ensure_dir(out_wav_path)

        # Nếu file output đã tồn tại thì bỏ qua xử lý nặng để tiết kiệm thời gian
        if out_wav_path.exists():
            # Tải meta để tính lại duration chính xác từ file out (phòng khi lệch)
            wav_16k, sr_out = torchaudio.load(str(out_wav_path))
            if sr_out != FINAL_SR:
                return False, {
                    "index": i,
                    "error": f"Existing output has wrong SR={sr_out}, expected {FINAL_SR}",
                    "source": str(in_path),
                    "dest": str(out_wav_path),
                }
            if force_mono and wav_16k.shape[0] > 1:
                wav_16k = wav_16k.mean(dim=0, keepdim=True)
                torchaudio.save(str(out_wav_path), wav_16k, FINAL_SR)

            num_samples = wav_16k.shape[-1]
            duration = float(num_samples) / float(FINAL_SR)
        else:
            # Load
            wav, sr = torchaudio.load(str(in_path))

            # Mono hóa nếu cần
            if force_mono and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # Downsample -> 8k
            if sr != TARGET_LOW_SR:
                wav_8k = torchaudio.functional.resample(wav, sr, TARGET_LOW_SR)
            else:
                wav_8k = wav

            # Upsample -> 16k
            if TARGET_LOW_SR != FINAL_SR:
                wav_16k = torchaudio.functional.resample(wav_8k, TARGET_LOW_SR, FINAL_SR)
            else:
                wav_16k = wav_8k

            # Save result (16k) to new location
            torchaudio.save(str(out_wav_path), wav_16k, FINAL_SR)
            num_samples = wav_16k.shape[-1]
            duration = float(num_samples) / float(FINAL_SR)

        # Update JSON entry
        new_item = dict(item)
        new_item["audio_filepath"] = str(out_wav_path)
        new_item["sample_rate"] = FINAL_SR
        new_item["duration"] = duration
        return True, new_item

    except Exception as e:
        return False, {
            "index": i,
            "error": f"{type(e).__name__}: {e}",
            "source": item.get("audio_filepath", "<unknown>"),
            "dest": str(out_root),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Input NeMo JSONL manifest path")
    ap.add_argument("--output-audio-root", required=True, help="Root for processed audio (mirrors subfolders)")
    ap.add_argument("--output-manifest", required=True, help="Output manifest path to write")
    ap.add_argument("--num-workers", type=int, default=4, help="Parallel workers")
    ap.add_argument("--chunksize", type=int, default=16, help="Multiprocessing imap chunksize")
    ap.add_argument("--skip-existing", action="store_true", help="(Deprecated – kept for compatibility)")
    ap.add_argument("--force-mono", action="store_true", help="Convert to mono by averaging channels")
    args = ap.parse_args()

    in_manifest = Path(os.path.expanduser(args.manifest))
    out_manifest = Path(os.path.expanduser(args.output_manifest))
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    # Read input lines
    items = []
    with open(in_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    jobs = [(i, items[i], args.output_audio_root, args.force_mono) for i in range(len(items))]

    successes = []
    failures = []

    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            for ok, payload in tqdm(
                pool.imap(_process_one, jobs, chunksize=args.chunksize),
                total=len(jobs),
                desc="Processing",
            ):
                if ok:
                    successes.append(payload)
                else:
                    failures.append(payload)
    else:
        for j in tqdm(jobs, desc="Processing"):
            ok, payload = _process_one(j)
            if ok:
                successes.append(payload)
            else:
                failures.append(payload)

    # Write updated manifest (only successes)
    with open(out_manifest, "w", encoding="utf-8") as f:
        for it in successes:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Summary
    print(f"\nWrote {len(successes)} entries to {out_manifest}")
    if failures:
        print(f"⚠️  {len(failures)} failures (first 10 shown):")
        for err in failures[:10]:
            print(f'  - idx={err.get("index")} src={err.get("source")} err="{err.get("error")}"')
    else:
        print("✅ No failures")

if __name__ == "__main__":
    main()
