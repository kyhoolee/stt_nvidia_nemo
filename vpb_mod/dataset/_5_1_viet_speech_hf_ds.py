#!/usr/bin/env python3
"""
Download + convert the gated HuggingFace dataset NhutP/VietSpeech into
NeMo-compatible JSONL manifests and 16kHz mono WAVs.

Output layout (default):
  <out_root>/audio/vietspeech/<split>/<shard>/xxxx.wav
  <out_root>/nemo_manifests/vietspeech/<split>_<k>.jsonl  (sharded)

Each JSONL line (NeMo ASR format):
{"audio_filepath": "/abs/path.wav", "duration": 12.345, "text": "...",
 "sample_rate": 16000, "dataset": "vietspeech"}

Notes
-----
1) This dataset is gated. First:
     $ huggingface-cli login
   or set env var HF_TOKEN=<your_token>. See: https://huggingface.co/docs/hub/security-tokens

2) You can also run with --streaming to avoid downloading all metadata at once.
   (Still writes WAVs + manifests to disk.)

3) Resume-safe: existing WAV files will be skipped by default unless --overwrite is set.

4) Dependencies:
     pip install datasets soundfile numpy tqdm torchaudio

5) For speed on large exports, increase --num-workers.

Example
-------
python download_vietspeech_to_nemo.py \
  --out-root ~/work/public_datasets/vi_small \
  --split train \
  --manifest-shard-size 250000 \
  --num-workers 8

"""
from __future__ import annotations
import argparse
import os
import sys
import json
import math
import shutil
from pathlib import Path
from functools import partial
from typing import Optional, Dict, Any

import numpy as np
from tqdm import tqdm

import soundfile as sf

try:
    import torchaudio
    TORCHAUDIO_OK = True
except Exception:
    TORCHAUDIO_OK = False

from datasets import load_dataset, Audio, IterableDataset, Dataset


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_wav(array: np.ndarray, sr: int, out_path: Path, target_sr: int = 16000) -> float:
    """Write mono 16k WAV. Returns duration (seconds)."""
    # Ensure mono: average channels if needed
    if array.ndim == 2:
        array = array.mean(axis=0)
    # Resample if needed
    if sr != target_sr:
        if TORCHAUDIO_OK:
            import torch
            wav = torch.tensor(array, dtype=torch.float32).unsqueeze(0)  # [1, T]
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler(wav)
            array = wav.squeeze(0).numpy()
            sr = target_sr
        else:
            # Fallback: naive linear interpolation
            duration = array.shape[-1] / sr
            new_len = int(round(duration * target_sr))
            xp = np.linspace(0.0, 1.0, array.shape[-1], endpoint=False)
            xq = np.linspace(0.0, 1.0, new_len, endpoint=False)
            array = np.interp(xq, xp, array).astype(np.float32)
            sr = target_sr

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path.as_posix(), array, sr, subtype="PCM_16")
    duration = float(len(array) / sr)
    return duration


def _process_one(idx: int, example: Dict[str, Any], split: str, audio_root: Path,
                 overwrite: bool, shard_mod: int, start_index: int) -> Optional[Dict[str, Any]]:
    """Convert one HF row -> write WAV -> return NeMo manifest dict."""
    # Read audio
    audio = example["audio"]  # {'path':..., 'array': np.ndarray, 'sampling_rate': int}
    text = example.get("transcription", "").strip()
    arr = audio["array"]
    sr = int(audio["sampling_rate"])

    # Path scheme: shard folders to keep dirs manageable
    global_idx = start_index + idx
    shard = f"shard_{(global_idx // shard_mod):04d}"
    wav_name = f"vietspeech_{split}_{global_idx:09d}.wav"
    out_wav = audio_root / split / shard / wav_name

    if out_wav.exists() and not overwrite:
        # Duration from file if exists; cheap parse (skip, approximate)
        try:
            f = sf.SoundFile(out_wav.as_posix())
            duration = float(len(f) / f.samplerate)
        except Exception:
            # If failing, rewrite
            duration = write_wav(arr, sr, out_wav, target_sr=16000)
    else:
        duration = write_wav(arr, sr, out_wav, target_sr=16000)

    return {
        "audio_filepath": str(out_wav.resolve()),
        "duration": round(duration, 3),
        "text": text,
        "sample_rate": 16000,
        "dataset": "vietspeech",
    }


def _iter_hf(ds, desc: str):
    if isinstance(ds, IterableDataset):
        return tqdm(ds, desc=desc, unit="ex")
    else:
        return tqdm(range(len(ds)), desc=desc, unit="ex")


def export_split(
    split: str,
    out_root: Path,
    manifest_shard_size: int = 200_000,
    streaming: bool = False,
    num_workers: int = 4,
    limit: Optional[int] = None,
    overwrite: bool = False,
    start_index: int = 0,
    hf_token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
):
    """Export one split to WAV + sharded NeMo manifests."""
    # Load HF dataset
    load_kwargs = dict(
        name=None,
        split=split,
        streaming=streaming,
        token=hf_token,
    )
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset("NhutP/VietSpeech", **load_kwargs)

    # Cast audio to 16k mono decoding
    if isinstance(ds, IterableDataset):
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    else:
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    audio_root = out_root / "audio" / "vietspeech"
    mani_root = out_root / "nemo_manifests" / "vietspeech"
    _ensure_dir(audio_root / split)
    _ensure_dir(mani_root)

    # Shard writing
    shard_idx = 0
    written_in_shard = 0
    shard_path = mani_root / f"{split}_{shard_idx:03d}.jsonl"
    cur_fp = open(shard_path, "a", encoding="utf-8")

    # Worker pool (threaded; soundfile is C so GIL is less problematic)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    shard_mod = 5000  # directory fanout

    # Iterate
    total = (None if isinstance(ds, IterableDataset) else len(ds))
    iterator = _iter_hf(ds, desc=f"Export {split}")

    # Helper to fetch row by index for map-like usage
    def get_row(i_or_row):
        if isinstance(ds, IterableDataset):
            return i_or_row
        else:
            i = i_or_row
            return ds[i]

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        count = 0
        for i_or_row in iterator:
            row = get_row(i_or_row)
            fut = ex.submit(
                _process_one,
                idx=count,
                example=row,
                split=split,
                audio_root=audio_root,
                overwrite=overwrite,
                shard_mod=shard_mod,
                start_index=start_index,
            )
            futures.append(fut)
            count += 1
            if limit is not None and count >= limit:
                break

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Writing manifests", unit="ex"):
            rec = fut.result()
            if rec is None:
                continue
            # Rotate shard file
            nonlocal_written = written_in_shard + 1
            if nonlocal_written > manifest_shard_size:
                cur_fp.close()
                shard_idx += 1
                written_in_shard = 0
                shard_path = mani_root / f"{split}_{shard_idx:03d}.jsonl"
                cur_fp = open(shard_path, "a", encoding="utf-8")
            # Write line
            cur_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written_in_shard += 1

    cur_fp.close()


def main():
    p = argparse.ArgumentParser(description="Export NhutP/VietSpeech to NeMo manifests + WAVs")
    p.add_argument("--out-root", type=Path, required=True, help="Output root directory")
    p.add_argument("--split", type=str, default="train", help="HF split: e.g., train (only split available)")
    p.add_argument("--manifest-shard-size", type=int, default=200_000, help="Lines per JSONL shard")
    p.add_argument("--num-workers", type=int, default=4, help="Parallel audio writers")
    p.add_argument("--limit", type=int, default=None, help="For debugging; limit number of examples")
    p.add_argument("--overwrite", action="store_true", help="Rewrite existing WAVs")
    p.add_argument("--start-index", type=int, default=0, help="Global index offset if merging runs")
    p.add_argument("--streaming", action="store_true", help="Use HF streaming mode")
    p.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"), help="HF token for gated dataset")

    args = p.parse_args()

    export_split(
        split=args.split,
        out_root=args.out_root,
        manifest_shard_size=args.manifest_shard_size,
        streaming=args.streaming,
        num_workers=args.num_workers,
        limit=args.limit,
        overwrite=args.overwrite,
        start_index=args.start_index,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
    )

    print("✅ Done.")


if __name__ == "__main__":
    main()
