#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Common Voice 8.0 (Vietnamese) and export to NeMo JSONL manifests with 16kHz WAV audio.

Requirements:
  pip install datasets soundfile numpy tqdm

Usage:
  # Option A: pass your HF token as an env var
  export HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
  python download_commonvoice_vi_to_nemo.py --out-root ~/work/public_datasets/vi_small

  # Option B: pass the token directly
  python download_commonvoice_vi_to_nemo.py --out-root ~/work/public_datasets/vi_small --hf-token hf_XXXXXXXXXXXXXXXX

This will create:
  {out-root}/audio/common_voice_8_0_vi/{train,validation,test}/*.wav
  {out-root}/nemo_manifests/common_voice_8_0_vi/{train,validation,test}.jsonl

Notes:
- Only uses the {train, validation, test} splits by default.
- Audio is written at 16 kHz mono WAV.
- Text is lightly normalized (strip outer quotes; append trailing punctuation). Use --no-text-normalize to disable.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm
import soundfile as sf

PUNCT = {".", "?", "!"}

def normalize_text(t: str) -> str:
    if not t:
        return t
    t = t.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1].strip()
    if t and t[-1] not in PUNCT:
        t = t + "."
    return t

def ensure_mono(audio: np.ndarray) -> np.ndarray:
    # audio shape can be (n,) or (n, ch)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise ValueError(f"Unexpected audio shape: {audio.shape}")

def export_split(ds, split_name: str, out_audio_dir: Path, manifest_path: Path, text_normalize: bool, dataset_tag: str):
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as mf:
        for i in tqdm(range(len(ds)), desc=f"Export {split_name}", unit="utt"):
            ex = ds[i]
            sent = ex.get("sentence", "")
            if text_normalize:
                sent = normalize_text(sent)

            audio = ex["audio"]
            arr = audio["array"]
            sr = audio["sampling_rate"]

            # force mono if needed
            arr = ensure_mono(arr)

            # deterministic filename
            base = f"cv8_vi_{split_name}_{i:08d}.wav"
            out_path = out_audio_dir / base

            # write 16-bit PCM WAV
            sf.write(out_path.as_posix(), arr, sr, subtype="PCM_16")

            duration = len(arr) / float(sr)
            line = {
                "audio_filepath": out_path.as_posix(),
                "duration": round(duration, 3),
                "text": sent,
                "sample_rate": sr,
                "dataset": dataset_tag,
            }
            mf.write(json.dumps(line, ensure_ascii=False) + "\n")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="Root folder to store audio and manifests")
    ap.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_TOKEN", None),
                    help="Hugging Face token (or set HUGGINGFACE_TOKEN env var). You must have access to the gated dataset.")
    ap.add_argument("--language", default="vi", help="Locale/language code (default: vi)")
    ap.add_argument("--sampling-rate", type=int, default=16000, help="Target sampling rate for audio decoding/export")
    ap.add_argument("--splits", default="train,validation,test", help="Comma-separated splits to export")
    ap.add_argument("--no-text-normalize", action="store_true", help="Disable light text normalization")
    ap.add_argument("--max-items", type=int, default=None, help="Debug: limit items per split")
    # NEW: cho phép tự tạo dev nếu validation rỗng
    ap.add_argument("--auto-dev-ratio", type=float, default=0.0,
                    help="If >0 and 'validation' split is empty, create a dev set by sampling this ratio from train (e.g., 0.05).")
    ap.add_argument("--auto-dev-seed", type=int, default=1337, help="Random seed for auto-dev sampling")
    args = ap.parse_args()

    if not args.hf_token:
        raise SystemExit("❌ Missing HF token. Provide --hf-token or set HUGGINGFACE_TOKEN env var.")

    out_root = Path(os.path.expanduser(args.out_root)).resolve()
    dataset_tag = f"common_voice_8_0_{args.language}"
    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    text_normalize = not args.no_text_normalize

    print(f"Out root: {out_root}")
    print(f"Dataset tag: {dataset_tag}")
    print(f"Splits (requested): {requested_splits}")
    print("Loading dataset from Hugging Face...")

    # Helper to load 1 split, trả về None nếu split rỗng
    def load_split_or_none(split_name: str):
        try:
            ds_local = load_dataset(
                "mozilla-foundation/common_voice_8_0",
                args.language,
                split=split_name,
                token=args.hf_token,
                trust_remote_code=True
            )
            # Nếu thực sự rỗng, datasets sẽ ném ValueError trước đó.
            return ds_local
        except ValueError as e:
            if "corresponds to no data" in str(e):
                print(f"⚠️  Split '{split_name}' is empty. Skipping.")
                return None
            raise

    # 1) Export các split có dữ liệu
    available_splits = {}
    for split in requested_splits:
        print(f"\n=== Split: {split} ===")
        ds = load_split_or_none(split)
        if ds is None:
            available_splits[split] = 0
            continue

        ds = ds.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
        if args.max_items is not None:
            ds = ds.select(range(min(args.max_items, len(ds))))

        out_audio_dir = out_root / "audio" / dataset_tag / split
        manifest_path = out_root / "nemo_manifests" / dataset_tag / f"{split}.jsonl"
        export_split(ds, split, out_audio_dir, manifest_path, text_normalize, dataset_tag)
        available_splits[split] = len(ds)

    # 2) Nếu validation rỗng và user muốn auto-dev từ train
    need_auto_dev = ("validation" in requested_splits and available_splits.get("validation", 0) == 0 and args.auto_dev_ratio > 0.0)

    if need_auto_dev:
        print("\n=== Auto-create validation from train ===")
        # load lại full train (không cắt theo max_items để đủ dữ liệu cho sampling)
        ds_train_full = load_split_or_none("train")
        if ds_train_full is None or len(ds_train_full) == 0:
            print("❌ Cannot create auto-dev: train split is empty.")
        else:
            # resample + chọn ngẫu nhiên theo tỉ lệ
            ds_train_full = ds_train_full.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
            import random
            random.seed(args.auto_dev_seed)
            idxs = list(range(len(ds_train_full)))
            random.shuffle(idxs)
            k = max(1, int(len(ds_train_full) * args.auto_dev_ratio))
            dev_indices = idxs[:k]
            ds_dev = ds_train_full.select(dev_indices)

            if args.max_items is not None:
                ds_dev = ds_dev.select(range(min(args.max_items, len(ds_dev))))

            out_audio_dir = out_root / "audio" / dataset_tag / "validation"
            manifest_path = out_root / "nemo_manifests" / dataset_tag / "validation.jsonl"
            export_split(ds_dev, "validation", out_audio_dir, manifest_path, text_normalize, dataset_tag)
            print(f"✅ Auto-dev created: {len(ds_dev)} items.")

    print("\n✅ Done.")
    print(f"Audio root: {out_root / 'audio' / dataset_tag}")
    print(f"Manifests:  {out_root / 'nemo_manifests' / dataset_tag}")

if __name__ == "__main__":
    main()
