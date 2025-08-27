#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split existing NeMo ASR JSONL manifests into train/dev/test.

- Works directly on your current files like:
    ~/work/public_datasets/vi_small/nemo_manifests/vietspeech/train_*.jsonl

- Two split modes:
    1) hash: deterministic by hashing audio_filepath (default; scalable)
    2) random: shuffle all records with a seed, then cut by ratios

- Optional filters: min/max duration, drop empty texts.

Output:
    <out_dir>/train.jsonl
    <out_dir>/dev.jsonl
    <out_dir>/test.jsonl

Examples
--------
Hash-based (recommended, deterministic):
python split_nemo_manifest.py \
  --in-glob "~/work/public_datasets/vi_small/nemo_manifests/vietspeech/train_*.jsonl" \
  --out-dir  "~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits" \
  --train 0.98 --dev 0.01 --test 0.01

Random-based (if you prefer pure random split):
python split_nemo_manifest.py \
  --in-glob "~/work/public_datasets/vi_small/nemo_manifests/vietspeech/train_*.jsonl" \
  --out-dir  "~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits_random" \
  --mode random --seed 2025 --train 0.98 --dev 0.01 --test 0.01
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import glob
import hashlib
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Any, Optional


def iter_jsonl(files: List[Path]) -> Iterable[Dict[str, Any]]:
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    sys.stderr.write(f"[WARN] Skip bad line in {fp}: {e}\n")


def hash01(s: str) -> float:
    """Stable float in [0,1) from md5(s)."""
    m = hashlib.md5(s.encode("utf-8")).hexdigest()
    # Use 8 bytes (16 hex chars) to form a 64-bit int
    n = int(m[:16], 16)
    return (n & ((1 << 64) - 1)) / float(1 << 64)


def choose_bucket_hash(p: float, train: float, dev: float, test: float) -> str:
    assert abs((train + dev + test) - 1.0) < 1e-6, "Ratios must sum to 1"
    if p < train:
        return "train"
    elif p < train + dev:
        return "dev"
    else:
        return "test"


def parse_args():
    ap = argparse.ArgumentParser(description="Split NeMo JSONL manifests into train/dev/test")
    ap.add_argument("--in-glob", type=str, required=True,
                    help="Glob for input JSONL files (e.g., '~/.../vietspeech/train_*.jsonl')")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for split manifests")
    ap.add_argument("--mode", choices=["hash", "random"], default="hash",
                    help="Split mode (hash: deterministic; random: shuffle with seed)")
    ap.add_argument("--seed", type=int, default=2025, help="Random seed (for mode=random)")
    ap.add_argument("--train", type=float, default=0.98, help="Train ratio")
    ap.add_argument("--dev", type=float, default=0.01, help="Dev ratio")
    ap.add_argument("--test", type=float, default=0.01, help="Test ratio")
    ap.add_argument("--min-duration", type=float, default=0.0, help="Drop samples shorter than this (sec)")
    ap.add_argument("--max-duration", type=float, default=1e9, help="Drop samples longer than this (sec)")
    ap.add_argument("--drop-empty-text", action="store_true", help="Drop records with empty/whitespace text")
    ap.add_argument("--dry-run", action="store_true", help="Compute stats but do not write files")
    return ap.parse_args()


def main():
    args = parse_args()
    assert abs((args.train + args.dev + args.test) - 1.0) < 1e-6, "Ratios must sum to 1"

    in_paths = sorted(Path(p).expanduser() for p in glob.glob(os.path.expanduser(args.in_glob)))
    if not in_paths:
        print(f"[ERR] No input files matched: {args.in_glob}", file=sys.stderr)
        sys.exit(1)

    # Expand ~ to absolute home directory
    args.out_dir = args.out_dir.expanduser()
    args.in_glob = os.path.expanduser(args.in_glob)

    assert abs((args.train + args.dev + args.test) - 1.0) < 1e-6, "Ratios must sum to 1"

    in_paths = sorted(Path(p).expanduser() for p in glob.glob(args.in_glob))
    if not in_paths:
        print(f"[ERR] No input files matched: {args.in_glob}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)


    out_files = {
        "train": args.out_dir / "train.jsonl",
        "dev":   args.out_dir / "dev.jsonl",
        "test":  args.out_dir / "test.jsonl",
    }

    # Collect or stream depending on mode
    counts = {"train": 0, "dev": 0, "test": 0, "dropped": 0, "total": 0}

    if args.mode == "hash":
        fps = {} if args.dry_run else {k: v.open("w", encoding="utf-8") for k, v in out_files.items()}
        try:
            for rec in iter_jsonl(in_paths):
                counts["total"] += 1
                # Basic filters
                dur = float(rec.get("duration", 0.0))
                txt = (rec.get("text") or "").strip()
                if dur < args.min_duration or dur > args.max_duration:
                    counts["dropped"] += 1
                    continue
                if args.drop_empty_text and len(txt) == 0:
                    counts["dropped"] += 1
                    continue

                key = rec.get("audio_filepath") or json.dumps(rec, ensure_ascii=False)
                p = hash01(key)
                bucket = choose_bucket_hash(p, args.train, args.dev, args.test)
                counts[bucket] += 1

                if not args.dry_run:
                    fps[bucket].write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            if not args.dry_run:
                for f in fps.values():
                    f.close()

    else:  # random mode
        # Load all (can be heavier on RAM for very large datasets)
        all_recs: List[Dict[str, Any]] = []
        for rec in iter_jsonl(in_paths):
            counts["total"] += 1
            dur = float(rec.get("duration", 0.0))
            txt = (rec.get("text") or "").strip()
            if dur < args.min_duration or dur > args.max_duration:
                counts["dropped"] += 1
                continue
            if args.drop_empty_text and len(txt) == 0:
                counts["dropped"] += 1
                continue
            all_recs.append(rec)

        random.Random(args.seed).shuffle(all_recs)
        n = len(all_recs)
        n_train = int(round(args.train * n))
        n_dev   = int(round(args.dev   * n))
        n_test  = n - n_train - n_dev

        splits = {
            "train": all_recs[:n_train],
            "dev":   all_recs[n_train:n_train + n_dev],
            "test":  all_recs[n_train + n_dev:],
        }
        for k, v in splits.items():
            counts[k] = len(v)

        if not args.dry_run:
            for k, path in out_files.items():
                with path.open("w", encoding="utf-8") as f:
                    for rec in splits[k]:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Report
    kept = counts["train"] + counts["dev"] + counts["test"]
    print("=== Split Summary ===")
    print(f"Input files    : {len(in_paths)}")
    print(f"Total read     : {counts['total']}")
    print(f"Dropped        : {counts['dropped']}")
    print(f"Kept           : {kept}")
    print(f"  Train        : {counts['train']}")
    print(f"  Dev          : {counts['dev']}")
    print(f"  Test         : {counts['test']}")
    if not args.dry_run:
        print("Outputs:")
        for k, p in out_files.items():
            print(f"  {k:5s} -> {p}")
    print("✅ Done.")
    

if __name__ == "__main__":
    main()
