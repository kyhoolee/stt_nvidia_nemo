#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a NeMo ASR manifest's train set into new train + dev manifests.

- Input: a NeMo JSONL manifest (each line is a JSON with keys like
  audio_filepath, duration, text, sample_rate, dataset)
- Output: two manifests: train_out.jsonl and dev_out.jsonl
- Does NOT move/copy audio files; only splits manifest lines.

Examples:
  python split_train_to_dev.py \
    --train-manifest ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/train.jsonl \
    --train-out     ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/train.jsonl \
    --dev-out       ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/dev.jsonl \
    --dev-ratio 0.05 \
    --seed 1337 \
    --stratify-duration

Tips:
- Use --backup to keep a copy of the original train manifest as *.bak
- Use --dry-run to preview counts without writing files.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

def read_manifest(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_manifest(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def duration_bins(durs: List[float], num_bins: int = 5) -> List[float]:
    """Compute quantile cut points to form approximately equal-sized bins."""
    if not durs:
        return []
    # Simple quantiles
    qs = [i / num_bins for i in range(1, num_bins)]
    sorted_d = sorted(durs)
    cuts = []
    for q in qs:
        idx = int(round(q * (len(sorted_d) - 1)))
        cuts.append(sorted_d[idx])
    return cuts

def assign_bin(d: float, cuts: List[float]) -> int:
    # Return bin index in [0..len(cuts)]
    for i, c in enumerate(cuts):
        if d <= c:
            return i
    return len(cuts)

def stratified_split(
    items: List[Dict[str, Any]],
    dev_ratio: float,
    seed: int,
    num_bins: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split with approximately preserved duration distribution."""
    durs = [float(ex.get("duration", 0.0)) for ex in items]
    cuts = duration_bins(durs, num_bins=num_bins)

    # Group by bins
    buckets = {}
    for ex in items:
        b = assign_bin(float(ex.get("duration", 0.0)), cuts)
        buckets.setdefault(b, []).append(ex)

    rng = random.Random(seed)
    dev, train = [], []
    for b, bucket in buckets.items():
        rng.shuffle(bucket)
        k = max(1, int(round(len(bucket) * dev_ratio))) if len(bucket) > 0 else 0
        dev.extend(bucket[:k])
        train.extend(bucket[k:])

    return train, dev

def simple_split(
    items: List[Dict[str, Any]],
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    k = max(1, int(round(len(items) * dev_ratio))) if len(items) > 0 else 0
    dev_idx = set(idxs[:k])
    dev = [items[i] for i in dev_idx]
    train = [items[i] for i in idxs[k:]]
    return train, dev

def sum_duration(items: List[Dict[str, Any]]) -> float:
    return float(sum(float(ex.get("duration", 0.0)) for ex in items))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True, help="Path to existing train manifest (JSONL)")
    ap.add_argument("--train-out", required=True, help="Path to write NEW train manifest (JSONL)")
    ap.add_argument("--dev-out", required=True, help="Path to write dev manifest (JSONL)")
    ap.add_argument("--dev-ratio", type=float, default=0.05, help="Fraction of train to move to dev [0,1]")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument("--stratify-duration", action="store_true", help="Preserve duration distribution across splits")
    ap.add_argument("--backup", action="store_true", help="Write a backup copy of the original train as *.bak")
    ap.add_argument("--dry-run", action="store_true", help="Only print stats, do not write files")
    args = ap.parse_args()

    train_manifest_path = Path(args.train_manifest).expanduser().resolve()
    train_out_path = Path(args.train_out).expanduser().resolve()
    dev_out_path = Path(args.dev_out).expanduser().resolve()

    print(f"Reading: {train_manifest_path}")
    items = read_manifest(train_manifest_path)
    n = len(items)
    dur_sum = sum_duration(items)
    print(f"Loaded {n} items. Total duration ~ {dur_sum/3600:.2f} hours.")

    if args.stratify_duration:
        new_train, dev = stratified_split(items, args.dev_ratio, args.seed)
    else:
        new_train, dev = simple_split(items, args.dev_ratio, args.seed)

    # Sort optionally to keep deterministic order by audio path (nice-to-have)
    new_train = sorted(new_train, key=lambda x: x.get("audio_filepath", ""))
    dev = sorted(dev, key=lambda x: x.get("audio_filepath", ""))

    print(f"Split result:")
    print(f"  New train: {len(new_train)} items, {sum_duration(new_train)/3600:.2f} h")
    print(f"  Dev      : {len(dev)} items, {sum_duration(dev)/3600:.2f} h")

    if args.dry_run:
        print("Dry-run mode ON. No files written.")
        return

    # Backup original train manifest if requested
    if args.backup:
        backup_path = train_manifest_path.with_suffix(train_manifest_path.suffix + ".bak")
        backup_path.write_text(train_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup original -> {backup_path}")

    # Write outputs
    write_manifest(train_out_path, new_train)
    write_manifest(dev_out_path, dev)
    print(f"Wrote:\n  {train_out_path}\n  {dev_out_path}\n✅ Done.")

if __name__ == "__main__":
    main()
