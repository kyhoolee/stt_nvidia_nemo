
#!/usr/bin/env python3
'''
Merge multiple NeMo JSONL manifests from different datasets into unified train/dev/test splits.

Usage examples:
  python merge_manifests.py \
      --manifest-root ~/work/public_datasets/vi_small/nemo_manifests \
      --datasets fpt_fosd infore lsvsc vais1000 vietmed vivos vlsp2020 \
      --train-files  fpt_fosd/fpt_fosd_train.jsonl infore/infore_train.jsonl lsvsc/lsvsc_train.jsonl vais1000/vais1000_train.jsonl vietmed/vietmed_train.jsonl vivos/vivos_train.jsonl vlsp2020/vlsp2020_train.jsonl \
      --dev-files    lsvsc/lsvsc_dev.jsonl vietmed/vietmed_dev.jsonl \
      --test-files   lsvsc/lsvsc_test.jsonl vietmed/vietmed_test.jsonl vivos/vivos_test.jsonl \
      --out-dir      ~/work/public_datasets/vi_small/nemo_manifests_merged \
      --seed 20250819 \
      --max-seconds-per-split 0 \
      --shuffle

Notes:
- If a split has no --*_files, it will be left empty.
- Use --max-per-dataset to cap per-dataset items on each split to reduce dominance.
- Use --max-seconds-per-split to cap total audio seconds for each split (0 = no cap).
- You can filter by duration using --min-dur and --max-dur (in seconds; 0 = ignore).
'''

import argparse
import json
import os
import random
from pathlib import Path
from collections import defaultdict

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def load_split(manifest_root, file_list, dataset_tag, min_dur, max_dur):
    out = []
    for rel in file_list:
        p = Path(os.path.expanduser(manifest_root)) / rel
        ds_name = dataset_tag.get(rel, None)
        for ex in read_jsonl(p):
            dur = float(ex.get("duration", 0))
            if min_dur and dur < min_dur:
                continue
            if max_dur and dur > max_dur:
                continue
            # annotate dataset if absent
            if "dataset" not in ex:
                ex["dataset"] = ds_name or rel.split("/")[0]
            out.append(ex)
    return out

def cap_by_dataset(examples, max_per_dataset=0):
    if max_per_dataset <= 0:
        return examples
    buckets = defaultdict(list)
    for ex in examples:
        buckets[ex.get("dataset", "unknown")].append(ex)
    capped = []
    for ds, arr in buckets.items():
        if max_per_dataset > 0 and len(arr) > max_per_dataset:
            capped.extend(arr[:max_per_dataset])
        else:
            capped.extend(arr)
    return capped

def cap_by_seconds(examples, max_seconds=0.0):
    if max_seconds <= 0:
        return examples
    tot = 0.0
    out = []
    for ex in examples:
        d = float(ex.get("duration", 0))
        if tot + d <= max_seconds:
            out.append(ex)
            tot += d
        else:
            break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-root", required=True, help="Root folder containing dataset manifest subfolders")
    ap.add_argument("--datasets", nargs="*", default=[], help="Optional list of dataset names (for reference only)")
    ap.add_argument("--train-files", nargs="*", default=[], help="Relative JSONL paths for TRAIN split")
    ap.add_argument("--dev-files",   nargs="*", default=[], help="Relative JSONL paths for DEV/VAL split")
    ap.add_argument("--test-files",  nargs="*", default=[], help="Relative JSONL paths for TEST split")
    ap.add_argument("--out-dir", required=True, help="Output dir for merged manifests")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 to skip shuffle unless --shuffle used)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle each split before optional capping by seconds")
    ap.add_argument("--max-per-dataset", type=int, default=0, help="Cap items per dataset per split (0 = unlimited)")
    ap.add_argument("--max-seconds-per-split", type=float, default=0, help="Cap total seconds per split (0 = unlimited)")
    ap.add_argument("--min-dur", type=float, default=0, help="Filter: min duration in seconds (0 = ignore)")
    ap.add_argument("--max-dur", type=float, default=0, help="Filter: max duration in seconds (0 = ignore)")
    args = ap.parse_args()

    # Annotate dataset tag per file (first path component as dataset)
    dataset_tag = {}
    for rel in args.train_files + args.dev_files + args.test_files:
        dataset_tag[rel] = rel.split("/")[0] if "/" in rel else "unknown"

    splits = {
        "train": load_split(args.manifest_root, args.train_files, dataset_tag, args.min_dur, args.max_dur),
        "dev":   load_split(args.manifest_root, args.dev_files,   dataset_tag, args.min_dur, args.max_dur),
        "test":  load_split(args.manifest_root, args.test_files,  dataset_tag, args.min_dur, args.max_dur),
    }

    if args.shuffle:
        rng = random.Random(args.seed if args.seed != 0 else None)
        for k in splits:
            rng.shuffle(splits[k])

    # Apply per-dataset cap, then total seconds cap
    for k in list(splits.keys()):
        splits[k] = cap_by_dataset(splits[k], args.max_per_dataset)
        splits[k] = cap_by_seconds(splits[k], args.max_seconds_per_split)

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, arr in splits.items():
        out_path = out_dir / f"merged_{k}.jsonl"
        write_jsonl(out_path, arr)
        hours = sum(float(x.get("duration", 0)) for x in arr) / 3600.0
        print(f"[{k}] {len(arr)} items | {hours:.2f} hours -> {out_path}")

if __name__ == "__main__":
    main()
