#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPB NeMo manifest: enrich + split by CLID (no time)

- Chỉ tránh leakage theo người dùng (phone_number/CLID)
- Chia group-wise theo CLID (fallback: audio_name nếu thiếu CLID)
- Xuất 3 biến thể: all / right_only / left_only

Usage
-----
python vpb_manifest_split_by_clid.py \
  --in /path/to/label_batch_092025 \
  --out-dir /path/to/splits_by_clid_tripack \
  --train-ratio 0.90 --val-ratio 0.05 --test-ratio 0.05 \
  --seed 42
"""

import json
import gzip
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re

# ---------- Regex ----------
RE_UTT = re.compile(
    r'^(?P<audio_name>.+)___(?P<start_ms>\d{6,})___(?P<channel>left|right)___(?P<end_ms>\d{6,})\.wav$'
)
RE_AUDIO_NAME = re.compile(
    r'^E_(?P<agent_username>[A-Za-z0-9]+)_D_(?P<date>\d{4}-\d{2}-\d{2})_H_(?P<time_hms>\d{6})_(?P<time_ms3>\d{3})_CLID_(?P<clid>\d+)$'
)

def _open_reader(fp: Path):
    return gzip.open(fp, "rt", encoding="utf-8") if str(fp).endswith(".gz") else open(fp, "r", encoding="utf-8")

def _open_writer(fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(fp, "wt", encoding="utf-8") if str(fp).endswith(".gz") else open(fp, "w", encoding="utf-8")

def parse_utt_id(utt_id: str) -> Optional[Dict]:
    m1 = RE_UTT.match(utt_id)
    if not m1:
        return None
    audio_name = m1.group("audio_name")
    start_ms   = int(m1.group("start_ms"))
    end_ms     = int(m1.group("end_ms"))
    channel    = m1.group("channel")

    m2 = RE_AUDIO_NAME.match(audio_name)
    if not m2:
        return {
            "audio_name": audio_name,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "channel": channel,
            "seg_ms": end_ms - start_ms,
            # thiếu CLID → sẽ fallback group theo audio_name
            "agent_username": None, "date": None, "time_hms": None, "time_ms3": None, "clid": None,
            "call_dt": None,
            "call_key": f"{audio_name}|None|None|None",
        }

    agent_username = m2.group("agent_username")
    date_str       = m2.group("date")
    time_hms       = m2.group("time_hms")
    time_ms3       = m2.group("time_ms3")
    clid           = m2.group("clid")
    try:
        call_dt_iso = datetime.strptime(f"{date_str} {time_hms}", "%Y-%m-%d %H%M%S").isoformat()
    except Exception:
        call_dt_iso = None

    return {
        "audio_name": audio_name,
        "agent_username": agent_username,
        "date": date_str,
        "time_hms": time_hms,
        "time_ms3": time_ms3,
        "clid": clid,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "channel": channel,
        "seg_ms": end_ms - start_ms,
        "call_dt": call_dt_iso,
        "call_key": f"{agent_username}|{date_str}|{time_hms}|{clid}",
    }

def enrich_manifest(in_fp: Path) -> List[Dict]:
    rows: List[Dict] = []
    with _open_reader(in_fp) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            parsed = parse_utt_id(obj.get("utt_id", "") or obj.get("audio_id", ""))
            if not parsed:
                continue
            obj.update(parsed)
            rows.append(obj)
    return rows

def _assign_groups_by_clid(rows: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """
    Group-wise split theo CLID. Nếu CLID None → dùng audio_name để group,
    để vẫn không rò rỉ trong cùng 1 cuộc gọi.
    """
    # collect groups
    group_to_rows: Dict[str, List[Dict]] = {}
    for r in rows:
        g = r.get("clid") or f"CALL::{r.get('audio_name')}"
        group_to_rows.setdefault(g, []).append(r)

    groups = list(group_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    # Normalize ratios just in case
    s = train_ratio + val_ratio + test_ratio
    train_ratio_n, val_ratio_n, test_ratio_n = (train_ratio/s, val_ratio/s, test_ratio/s)

    n_test  = int(round(test_ratio_n * n))
    n_val   = int(round(val_ratio_n  * n))
    n_train = n - n_test - n_val
    if n_train < 0:  # guard
        n_train = max(0, n - n_test - n_val)

    test_groups = set(groups[:n_test])
    val_groups  = set(groups[n_test:n_test+n_val])
    train_groups= set(groups[n_test+n_val:])

    split = {"train": [], "val": [], "test": []}
    for g in groups:
        bucket = "train" if g in train_groups else ("val" if g in val_groups else "test")
        split[bucket].extend(group_to_rows[g])
    return split

def _write_jsonl(rows: List[Dict], fp: Path):
    with _open_writer(fp) as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def _filter_channel(rows: List[Dict], channel: Optional[str]) -> List[Dict]:
    if not channel:
        return rows
    ch = channel.lower()
    if ch not in {"left", "right"}:
        return rows
    return [r for r in rows if r.get("channel") == ch]

def run(in_path: Path, out_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) Enrich once
    rows = enrich_manifest(in_path)

    # 2) Group-wise split by CLID (fallback audio_name)
    split_all = _assign_groups_by_clid(rows, train_ratio, val_ratio, test_ratio, seed)

    # 3) Write tripack
    # 3.1 all
    out_all = out_dir / "all"
    _write_jsonl(split_all["train"], out_all / "train.jsonl")
    _write_jsonl(split_all["val"],   out_all / "val.jsonl")
    _write_jsonl(split_all["test"],  out_all / "test.jsonl")

    # 3.2 right_only (user)
    out_right = out_dir / "right_only"
    _write_jsonl(_filter_channel(split_all["train"], "right"), out_right / "train.jsonl")
    _write_jsonl(_filter_channel(split_all["val"],   "right"), out_right / "val.jsonl")
    _write_jsonl(_filter_channel(split_all["test"],  "right"), out_right / "test.jsonl")

    # 3.3 left_only (agent)
    out_left = out_dir / "left_only"
    _write_jsonl(_filter_channel(split_all["train"], "left"), out_left / "train.jsonl")
    _write_jsonl(_filter_channel(split_all["val"],   "left"), out_left / "val.jsonl")
    _write_jsonl(_filter_channel(split_all["test"],  "left"), out_left / "test.jsonl")

    # 4) report
    print("[SPLIT BY CLID] (group-wise, no time)")
    print(f"  groups = CLID (fallback CALL::<audio_name>)")
    print(f"  ratios(train/val/test) = {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f} (seed={seed})")
    print(f"  all        : train={len(split_all['train'])}, val={len(split_all['val'])}, test={len(split_all['test'])}")
    print(f"  right_only : train={len(_filter_channel(split_all['train'],'right'))}, "
          f"val={len(_filter_channel(split_all['val'],'right'))}, "
          f"test={len(_filter_channel(split_all['test'],'right'))}")
    print(f"  left_only  : train={len(_filter_channel(split_all['train'],'left'))}, "
          f"val={len(_filter_channel(split_all['val'],'left'))}, "
          f"test={len(_filter_channel(split_all['test'],'left'))}")

def build_cli():
    ap = argparse.ArgumentParser(description="VPB NeMo manifest: enrich + split by CLID (no time)")
    ap.add_argument("--in", dest="infile", required=True, help="Input JSONL manifest path")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--train-ratio", type=float, default=0.90)
    ap.add_argument("--val-ratio",   type=float, default=0.05)
    ap.add_argument("--test-ratio",  type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    return ap

def main():
    ap = build_cli()
    args = ap.parse_args()
    run(
        in_path=Path(args.infile),
        out_dir=Path(args.out_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
