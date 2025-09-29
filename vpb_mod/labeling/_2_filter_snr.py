#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lọc manifest theo percentile SNR.
Ví dụ: giữ top 10% file có snr_db cao nhất.
"""

import argparse, json
import numpy as np
from pathlib import Path

def filter_manifest(manifest_path: Path, out_path: Path, percentile: float):
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "snr_db" in rec and rec["snr_db"] is not None:
                records.append(rec)

    if not records:
        print(f"[WARN] {manifest_path}: không có record nào có snr_db.")
        return

    snrs = np.array([r["snr_db"] for r in records])
    thr = np.percentile(snrs, percentile)
    print(f"[INFO] {manifest_path}: percentile={percentile}, threshold={thr:.3f} dB")

    # lọc
    filtered = [r for r in records if r["snr_db"] >= thr]

    with open(out_path, "w", encoding="utf-8") as fo:
        for r in filtered:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] {len(filtered)}/{len(records)} records written to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=str, help="Input manifest .jsonl (đã có snr_db)")
    ap.add_argument("--percentile", type=float, default=90,
                    help="Giữ các record có snr_db >= percentile này (default=90)")
    ap.add_argument("--out", type=str, default="", help="Output file (default = *.top.jsonl)")
    args = ap.parse_args()

    in_path = Path(args.manifest)
    out_path = Path(args.out) if args.out else in_path.with_suffix(".top.jsonl")
    filter_manifest(in_path, out_path, args.percentile)
