#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tính corpus WER bằng jiwer.wer cho model gốc (original_text) so với human text trong các split cũ.

Inputs:
  --new-manifest  : File JSONL (gộp) có original_text hoặc thư mục chứa nhiều JSONL per-batch.
  --splits-root   : Thư mục có cấu trúc:
                      all/{train.jsonl,val.jsonl,test.jsonl}
                      left_only/{train.jsonl,val.jsonl,test.jsonl}
                      right_only/{train.jsonl,val.jsonl,test.jsonl}
  --out           : (tuỳ chọn) ghi TSV summary.

Chuẩn hoá:
  - MẶC ĐỊNH: chỉ lowercase (khớp code bạn).
  - Tuỳ chọn:
      --normalize-strong : dùng pipeline jiwer (lowercase + remove punctuation + collapse spaces).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict
import sys

from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] skip bad json in {path}: {e}", file=sys.stderr)

def load_manifest_with_original(manifest: Path) -> Dict[str, str]:
    """
    Nếu 'manifest' là thư mục -> đọc tất cả *.jsonl trong đó (không đệ quy).
    Trả về: utt_id -> original_text
    """
    files: List[Path] = []
    if manifest.is_dir():
        files = sorted(manifest.glob("*.jsonl"))
    else:
        files = [manifest]
    utt2orig: Dict[str, str] = {}
    n = 0
    for fp in files:
        for rec in read_jsonl(fp):
            uid = rec.get("utt_id")
            if not uid:
                continue
            utt2orig[uid] = rec.get("original_text", "") or ""
            n += 1
    print(f"[INFO] Loaded {n} items with original_text from {len(files)} file(s).")
    return utt2orig

def find_split_files(splits_root: Path) -> List[Tuple[str, Path]]:
    results = []
    for group in ["all", "left_only", "right_only"]:
        gdir = splits_root / group
        if not gdir.is_dir():
            continue
        for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            fp = gdir / name
            if fp.exists():
                results.append((f"{group}/{name}", fp))
    if not results:
        for fp in splits_root.rglob("*.jsonl"):
            rel = fp.relative_to(splits_root).as_posix()
            results.append((rel, fp))
    return sorted(results)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-manifest", type=Path, required=True,
                    help="File JSONL (gộp) có original_text hoặc thư mục chứa nhiều JSONL.")
    ap.add_argument("--splits-root", type=Path, required=True,
                    help="Thư mục splits (all/, left_only/, right_only/...).")
    ap.add_argument("--out", type=Path, default=None, help="(Optional) Ghi TSV summary.")
    ap.add_argument("--normalize-strong", action="store_true",
                    help="Lowercase + remove punctuation + collapse spaces (jiwer Compose).")
    args = ap.parse_args()

    if not args.new_manifest.exists():
        ap.error(f"--new-manifest not found: {args.new_manifest}")
    if not args.splits_root.exists():
        ap.error(f"--splits-root not found: {args.splits_root}")

    # Chọn transform để khớp code bạn: mặc định chỉ lowercase
    if args.normalize_strong:
        truth_tf = hypothesis_tf = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    else:
        truth_tf = hypothesis_tf = Compose([ToLowerCase()])

    utt2orig = load_manifest_with_original(args.new_manifest)
    split_files = find_split_files(args.splits_root)
    if not split_files:
        ap.error(f"No split jsonl found under: {args.splits_root}")

    rows = []
    group_aggr = defaultdict(lambda: {"pred": [], "ref": [], "matched": 0, "missing": 0})

    for label, fp in split_files:

        refs, hyps = [], []
        matched = missing = skipped_empty = 0

        for rec in read_jsonl(fp):
            uid = rec.get("utt_id")
            gt_text = rec.get("text", "")                  # ✅ ground truth (human)
            if not uid or uid not in utt2orig:
                missing += 1
                continue
            pred_text = utt2orig[uid]                      # ✅ prediction (original_text)

            # Chuẩn hoá giống code bạn: chỉ lowercase
            gt_norm   = (gt_text or "").strip().lower()
            pred_norm = (pred_text or "").strip().lower()

            if not gt_norm:
                skipped_empty += 1
                continue

            refs.append(gt_norm)    # jiwer expects refs = ground truth
            hyps.append(pred_norm)  # jiwer expects hyps = hypothesis (prediction)
            matched += 1

        score = wer(refs, hyps) if matched > 0 else 0.0
        rows.append((label, matched, missing, score))

        print(f"[DONE] {label}: matched={matched}, missing={missing}, skipped_empty={skipped_empty}, WER={score:.6f}")

        grp = label.split("/", 1)[0]
        group_aggr[grp]["pred"].extend(hyps)
        group_aggr[grp]["ref"].extend(refs)
        group_aggr[grp]["matched"] += matched
        group_aggr[grp]["missing"] += missing

    # Tóm tắt theo GROUP

    print("\n== GROUP SUMMARY (corpus WER, jiwer) ==")
    for grp, d in sorted(group_aggr.items()):
        g_refs_raw = d["ref"]
        g_hyps_raw = d["pred"]

        # Chuẩn hoá giống per-file: strip + lower, bỏ câu rỗng
        g_refs = []
        g_hyps = []
        for r, h in zip(g_refs_raw, g_hyps_raw):
            r2 = (r or "").strip().lower()
            h2 = (h or "").strip().lower()
            if not r2:
                continue  # tránh lỗi jiwer
            g_refs.append(r2)
            g_hyps.append(h2)

        g_matched = len(g_refs)
        g_missing = d["missing"]
        g_score = wer(g_refs, g_hyps) if g_matched > 0 else 0.0

        print(f"{grp:10s} | matched={g_matched:7d}  missing={g_missing:7d}  WER={g_score:.6f}")
            
    # --- Ghi TSV: per-file + group summary ---
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)

        # chuẩn bị group summary để ghi file (không tính lại lần nữa)
        group_rows = []
        for grp, d in sorted(group_aggr.items()):
            g_refs_raw = d["ref"]
            g_hyps_raw = d["pred"]

            # chuẩn hoá giống per-file: strip+lower, bỏ câu rỗng
            g_refs, g_hyps = [], []
            for r, h in zip(g_refs_raw, g_hyps_raw):
                r2 = (r or "").strip().lower()
                h2 = (h or "").strip().lower()
                if not r2:
                    continue
                g_refs.append(r2)
                g_hyps.append(h2)

            g_matched = len(g_refs)
            g_missing = d["missing"]
            g_score = wer(g_refs, g_hyps) if g_matched > 0 else 0.0
            group_rows.append((grp, g_matched, g_missing, g_score))

        with args.out.open("w", encoding="utf-8") as f:
            # per-file
            f.write("label\tmatched\tmissing\tWER\n")
            for (label, matched, missing, score) in rows:
                f.write(f"{label}\t{matched}\t{missing}\t{score:.6f}\n")

            # group summary (ghi tiếp dưới, prefix GROUP:)
            f.write("\n# group_summary\n")
            f.write("group\tmatched\tmissing\tWER\n")
            for (grp, g_matched, g_missing, g_score) in group_rows:
                f.write(f"{grp}\t{g_matched}\t{g_missing}\t{g_score:.6f}\n")

        print(f"[OK] Wrote summary TSV -> {args.out}")


if __name__ == "__main__":
    main()
