#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, gzip, sys
from pathlib import Path
from collections import defaultdict

ID_KEYS_DEFAULT = ["utt_id", "id", "uid", "sample_id", "audio_id"]

def open_any(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")

def pick_instance_id(obj, id_keys):
    # Ưu tiên khóa id
    for k in id_keys:
        v = obj.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    # Fallback: audio_filepath (+ offset|start_time|start|duration|end_time)
    ap = obj.get("audio_filepath")
    if ap:
        base = str(ap).strip()
        off  = obj.get("offset") or obj.get("start_time") or obj.get("start")
        if off is not None:
            return f"{base}::offset={off}"
        dur = obj.get("duration") or obj.get("end_time")
        if dur is not None:
            return f"{base}::dur={dur}"
        return base
    return None

def scan_file_ids(path: Path, id_keys):
    ids = []
    id2lines = defaultdict(list)
    with open_any(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"JSON parse error at {path}:{lineno} -> {e}")
            inst_id = pick_instance_id(obj, id_keys)
            if inst_id is None:
                raise RuntimeError(f"Missing instance id at {path}:{lineno} (no keys {id_keys}+fallbacks)")
            ids.append(inst_id)
            id2lines[inst_id].append(lineno)
    return ids, id2lines

def write_clean_without_overlaps(src_path: Path, out_path: Path, anchor_set, id_keys):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with open_any(src_path) as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            inst_id = pick_instance_id(obj, id_keys)
            if inst_id in anchor_set:
                dropped += 1
                continue
            f_out.write(line)
            kept += 1
    return kept, dropped

def main():
    ap = argparse.ArgumentParser(
        description="Check data leakage by finding instance_id overlaps with an anchor manifest (e.g., train)."
    )
    ap.add_argument("--anchor", required=True, type=Path,
                    help="Anchor manifest (e.g., train_meta_nemo.jsonl). Overlaps are computed against this file only.")
    ap.add_argument("others", nargs="+", type=Path,
                    help="Other manifests to check for overlap with the anchor.")
    ap.add_argument("--id-keys", nargs="*", default=ID_KEYS_DEFAULT,
                    help=f"Priority id keys (default: {ID_KEYS_DEFAULT})")
    ap.add_argument("--summary-tsv", type=Path, default=Path("overlap_with_anchor.summary.tsv"),
                    help="Output TSV (summary per file).")
    ap.add_argument("--details-tsv", type=Path, default=Path("overlap_with_anchor.details.tsv"),
                    help="Output TSV (detailed overlaps).")
    ap.add_argument("--write-clean", type=Path, default=None,
                    help="If set, write cleaned copies (without overlaps) of 'others' into this OUTDIR, preserving filenames.")
    args = ap.parse_args()

    anchor = args.anchor.expanduser().resolve()
    others = [p.expanduser().resolve() for p in args.others]
    id_keys = args.id_keys

    # Load anchor ids
    anchor_ids, anchor_id2lines = scan_file_ids(anchor, id_keys)
    anchor_set = set(anchor_ids)
    print(f"[ANCHOR] {anchor} -> lines={len(anchor_ids)} unique_ids={len(set(anchor_ids))}")

    # Prepare outputs
    details_rows = []
    summary_rows = []

    for fp in others:
        ids, id2lines = scan_file_ids(fp, id_keys)
        uniq_ids = set(ids)
        overlaps = uniq_ids.intersection(anchor_set)

        # Summary
        summary_rows.append({
            "file": str(fp),
            "lines": len(ids),
            "unique_ids": len(uniq_ids),
            "overlap_with_anchor": len(overlaps),
            "overlap_ratio_unique": f"{(len(overlaps)/max(1,len(uniq_ids))):.6f}",
        })

        # Details
        for inst in sorted(overlaps):
            # liệt kê tất cả dòng ở file này + dòng ở anchor
            locs_other = ";".join(f"{fp}:{ln}" for ln in id2lines.get(inst, []))
            locs_anchor = ";".join(f"{anchor}:{ln}" for ln in anchor_id2lines.get(inst, []))
            details_rows.append((inst, locs_anchor, locs_other))

        # Optionally write clean file
        if args.write_clean is not None:
            rel = fp.name
            out_path = args.write_clean / rel
            kept, dropped = write_clean_without_overlaps(fp, out_path, anchor_set, id_keys)
            print(f"[CLEAN] {fp} -> {out_path} | kept={kept}, dropped(overlap)={dropped}")

    # Write TSVs
    with args.summary_tsv.open("w", encoding="utf-8") as f:
        f.write("file\tlines\tunique_ids\toverlap_with_anchor\toverlap_ratio_unique\n")
        for r in summary_rows:
            f.write(f"{r['file']}\t{r['lines']}\t{r['unique_ids']}\t{r['overlap_with_anchor']}\t{r['overlap_ratio_unique']}\n")

    with args.details_tsv.open("w", encoding="utf-8") as f:
        f.write("instance_id\tanchor_locations\tother_locations\n")
        for inst, la, lo in details_rows:
            f.write(f"{inst}\t{la}\t{lo}\n")

    print("\n=== Overlap with Anchor (Summary) ===")
    for r in summary_rows:
        print(f"{r['file']} -> unique={r['unique_ids']}, overlap={r['overlap_with_anchor']} "
              f"({r['overlap_ratio_unique']})")
    print(f"\nSummary TSV : {args.summary_tsv}")
    print(f"Details TSV : {args.details_tsv}")
    if args.write_clean:
        print(f"Cleaned manifests written under: {args.write_clean}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
