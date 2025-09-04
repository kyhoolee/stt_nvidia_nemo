#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from jiwer import compute_measures, Compose, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation

# -------------------- CONFIG --------------------

FILE_PATHS = [
    "/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta.json",
    "/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/next_day_test_meta_debug.json",
    "/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta.json",
    "/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta.json",
    "/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta.json",
]

TRANSFORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])

MAX_DEBUG_PRINT = 3  # số sample in ra để check

# -------------------- READ RECORDS --------------------

def read_records(p: Path) -> List[Dict[str, Any]]:
    txt = p.read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        print(f"[DEBUG] File {p.name} is empty")
        return []

    try:
        data = json.loads(txt)
        if isinstance(data, list):
            print(f"[DEBUG] File {p.name} parsed as JSON array, total={len(data)}")
            if data:
                print("  Sample[0]:", json.dumps(data[0], ensure_ascii=False))
            return data
        if isinstance(data, dict):
            for k in ("data", "items", "records"):
                if k in data and isinstance(data[k], list):
                    print(f"[DEBUG] File {p.name} parsed as JSON object with key '{k}', total={len(data[k])}")
                    if data[k]:
                        print("  Sample[0]:", json.dumps(data[k][0], ensure_ascii=False))
                    return data[k]
    except Exception as e:
        print(f"[DEBUG] File {p.name} not JSON array/object, fallback JSONL. Error={e}")

    out = []
    for i, line in enumerate(txt.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            out.append(obj)
            if len(out) <= MAX_DEBUG_PRINT:
                print(f"[DEBUG] File {p.name} JSONL sample line {i+1}: {json.dumps(obj, ensure_ascii=False)}")
        except json.JSONDecodeError:
            print(f"[DEBUG] File {p.name} JSON decode error at line {i+1}: {line[:80]}...")
    print(f"[DEBUG] File {p.name} parsed as JSONL, total={len(out)}")
    return out

# -------------------- VALIDATE --------------------

def validate_and_collect_pairs(records: List[Dict[str, Any]], file_path: Path, strict: bool):
    stats = {
        "total": len(records),
        "valid": 0,
        "skip_empty_ref": 0,
        "skip_missing": 0,
        "examples": []
    }
    refs, hyps = [], []

    for idx, r in enumerate(records):
        ref = r.get("text")        # ground truth
        hyp = r.get("base_text")   # model prediction
        utt_id = r.get("utt_id")

        if not isinstance(ref, str) or not isinstance(hyp, str):
            stats["skip_missing"] += 1
            continue

        ref_t = TRANSFORM(ref).strip()
        hyp_t = TRANSFORM(hyp).strip()

        if ref_t == "":
            stats["skip_empty_ref"] += 1
            if len(stats["examples"]) < MAX_DEBUG_PRINT:
                stats["examples"].append({
                    "idx": idx,
                    "utt_id": utt_id,
                    "reason": "empty_ref",
                    "ref_raw": ref,
                    "hyp_raw": hyp
                })
            if strict:
                raise ValueError(f"[STRICT] Empty reference at idx={idx}, utt_id={utt_id}, file={file_path}")
            continue

        refs.append(ref_t)
        hyps.append(hyp_t)
        stats["valid"] += 1

    return refs, hyps, stats

# -------------------- FILE LEVEL --------------------

def file_wer(path: Path, strict: bool = False) -> Tuple[float, Dict[str, Any], List[str], List[str]]:
    records = read_records(path)
    refs, hyps, stats = validate_and_collect_pairs(records, path, strict)
    if stats["valid"] == 0:
        return float("nan"), stats, [], []
    measures = compute_measures(refs, hyps)
    stats["wer_measures"] = measures
    return measures["wer"], stats, refs, hyps

# -------------------- MAIN --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out-tsv", type=Path, default=Path("wer_report.tsv"),
                    help="File TSV để ghi kết quả")
    args = ap.parse_args()

    print("=== WER report per file ===")
    all_refs, all_hyps = [], []
    rows = []  # để xuất TSV

    for fp in FILE_PATHS:
        p = Path(fp)
        if not p.exists():
            print(f"[WARN] Not found: {p}")
            continue

        try:
            wer, stats, refs, hyps = file_wer(p, strict=args.strict)
        except ValueError as e:
            print(f"[ERROR][STRICT] {p}\n{e}")
            return

        wer_str = f"{wer:.4f}" if wer == wer else "NaN"
        print(f"\nFile: {p}")
        print(f"  total={stats['total']} | valid={stats['valid']} "
              f"| skip_empty_ref={stats['skip_empty_ref']} "
              f"| skip_missing={stats['skip_missing']}")
        print(f"  WER={wer_str}")

        if stats["examples"]:
            print("  --- Examples of skipped ---")
            for e in stats["examples"]:
                print(f"   - idx={e['idx']} utt_id={e['utt_id']} reason={e['reason']} "
                      f"ref_raw='{e['ref_raw']}' hyp_raw='{e['hyp_raw']}'")

        # append row cho TSV
        rows.append({
            "file": str(p),
            "total": stats['total'],
            "valid": stats['valid'],
            "skip_empty_ref": stats['skip_empty_ref'],
            "skip_missing": stats['skip_missing'],
            "wer": wer_str
        })

        # gom overall
        all_refs.extend(refs)
        all_hyps.extend(hyps)

    # Overall
    if all_refs:
        overall = compute_measures(all_refs, all_hyps)
        wer_overall = f"{overall['wer']:.4f}"
        print("\n=== Overall (all files combined) ===")
        print(f"WER={wer_overall} | total_pairs={len(all_refs)}")

        rows.append({
            "file": "OVERALL",
            "total": len(all_refs),
            "valid": len(all_refs),
            "skip_empty_ref": 0,
            "skip_missing": 0,
            "wer": wer_overall
        })
    else:
        print("\n=== Overall ===\nNo valid pairs.")

    # ---- Xuất TSV ----
    out_fp = args.out_tsv
    with out_fp.open("w", encoding="utf-8") as f:
        header = ["file", "total", "valid", "skip_empty_ref", "skip_missing", "wer"]
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row[h]) for h in header) + "\n")
    print(f"\n[INFO] WER report saved to {out_fp}")

if __name__ == "__main__":
    main()