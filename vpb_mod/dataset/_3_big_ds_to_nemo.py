#!/usr/bin/env python3
# convert_vi_voice_to_nemo.py
import os, sys, json, argparse, logging
from glob import glob

# tqdm optional: nếu không có thì fallback sang không dùng progress bar
try:
    from tqdm import tqdm
    def iter_progress(it, total=None, desc=None):
        return tqdm(it, total=total, desc=desc, ncols=100)
except Exception:
    def iter_progress(it, total=None, desc=None):
        return it

def is_abs(p: str) -> bool:
    return os.path.isabs(os.path.expanduser(p)) or (p or "").startswith("~")

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

def guess_split_from_path(in_path: str) -> str:
    norm = in_path.replace("\\", "/")
    for cand in ("train", "dev", "test"):
        if f"/{cand}/" in norm:
            return cand
    base = os.path.basename(in_path)
    if "train" in base: return "train"
    if "dev" in base:   return "dev"
    if "test" in base:  return "test"
    return "train"

def convert_file(
    in_path: str,
    out_path: str,
    audio_root: str,
    sample_rate: int = 16000,
    skip_missing: bool = True,
    log_every: int = 2000,
    verbose: bool = False,
    dry_run: bool = False,
    max_records: int | None = None,
):
    split = guess_split_from_path(in_path)
    stats = {
        "in": 0, "out": 0, "miss": 0,
        "miss_abs": 0, "miss_rel": 0, "parse_err": 0
    }

    if not dry_run:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fout = open(out_path, "w", encoding="utf-8")
    else:
        fout = None

    # đếm tổng dòng để hiển thị progress nếu có thể
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = None

    if verbose:
        logging.debug(f"[FILE] {in_path} -> split={split} out={out_path}")

    with open(in_path, "r", encoding="utf-8") as fin:
        iterator = iter_progress(fin, total=total_lines, desc=f"{os.path.basename(in_path)}")
        for idx, line in enumerate(iterator, start=1):
            if max_records is not None and stats["in"] >= max_records:
                break

            line = line.strip()
            if not line:
                continue
            stats["in"] += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                stats["parse_err"] += 1
                if verbose:
                    logging.warning(f"[PARSE] {in_path}:{stats['in']} -> {e}")
                continue

            # fields nguồn
            text = (obj.get("text") or "").strip()
            duration = obj.get("duration", None)
            dataset = obj.get("dataset", "vi_voice")
            wav_field = obj.get("wav") or obj.get("audio") or obj.get("audio_filepath")

            if not wav_field:
                stats["miss"] += 1
                if verbose:
                    logging.warning(f"[MISS] No 'wav' field @ {in_path}:{stats['in']}")
                continue

            if is_abs(wav_field):
                audio_fp = os.path.expanduser(wav_field)
                rel = False
            else:
                audio_fp = os.path.join(audio_root, split, wav_field)
                rel = True

            audio_fp = os.path.realpath(os.path.expanduser(audio_fp))

            if not os.path.exists(audio_fp):
                stats["miss"] += 1
                if rel: stats["miss_rel"] += 1
                else:   stats["miss_abs"] += 1

                msg = f"[MISS] Not found: {audio_fp}"
                if skip_missing:
                    if verbose:
                        logging.warning(msg)
                    continue
                else:
                    logging.warning(msg + " (kept)")

            out_obj = {
                "audio_filepath": audio_fp,
                "duration": duration,
                "text": text,
                "sample_rate": sample_rate,
                "dataset": dataset
            }

            if not dry_run:
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            stats["out"] += 1

            if log_every and (stats["in"] % log_every == 0):
                logging.info(
                    f"[{split}] {os.path.basename(in_path)} | seen={stats['in']:,} -> out={stats['out']:,} miss={stats['miss']:,}"
                )
                # in một mẫu minh họa
                if verbose:
                    logging.debug(f"Sample #{stats['in']}: text='{text[:80]}' | audio='{audio_fp}' | dur={duration}")

    if fout:
        fout.close()

    # báo cáo per-file
    logging.info(
        f"[DONE] {os.path.basename(in_path)} (split={split}) | in={stats['in']:,}, out={stats['out']:,}, "
        f"miss={stats['miss']:,} (rel={stats['miss_rel']:,}, abs={stats['miss_abs']:,}), parse_err={stats['parse_err']:,}"
    )
    return stats, split

def main():
    ap = argparse.ArgumentParser(description="Convert vi_voice manifests to NeMo JSONL with debug & progress")
    ap.add_argument("--in-root", required=True, help="Thư mục manifest vi_voice (train/dev/test)")
    ap.add_argument("--audio-root", required=True, help="Thư mục audio/vi_voice (train/dev/test)")
    ap.add_argument("--out-root", required=True, help="Thư mục output NeMo JSONL")
    ap.add_argument("--pattern", default="vi_voice_*_manifest.json", help="Pattern file manifest nguồn")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--skip-missing", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true", help="In log chi tiết")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ đếm, không ghi file")
    ap.add_argument("--log-every", type=int, default=2000, help="In tiến độ mỗi N dòng (0 = tắt)")
    ap.add_argument("--max-records", type=int, default=None, help="Giới hạn số record để test nhanh")
    args = ap.parse_args()

    setup_logging(args.verbose)

    in_root   = os.path.realpath(os.path.expanduser(args.in_root))
    audio_root= os.path.realpath(os.path.expanduser(args.audio_root))
    out_root  = os.path.realpath(os.path.expanduser(args.out_root))

    if not os.path.isdir(in_root):
        logging.error(f"in-root not found: {in_root}"); sys.exit(1)
    if not os.path.isdir(audio_root):
        logging.error(f"audio-root not found: {audio_root}"); sys.exit(1)
    os.makedirs(out_root, exist_ok=True)

    # gom tất cả manifest theo split
    candidates = []
    for split in ("train", "dev", "test"):
        candidates += glob(os.path.join(in_root, split, args.pattern))
    if not candidates:
        candidates = glob(os.path.join(in_root, args.pattern))
    if not candidates:
        logging.error(f"No input manifests found with pattern '{args.pattern}' under {in_root}")
        sys.exit(2)

    total = {"in":0, "out":0, "miss":0, "miss_abs":0, "miss_rel":0, "parse_err":0}
    for src in sorted(candidates):
        base = os.path.splitext(os.path.basename(src))[0]
        split = guess_split_from_path(src)
        dst  = os.path.join(out_root, "vi_voice", split, f"{base}.jsonl")
        stats, _ = convert_file(
            in_path=src,
            out_path=dst,
            audio_root=audio_root,
            sample_rate=args.sample_rate,
            skip_missing=args.skip_missing,
            log_every=args.log_every,
            verbose=args.verbose,
            dry_run=args.dry_run,
            max_records=args.max_records
        )
        for k in total: total[k] += stats[k]

    logging.info("==== SUMMARY (ALL FILES) ====")
    logging.info(
        f"Inputs : {total['in']:,}\n"
        f"Outputs: {total['out']:,}\n"
        f"Missing: {total['miss']:,} (rel={total['miss_rel']:,}, abs={total['miss_abs']:,})\n"
        f"ParseErr: {total['parse_err']:,}\n"
        f"Out dir: {out_root}"
    )

if __name__ == "__main__":
    main()
