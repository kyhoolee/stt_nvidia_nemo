#!/usr/bin/env python3
import argparse, json, shutil, random, hashlib
from pathlib import Path

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst.name
    # tránh trùng tên: thêm hash 8 ký tự
    stem, suf = dst.stem, dst.suffix
    h = hashlib.md5(str(src).encode()).hexdigest()[:8]
    new = dst.with_name(f"{stem}__{h}{suf}")
    shutil.copy2(src, new)
    return new.name

def main():
    ap = argparse.ArgumentParser(description="Pack a small manifest + audio samples into a portable folder.")
    ap.add_argument("--manifest", required=True, help="Path to source NeMo jsonl manifest")
    ap.add_argument("--out-dir", required=True, help="Output folder to place subset and audio")
    ap.add_argument("--limit", type=int, default=64, help="Number of samples to extract")
    ap.add_argument("--random-seed", type=int, default=None, help="If set, pick random subset instead of first N")
    args = ap.parse_args()

    src_manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    out_manifest = out_dir / "manifest_subset.jsonl"
    readme = out_dir / "README.md"

    # đọc manifest nguồn
    rows = []
    with src_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) == 0:
        raise ValueError("Manifest rỗng!")

    # chọn subset
    if args.random_seed is not None:
        random.Random(args.random_seed).shuffle(rows)
        chosen = rows[:args.limit]
    else:
        chosen = rows[:args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)

    # copy audio + viết manifest mới (đường dẫn tương đối)
    kept = []
    for i, row in enumerate(chosen):
        src = Path(row["audio_filepath"])
        if not src.exists():
            print(f"[WARN] Missing file: {src} -> skip")
            continue
        dst_name = safe_copy(src, audio_dir / src.name)
        # manifest mới: audio_filepath = relative
        new_row = dict(row)
        new_row["audio_filepath"] = f"audio/{dst_name}"
        kept.append(new_row)

    # ghi manifest subset
    with out_manifest.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # README
    with readme.open("w", encoding="utf-8") as f:
        f.write(
            "# VPB ASR Portable Subset\n\n"
            "- `manifest_subset.jsonl`: manifest trỏ tới đường dẫn tương đối trong thư mục này.\n"
            "- Thư mục `audio/`: chứa toàn bộ WAV cần thiết.\n\n"
            "## Cách dùng nhanh (ví dụ NeMo/py)\n"
            "Duyệt và in 5 dòng đầu:\n\n"
            "```python\n"
            "import json\n"
            "from pathlib import Path\n"
            "m = Path('manifest_subset.jsonl')\n"
            "for i, line in enumerate(m.open('r', encoding='utf-8')):\n"
            "    if i==5: break\n"
            "    print(json.loads(line))\n"
            "```\n"
        )

    print(f"[DONE] Wrote {len(kept)} items -> {out_manifest}")
    print(f"[INFO] Audio folder: {audio_dir}")

if __name__ == "__main__":
    main()
