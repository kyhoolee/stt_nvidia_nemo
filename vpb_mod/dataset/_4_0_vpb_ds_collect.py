#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

def read_manifest_any(path: Path) -> Tuple[List[dict], str]:
    """
    Đọc manifest nội bộ có thể ở 2 dạng:
    - JSON array: [..., ...]
    - JSONL: {..}\n{..}\n
    Trả về (entries, fmt) với fmt in {"json_array", "jsonl"}.
    """
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):  # JSON array
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} không phải list JSON.")
        return data, "json_array"
    else:  # JSONL
        entries = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries, "jsonl"

def write_manifest_any(entries: List[dict], path: Path, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json_array":
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "jsonl":
        with path.open("w", encoding="utf-8") as fo:
            for e in entries:
                fo.write(json.dumps(e, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unknown manifest format: {fmt}")

def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError("copy_mode chỉ nhận 'copy' hoặc 'symlink'")

def process_manifest(
    manifest_path: Path,
    base_audio_root: Path,
    output_root: Path,
    copy_mode: str = "copy",
    keep_raw_manifest: bool = True,
) -> Tuple[int, int]:
    """
    Đọc manifest, lọc các entry có audio tồn tại, copy/symlink audio sang clean_root/audio/,
    và ghi manifest đã lọc (giữ nguyên format gốc) vào clean_root/manifest/.
    """
    entries, fmt = read_manifest_any(manifest_path)

    # Out paths
    # - manifest đã lọc: giữ tên file, nhưng đặt trong output_root/manifest/<relative-to-some-parent>
    # Đơn giản: dùng thư mục con là tên parent + tên file để tránh đè nhau.
    dataset_dir_name = manifest_path.parent.name
    out_manifest = output_root / "manifest" / dataset_dir_name / manifest_path.name

    # Optional: lưu raw manifest (nguyên trạng) để đối chiếu
    if keep_raw_manifest:
        raw_copy_path = output_root / "manifest_raw" / dataset_dir_name / manifest_path.name
        raw_copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, raw_copy_path)

    # Lọc + copy audio
    filtered = []
    missing = []
    copied = 0

    for e in entries:
        # BẮT BUỘC: manifest nội bộ có field audio_path (relative)
        rel_audio = Path(e["audio_path"])
        src = base_audio_root / rel_audio
        if src.exists():
            # copy/symlink sang clean_root/audio/<rel_audio>
            dst = output_root / "audio" / rel_audio
            copy_or_link(src, dst, copy_mode)
            copied += 1
            # GIỮ format entry y nguyên (không đổi trường, không đổi audio_path)
            # => chỉ append lại
            filtered.append(e)
        else:
            missing.append(str(rel_audio))

    # Ghi manifest đã lọc đúng format gốc
    write_manifest_any(filtered, out_manifest, fmt)

    # Ghi báo cáo nhỏ để bạn soát nhanh
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{dataset_dir_name}__{manifest_path.stem}__report.tsv"
    with report_path.open("w", encoding="utf-8") as fo:
        fo.write("status\trel_audio_path\n")
        for e in filtered:
            fo.write(f"copied\t{e['audio_path']}\n")
        for m in missing:
            fo.write(f"missing\t{m}\n")

    print(f"✅ {manifest_path} → {out_manifest} | kept {len(filtered)} / {len(entries)} | missing {len(missing)}")
    if missing:
        miss_log = output_root / "reports" / f"{dataset_dir_name}__{manifest_path.stem}__missing.txt"
        miss_log.write_text("\n".join(missing), encoding="utf-8")
        print(f"⚠️ Missing list: {miss_log}")

    return len(filtered), len(missing)

def main():
    ap = argparse.ArgumentParser(description="Gom audio + manifest nội bộ sang folder sạch để share (giữ nguyên format manifest).")
    ap.add_argument("--base-audio-root", required=True, help="Thư mục gốc chứa audio (nơi audio_path là relative).")
    ap.add_argument("--output-root", required=True, help="Thư mục clean để gom dữ liệu.")
    ap.add_argument("--copy-mode", default="copy", choices=["copy", "symlink"], help="Copy thật hoặc tạo symlink.")
    ap.add_argument("--keep-raw-manifest", action="store_true", help="Copy thêm bản manifest gốc vào manifest_raw/ để đối chiếu.")
    ap.add_argument("manifests", nargs="+", help="Danh sách path tới các manifest nội bộ.")
    args = ap.parse_args()

    base_audio_root = Path(os.path.expanduser(args.base_audio_root)).resolve()
    output_root = Path(os.path.expanduser(args.output_root)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_missing = 0
    for m in args.manifests:
        kept, miss = process_manifest(
            Path(m).resolve(),
            base_audio_root,
            output_root,
            copy_mode=args.copy_mode,
            keep_raw_manifest=args.keep_raw_manifest,
        )
        total_kept += kept
        total_missing += miss

    print(f"\n==== SUMMARY ====")
    print(f"Kept: {total_kept} | Missing: {total_missing}")
    print(f"Audio out: {output_root / 'audio'}")
    print(f"Manifest out: {output_root / 'manifest'}")

if __name__ == "__main__":
    main()
