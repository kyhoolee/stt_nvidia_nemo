#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VietSpeech EFS materialization tools (with progress debug).
"""

from __future__ import annotations
import argparse, json, os, sys, shutil
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SPLITS = ["train", "dev", "test"]

def expand(p: Path | str) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_manifest(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                sys.stderr.write(f"[WARN] {path}:{ln} bad JSONL: {e}\n")

def find_vietspeech_subpath(src_audio: Path) -> Tuple[str, Path]:
    parts = src_audio.parts
    idx = None
    for i, p in enumerate(parts):
        if p == "vietspeech":
            idx = i
            break
    if idx is None or idx + 2 >= len(parts):
        raise ValueError(f"Cannot locate 'vietspeech/<split>' in path: {src_audio}")
    split = parts[idx + 1]
    sub = Path(*parts[idx:])  # vietspeech/<split>/...
    return split, sub

def map_dst_audio(src_audio: Path, dst_audio_root: Path) -> Path:
    split, sub = find_vietspeech_subpath(src_audio)
    return (dst_audio_root / sub).resolve()

# ----------------------------- copy-audio -----------------------------

def do_copy(src: Path, dst: Path, mode: str, overwrite: bool):
    if dst.exists():
        if not overwrite:
            return
        if dst.is_file():
            dst.unlink()
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def cmd_copy_audio(args):
    src_dir = expand(args.src_manifest_dir)
    dst_root = expand(args.dst_root)
    dst_audio_root = dst_root / "audio"
    ensure_dir(dst_audio_root)

    src_manifests = {s: src_dir / f"{s}.jsonl" for s in SPLITS}
    for s, p in src_manifests.items():
        if not p.exists():
            print(f"[ERR] Missing manifest: {p}", file=sys.stderr)
            sys.exit(1)

    pairs: Dict[Path, Path] = {}
    total_records = 0
    for s in SPLITS:
        for rec in read_manifest(src_manifests[s]):
            total_records += 1
            src_fp = expand(rec["audio_filepath"])
            dst_fp = map_dst_audio(src_fp, dst_audio_root)
            pairs[src_fp] = dst_fp

    print(f"[copy-audio] unique WAV files: {len(pairs)} (from {total_records} records)")

    copied = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(do_copy, s, d, args.mode, args.overwrite): (s, d) for s, d in pairs.items()}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Copying audio", unit="file"):
            try:
                fut.result()
                copied += 1
            except Exception as e:
                errors += 1
                src, dst = futs[fut]
                sys.stderr.write(f"[ERR] copy {src} -> {dst} : {e}\n")

    print("=== copy-audio Summary ===")
    print(f"Dst root   : {dst_root}")
    print(f"Files total: {len(pairs)}")
    print(f"Copied/lnk : {copied - errors}")
    print(f"Errors     : {errors}")
    print("✅ Done.")

# --------------------------- remap-manifest ---------------------------

def cmd_remap_manifest(args):
    src_dir = expand(args.src_manifest_dir)
    dst_root = expand(args.dst_root)
    out_dir = expand(args.out_manifest_dir) if args.out_manifest_dir else (dst_root / "manifest")
    ensure_dir(out_dir)

    dst_audio_root = dst_root / "audio"
    src_manifests = {s: src_dir / f"{s}.jsonl" for s in SPLITS}
    for s, p in src_manifests.items():
        if not p.exists():
            print(f"[ERR] Missing manifest: {p}", file=sys.stderr)
            sys.exit(1)

    total = 0
    for s in SPLITS:
        recs_out: List[Dict[str, Any]] = []
        for rec in tqdm(read_manifest(src_manifests[s]), desc=f"Remapping {s}", unit="rec"):
            src_fp = expand(rec["audio_filepath"])
            dst_fp = map_dst_audio(src_fp, dst_audio_root)
            new_rec = dict(rec)
            new_rec["audio_filepath"] = str(dst_fp)
            recs_out.append(new_rec)
        out_path = out_dir / f"{s}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in recs_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(recs_out)
        print(f"[remap] {s}: {len(recs_out)} -> {out_path}")

    print("=== remap-manifest Summary ===")
    print(f"Out dir    : {out_dir}")
    print(f"Total recs : {total}")
    print("✅ Done.")

# -------------------------------- main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="VietSpeech EFS materialization tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_copy = sub.add_parser("copy-audio", help="Copy (or link) WAVs to EFS, preserving tree structure")
    ap_copy.add_argument("--src-manifest-dir", type=Path, required=True)
    ap_copy.add_argument("--dst-root", type=Path, required=True)
    ap_copy.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
    ap_copy.add_argument("--overwrite", action="store_true")
    ap_copy.add_argument("--num-workers", type=int, default=8)
    ap_copy.set_defaults(func=cmd_copy_audio)

    ap_remap = sub.add_parser("remap-manifest", help="Remap manifest audio_filepath sang EFS")
    ap_remap.add_argument("--src-manifest-dir", type=Path, required=True)
    ap_remap.add_argument("--dst-root", type=Path, required=True)
    ap_remap.add_argument("--out-manifest-dir", type=Path, default=None)
    ap_remap.set_defaults(func=cmd_remap_manifest)

    args = ap.parse_args()
    if hasattr(args, "src_manifest_dir"): args.src_manifest_dir = expand(args.src_manifest_dir)
    if hasattr(args, "dst_root"): args.dst_root = expand(args.dst_root)
    if hasattr(args, "out_manifest_dir") and args.out_manifest_dir: args.out_manifest_dir = expand(args.out_manifest_dir)
    args.func(args)

if __name__ == "__main__":
    main()
