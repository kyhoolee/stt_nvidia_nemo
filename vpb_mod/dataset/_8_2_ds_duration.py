#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sum durations from NeMo-like JSONL manifests.

Features:
- Walk through a root directory and read *.jsonl / *.jsonl.gz
- For each manifest file: sum durations (seconds) and print human HH:MM:SS
- Aggregate by dataset (parent folder name) and overall ALL
- Robust to various field names: duration or compute from audio when missing
- Optionally write a TSV summary

Usage:
  python sum_manifest_duration.py --root ./manifest/splits_by_clid_tripack --out summary.tsv
"""

from __future__ import annotations
import argparse, json, gzip, sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Optional libs to compute duration if missing in manifest
# - soundfile is fastest/most robust for WAV/FLAC
# - wave is stdlib fallback for WAV
try:
    import soundfile as sf  # pip install soundfile
except Exception:
    sf = None
import contextlib
import wave

# ---------- Helpers ----------

def human_time(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"

def open_jsonl(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def get_field(d: Dict, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def audio_duration_from_file(audio_path: str) -> Optional[float]:
    """Try to compute duration (seconds) from an audio file path."""
    p = Path(audio_path)
    if not p.exists():
        return None
    # Try soundfile first (supports many formats)
    if sf is not None:
        try:
            with sf.SoundFile(str(p)) as f:
                frames = len(f)
                sr = f.samplerate
                if sr and sr > 0:
                    return frames / float(sr)
        except Exception:
            pass
    # Fallback: wave (WAV only)
    try:
        with contextlib.closing(wave.open(str(p), "rb")) as w:
            frames = w.getnframes()
            sr = w.getframerate()
            if sr and sr > 0:
                return frames / float(sr)
    except Exception:
        pass
    return None

def sum_manifest(path: Path) -> Tuple[int, float]:
    """
    Return (num_records, total_duration_seconds) for a manifest file.
    Manifest lines should be JSON dicts. Duration keys tried:
        "duration", "audio_duration", "dur"
    If missing, try to compute from audio file using:
        "audio_filepath", "audio_path", "audio"
    """
    n = 0
    total = 0.0
    with open_jsonl(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n += 1
            dur = get_field(obj, "duration", "audio_duration", "dur", default=None)
            if isinstance(dur, (int, float)):
                total += float(dur)
                continue
            # Try compute from audio path
            apath = get_field(obj, "audio_filepath", "audio_path", "audio", default=None)
            if apath:
                d = audio_duration_from_file(apath)
                if d is not None:
                    total += d
                    continue
            # If we get here, no duration info
    return n, total

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder that contains subfolders (e.g., all/, left_only/, right_only/).")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write TSV summary.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Find manifest files
    manifest_files = sorted(
        [p for p in root.rglob("*") if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))]
    )
    if not manifest_files:
        print(f"[WARN] No *.jsonl or *.jsonl.gz found under {root}", file=sys.stderr)
        sys.exit(0)

    # Stats containers
    file_stats: Dict[str, Dict[str, float]] = {}   # file -> {"n": int, "sec": float}
    ds_stats: Dict[str, Dict[str, float]] = {}     # dataset (parent dir) -> {"n": int, "sec": float}
    total_n = 0
    total_sec = 0.0

    print("== Per-file summary ==")
    for mf in manifest_files:
        ds_name = mf.parent.name  # assumes structure .../<dataset>/<file>.jsonl
        n, sec = sum_manifest(mf)

        file_stats[str(mf)] = {"n": n, "sec": sec}
        total_n += n
        total_sec += sec

        if ds_name not in ds_stats:
            ds_stats[ds_name] = {"n": 0, "sec": 0.0}
        ds_stats[ds_name]["n"] += n
        ds_stats[ds_name]["sec"] += sec

        print(f"- {mf}: n={n:,}, sec={sec:,.2f} ({human_time(sec)})")

    print("\n== Per-dataset summary ==")
    for ds in sorted(ds_stats.keys()):
        n = ds_stats[ds]["n"]
        sec = ds_stats[ds]["sec"]
        print(f"* {ds:>10s}: n={n:,}, sec={sec:,.2f} ({human_time(sec)})")

    print("\n== OVERALL ==")
    print(f"Total records: {total_n:,}")
    print(f"Total seconds: {total_sec:,.2f}")
    print(f"Total (HH:MM:SS): {human_time(total_sec)}")

    # Optional TSV
    if args.out:
        outp = Path(args.out)
        lines = []
        # Header
        lines.append("level\tname\tn_records\ttotal_seconds\thhmmss")
        # Per-file
        for mf, st in file_stats.items():
            lines.append(f"file\t{mf}\t{int(st['n'])}\t{st['sec']:.6f}\t{human_time(st['sec'])}")
        # Per-dataset
        for ds, st in ds_stats.items():
            lines.append(f"dataset\t{ds}\t{int(st['n'])}\t{st['sec']:.6f}\t{human_time(st['sec'])}")
        # Overall
        lines.append(f"overall\tALL\t{total_n}\t{total_sec:.6f}\t{human_time(total_sec)}")
        outp.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[OK] Wrote TSV summary → {outp}")

if __name__ == "__main__":
    main()
