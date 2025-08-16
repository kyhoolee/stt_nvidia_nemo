#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create NVIDIA NeMo ASR manifests from your localized Vietnamese ASR corpora.

This script assumes you have already materialized WAV files to a folder tree like:

  OUT_ROOT/
    <dataset_name>/
      train|dev|validation|test/
        audio/
          <utt_id>.wav
    manifests/
      <dataset_name>/
        <dataset_name>_train.jsonl
        <dataset_name>_dev.jsonl (or _validation.jsonl)
        <dataset_name>_test.jsonl

Where those per-dataset JSONLs (created by your downloader) contain rows like:
{
  "dataset": "vivos",
  "split": "train",
  "utt_id": "VIVOSSPK01_001",
  "wav": "/abs/path/.../VIVOSSPK01_001.wav",
  "text": "một câu ví dụ ..."
}

This tool will:
  1) Read all JSONL(s) for a dataset to get (wav_path|stem) -> transcript
  2) Walk the corresponding audio/ directory, compute duration (and sample rate)
  3) Write a NeMo manifest JSONL with items of the form:
        {"audio_filepath": <abs path>, "duration": <float sec>, "text": <str>, "sample_rate": <int>}

You can also optionally merge multiple datasets into a single manifest per split.

Design notes / gotchas
----------------------
- Some corpora have only train+test, or train only. We **never fabricate** splits.
  The script only produces manifests for splits that actually exist on disk. If a dataset
  lacks `dev` but has `validation`, we map that to the `dev` bucket (common convention).
- Text alignment: we build a transcript map using both absolute file paths **and** filename stems.
  This makes the step robust whether your source manifest stores absolute `wav` or just names.
- Duration: we read with `soundfile` info; if duration is missing or zero, we fall back to reading
  samples and computing `len(samples)/sr`.
- Filtering: you can filter by min/max duration and require a specific sample rate.

Example usages
--------------
# Create manifests for all datasets found under OUT_ROOT using defaults
python create_nemo_manifest.py \
  --root ~/work/public_datasets/vi_small

# Only build for a subset and filter out durations < 0.2s or > 30s
python create_nemo_manifest.py \
  --root ~/work/public_datasets/vi_small \
  --datasets vivos lsvsc \
  --min-duration 0.2 --max-duration 30

# Enforce 16kHz audio only and lowercase transcripts
python create_nemo_manifest.py \
  --root ~/work/public_datasets/vi_small \
  --ensure-sr 16000 \
  --lowercase

# Also produce merged manifests across datasets (one per split)
python create_nemo_manifest.py \
  --root ~/work/public_datasets/vi_small \
  --merge-out merged_manifests

"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import soundfile as sf
from tqdm import tqdm


# --------------------------- Data Structures ---------------------------
@dataclass
class Row:
    """One line in a NeMo manifest.

    NeMo accepts JSONL with (at least):
      - audio_filepath: absolute path to wav/flac
      - duration: seconds (float)
      - text: transcript (UTF-8)
    We also optionally include sample_rate for sanity/debug.
    """
    audio_filepath: str
    duration: float
    text: str
    sample_rate: Optional[int] = None


# --------------------------- Helpers ---------------------------
def find_datasets(root: Path) -> List[str]:
    """List dataset folder names under root (excluding 'manifests' and non-dirs).

    We treat a folder as a dataset if it contains at least one split directory
    among {train, dev, validation, test} which itself contains an `audio/` subdir.
    """
    out: List[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name != "manifests":
            if any((p / s / "audio").exists() for s in ("train", "dev", "validation", "test")):
                out.append(p.name)
    return out


def load_source_texts(man_dir: Path) -> Dict[str, str]:
    """Build a map (absolute wav path -> text) and (stem -> text) from your
    source JSONLs. Having both keys makes us robust against path changes.

    If `man_dir` does not exist (e.g., a purely audio-only corpus), we return an empty
    mapping; such utterances will be dropped later as "no_text".
    """
    mapping: Dict[str, str] = {}
    if not man_dir.exists():
        return mapping

    for jl in sorted(man_dir.glob("*.jsonl")):
        with jl.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue  # skip corrupt lines but keep going
                wav = obj.get("wav")
                text = obj.get("text")
                if not wav or text is None:
                    continue
                mapping[os.path.abspath(wav)] = text
                mapping.setdefault(Path(wav).stem, text)
    return mapping


def normalize_text(s: Optional[str], lowercase: bool, strip: bool) -> Optional[str]:
    """Trim/lowercase transcript if requested. Keep None/empty as signals to drop."""
    if s is None:
        return None
    out = s
    if strip:
        out = out.strip()
    if lowercase:
        out = out.lower()
    return out


def iter_audio_files(ds_dir: Path, splits: Iterable[str]) -> List[Tuple[str, Path]]:
    """Return list of (split_name, wav_path) for requested splits that exist.

    - If caller asks for `dev` but dataset uses `validation`, we transparently
      iterate `validation` files but keep the logical split as `dev` later.
    - We do not invent missing splits.
    """
    results: List[Tuple[str, Path]] = []
    for sp in splits:
        audio_dir = ds_dir / sp / "audio"
        if not audio_dir.exists():
            # bridge dev <-> validation naming
            alt = "validation" if sp == "dev" else ("dev" if sp == "validation" else None)
            if alt:
                audio_dir = ds_dir / alt / "audio"
        if audio_dir.exists():
            for wav in sorted(audio_dir.glob("*.wav")):
                results.append((sp, wav))
    return results


def sf_info(wav_path: Path) -> Tuple[float, int]:
    """Read duration and sample rate using soundfile.

    Some containers (or certain encoders) may not populate `info.duration`.
    In that case, we read the samples and compute duration as a fallback.
    """
    info = sf.info(str(wav_path))
    duration = getattr(info, "duration", None)
    if duration is None or duration == 0:
        data, sr = sf.read(str(wav_path))
        duration = len(data) / float(sr)
        return duration, sr
    return float(info.duration), int(info.samplerate)


def build_rows_for_dataset(
    root: Path,
    dataset: str,
    splits: Iterable[str],
    lowercase: bool,
    strip_text: bool,
    ensure_sr: Optional[int],
    min_dur: Optional[float],
    max_dur: Optional[float],
    source_manifest_dir: Optional[Path] = None,
) -> Dict[str, List[Row]]:
    """Create NeMo rows per split for a dataset.

    Returns: dict split -> list[Row]

    Skipping policy:
      - If transcript is missing/empty -> drop (counted in `skipped_no_text`).
      - If `ensure_sr` is set and file's SR != ensure_sr -> drop (counted in `skipped_sr`).
      - If duration outside [min_dur, max_dur] (when provided) -> drop (counted in `skipped_dur`).
    """
    ds_dir = root / dataset
    if source_manifest_dir is None:
        source_manifest_dir = root / "manifests" / dataset

    text_map = load_source_texts(source_manifest_dir)

    per_split: Dict[str, List[Row]] = {"train": [], "dev": [], "test": []}

    pairs = iter_audio_files(ds_dir, splits)
    skipped_no_text = 0
    skipped_sr = 0
    skipped_dur = 0

    for want_split, wav_path in tqdm(pairs, desc=f"{dataset}: scanning", unit="wav"):
        abs_wav = os.path.abspath(str(wav_path))
        # lookup by absolute path first, then by stem
        text = text_map.get(abs_wav)
        if text is None:
            text = text_map.get(wav_path.stem)
        text = normalize_text(text, lowercase=lowercase, strip=strip_text)
        if not text:
            skipped_no_text += 1
            continue

        try:
            duration, sr = sf_info(wav_path)
        except Exception:
            # unreadable/broken file -> skip silently
            continue

        if ensure_sr is not None and sr != ensure_sr:
            skipped_sr += 1
            continue

        if (min_dur is not None and duration < min_dur) or (max_dur is not None and duration > max_dur):
            skipped_dur += 1
            continue

        # Normalize validation -> dev bucket name for output
        key = "dev" if want_split in ("dev", "validation") else want_split
        if key not in per_split:
            per_split[key] = []
        per_split[key].append(Row(audio_filepath=abs_wav, duration=duration, text=text, sample_rate=sr))

    # Summary for quick triage
    total = sum(len(v) for v in per_split.values())
    print(
        f"[{dataset}] built {total} rows | "
        f"train={len(per_split.get('train', []))}, dev={len(per_split.get('dev', []))}, test={len(per_split.get('test', []))} | "
        f"skipped: no_text={skipped_no_text}, sr={skipped_sr}, dur={skipped_dur}"
    )
    return per_split


def write_manifest(rows: List[Row], out_path: Path, include_sr: bool = True) -> None:
    """Write a NeMo JSONL. We keep the file minimal and stable for reproducibility."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "audio_filepath": r.audio_filepath,
                "duration": float(r.duration),
                "text": r.text,
            }
            if include_sr and r.sample_rate is not None:
                obj["sample_rate"] = int(r.sample_rate)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {out_path} ({len(rows)} items)")


def write_dataset_manifests(
    root: Path,
    dataset: str,
    out_dir: Path,
    splits: Iterable[str],
    lowercase: bool,
    strip_text: bool,
    ensure_sr: Optional[int],
    min_dur: Optional[float],
    max_dur: Optional[float],
    source_manifest_dir: Optional[Path],
) -> Dict[str, Path]:
    """Build and write per-split manifests for one dataset.

    Returns a mapping split -> written manifest path (only for splits that produced rows).
    """
    per_split = build_rows_for_dataset(
        root=root,
        dataset=dataset,
        splits=splits,
        lowercase=lowercase,
        strip_text=strip_text,
        ensure_sr=ensure_sr,
        min_dur=min_dur,
        max_dur=max_dur,
        source_manifest_dir=source_manifest_dir,
    )

    out_paths: Dict[str, Path] = {}
    for sp, rows in per_split.items():
        if not rows:
            continue  # do not emit empty files
        out_path = out_dir / dataset / f"{dataset}_{sp}.jsonl"
        write_manifest(rows, out_path)
        out_paths[sp] = out_path
    return out_paths


def merge_manifests(across: Dict[str, Dict[str, Path]], merge_out: Path) -> None:
    """Merge per-dataset manifests into one per split under `merge_out`.

    Input `across`: dataset -> {split -> path}. We simply concatenate JSONLs (one item per line).
    """
    tmp: Dict[str, List[Path]] = {"train": [], "dev": [], "test": []}
    for _ds, split_map in across.items():
        for sp, p in split_map.items():
            if sp not in tmp:
                tmp[sp] = []
            tmp[sp].append(p)

    merge_out.mkdir(parents=True, exist_ok=True)

    for sp, paths in tmp.items():
        if not paths:
            continue
        out_path = merge_out / f"merged_{sp}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as out_f:
            for p in paths:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        out_f.write(line)
                        count += 1
        print(f"✓ Merged {len(paths)} files -> {out_path} ({count} items)")


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    """Command-line interface definition.

    Key flags to remember when debugging:
      --ensure-sr 16000         # quickly filter out wrong SR files
      --min-duration 0.2        # drop short blips
      --max-duration 30         # drop very long turns
      --lowercase               # normalize text
      --no-strip                # keep exact whitespace (useful for diffing)
      --merge-out <dir>         # also emit merged manifests
    """
    p = argparse.ArgumentParser(
        description="Create NeMo ASR manifests from localized WAV datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=Path, required=True, help="Root containing <dataset>/<split>/audio and manifests/<dataset>")
    p.add_argument("--datasets", nargs="*", default=None, help="Datasets to process. If omitted, auto-discover under --root")
    p.add_argument("--splits", nargs="*", default=["train", "dev", "test"], help="Which splits to build")
    p.add_argument("--output-dir", type=Path, default=None, help="Where to place NeMo manifests. Default: <root>/nemo_manifests")
    p.add_argument("--source-manifest-dir", type=Path, default=None, help="Override source manifest root (expects per-dataset subfolders)")

    p.add_argument("--min-duration", type=float, default=None, help="Drop utterances shorter than this (seconds)")
    p.add_argument("--max-duration", type=float, default=None, help="Drop utterances longer than this (seconds)")
    p.add_argument("--ensure-sr", type=int, default=None, help="Keep only audio with this sample rate (e.g., 16000)")
    p.add_argument("--lowercase", action="store_true", help="Lowercase transcripts")
    p.add_argument("--no-strip", action="store_true", help="Do NOT strip whitespace in transcripts")

    p.add_argument("--merge-out", type=Path, default=None, help="If set, also create merged manifests across datasets into this folder")

    return p.parse_args()


def main():
    args = parse_args()

    root: Path = args.root.expanduser()
    out_dir: Path = (args.output_dir or (root / "nemo_manifests")).expanduser()
    source_root: Optional[Path] = args.source_manifest_dir.expanduser() if args.source_manifest_dir else None

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = find_datasets(root)
        if not datasets:
            raise SystemExit(f"No datasets found under {root}")

    print("=== PLAN ===")
    print(f"Root         : {root}")
    print(f"Datasets     : {', '.join(datasets)}")
    print(f"Splits       : {', '.join(args.splits)}")
    print(f"Output dir   : {out_dir}")
    print(f"Source man dir root: {source_root or (root / 'manifests')} (per-dataset subfolders expected)")
    print(f"Filters      : min_dur={args.min_duration}, max_dur={args.max_duration}, ensure_sr={args.ensure_sr}")
    print(f"Text opts    : lowercase={args.lowercase}, strip={not args.no_strip}")
    if args.merge_out:
        print(f"Merged out   : {args.merge_out}")
    print("============")

    across: Dict[str, Dict[str, Path]] = {}

    for ds in datasets:
        ds_source_dir = (source_root / ds) if source_root else (root / "manifests" / ds)
        paths = write_dataset_manifests(
            root=root,
            dataset=ds,
            out_dir=out_dir,
            splits=args.splits,
            lowercase=args.lowercase,
            strip_text=not args.no_strip,
            ensure_sr=args.ensure_sr,
            min_dur=args.min_duration,
            max_dur=args.max_duration,
            source_manifest_dir=ds_source_dir,
        )
        across[ds] = paths

    if args.merge_out:
        merge_manifests(across, args.merge_out.expanduser())

    print("All done.")


if __name__ == "__main__":
    main()
