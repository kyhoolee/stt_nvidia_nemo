#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix/Convert VPB manifests.

Modes
-----
1) Default (test/infer friendly JSONL):
   - Input: JSON array OR JSONL
   - Output: JSONL with fields: audio_filepath (absolute), text, (keep: duration, sample_rate, dataset if present)
   - Use when preparing test/eval manifests.

2) Train mode (--to-nemo-train):
   - Output lines follow NeMo train format strictly:
       {"audio_filepath": "...", "duration": 3.384, "text": "...", "sample_rate": 16000, "dataset": "NAME"}
   - Requires --dataset-name
   - Will compute duration/sample_rate if missing (soundfile -> librosa), else use --assume-sr and warn if duration unknown.
"""

import argparse, json, os, sys
from pathlib import Path

# Optional deps
_sf = None
_lb = None
try:
    import soundfile as _sf  # type: ignore
except Exception:
    _sf = None
try:
    import librosa as _lb  # type: ignore
except Exception:
    _lb = None


def read_manifest(path: Path):
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):  # JSON array
        data = json.loads(text)
        for item in data:
            yield item
    else:  # JSONL
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"[WARN] Bad JSON at line {ln} in {path}: {e}\n")


def is_abs(p: str) -> bool:
    # consider s3:// and gs:// as absolute too
    return os.path.isabs(p) or p.startswith(("s3://", "gs://", "http://", "https://"))


def compute_audio_meta(audio_fp: str):
    """
    Try to return (duration_sec: float|None, sample_rate: int|None) by probing file.
    Priority: soundfile -> librosa. If both fail, return (None, None).
    """
    # Remote URIs (s3/https) not supported here
    if audio_fp.startswith(("s3://", "gs://", "http://", "https://")):
        return None, None

    # Try soundfile
    if _sf is not None:
        try:
            with _sf.SoundFile(audio_fp) as sfh:
                sr = int(sfh.samplerate)
                dur = float(len(sfh)) / float(sr) if sr > 0 else None
                return dur, sr
        except Exception:
            pass

    # Try librosa
    if _lb is not None:
        try:
            y, sr = _lb.load(audio_fp, sr=None, mono=True)  # keep native sr
            dur = float(len(y)) / float(sr) if sr and len(y) else None
            return dur, int(sr) if sr else None
        except Exception:
            pass

    return None, None


def main():
    ap = argparse.ArgumentParser(description="Normalize/convert VPB manifests.")
    ap.add_argument("--input", "-i", required=True, type=Path, help="Input manifest (.json or .jsonl)")
    ap.add_argument("--audio-base", "-a", required=True, type=Path,
                    help="Base folder that `audio_path` is relative to (e.g., /home/ubuntu/work/clean_dataset_vpb/audio)")
    ap.add_argument("--output", "-o", type=Path,
                    help="Output JSONL path (default: <input_name>_fixed.jsonl in same dir)")
    ap.add_argument("--fail-missing", action="store_true",
                    help="If set, fail when audio file is missing instead of warning.")

    # Train-mode flags
    ap.add_argument("--to-nemo-train", action="store_true",
                    help="Convert to strict NeMo train manifest format.")
    ap.add_argument("--dataset-name", type=str, default=None,
                    help="Dataset name to fill `dataset` field in train mode (required if --to-nemo-train).")
    ap.add_argument("--assume-sr", type=int, default=16000,
                    help="Assumed sample_rate if not readable from file. Default: 16000.")

    args = ap.parse_args()

    in_path: Path = args.input
    audio_base: Path = args.audio_base
    default_suffix = "_train.jsonl" if args.to_nemo_train else "_fixed.jsonl"
    out_path: Path = args.output or in_path.with_name(in_path.stem + default_suffix)

    # Safety: ensure base exists (only for local paths)
    if not str(audio_base).startswith(("s3://", "gs://", "http://", "https://")) and not audio_base.exists():
        sys.stderr.write(f"[ERROR] audio_base does not exist: {audio_base}\n")
        sys.exit(2)

    if args.to_nemo_train and not args.dataset_name:
        sys.stderr.write("[ERROR] --to-nemo-train requires --dataset-name\n")
        sys.exit(2)

    n_in = n_out = n_missing = n_meta_fail = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in read_manifest(in_path):
            n_in += 1
            # prefer `text`, else fallback to `base_text`
            text_val = ex.get("text")
            if (text_val is None or str(text_val).strip() == "") and "base_text" in ex:
                text_val = ex.get("base_text")
            if text_val is None:
                text_val = ""  # keep non-null

            # compute absolute audio_filepath
            src_path = ex.get("audio_filepath") or ex.get("audio_path")
            if not src_path:
                sys.stderr.write(f"[WARN] Example {n_in} missing audio_path/audio_filepath. Skipped.\n")
                continue

            if is_abs(src_path):
                audio_fp = src_path
            else:
                audio_fp = str((audio_base / src_path).resolve())

            # existence check for local files
            if not is_abs(src_path) and not Path(audio_fp).exists():
                n_missing += 1
                msg = f"[WARN] Missing audio file: {audio_fp}"
                if args.fail_missing:
                    sys.stderr.write(msg + "\n")
                    sys.exit(3)
                else:
                    sys.stderr.write(msg + " (continuing)\n")
                    # still write a line, but duration/meta may be None

            if args.to_nemo_train:
                # Train format: ensure duration & sample_rate
                duration = ex.get("duration")
                sample_rate = ex.get("sample_rate")

                if duration is None or sample_rate is None:
                    dur2, sr2 = compute_audio_meta(audio_fp)
                    if duration is None:
                        duration = dur2
                    if sample_rate is None:
                        sample_rate = sr2

                if sample_rate is None:
                    sample_rate = args.assume_sr  # fallback

                if duration is None:
                    n_meta_fail += 1
                    # keep going but warn
                    sys.stderr.write(f"[WARN] Could not read duration for: {audio_fp} (writing null)\n")

                out = {
                    "audio_filepath": audio_fp,
                    "duration": duration,                 # may be None if unreadable
                    "text": text_val,
                    "sample_rate": int(sample_rate) if sample_rate is not None else None,
                    "dataset": args.dataset_name,
                }
            else:
                # Test/infer friendly format (keep useful known fields)
                out = {
                    "utt_id": ex.get("utt_id"),
                    "audio_filepath": audio_fp,
                    "text": text_val,
                }
                for k in ("duration", "sample_rate", "dataset"):
                    if k in ex:
                        out[k] = ex[k]

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    sys.stderr.write(f"[DONE] Read {n_in} | Wrote {n_out} | Missing files: {n_missing} | Meta-failed: {n_meta_fail}\n")
    print(str(out_path))


if __name__ == "__main__":
    main()
