#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bulk convert VPB JSON array manifests -> NeMo JSONL manifests.

- Base audio folder: ~/work/vpb_dataset
- Output root: ~/work/public_datasets/vi_small/nemo_manifests/vpb_ds
- Mirroring subpaths after base_audio, and change .json -> .jsonl
- Prefer "text", fallback to "base_text"
- audio_filepath trong output sẽ dùng prefix "~" thay cho "/home/kylh"
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

# ==== CONFIG ====
BASE_AUDIO = Path("~/work/vpb_dataset").expanduser().resolve()
OUT_ROOT   = Path("~/work/public_datasets/vi_small/nemo_manifests/vpb_ds").expanduser().resolve()

INPUTS = [
    "~/work/vpb_dataset/standard_test_2/test_meta.json",
    "~/work/vpb_dataset/standard_test/test_meta.json",
    "~/work/vpb_dataset/standard_test/next_day_test_meta_debug.json",
    "~/work/vpb_dataset/manifest_vpb_right_2/train_meta.json",
    "~/work/vpb_dataset/manifest_vpb_right_2/valid_meta.json",
]

DEFAULT_SR = 16000
SKIP_MISSING = True

def get_audio_info(p: Path) -> tuple[Optional[int], Optional[float]]:
    """Return (sample_rate, duration_seconds)."""
    try:
        import soundfile as sf
        info = sf.info(str(p))
        sr = int(info.samplerate) if info.samplerate else None
        dur = float(info.frames) / float(info.samplerate) if info.frames and info.samplerate else None
        return sr, dur
    except Exception:
        try:
            import wave, contextlib
            with contextlib.closing(wave.open(str(p), "rb")) as w:
                sr = w.getframerate()
                frames = w.getnframes()
                dur = frames / float(sr) if sr else None
                return int(sr), dur
        except Exception:
            return None, None

def as_user_path(p: Path) -> str:
    """Chuyển /home/kylh/... thành ~/..."""
    p_abs = str(p)
    home = str(Path("~").expanduser())
    if p_abs.startswith(home):
        return "~" + p_abs[len(home):]
    return p_abs

def convert_one(in_path: Path, base_audio: Path, out_root: Path) -> Path:
    in_path = in_path.expanduser().resolve()
    rel = in_path.relative_to(base_audio)  # path sau ~/work/vpb_dataset
    out_path = (out_root / rel).with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise RuntimeError(f"Input is not a JSON array: {in_path}")

    n_in, n_ok, n_skip = len(data), 0, 0

    with open(out_path, "w", encoding="utf-8") as w:
        for item in data:
            rel_audio = item.get("audio_path") or item.get("path") or item.get("audio")
            if not rel_audio:
                n_skip += 1
                continue

            abs_audio = (base_audio / rel_audio).resolve()
            if not abs_audio.exists() and SKIP_MISSING:
                n_skip += 1
                continue

            txt = item.get("text") or item.get("base_text", "")

            sr, dur = get_audio_info(abs_audio)
            if sr is None:
                sr = DEFAULT_SR
            if dur is None:
                dur = 0.0

            rec = {
                "audio_filepath": as_user_path(abs_audio),
                "duration": float(dur),
                "text": txt if isinstance(txt, str) else "",
                "sample_rate": int(sr),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"✅ {in_path}  →  {out_path} | in={n_in} ok={n_ok} skip={n_skip}")
    return out_path

def main():
    print(f"Base audio: {BASE_AUDIO}")
    print(f"Output root: {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for p in INPUTS:
        convert_one(Path(p), BASE_AUDIO, OUT_ROOT)

if __name__ == "__main__":
    main()
