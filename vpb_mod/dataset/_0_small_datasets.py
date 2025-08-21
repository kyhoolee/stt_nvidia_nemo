#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from shutil import copy2
from typing import Optional, Dict, List

from datasets import load_dataset, Audio, get_dataset_split_names
import soundfile as sf
from tqdm import tqdm


# === Chỉnh OUTPUT theo máy của bạn ===
OUT_ROOT = Path("~/work/public_datasets/vi_small").expanduser()

# Nguồn HF (không dùng trust_remote_code)
SOURCES: Dict[str, List[Dict[str, Optional[str]]]] = {
    "fpt_fosd":      [{"path": "doof-ferb/fpt_fosd", "config": None}],
    "infore":        [{"path": "doof-ferb/infore1_25hours", "config": None}],
    "lsvsc":         [{"path": "doof-ferb/LSVSC", "config": None}],
    "speech_massive":[{"path": "doof-ferb/Speech-MASSIVE_vie", "config": None}],
    "vais1000":      [{"path": "doof-ferb/vais1000", "config": None}],
    "vietmed":       [{"path": "leduckhai/VietMed", "config": "default"}],  # không còn 'labeled'
    "vivos":         [{"path": "AILAB-VNUHCM/vivos", "config": None},
                      {"path": "SEACrowd/vivos", "config": None}],
    "vlsp2020":      [{"path": "doof-ferb/vlsp2020_vinai_100h", "config": None}],
}

# Thứ tự ưu tiên split khi có
SPLIT_ORDER = ["train", "validation", "dev", "test"]  # một số repo dùng 'validation' thay cho 'dev'

TEXT_KEYS = ["text", "sentence", "transcription", "raw_text", "norm_text", "label"]

def ensure_decode_false(ds):
    # cast_column(decode=False) để có 'path' nếu nguồn hỗ trợ
    try:
        if "audio" in ds.features and isinstance(ds.features["audio"], Audio):
            ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    return ds

def write_wav_from_array(array, sr, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), array, sr)

def copy_or_decode_split(dataset_name: str, hf_path: str, config: Optional[str], split: str):
    print(f"→ Loading {dataset_name} | {hf_path} | config={config} | split={split}")
    ds = load_dataset(hf_path, config, split=split)

    # Thử lấy đường dẫn file gốc
    ds = ensure_decode_false(ds)

    out_dir = OUT_ROOT / dataset_name / split / "audio"
    man_dir = OUT_ROOT / "manifests" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    # tìm cột text
    found_text_key = None
    for k in TEXT_KEYS:
        if k in ds.features:
            found_text_key = k
            break

    manifest = []
    need_decode = False

    # First pass: xem có 'audio.path' usable không
    # nếu phần lớn không có path → chuyển qua decode=True
    with_path = 0
    total_probe = min(10, len(ds))
    for i in range(total_probe):
        ex = ds[i]
        p = ex.get("audio", {}).get("path") if isinstance(ex.get("audio"), dict) else None
        if p and os.path.exists(p):
            with_path += 1
    if with_path < total_probe:  # nhiều phần tử không có path
        need_decode = True
        ds = ds.cast_column("audio", Audio(decode=True))

    for ex in tqdm(ds, desc=f"Saving {dataset_name}/{split}", unit="utt"):
        # xác định id
        ex_id = ex.get("id") or ex.get("utt_id") or ex.get("key")
        if not ex_id:
            # fallback: từ tên tệp hoặc số thứ tự
            if not need_decode and isinstance(ex.get("audio"), dict) and ex["audio"].get("path"):
                ex_id = Path(ex["audio"]["path"]).stem
            else:
                ex_id = f"utt_{len(manifest):08d}"

        # đường đích
        dst = out_dir / f"{ex_id}.wav"

        if not need_decode:
            src_path = None
            if isinstance(ex.get("audio"), dict):
                src_path = ex["audio"].get("path")
            if src_path and os.path.exists(src_path):
                if not dst.exists():
                    copy2(src_path, dst)
            else:
                # trường hợp hy hữu: mẩu riêng lẻ không có path → decode mẩu này
                audio = ex["audio"]
                array = audio["array"]
                sr = audio["sampling_rate"]
                write_wav_from_array(array, sr, dst)
        else:
            audio = ex["audio"]
            array = audio["array"]
            sr = audio["sampling_rate"]
            write_wav_from_array(array, sr, dst)

        text_val = ex.get(found_text_key) if found_text_key else None
        if text_val is not None and not isinstance(text_val, str):
            text_val = str(text_val)

        manifest.append({
            "dataset": dataset_name,
            "split": split,
            "utt_id": str(ex_id),
            "wav": str(dst),
            "text": text_val,
        })

    # ghi manifest JSONL
    with open(man_dir / f"{dataset_name}_{split}.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def download_one(dataset_name: str):
    candidates = SOURCES[dataset_name]
    last_err = None
    for cand in candidates:
        try:
            hf_path, config = cand["path"], cand["config"]

            # Lấy danh sách split có thật trên repo
            try:
                available = get_dataset_split_names(hf_path, config)
            except Exception:
                # fallback: assume at least train
                available = ["train"]

            # Chuẩn hóa: map 'validation' -> 'dev' (song song ghi ra folder 'dev' nếu muốn)
            normalized = []
            for s in available:
                if s == "validation":
                    normalized.append("dev")
                else:
                    normalized.append(s)

            # Duyệt theo ưu tiên
            seen = set()
            for s in SPLIT_ORDER:
                # chấp nhận nếu s có thật hoặc (s=='dev' và source dùng 'validation')
                if s in normalized and s not in seen:
                    real_split = "validation" if s == "dev" and "validation" in available else s
                    copy_or_decode_split(dataset_name, hf_path, config, real_split)
                    seen.add(s)
            print(f"✓ Done {dataset_name} via {hf_path} (config={config})")
            return
        except Exception as e:
            last_err = e
            print(f"  ✗ Failed with {cand['path']} (config={cand['config']}): {e}")
    raise RuntimeError(f"All sources failed for {dataset_name}: {last_err}")

def summarize():
    print("\n" + "="*80)
    print("SUMMARY")
    for ds_dir in sorted(OUT_ROOT.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == "manifests":
            continue
        print(f"📁 {ds_dir.name}")
        for split in ["train", "dev", "test"]:
            ad = ds_dir / split / "audio"
            if ad.exists():
                n = sum(1 for _ in ad.glob("*.wav"))
                print(f"  - {split:<5}: {n} wav")
        print()

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    order = ['vivos']
    # ["fpt_fosd", "infore", "lsvsc", "speech_massive", "vais1000", "vietmed", "vivos", "vlsp2020"]
    for name in order:
        print("=" * 80)
        print(f"Downloading dataset: {name}")
        download_one(name)
    summarize()

if __name__ == "__main__":
    main()
