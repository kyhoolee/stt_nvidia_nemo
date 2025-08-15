import os
import json
from pathlib import Path
from collections import defaultdict

# Chỉnh lại đường dẫn output của script tải dữ liệu
OUT_ROOT = Path("../public_datasets/vi_small")
MANIFEST_DIR = OUT_ROOT / "manifests"

def summarize_dataset(dataset_name):
    dataset_dir = OUT_ROOT / dataset_name
    manifest_dir = MANIFEST_DIR / dataset_name

    print(f"📁 Dataset: {dataset_name}")

    # Thống kê từng split
    for split_dir in sorted(dataset_dir.glob("*")):
        audio_dir = split_dir / "audio"
        if audio_dir.exists():
            wav_count = sum(1 for f in audio_dir.glob("*.wav"))
        else:
            wav_count = 0

        manifest_path = manifest_dir / f"{dataset_name}_{split_dir.name}.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_count = sum(1 for _ in f)
        else:
            manifest_count = 0

        print(f"  🔹 {split_dir.name}: {wav_count} wav files | {manifest_count} manifest entries")
    print()

def main():
    if not OUT_ROOT.exists():
        print(f"❌ Output root {OUT_ROOT} not found")
        return

    for ds_dir in sorted(OUT_ROOT.iterdir()):
        if ds_dir.is_dir() and ds_dir.name != "manifests":
            summarize_dataset(ds_dir.name)

if __name__ == "__main__":
    main()
