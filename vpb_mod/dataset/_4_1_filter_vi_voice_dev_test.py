#!/usr/bin/env python3
import json
import os

BASE_DIR = os.path.expanduser(
    "~/work/public_datasets/vi_small/nemo_manifests_big/vi_voice"
)
SPLITS = ["test", "dev"]

for split in SPLITS:
    in_path = os.path.join(BASE_DIR, split, f"vi_voice_{split}_manifest.jsonl")
    out_path = os.path.join(BASE_DIR, split, f"vi_voice_{split}_manifest_origin.jsonl")

    if not os.path.exists(in_path):
        print(f"⚠️ File không tồn tại: {in_path}")
        continue

    kept, total = 0, 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                sample = json.loads(line)
                if "/origin/" in sample.get("audio_filepath", ""):
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    kept += 1
            except json.JSONDecodeError:
                continue

    print(f"✅ {split}: {kept}/{total} lines kept → {out_path}")
