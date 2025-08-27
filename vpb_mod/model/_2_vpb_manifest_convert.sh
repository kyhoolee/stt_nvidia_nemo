#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ============ CONFIG ============
AUDIO_BASE="/home/ubuntu/work/clean_dataset_vpb/audio"
MANIFEST_ROOT="/home/ubuntu/work/clean_dataset_vpb/manifest"
MODE="train"   # "test" => _nemo.jsonl, "train" => _train.jsonl

# ============ LOOP CONVERT ============
for f in ${MANIFEST_ROOT}/*/*.json; do
    if [[ ! -f "$f" ]]; then
        continue
    fi

    if [[ "$MODE" == "train" ]]; then
        out="${f%.json}_train.jsonl"
        dataset_name="$(basename "$(dirname "$f")")_$(basename "${f%.json}")"
        echo "Processing (train) $f -> $out (dataset=${dataset_name})"
        python -m vpb_mod.model._2_vpb_manifest_convert \
            --input "$f" \
            --audio-base "$AUDIO_BASE" \
            --to-nemo-train \
            --dataset-name "$dataset_name" \
            --output "$out"
    else
        out="${f%.json}_nemo.jsonl"
        echo "Processing (test) $f -> $out"
        python -m vpb_mod.model._2_vpb_manifest_convert \
            --input "$f" \
            --audio-base "$AUDIO_BASE" \
            --output "$out"
    fi
done
