BASE=~/work/public_datasets/vi_small
IN=$BASE/nemo_manifests_merged
OUT_AUDIO=$BASE/audio_telephony_sim
OUT_MANIFEST=$BASE/nemo_manifests_processed_merged

mkdir -p "$OUT_MANIFEST"

# train dev test

for SPLIT in dev test train; do 
  python -m vpb_mod.dataset.preprocess_audio_telephony \
    --manifest "$IN/merged_${SPLIT}.jsonl" \
    --output-audio-root "$OUT_AUDIO" \
    --output-manifest   "$OUT_MANIFEST/merged_${SPLIT}.jsonl" \
    --num-workers 8 --chunksize 16 --force-mono
done
