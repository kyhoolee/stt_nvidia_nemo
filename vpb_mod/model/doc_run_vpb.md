# 1) standard_test_2
python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --test-manifest /home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test_2/test_meta.jsonl \
  --devices 1 \
  --precision 16 \
  --batch-size 64 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments \
  --exp-name vpb_asr_fastconformer_testonly_standard_test_2 \
  --nemo ../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo \
  --test-only

====================================================================================================
✅ Finished testing.
✨ Final WER for the test set: 0.6717
====================================================================================================


# 2) standard_test
python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --test-manifest /home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test/test_meta.jsonl \
  --devices 1 \
  --precision 16 \
  --batch-size 64 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments \
  --exp-name vpb_asr_fastconformer_testonly_standard_test \
  --nemo ../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo \
  --test-only

====================================================================================================
🔍 Running manual transcription and WER calculation...
====================================================================================================
✅ Finished testing.
✨ Final WER for the test set: 0.7168
====================================================================================================

# 3) next_day_test_meta_debug
python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --test-manifest /home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test/next_day_test_meta_debug.jsonl \
  --devices 1 \
  --precision 16 \
  --batch-size 64 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments \
  --exp-name vpb_asr_fastconformer_testonly_next_day_test_debug \
  --nemo ../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo \
  --test-only

====================================================================================================
✅ Finished testing.
✨ Final WER for the test set: 0.6169
====================================================================================================

# 4) manifest_vpb_right_2/train_meta.jsonl
python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --test-manifest /home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/manifest_vpb_right_2/train_meta.jsonl \
  --devices 1 \
  --precision 16 \
  --batch-size 64 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments \
  --exp-name vpb_asr_fastconformer_testonly_vpb_right2_train \
  --nemo ../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo \
  --test-only

====================================================================================================
✅ Finished testing.
✨ Final WER for the test set: 0.6823
====================================================================================================

# 5) manifest_vpb_right_2/valid_meta.jsonl
python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --test-manifest /home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/manifest_vpb_right_2/valid_meta.jsonl \
  --devices 1 \
  --precision 16 \
  --batch-size 64 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments \
  --exp-name vpb_asr_fastconformer_testonly_vpb_right2_valid \
  --nemo ../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo \
  --test-only

====================================================================================================
✅ Finished testing.
✨ Final WER for the test set: 0.6735
====================================================================================================