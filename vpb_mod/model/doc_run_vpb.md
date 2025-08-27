# 1) standard_test_2
python -m vpb_mod.model._2_fastformer_infer \
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
python -m vpb_mod.model._2_fastformer_infer \
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
python -m vpb_mod.model._2_fastformer_infer \
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
python -m vpb_mod.model._2_fastformer_infer \
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
python -m vpb_mod.model._2_fastformer_infer \
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



/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test_2/test_meta.jsonl
[
  {
    "utt_id": "E_huongds_D_2025-06-02_H_091735_844_CLID_0813494717___000004962___right___000005694",
    "audio_path": "archive_2/wavs/E_huongds_D_2025-06-02_H_091735_844_CLID_0813494717___000004962___right___000005694.wav",
    "text": "phải chị ơi",
    "base_text": "phải lên"
  },...

/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test/test_meta.jsonl
[
  {
    "utt_id": "E_DUCLV8_D_2025-02-11_H_080820_963_CLID_0981283053__left",
    "audio_path": "standard_test/wavs/E_DUCLV8_D_2025-02-11_H_080820_963_CLID_0981283053__left.wav",
    "text": "alo ạ anh ro him đang nghe máy hả anh em chào anh em là đức bên ngân hàng vp banh ạ bên ngân hàng gọi thông báo sớm khoản vay ô tô với khoản vay tín chấp ấy sắp đến hạn thanh toán là ngày mười lăm tháng hai anh nhận được tin nhắn chưa ạ ờ mười bảy triệu sáu trăm năm mươi nghìn ý thì anh để ý thanh toán đúng hạn giúp em anh nhá để tránh phát sinh kỳ lãi phạt với ảnh hưởng đến lịch sử tín dụng ấy ạ dạ vâng ạ dạ rồi em cảm ơn anh nhá vâng em chào anh ạ",
    "base_text": "alo ạ anh ro him đang nghe máy hả anh em chào anh em là đức minh ngân hàng vp banh ạ đinh ngân hàng gọi thông báo chở mấy với khoản vay ô tô với khoản vay tín chấp ấy sắp đến hạn thanh toán là ngày mười lăm tháng hai anh nhận được tin nhắn chưa ạ ờ mười bảy triệu sáu trăm năm mươi nghìn ý thì anh để ý thanh toán hữu hạn giúp em anh nhá để tránh phát sinh kỳ lãi phạt với ảnh hưởng đến lịch sử tín dụng ấy ạ dạ vâng ạ dạ rồi em cảm ơn anh nhá vâng em chào anh ạ"
  },...

/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test/next_day_test_meta_debug.jsonl
[
  {
    "utt_id": "E_huongds_D_2025-04-09_H_092332_720_CLID_0942487879___000027990___right___000029162",
    "audio_path": "archive/tts_dataset_best_call_agent_audio/tts_dataset_best_call_agent_audio/wavs/E_huongds_D_2025-04-09_H_092332_720_CLID_0942487879___000027990___right___000029162.wav",
    "text": "nói gì không có nghe",
    "gold_corrected": "nói gì không có nghe",
    "pred_old": "mới gì không có ăn hả"
  },...

/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/manifest_vpb_right_2/train_meta.jsonl
[
  {
    "utt_id": "E_huongds_D_2025-06-02_H_091735_844_CLID_0813494717___000004962___right___000005694",
    "audio_path": "archive_2/wavs/E_huongds_D_2025-06-02_H_091735_844_CLID_0813494717___000004962___right___000005694.wav",
    "text": "phải chị ơi",
    "base_text": "phải lên",
    "snr": 28.16,
    "snr_bucket": "clean"
  },...

/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/manifest_vpb_right_2/valid_meta.jsonl
[
  {
    "utt_id": "E_huytq37_D_2025-06-16_H_183906_901_CLID_0982317861___000404514___right___000412094",
    "audio_path": "archive_2/wavs/E_huytq37_D_2025-06-16_H_183906_901_CLID_0982317861___000404514___right___000412094.wav",
    "text": "hơn hơn một trăm bảy mươi triệu mà anh trả chín mươi triệu thì còn hơn tám mươi triệu là khác nào anh trả hai mươi tư triệu tín tín thấp còn anh trả còn số còn lại anh trả là một thẻ",
    "base_text": "hơn một trăm bảy mươi triệu mà anh trả chín mươi triệu thì còn hơn tám mươi triệu là khác nào anh trả hai mươi tư triệu tính chín thấp còn anh trả còn số còn anh trả là một thẻ",
    "snr": 31.12,
    "snr_bucket": "clean"
  },...