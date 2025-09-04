
mf = OrderedDict([
    ("standard_test_2",      manifest_root / "standard_test_2" / "test_meta_nemo.jsonl"),
    ("standard_test",        manifest_root / "standard_test"   / "test_meta_nemo.jsonl"),
    ("next_day_test_debug",  manifest_root / "standard_test"   / "next_day_test_meta_debug_nemo.jsonl"),
    ("vpb_right2_train",     manifest_root / "manifest_vpb_right_2" / "train_meta_nemo.jsonl"),
    ("vpb_right2_valid",     manifest_root / "manifest_vpb_right_2" / "valid_meta_nemo.jsonl"),
])

FILE_PATHS = [
    "/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta.json",
    "/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/next_day_test_meta_debug.json",
    "/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta.json",
    "/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta.json",
    "/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta.json",
]

## 0. Chunkformer (in-used model)

python -m vpb_mod.model._3_vpb_origin_performance

file	total	valid	skip_empty_ref	skip_missing	wer
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta.json	2993	2993	0	0	0.2499
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/next_day_test_meta_debug.json	1650	1650	0	0	0.2076
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta.json	29	29	0	0	0.1613
/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta.json	3072	3072	0	0	0.2420
/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta.json	630	630	0	0	0.2696
OVERALL	8374	8374	0	0	0.2350


==== Tính trùng lặp của tập dữ liệu 
[ANCHOR] /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta_nemo.jsonl -> lines=3072 unique_ids=3072

=== Overlap with Anchor (Summary) ===
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta_nemo.jsonl -> unique=2993, overlap=2096 (0.700301)
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl -> unique=29, overlap=0 (0.000000)
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/next_day_test_meta_debug_nemo.jsonl -> unique=1650, overlap=0 (0.000000)
/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_nemo.jsonl -> unique=630, overlap=0 (0.000000)




## 1. Model trained by viet_speech dataset 
python -m vpb_mod.model._2_fastformer_infer \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vietspeech/vpb_asr_fastconformer/2025-08-25_07-42-00/checkpoints/vpb_asr_fastconformer.nemo


model	dataset	wer	log_path	hard_samples
vpb_asr_fastconformer	standard_test_2	0.3546227461288865	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__standard_test_2__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__standard_test_2__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	standard_test	0.3378060263653484	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__standard_test__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__standard_test__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	next_day_test_debug	0.34140641597620563	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__next_day_test_debug__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__next_day_test_debug__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_right2_train	0.3633193573830379	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__vpb_right2_train__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__vpb_right2_train__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_right2_valid	0.39017875487980275	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__vpb_right2_valid__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_023032/hardfix__vpb_right2_valid__vpb_asr_fastconformer_hard.tsv


## 2. Model fine-tuned by pseudo label

python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo


model	dataset	wer	log_path	hard_samples
vpb_asr_fastconformer_ft_poc_qc_v1	standard_test_2	0.30402102928492214	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	standard_test	0.2582391713747646	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	next_day_test_debug	0.2687486721903548	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_right2_train	0.29557059238656647	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_right2_valid	0.3240189028148757	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_025447/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv


## 3. Model fine-tuned by vpb train ds AFTER pseudo fine-tune 

Sử dụng tập train vpb_right2_train

python -m vpb_mod.model._2_fastformer_infer \
  --devices -3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v2/2025-09-03_03-23-34/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v2.nemo


model	dataset	wer	log_path	hard_samples
vpb_asr_fastconformer_ft_poc_qc_v2	standard_test_2	0.24204214071548857	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	standard_test	0.4293785310734463	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	next_day_test_debug	0.26485376389774096	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_right2_train	0.24563078583585868	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_right2_valid	0.2821039654818163	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250904_041334/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv


