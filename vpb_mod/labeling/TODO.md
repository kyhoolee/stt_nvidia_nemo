

python -m vpb_mod.labeling._1_vpb_snr \
  --vad auto --top-db 22 --aggr 2 --min-gap-ms 60 --min-len-ms 100 \
  --jobs 8 --log-every 25 --log-interval 1.5 \
  --summary-csv /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/snr_test_summary.csv \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.jsonl



python -m vpb_mod.labeling._1_vpb_snr \
  --vad auto --top-db 22 --aggr 2 --min-gap-ms 60 --min-len-ms 100 \
  --jobs 8 --log-every 25 --log-interval 1.5 \
  --summary-csv /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/snr_valid_summary.csv \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.jsonl



python -m vpb_mod.labeling._1_vpb_snr \
  --vad auto --top-db 22 --aggr 2 --min-gap-ms 60 --min-len-ms 100 \
  --jobs 8 --log-every 25 --log-interval 1.5 \
  --summary-csv /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/snr_train_summary.csv \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.jsonl


====================

python -m vpb_mod.labeling._2_filter_snr \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.with_snr.jsonl \
  --percentile 60

python -m vpb_mod.labeling._2_filter_snr \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.with_snr.jsonl \
  --percentile 60

python -m vpb_mod.labeling._2_filter_snr \
  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.with_snr.jsonl \
  --percentile 60


========

/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.with_snr.top.jsonl
/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.with_snr.top.jsonl
/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.with_snr.top.jsonl

=========


python add_model_text_to_manifests.py \
  --nemo /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo \
  --manifests \
    /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.with_snr.top.jsonl \
    /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.with_snr.top.jsonl \
    /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.with_snr.top.jsonl \
  --batch-size 16 \
  --precision 32 \
  --out-suffix .with_model


============


python -m vpb_mod.labeling._3_fcm_label \
  --nemo /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo \
  --manifests \
    /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.with_snr.top.jsonl \
    /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.with_snr.top.jsonl \
    /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.with_snr.top.jsonl \
  --batch-size 8 \
  --precision 32 \
  --out-suffix .with_model




python -m vpb_mod.labeling._3_fcm_label \
  --nemo /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo \
  --manifests \
    /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.with_snr.top.jsonl \
  --batch-size 16 \
  --precision 32 \
  --out-suffix .with_model


/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.with_snr.top.with_model.jsonl
/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.with_snr.top.with_model.jsonl
/home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.with_snr.top.with_model.jsonl


