

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