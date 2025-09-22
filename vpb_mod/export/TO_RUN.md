python -m vpb_mod.export._0_export_rnnt_core_onnx \
  --nemo /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo \
  --out  vpb_mod/export/asr_deploy


python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo



================


python -m vpb_mod.export.export_rnnt_core_onnx \
  --nemo /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo \
  --out  /home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy \
  --opset 17 \
  --skip-preproc


python -m vpb_mod.export.probe_rnnt_io --nemo /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo


===================


DEPLOY=/home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy
MANIFROOT=/home/ubuntu/work/clean_dataset_vpb/manifest

# chạy 1 hoặc nhiều manifest (NAME=path để đặt nhãn)
python -m vpb_mod.export.run_onnx_infer_manifest \
  --deploy /home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy \
  --manifest standard_test_2=/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta_nemo.jsonl \
  --manifest standard_test=/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl \
  --batch-size 8 \
  --max-u 256 \
  --hard-topk 50 \
  --out-dir ./onnx_eval_logs


=======

python -m vpb_mod.export.probe_onnx_io /home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy


python -m vpb_mod.export.probe_tokenizer_decode --nemo /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo


python -m vpb_mod.export.run_onnx_infer_manifest \
  --deploy /home/ubuntu/work/stt_nvidia_nemo/vpb_mod/export/asr_deploy \
  --manifest standard_test_2=/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta_nemo.jsonl