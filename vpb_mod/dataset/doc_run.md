## DATA 

(nemo) kylh@educa:~/work/stt_nvidia_nemo$  python -m vpb_mod.dataset._0_small_datasets_verify
📁 Dataset: fpt_fosd
  🔹 train: 25917 wav files | 25917 manifest entries

📁 Dataset: infore
  🔹 train: 14935 wav files | 14935 manifest entries

📁 Dataset: lsvsc
  🔹 test: 5683 wav files | 5683 manifest entries
  🔹 train: 45458 wav files | 45458 manifest entries
  🔹 validation: 5682 wav files | 5682 manifest entries

📁 Dataset: nemo_manifests
  🔹 fpt_fosd: 0 wav files | 0 manifest entries
  🔹 infore: 0 wav files | 0 manifest entries
  🔹 lsvsc: 0 wav files | 0 manifest entries
  🔹 vais1000: 0 wav files | 0 manifest entries
  🔹 vietmed: 0 wav files | 0 manifest entries
  🔹 vivos: 0 wav files | 0 manifest entries
  🔹 vlsp2020: 0 wav files | 0 manifest entries

---------------------------- ERROR -> label missing dataset 
📁 Dataset: speech_massive
  🔹 test: 2974 wav files | 2974 manifest entries
  🔹 train: 115 wav files | 115 manifest entries
  🔹 validation: 2033 wav files | 2033 manifest entries
-----------------------------

📁 Dataset: vais1000
  🔹 train: 1000 wav files | 1000 manifest entries

📁 Dataset: vietmed
  🔹 dev: 2912 wav files | 2912 manifest entries
  🔹 test: 3437 wav files | 3437 manifest entries
  🔹 train: 2773 wav files | 2773 manifest entries

📁 Dataset: vivos
  🔹 test: 760 wav files | 760 manifest entries
  🔹 train: 11660 wav files | 11660 manifest entries

📁 Dataset: vlsp2020
  🔹 train: 56427 wav files | 56427 manifest entries


-----------------------------

(base) kylh@educa:~/work/public_datasets/vi_small/nemo_manifests$ tree 
.
├── fpt_fosd
│   └── fpt_fosd_train.jsonl
├── infore
│   └── infore_train.jsonl
├── lsvsc
│   ├── lsvsc_dev.jsonl
│   ├── lsvsc_test.jsonl
│   └── lsvsc_train.jsonl
├── vais1000
│   └── vais1000_train.jsonl
├── vietmed
│   ├── vietmed_dev.jsonl
│   ├── vietmed_test.jsonl
│   └── vietmed_train.jsonl
├── vivos
│   ├── vivos_test.jsonl
│   └── vivos_train.jsonl
├── vlsp2020
│   └── vlsp2020_train.jsonl
└── vpb_ds
    ├── manifest_vpb_right_2
    │   ├── train_meta.jsonl
    │   └── valid_meta.jsonl
    ├── standard_test
    │   ├── next_day_test_meta_debug.jsonl
    │   └── test_meta.jsonl
    └── standard_test_2
        └── test_meta.jsonl

11 directories, 17 files


-----------------------------

## CMD

tensorboard --logdir ./

python _1_nemo_manifest_format.py \
  --root ~/work/public_datasets/vi_small \
  --datasets vivos


python -m vpb_mod.dataset._1_nemo_manifest_format \
  --root ~/work/public_datasets/vi_small \
  --ensure-sr 16000 \
  --lowercase


python -m vpb_mod.dataset._1_nemo_manifest_format \
  --root ~/work/public_datasets/vi_small \
  --lowercase


## NEXT-LOGIC

- Convert to 8k -> then back to 16k 
- Merge-manifest then split train/dev/test 


python -m vpb_mod.dataset.merge_manifests \
  --manifest-root ~/work/public_datasets/vi_small/nemo_manifests \
  --datasets fpt_fosd infore lsvsc vais1000 vietmed vivos vlsp2020 \
  --train-files \
      fpt_fosd/fpt_fosd_train.jsonl \
      infore/infore_train.jsonl \
      lsvsc/lsvsc_train.jsonl \
      vais1000/vais1000_train.jsonl \
      vietmed/vietmed_train.jsonl \
      vivos/vivos_train.jsonl \
      vlsp2020/vlsp2020_train.jsonl \
  --dev-files \
      lsvsc/lsvsc_dev.jsonl \
      vietmed/vietmed_dev.jsonl \
  --test-files \
      lsvsc/lsvsc_test.jsonl \
      vietmed/vietmed_test.jsonl \
      vivos/vivos_test.jsonl \
  --out-dir ~/work/public_datasets/vi_small/nemo_manifests_merged \
  --seed 20250819 \
  --shuffle \
  --max-per-dataset 0 \
  --max-seconds-per-split 0 \
  --min-dur 0.2 --max-dur 30




python -m vpb_mod.dataset.merge_manifests \
  --manifest-root ~/work/public_datasets/vi_small/nemo_manifests \
  --datasets fpt_fosd lsvsc vais1000 vietmed vlsp2020 \
  --train-files \
      fpt_fosd/fpt_fosd_train.jsonl \
      lsvsc/lsvsc_train.jsonl \
      vais1000/vais1000_train.jsonl \
      vietmed/vietmed_train.jsonl \
      vlsp2020/vlsp2020_train.jsonl \
  --dev-files \
      lsvsc/lsvsc_dev.jsonl \
      vietmed/vietmed_dev.jsonl \
  --test-files \
      lsvsc/lsvsc_test.jsonl \
      vietmed/vietmed_test.jsonl \
  --out-dir ~/work/public_datasets/vi_small/nemo_manifests_merged \
  --seed 20250819 \
  --shuffle \
  --max-per-dataset 0 \
  --max-seconds-per-split 0 \
  --min-dur 0.2 --max-dur 30




python -m vpb_mod.dataset.merge_manifests \
  --manifest-root ~/work/public_datasets/vi_small/nemo_manifests \
  --datasets fpt_fosd infore lsvsc vais1000 vietmed vlsp2020 \
  --train-files \
      fpt_fosd/fpt_fosd_train.jsonl \
      infore/infore_train.jsonl \
      lsvsc/lsvsc_train.jsonl \
      vais1000/vais1000_train.jsonl \
      vietmed/vietmed_train.jsonl \
      vlsp2020/vlsp2020_train.jsonl \
  --dev-files \
      lsvsc/lsvsc_dev.jsonl \
      vietmed/vietmed_dev.jsonl \
  --test-files \
      lsvsc/lsvsc_test.jsonl \
      vietmed/vietmed_test.jsonl \
  --out-dir ~/work/public_datasets/vi_small/nemo_manifests_merged \
  --seed 20250819 \
  --shuffle \
  --max-per-dataset 0 \
  --max-seconds-per-split 0 \
  --min-dur 0.2 --max-dur 30


nohup bash ./sh_process_merge.sh > sh_process_merge.log 2>&1 &



'''
nohup python -m vpb_mod.dataset._3_big_ds_to_nemo \
  --in-root /mnt/efs/preprocess-4/manifest/vi_voice \
  --audio-root /mnt/efs/preprocess-4/audio/vi_voice \
  --out-root /home/ubuntu/work/public_datasets/vi_small/nemo_manifests_big \
  --skip-missing --verbose --log-every 5000 > log_manifest_vi_voice.log 2>&1 &


nohup python -m vpb_mod.dataset._3_big_ds_to_nemo \
  --in-root /mnt/efs/preprocess-3/manifest/viet_bud500 \
  --audio-root /mnt/efs/preprocess-3/audio/viet_bud500 \
  --out-root /home/ubuntu/work/public_datasets/vi_small/nemo_manifests_big \
  --skip-missing --verbose --log-every 5000 > log_manifest_viet_bud500.log 2>&1 &


------------------------


python  -m vpb_mod.dataset._3_big_ds_to_nemo \
  --dataset vivos \
  --in-root   /mnt/efs/preprocess-3/manifest/vivos \
  --audio-root /mnt/efs/preprocess-3/audio/vivos \
  --out-root  /home/ubuntu/work/public_datasets/vi_small/nemo_manifests_big \
  --pattern "{dataset}_manifest.json" \
  --skip-missing

--------------------------------------------------------------------



nohup python  -m vpb_mod.dataset._3_big_ds_to_nemo \
  --dataset viet_bud500 \
  --in-root   /mnt/efs/preprocess-3/manifest/viet_bud500 \
  --audio-root /mnt/efs/preprocess-3/audio/viet_bud500 \
  --out-root  /home/ubuntu/work/public_datasets/vi_small/nemo_manifests_big \
  --pattern "{dataset}_manifest.json" \
  --skip-missing --verbose --log-every 5000 > log_manifest_viet_bud500.log 2>&1 &


nohup python -m vpb_mod.dataset._3_big_ds_to_nemo \
  --dataset viet_bud500 \
  --in-root   /mnt/efs/preprocess-3/manifest/viet_bud500 \
  --audio-root /mnt/efs/preprocess-3/audio/viet_bud500 \
  --out-root  /home/ubuntu/work/public_datasets/vi_small/nemo_manifests_big \
  --skip-missing --verbose --log-every 5000 \
  > log_manifest_viet_bud500.log 2>&1 &



'''
