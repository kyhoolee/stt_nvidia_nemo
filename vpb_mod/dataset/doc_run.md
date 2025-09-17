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


## STANDARD-DATASET 

(base) ubuntu@ip-10-0-14-129:~/work/public_datasets/vi_small/nemo_manifests/vietspeech$ ls
train_000.jsonl  train_002.jsonl  train_004.jsonl
train_001.jsonl  train_003.jsonl
(base) ubuntu@ip-10-0-14-129:~/work/public_datasets/vi_small/nemo_manifests/vietspeech$ head -n 2 train_000.jsonl 
{"audio_filepath": "/home/ubuntu/work/public_datasets/vi_small/audio/vietspeech/train/shard_0179/vietspeech_train_000899244.wav", "duration": 4.512, "text": "chứ chẳng có ý gì đâu san ngạc nhiên rồi ngẫm nghĩ rồi y hỏi", "sample_rate": 16000, "dataset": "vietspeech"}
{"audio_filepath": "/home/ubuntu/work/public_datasets/vi_small/audio/vietspeech/train/shard_0055/vietspeech_train_000277941.wav", "duration": 5.664, "text": "nhưng kẻ trộm sách giỏi việc đọc và phá hủy những quyển sách hơn là đưa ra những giả thiết", "sample_rate": 16000, "dataset": "vietspeech"}

-------------------------


python -m vpb_mod.dataset._5_1_viet_speech_hf_ds \
  --out-root ~/work/public_datasets/vi_small \
  --split train \
  --manifest-shard-size 250000 \
  --num-workers 8

python -m vpb_mod.dataset._5_2_viet_speech_split \
  --in-glob "~/work/public_datasets/vi_small/nemo_manifests/vietspeech/train_*.jsonl" \
  --out-dir  "~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits" \
  --train 0.90 --dev 0.05 --test 0.05


-------

127.0.0.1:/                     8.0E  2.6T  8.0E   1% /mnt/efs

-------

(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo$ head -n 1 ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits/test.jsonl
{"audio_filepath": "/home/ubuntu/work/public_datasets/vi_small/audio/vietspeech/train/shard_0116/vietspeech_train_000582586.wav", "duration": 5.75, "text": "nguồn năng lượng tích cực của bạn rất có giá trị và nó không thể lãng phí cho những người như vậy", "sample_rate": 16000, "dataset": "vietspeech"}

-------

(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo$ python -m vpb_mod.dataset._5_2_viet_speech_split \
  --in-glob "~/work/public_datasets/vi_small/nemo_manifests/vietspeech/train_*.jsonl" \
  --out-dir  "~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits" \
  --train 0.90 --dev 0.05 --test 0.05
=== Split Summary ===
Input files    : 5
Total read     : 1026047
Dropped        : 0
Kept           : 1026047
  Train        : 923491
  Dev          : 51309
  Test         : 51247
Outputs:
  train -> ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits/train.jsonl
  dev   -> ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits/dev.jsonl
  test  -> ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits/test.jsonl
✅ Done.


---------

python  -m vpb_mod.dataset._5_3_move_data_s3 copy-audio \
  --src-manifest-dir ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits \
  --dst-root /mnt/efs/share-ds/vietspeech \
  --num-workers 8


---------

python -m vpb_mod.dataset._5_3_move_data_s3 remap-manifest \
  --src-manifest-dir ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits \
  --dst-root /mnt/efs/share-ds/vietspeech \
  --out-manifest-dir /mnt/efs/share-ds/vietspeech/manifest


### Common-voice

python -m vpb_mod.dataset._6_1_common_voice_hf_ds \
  --out-root ~/work/public_datasets/vi_small

# 1) Tách 5% từ train thành dev, giữ phân phối duration (khuyến nghị)
python  -m vpb_mod.dataset._6_2_common_voice_split \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/train.jsonl \
  --train-out     ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/train.jsonl \
  --dev-out       ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/dev.jsonl \
  --dev-ratio 0.05 \
  --stratify-duration \
  --backup

# 2) Chạy thử xem thống kê (không ghi file)
python split_train_to_dev.py \
  --train-manifest .../train.jsonl \
  --train-out .../train.jsonl \
  --dev-out .../dev.jsonl \
  --dev-ratio 0.1 \
  --dry-run



#### VPB-dataset NEMO_MANIFEST

python -m vpb_mod.dataset._8_vpb_label_manifest \
  --data-root /home/ubuntu/workspace/col.tool_label_speech_to_text/data/ods \
  --out /home/ubuntu/work/clean_dataset_vpb/manifest/label_batch_092025





(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo$ 

python -m vpb_mod.dataset._8_1_split_vpb_ds \
  --in /home/ubuntu/work/clean_dataset_vpb/manifest/label_batch_092025 \
  --out-dir /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack \
  --train-ratio 0.80 --val-ratio 0.1 --test-ratio 0.1 \
  --seed 42

[SPLIT BY CLID] (group-wise, no time)
  groups = CLID (fallback CALL::<audio_name>)
  ratios(train/val/test) = 0.80/0.10/0.10 (seed=42)
  all        : train=83633, val=10350, test=10584
  right_only : train=36565, val=4572, test=4638
  left_only  : train=47068, val=5778, test=5946

(base) ubuntu@ip-10-0-14-129:~/work/clean_dataset_vpb/manifest/splits_by_clid_tripack$ tree
.
├── all
│   ├── test.jsonl
│   ├── train.jsonl
│   └── val.jsonl
├── left_only
│   ├── test.jsonl
│   ├── train.jsonl
│   └── val.jsonl
└── right_only
    ├── test.jsonl
    ├── train.jsonl
    └── val.jsonl

3 directories, 9 files


python -m vpb_mod.dataset._8_2_ds_duration --root /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack --out summary.tsv


== Per-dataset summary ==
*        all: n=104,567, sec=198,655.53 (55:10:56)
*  left_only: n=58,792, sec=135,077.30 (37:31:17)
* right_only: n=45,775, sec=63,578.23 (17:39:38)