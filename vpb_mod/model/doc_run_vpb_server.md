Đã refactor xong thành file `train_fastconformer_vpb.py` (mình đặt trên canvas để bạn xem/chỉnh trực tiếp).

## Có gì mới & sạch hơn

* Tách bạch rõ 5 phần: tokenizer → config → LR/warmup scaling → trainer/loggers → train/test.
* Dùng `json.loads` thay `eval` khi đọc manifest (an toàn hơn).
* Hỗ trợ preset kích thước mô hình (`--size {small, medium, large}`) khớp bảng của NeMo:

  * `small`: d\_model=176, n\_heads=4, n\_layers=16, pred/joint=320
  * `medium`: d\_model=256, n\_heads=4, n\_layers=16, pred/joint=640
  * `large`: d\_model=512, n\_heads=8, n\_layers=17, pred/joint=640
* Scale LR theo global batch size (per-GPU \* devices \* accumulate) + warmup \~5% tổng steps.
* Cho phép tắt SpecAug nhanh bằng `--disable-specaug` khi debug/không ổn định.
* `exp_manager` thiết lập logdir + checkpoint; tự gắn thêm CSVLogger bên cạnh TensorBoard.
* CLI rõ ràng: chỉ định manifest train/val/test, tokenizer dir, vocab size, epochs, devices, precision, batch size, accumulate, max\_duration…

## Cách chạy nhanh (ví dụ với merged manifests)

```bash
python train_fastconformer_vpb.py \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/merged_manifests/merged_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/merged_manifests/merged_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/merged_manifests/merged_test.jsonl \
  --tokenizer-dir  ./tokenizers \
  --vocab-size 256 \
  --size small \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 32 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ./experiments \
  --exp-name vpb_asr_fastconformer
```

## Nếu muốn giữ spec-augment mặc định

Bỏ `--disable-specaug` (mặc định là bật). Khi cần tắt để ổn định giai đoạn đầu thì thêm flag đó.

## Gợi ý kiểm tra khớp data

* Train/val/test manifest phải theo schema NeMo: `audio_filepath`, `duration`, `text` (có thể thêm `sample_rate`).
* Tokenizer sẽ đọc **train manifest** để gom corpus. Có `--lowercase-text` nếu muốn chuẩn hóa.

Cần mình bổ sung preset `xl`/`xxl` theo YAML gốc, hay thêm tuỳ chọn `--xscaling False` cho biến thể 1.2B thì nói mình thêm luôn nhé.


------------------------------------------------

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

📁 Dataset: speech_massive
  🔹 test: 2974 wav files | 2974 manifest entries
  🔹 train: 115 wav files | 115 manifest entries
  🔹 validation: 2033 wav files | 2033 manifest entries

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



## RUN 


### TRAIN 

python train_fastconformer_vpb.py \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/merged_manifests/merged_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/merged_manifests/merged_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/merged_manifests/merged_test.jsonl \
  --tokenizer-dir  ./tokenizers \
  --vocab-size 256 \
  --size small \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 32 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments \
  --exp-name vpb_asr_fastconformer


python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests/vietmed/vietmed_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests/vietmed/vietmed_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests/vietmed/vietmed_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietmed \
  --vocab-size 256 \
  --size small \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 32 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vietmed \
  --exp-name vpb_asr_fastconformer



pip install \       
  pytorch-lightning==2.2.5 \
  hydra-core==1.3.2 \
  omegaconf==2.3.0 \
  sentencepiece==0.2.0 \
  soundfile==0.12.1 \
  librosa==0.10.2.post1 \
  einops==0.8.0 \
  torchaudio-augmentations==0.2.4




python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/lsvsc \
  --vocab-size 256 \
  --size small_0 \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 128 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/lsvsc \
  --exp-name vpb_asr_fastconformer


nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/lsvsc \
  --vocab-size 256 \
  --size small_0 \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 128 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/lsvsc \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/lsvsc.log 2>&1 &


### TRAIN-FULL 


nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/merged \
  --vocab-size 256 \
  --size small_0 \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 256 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/merged \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/merged.log 2>&1 &



nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/merged \
  --vocab-size 512 \
  --size medium \
  --epochs 200 \
  --devices -1 \
  --precision 16 \
  --batch-size 128 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/merged_2 \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/merged_2_medium.log 2>&1 &


nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/merged \
  --vocab-size 512 \
  --size medium \
  --epochs 200 \
  --devices -1 \
  --precision 16 \
  --batch-size 128 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/merged_2 \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/merged_2_medium.log 2>&1 &


--------------------------

ls /mnt/efs/


--------------------------


nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/merged_3_large \
  --vocab-size 1024 \
  --size large \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 128 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/merged_3_large \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/merged_3_large.log 2>&1 &





nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/merged_3_large \
  --vocab-size 1024 \
  --size large \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 64 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/merged_3_large \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/merged_3_large.log 2>&1 &

---------------------------------

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests_processed_merged/merged_test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/merged_3_large \
  --vocab-size 1024 \
  --size large \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 64 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/merged_3_large \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/merged_3_large.log 2>&1 &

(base) ubuntu@ip-10-0-14-129:~/work/public_datasets/vi_small/nemo_manifests_big/vi_voice$ tree
.
├── dev
│   ├── vi_voice_dev_manifest.jsonl
│   └── vi_voice_dev_manifest_origin.jsonl
├── test
│   ├── vi_voice_dev_manifest.jsonl
│   ├── vi_voice_test_manifest.jsonl
│   └── vi_voice_test_manifest_origin.jsonl
└── train
    └── vi_voice_train_manifest.jsonl


-------------------------------------

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests_big/vi_voice/train/vi_voice_train_manifest.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests_big/vi_voice/dev/vi_voice_dev_manifest_origin.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests_big/vi_voice/test/vi_voice_test_manifest_origin.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vi_voice \
  --vocab-size 1024 \
  --size large \
  --epochs 20 \
  --devices -1 \
  --precision 16 \
  --batch-size 64 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vi_voice \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/vi_voice.log 2>&1 &


--------------------------------------

#### VIET_SPEECH

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits/train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits/dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests/vietspeech_splits/test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 \
  --size large \
  --epochs 20 \
  --devices -1 \
  --precision 16 \
  --batch-size 64 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vietspeech \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/vietspeech.log 2>&1 &



#### COMMON_VOICE 

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/train.jsonl \
  --val-manifest   ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/dev.jsonl \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests/common_voice_8_0_vi/test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/common_voice_vi \
  --vocab-size 1024 \
  --size large \
  --epochs 20 \
  --devices -1 \
  --precision 16 \
  --batch-size 64 \
  --accumulate-grad-batches 1 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/common_voice_vi \
  --exp-name vpb_asr_fastconformer > vpb_mod/logs/common_voice_vi.log 2>&1 &



### TEST 


[
    "/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test_2/test_meta.jsonl",
    "/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test/test_meta.jsonl",
    "/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/standard_test/next_day_test_meta_debug.jsonl",
    "/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/manifest_vpb_right_2/train_meta.jsonl",
    "/home/kylh/work/public_datasets/vi_small/nemo_manifests/vpb_ds/manifest_vpb_right_2/valid_meta.jsonl",
]




# Initialize the model -> @NOTE vẫn bám theo code mẫu 
model = nemo_asr.models.EncDecRNNTBPEModel(cfg=config.model, trainer=trainer)
# model.summarize()
# print(model)

# Load pre-trained weights if available
ckpt_dir = "..."
pretrained_model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(os.path.join(ckpt_dir,vpb_asr_fastconformer_transducer_bpe--val_wer=0.3318-epoch=100-last.ckptt"), map_location='cpu')
model.encoder.load_state_dict(pretrained_model.encoder.state_dict(), strict=True)


------------------


python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_test.jsonl \
  --devices 1 \
  --precision 16 \
  --batch-size 64 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments \
  --exp-name vpb_asr_fastconformer_testonly_ckpt100 \
  --ckpt ../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer--val_wer=0.1370-epoch=100-last.ckpt \
  --test-only


python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --test-manifest  ~/work/public_datasets/vi_small/nemo_manifests/lsvsc/lsvsc_valid.jsonl \
  --devices 1 \
  --precision 16 \
  --batch-size 64 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments \
  --exp-name vpb_asr_fastconformer_testonly \
  --nemo ../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo \
  --test-only


---------------------

python -m vpb_mod.model._2_vpb_manifest_convert \
  --input /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta.json \
  --audio-base /home/ubuntu/work/clean_dataset_vpb/audio \
  --output /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta_nemo.jsonl


python -m vpb_mod.model._2_fastformer_infer \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_v1/2025-08-27_07-42-39/checkpoints/vpb_asr_fastconformer_ft_v1.nemo





python -m vpb_mod.model._2_fastformer_infer \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vietspeech/vpb_asr_fastconformer/2025-08-25_07-42-00/checkpoints/vpb_asr_fastconformer.nemo

python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v2/2025-09-03_03-23-34/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v2.nemo

/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v2/2025-09-03_03-23-34/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v2.nemo


model	dataset	wer	log_path
vpb_asr_fastconformer	standard_test_2	0.3547048917731137	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_042152/hardfix__standard_test_2__vpb_asr_fastconformer.log
vpb_asr_fastconformer	standard_test	0.3380414312617702	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_042152/hardfix__standard_test__vpb_asr_fastconformer.log
vpb_asr_fastconformer	next_day_test_debug	0.3419729480914949	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_042152/hardfix__next_day_test_debug__vpb_asr_fastconformer.log
vpb_asr_fastconformer	vpb_right2_train	0.3635269210012869	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_042152/hardfix__vpb_right2_train__vpb_asr_fastconformer.log
vpb_asr_fastconformer	vpb_right2_valid	0.3895623587425519	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_042152/hardfix__vpb_right2_valid__vpb_asr_fastconformer.log


MODE="train" bash vpb_mod/model/_2_vpb_manifest_convert.sh



#### FINE-TUNE VPB FROM VIET-SPEECH
export CUDA_VISIBLE_DEVICES=4,5,6,7


export CUDA_VISIBLE_DEVICES=4,5,6,7 

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta_train.jsonl \
  --val-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_train.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 \
  --size large \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 32 \
  --accumulate-grad-batches 2 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vpb_ft \
  --exp-name vpb_asr_fastconformer_ft_v1 \
  --init-from-nemo ../nemo_work/_1_small_vi_ds/experiments/vietspeech/vpb_asr_fastconformer/2025-08-25_07-42-00/checkpoints/vpb_asr_fastconformer.nemo \
  --freeze-encoder-ratio 0.2 \
  --unfreeze-at-epoch 2 \
  --grad-clip 1.0 \
  --fastemit-lambda 0.003  > vpb_mod/logs/vpb_ft.log 2>&1 &


============= PSEUDO_LABEL_FINE_TUNE

export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/train_meta_nemo.jsonl \
  --val-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/valid_meta_nemo.jsonl \
  --test-manifest  /home/ubuntu/work/clean_dataset_vpb/manifest/poc_qc_user/test_meta_nemo.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 \
  --size large \
  --epochs 100 \
  --devices -1 \
  --precision 16 \
  --batch-size 32 \
  --accumulate-grad-batches 2 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vpb_ft \
  --exp-name vpb_asr_fastconformer_ft_poc_qc_v1 \
  --init-from-nemo ../nemo_work/_1_small_vi_ds/experiments/vietspeech/vpb_asr_fastconformer/2025-08-25_07-42-00/checkpoints/vpb_asr_fastconformer.nemo \
  --freeze-encoder-ratio 0.2 \
  --unfreeze-at-epoch 2 \
  --grad-clip 1.0 \
  --fastemit-lambda 0.003 \
  > vpb_mod/logs/vpb_ft_poc_qc.log 2>&1 &


  

=============

model	dataset	wer	log_path
vpb_asr_fastconformer_ft_v1	standard_test	0.3100282485875706	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_121021/hardfix__standard_test__vpb_asr_fastconformer_ft_v1.log
vpb_asr_fastconformer_ft_v1	next_day_test_debug	0.2634374336095177	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_121021/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_v1.log
vpb_asr_fastconformer_ft_v1	vpb_right2_train	0.047033915895221885	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_121021/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_v1.log
vpb_asr_fastconformer_ft_v1	vpb_right2_valid	0.30675981097185123	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250827_121021/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_v1.log

===============

@TODO: viết script kiểm tra xem standard_test_2, standard_test, next_day_test_debug có bị duplicate data với vpb_right2_train ko ???
-> Đảm bảo dữ liệu ko bị leak (trực tiếp)

python -m vpb_mod.model._2_vpb_manifest_verify \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/next_day_test_meta_debug_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_nemo.jsonl


python -m vpb_mod.model._2_vpb_manifest_verify \
  --anchor /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/next_day_test_meta_debug_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_nemo.jsonl


(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo$ python -m vpb_mod.model._2_vpb_manifest_verify \
  --anchor /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/next_day_test_meta_debug_nemo.jsonl \
  /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_nemo.jsonl
[ANCHOR] /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta_nemo.jsonl -> lines=3072 unique_ids=3072

=== Overlap with Anchor (Summary) ===
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test_2/test_meta_nemo.jsonl -> unique=2993, overlap=2096 (0.700301)
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl -> unique=29, overlap=0 (0.000000)
/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/next_day_test_meta_debug_nemo.jsonl -> unique=1650, overlap=0 (0.000000)
/home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_nemo.jsonl -> unique=630, overlap=0 (0.000000)



[text](../../../nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo)

python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo \
  --hard-topk 80 --min-words 5

model	dataset	wer	log_path
vpb_asr_fastconformer_ft_poc_qc_v1	standard_test_2	0.3045139031502855	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_022954/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v1.log
vpb_asr_fastconformer_ft_poc_qc_v1	standard_test	0.2558851224105461	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_022954/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v1.log
vpb_asr_fastconformer_ft_poc_qc_v1	next_day_test_debug	0.26867785567594366	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_022954/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v1.log
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_right2_train	0.29606874507036407	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_022954/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v1.log
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_right2_valid	0.32525169508937746	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_022954/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v1.log


python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo \
  --denoise --df-sr 48000 \
  --hard-topk 80 --min-words 5


============

export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta_train.jsonl \
  --val-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_train.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 \
  --size large \
  --epochs 50 \
  --devices -1 \
  --precision 16 \
  --batch-size 64 \
  --accumulate-grad-batches 2 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vpb_ft \
  --exp-name vpb_asr_fastconformer_ft_poc_qc_v2 \
  --init-from-nemo ../nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo \
  --freeze-encoder-ratio 0.4 \
  --unfreeze-at-epoch 2 \
  --grad-clip 1.0 \
  --fastemit-lambda 0.003 \
  > vpb_mod/logs/vpb_ft_poc_qc_v2.log 2>&1 &

-------============-------

# Tạo session mới tên vpb_ft
tmux new -s vpb_ft
tmux attach -t vpb_ft


# Trong cửa sổ tmux, chạy lệnh sau:
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/train_meta_train.jsonl \
  --val-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/manifest_vpb_right_2/valid_meta_train.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 \
  --size large \
  --epochs 50 \
  --devices -1 \
  --precision 16 \
  --batch-size 64 \
  --accumulate-grad-batches 2 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vpb_ft \
  --exp-name vpb_asr_fastconformer_ft_poc_qc_v2 \
  --init-from-nemo ../nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo \
  --freeze-encoder-ratio 0.4 \
  --unfreeze-at-epoch 2 \
  --grad-clip 1.0 \
  --fastemit-lambda 0.003 \
  > vpb_mod/logs/vpb_ft_poc_qc_v2.log 2>&1


model	dataset	wer	log_path
vpb_asr_fastconformer_ft_poc_qc_v2	standard_test_2	0.24204214071548857	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_070603/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v2.log
vpb_asr_fastconformer_ft_poc_qc_v2	standard_test	0.4293785310734463	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_070603/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v2.log
vpb_asr_fastconformer_ft_poc_qc_v2	next_day_test_debug	0.26485376389774096	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_070603/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v2.log
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_right2_train	0.24563078583585868	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_070603/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v2.log
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_right2_valid	0.2821039654818163	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250903_070603/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v2.log
