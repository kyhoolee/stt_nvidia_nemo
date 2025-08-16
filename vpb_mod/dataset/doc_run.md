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


-----------------------------

## CMD

python _1_nemo_manifest_format.py \
  --root ~/work/public_datasets/vi_small \
  --datasets vivos


python -m vpb_mod.dataset._1_nemo_manifest_format \
  --root ~/work/public_datasets/vi_small \
  --ensure-sr 16000 \
  --lowercase