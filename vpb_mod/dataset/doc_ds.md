python -m vpb_mod.dataset._4_0_vpb_ds_collect \
  --base-audio-root /home/kylh/work/vpb_dataset \
  --output-root     /home/kylh/work/clean_dataset_vpb \
  --copy-mode       copy \
  --keep-raw-manifest \
  /home/kylh/work/vpb_dataset/standard_test_2/test_meta.json \
  /home/kylh/work/vpb_dataset/standard_test/test_meta.json \
  /home/kylh/work/vpb_dataset/standard_test/next_day_test_meta_debug.json \
  /home/kylh/work/vpb_dataset/manifest_vpb_right_2/train_meta.json \
  /home/kylh/work/vpb_dataset/manifest_vpb_right_2/valid_meta.json


