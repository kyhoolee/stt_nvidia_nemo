
## Vietspeech 
python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vietspeech/vpb_asr_fastconformer/2025-08-25_07-42-00/checkpoints/vpb_asr_fastconformer.nemo

/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/summary_20250916_050333.tsv

model	dataset	wer	log_path	hard_samples
vpb_asr_fastconformer	standard_test_2	0.3546227461288865	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__standard_test_2__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__standard_test_2__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	standard_test	0.3378060263653484	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__standard_test__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__standard_test__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	next_day_test_debug	0.34140641597620563	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__next_day_test_debug__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__next_day_test_debug__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_right2_train	0.3633193573830379	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_right2_train__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_right2_train__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_right2_valid	0.39017875487980275	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_right2_valid__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_right2_valid__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_all_train	0.38960049683172737	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_all_train__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_all_train__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_all_valid	0.38758716315334685	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_all_valid__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_all_valid__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_all_test	0.381030550014793	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_all_test__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_all_test__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_right_train	0.4569819998414083	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_right_train__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_right_train__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_right_valid	0.4568150911817009	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_right_valid__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_right_valid__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_right_test	0.4379526631319957	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_right_test__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_right_test__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_left_train	0.3639714374667323	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_left_train__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_left_train__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_left_valid	0.3596200262537581	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_left_valid__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_left_valid__vpb_asr_fastconformer_hard.tsv
vpb_asr_fastconformer	vpb_label_left_test	0.35972445404494086	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_left_test__vpb_asr_fastconformer.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_050333/hardfix__vpb_label_left_test__vpb_asr_fastconformer_hard.tsv




## Pseudo 

python -m vpb_mod.model._2_fastformer_infer \
  --devices 1 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo 


model	dataset	wer	log_path	hard_samples
vpb_asr_fastconformer_ft_poc_qc_v1	standard_test_2	0.30402102928492214	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	standard_test	0.2582391713747646	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	next_day_test_debug	0.2687486721903548	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_right2_train	0.29557059238656647	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_right2_valid	0.3240189028148757	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_all_train	0.3054520489456881	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_all_train__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_all_train__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_all_valid	0.30456363161384964	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_all_valid__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_all_valid__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_all_test	0.30387768541406196	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_all_test__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_all_test__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_right_train	0.3494259861320364	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_right_train__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_right_train__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_right_valid	0.34185292374211096	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_right_valid__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_right_valid__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_right_test	0.33991752434417993	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_right_test__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_right_test__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_left_train	0.2887704920776799	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_left_train__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_left_train__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_left_valid	0.28962411958163364	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_left_valid__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_left_valid__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v1	vpb_label_left_test	0.2902979373567609	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_left_test__vpb_asr_fastconformer_ft_poc_qc_v1.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_082811/hardfix__vpb_label_left_test__vpb_asr_fastconformer_ft_poc_qc_v1_hard.tsv



## Vpb-smallset
python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v2/2025-09-03_03-23-34/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v2.nemo


model	dataset	wer	log_path	hard_samples
vpb_asr_fastconformer_ft_poc_qc_v2	standard_test_2	0.24200106789337494	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__standard_test_2__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	standard_test	0.4284369114877589	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__standard_test__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	next_day_test_debug	0.26485376389774096	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_right2_train	0.24563078583585868	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_right2_valid	0.28272036161906716	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_all_train	0.2868487574354944	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_all_train__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_all_train__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_all_valid	0.2856439523340635	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_all_valid__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_all_valid__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_all_test	0.28444631080655475	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_all_test__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_all_test__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_right_train	0.3192935620578155	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_right_train__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_right_train__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_right_valid	0.31151713797552216	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_right_valid__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_right_valid__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_right_test	0.30803617839692277	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_right_test__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_right_test__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_left_train	0.2744641177317358	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_left_train__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_left_train__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_left_valid	0.2750857481615312	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_left_valid__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_left_valid__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv
vpb_asr_fastconformer_ft_poc_qc_v2	vpb_label_left_test	0.27554595505913965	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_left_test__vpb_asr_fastconformer_ft_poc_qc_v2.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_070643/hardfix__vpb_label_left_test__vpb_asr_fastconformer_ft_poc_qc_v2_hard.tsv



## Vpb-bigset-1

python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_vpb_ds_092025/2025-09-16_09-06-46/checkpoints/vpb_asr_fastconformer_ft_vpb_ds_092025.nemo



model	dataset	wer	log_path	hard_samples
vpb_asr_fastconformer_ft_vpb_ds_092025	standard_test_2	0.23579907175422024	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__standard_test_2__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__standard_test_2__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	standard_test	0.3987758945386064	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__standard_test__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__standard_test__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	next_day_test_debug	0.3152043056440762	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__next_day_test_debug__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_right2_train	0.2838640043173233	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_right2_train__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_right2_valid	0.2734744195603041	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_right2_valid__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_all_train	0.21552138220139153	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_all_train__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_all_train__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_all_valid	0.23389867974197698	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_all_valid__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_all_valid__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_all_test	0.23489439677798032	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_all_test__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_all_test__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_right_train	0.1896844906122521	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_right_train__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_right_train__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_right_valid	0.2551344189127933	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_right_valid__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_right_valid__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_right_test	0.2543576948400735	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_right_test__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_right_test__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_left_train	0.22537854060586582	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_left_train__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_left_train__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_left_valid	0.2254435614775502	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_left_valid__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_left_valid__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv
vpb_asr_fastconformer_ft_vpb_ds_092025	vpb_label_left_test	0.22761518400463635	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_left_test__vpb_asr_fastconformer_ft_vpb_ds_092025.log	/home/ubuntu/work/stt_nvidia_nemo/nemo_eval_hardfix/logs_20250916_115249/hardfix__vpb_label_left_test__vpb_asr_fastconformer_ft_vpb_ds_092025_hard.tsv



========================================================================

TRAIN-VALID-TEST on new DATA


## Fine-tuning from Pseudo-checkpoint

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/train.jsonl \
  --val-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/val.jsonl \
  --test-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 \
  --size large \
  --epochs 50 \
  --devices -1 \
  --precision 16 \
  --batch-size 32 \
  --accumulate-grad-batches 2 \
  --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vpb_ft \
  --exp-name vpb_asr_fastconformer_ft_vpb_ds_092025 \
  --init-from-nemo ../nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo \
  --freeze-encoder-ratio 0.8 \
  --unfreeze-at-epoch 2 \
  --grad-clip 1.0 \
  --fastemit-lambda 0.003 \
  > vpb_mod/logs/vpb_ft_ds_092025_freeze_80.log 2>&1 &


## Fine-tuning with different freezing option 

## 1. Freeze bottom then unblock 

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/train.jsonl \
  --val-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/val.jsonl \
  --test-manifest  /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 --size large --epochs 30 --devices -1 --precision 16 \
  --batch-size 32 --accumulate-grad-batches 2 --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vpb_ft \
  --exp-name vpb_asr_fastconformer_ft_vpb_ds_sched_A \
  --init-from-nemo ../nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo \
  --grad-clip 1.0 --fastemit-lambda 0.003 --freeze-dump \
  --stage 'e=0,enc_bottom_k=12,pre=1,subs=1,pos=1,dec_all=1,joint=1' \
  --stage 'e=6,enc_bottom_k=6'  --freeze-dump-stages \
  > vpb_mod/logs/vpb_bigset_ft_sched_A.log 2>&1 &



## 2. Mid/High only 

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/train.jsonl \
  --val-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/val.jsonl \
  --test-manifest  /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 --size large --epochs 15 --devices -1 --precision 16 \
  --batch-size 32 --accumulate-grad-batches 2 --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vpb_ft \
  --exp-name vpb_asr_fastconformer_ft_vpb_ds_sched_B \
  --init-from-nemo ../nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo \
  --grad-clip 1.0 --fastemit-lambda 0.003 --freeze-dump \
  --stage 'e=0,enc_bottom_k=11,pre=1,subs=1,pos=1,dec_all=1,joint=1'   --freeze-dump-stages \
  > vpb_mod/logs/vpb_bigset_ft_sched_B.log 2>&1 &



## 3. Sandwich 

nohup python -m vpb_mod.model._1_fastformer_trans_bpe \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --train-manifest /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/train.jsonl \
  --val-manifest   /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/val.jsonl \
  --test-manifest  /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/right_only/test.jsonl \
  --tokenizer-dir  ../nemo_work/_1_small_vi_ds/tokenizers/vietspeech \
  --vocab-size 1024 --size large --epochs 18 --devices -1 --precision 16 \
  --batch-size 32 --accumulate-grad-batches 2 --max-duration 17.0 \
  --exp-dir ../nemo_work/_1_small_vi_ds/experiments/vpb_ft \
  --exp-name vpb_asr_fastconformer_ft_vpb_ds_sched_C \
  --init-from-nemo ../nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_ft_poc_qc_v1/2025-08-29_09-18-14/checkpoints/vpb_asr_fastconformer_ft_poc_qc_v1.nemo \
  --grad-clip 1.0 --fastemit-lambda 0.003 --freeze-dump \
  --stage 'e=0,enc_bottom_k=8,enc_top_k=3,pre=1,subs=1,pos=1,dec_all=1,joint=1'   --freeze-dump-stages \
  > vpb_mod/logs/vpb_bigset_ft_sched_C.log 2>&1 &


