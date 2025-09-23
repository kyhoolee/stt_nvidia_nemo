import nemo.collections.asr as nemo_asr
from pathlib import Path

# --- Cấu hình ---
# 1. Đường dẫn đến file .nemo của bạn
nemo_file_path = Path("/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo")

# 2. Tên file ONNX đầu ra
onnx_output_path = "./vpb_fastconformer.onnx"
# -----------------

print(f"🚀 Bắt đầu quá trình export model...")
print(f"🧠 Đang tải model NeMo từ: {nemo_file_path}")

try:
    # Tải model từ file .nemo
    # Đối với model của bạn là EncDecRNNTBPEModel
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=str(nemo_file_path))
    asr_model.eval()

    print("==================================================")
    print(asr_model)
    print("==================================================")

    # Thực hiện export
    print(f"📦 Đang export sang ONNX:: {onnx_output_path}")
    asr_model.export(onnx_output_path)

    print(f"✅ Export thành công! File ONNX đã được lưu tại: {onnx_output_path}")

except Exception as e:
    print(f"❌ Đã xảy ra lỗi: {e}")

===============
(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo/vpb_mod/export_direct$ python _0_hardfix_export.py 
        🚀 Bắt đầu quá trình export model...
🧠 Đang tải model NeMo từ: /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo
[NeMo I 2025-09-23 02:13:59 mixins:181] Tokenizer SentencePieceTokenizer initialized with 1024 tokens
[NeMo W 2025-09-23 02:14:00 modelPT:180] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config : 
    manifest_filepath: /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/all/train.jsonl
    sample_rate: 16000
    batch_size: 64
    shuffle: true
    num_workers: 8
    pin_memory: true
    max_duration: 17.0
    min_duration: 0.1
    is_tarred: false
    tarred_audio_filepaths: null
    shuffle_n: 2048
    bucketing_strategy: fully_randomized
    bucketing_batch_size: null
    
[NeMo W 2025-09-23 02:14:00 modelPT:187] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s). 
    Validation config : 
    manifest_filepath: /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/all/val.jsonl
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    use_start_end_token: false
    num_workers: 8
    pin_memory: true
    return_transcripts: false
    
[NeMo W 2025-09-23 02:14:00 modelPT:194] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config : 
    manifest_filepath: /home/ubuntu/work/clean_dataset_vpb/manifest/splits_by_clid_tripack/all/test.jsonl
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    use_start_end_token: false
    num_workers: 8
    pin_memory: true
    return_transcripts: false
    
[NeMo I 2025-09-23 02:14:00 features:305] PADDING: 0
[NeMo I 2025-09-23 02:14:01 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0}
[NeMo I 2025-09-23 02:14:01 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0}
[NeMo W 2025-09-23 02:14:01 rnnt_loop_labels_computer:290] No conditional node support for Cuda.
    Cuda graphs with while loops are disabled, decoding speed will be slower
    Reason: No `cuda-python` module. Please do `pip install cuda-python>=12.3`
[NeMo I 2025-09-23 02:14:01 rnnt_models:226] Using RNNT Loss : warprnnt_numba
    Loss warprnnt_numba_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0}
[NeMo W 2025-09-23 02:14:01 rnnt_loop_labels_computer:290] No conditional node support for Cuda.
    Cuda graphs with while loops are disabled, decoding speed will be slower
    Reason: No `cuda-python` module. Please do `pip install cuda-python>=12.3`
[NeMo I 2025-09-23 02:14:04 save_restore_connector:275] Model EncDecRNNTBPEModel was successfully restored from /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo.


==================================================
EncDecRNNTBPEModel(
  (preprocessor): AudioToMelSpectrogramPreprocessor(
    (featurizer): FilterbankFeatures()
  )
  (encoder): ConformerEncoder(
    (pre_encode): ConvSubsampling(
      (out): Linear(in_features=2560, out_features=512, bias=True)
      (conv): Sequential(
        (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (4): ReLU(inplace=True)
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (7): ReLU(inplace=True)
      )
    )
    (pos_enc): RelPositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (layers): ModuleList(
      (0-16): 17 x ConformerLayer(
        (norm_feed_forward1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (feed_forward1): ConformerFeedForward(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
        (norm_conv): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (conv): ConformerConvolution(
          (pointwise_conv1): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (depthwise_conv): CausalConv1D(512, 512, kernel_size=(9,), stride=(1,), groups=512)
          (batch_norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Swish()
          (pointwise_conv2): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
        )
        (norm_self_att): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (self_attn): RelPositionMultiHeadAttention(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k): Linear(in_features=512, out_features=512, bias=True)
          (linear_v): Linear(in_features=512, out_features=512, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear_pos): Linear(in_features=512, out_features=512, bias=False)
        )
        (norm_feed_forward2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (feed_forward2): ConformerFeedForward(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm_out): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (decoder): RNNTDecoder(
    (prediction): ModuleDict(
      (embed): Embedding(1025, 640, padding_idx=1024)
      (dec_rnn): LSTMDropout(
        (lstm): LSTM(640, 640, dropout=0.2)
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
  )
  (joint): RNNTJoint(
    (pred): Linear(in_features=640, out_features=640, bias=True)
    (enc): Linear(in_features=512, out_features=640, bias=True)
    (joint_net): Sequential(
      (0): ReLU(inplace=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=640, out_features=1025, bias=True)
    )
    (_loss): RNNTLoss(
      (_loss): RNNTLossNumba()
    )
    (_wer): WER()
  )
  (loss): RNNTLoss(
    (_loss): RNNTLossNumba()
  )
  (spec_augmentation): SpectrogramAugmentation(
    (spec_augment): SpecAugment()
  )
  (wer): WER()
)
==================================================



📦 Đang export sang ONNX:: ./vpb_fastconformer.onnx
[NeMo I 2025-09-23 02:15:42 exportable:135] Successfully exported ConformerEncoder to ./encoder-vpb_fastconformer.onnx
[NeMo I 2025-09-23 02:15:42 exportable:135] Successfully exported RNNTDecoderJoint to ./decoder_joint-vpb_fastconformer.onnx
✅ Export thành công! File ONNX đã được lưu tại: ./vpb_fastconformer.onnx
(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo/vpb_mod/export_direct$ ls
_0_hardfix_export.py  __pycache__                           vpb_fastconformer_deployed
_1_hardfix_infer.py   decoder_joint-vpb_fastconformer.onnx
_2_export_part.py     encoder-vpb_fastconformer.onnx
(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo/vpb_mod/export_direct$ ls -l 
total 466160
-rw-rw-r-- 1 ubuntu ubuntu      1226 Sep 23 02:13 _0_hardfix_export.py
-rw-rw-r-- 1 ubuntu ubuntu      3382 Sep 22 12:10 _1_hardfix_infer.py
-rw-rw-r-- 1 ubuntu ubuntu      2209 Sep 23 01:59 _2_export_part.py
drwxrwxr-x 2 ubuntu ubuntu      4096 Sep 23 02:00 __pycache__
-rw-rw-r-- 1 ubuntu ubuntu  21337391 Sep 23 02:15 decoder_joint-vpb_fastconformer.onnx
-rw-rw-r-- 1 ubuntu ubuntu 455985980 Sep 23 02:15 encoder-vpb_fastconformer.onnx
drwxrwxr-x 4 ubuntu ubuntu      4096 Sep 22 12:13 vpb_fastconformer_deployed
(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo/vpb_mod/export_direct$ vim tmp.txt
(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo/vpb_mod/export_direct$ ls -l 
total 466160
-rw-rw-r-- 1 ubuntu ubuntu      1226 Sep 23 02:13 _0_hardfix_export.py
-rw-rw-r-- 1 ubuntu ubuntu      3382 Sep 22 12:10 _1_hardfix_infer.py
-rw-rw-r-- 1 ubuntu ubuntu      2209 Sep 23 01:59 _2_export_part.py
drwxrwxr-x 2 ubuntu ubuntu      4096 Sep 23 02:00 __pycache__
-rw-rw-r-- 1 ubuntu ubuntu  21337391 Sep 23 02:15 decoder_joint-vpb_fastconformer.onnx
-rw-rw-r-- 1 ubuntu ubuntu 455985980 Sep 23 02:15 encoder-vpb_fastconformer.onnx
drwxrwxr-x 4 ubuntu ubuntu      4096 Sep 22 12:13 vpb_fastconformer_deployed
(nemo) ubuntu@ip-10-0-14-129:~/work/stt_nvidia_nemo/vpb_mod/export_direct$ 

======================

mình hỏi chút 
sao model nemo fastconformer lại export 2 file onnx có tiền tố là encoder và decoder vậy nhỉ ? 



=======================


python _0_hardfix_infer.py \
  --nemo_model="/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo" \
  --dataset_manifest="/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl" \
  --batch_size=32 \
  --max_symbold_per_step=5 \
  --log




python _0_hardfix_infer.py \
  --onnx_encoder="./encoder-vpb_fastconformer.onnx" \
  --onnx_decoder="./decoder_joint-vpb_fastconformer.onnx" \
  --nemo_model="/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv_fixed.nemo" \
  --dataset_manifest="/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl" \
  --batch_size=32 \
  --max_symbold_per_step=5 \
  --log

