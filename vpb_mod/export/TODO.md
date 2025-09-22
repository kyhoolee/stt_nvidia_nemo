
## PERFORMANCE SUMMARY

dataset	chunkformer	vietspeech	pseudo_v1	smallset_v2	bigset_1	bigset_full_sched_eqv
standard_test_2	0.25	0.355	0.304	0.242	0.236	0.233
standard_test	0.161	0.338	0.258	0.428	0.399	0.219
next_day_test_debug	0.208	0.341	0.269	0.265	0.315	0.168
vpb_right2_train	0.242	0.363	0.296	0.246	0.284	0.233
vpb_right2_valid	0.27	0.39	0.324	0.283	0.274	0.242
vpb_label_all_train	0.215	0.39	0.306	0.287	0.216	0.085
vpb_label_all_valid	0.219	0.388	0.305	0.286	0.234	0.159
vpb_label_all_test	0.208	0.381	0.304	0.284	0.235	0.161
vpb_label_right_train	0.245	0.457	0.349	0.319	0.19	0.122
vpb_label_right_valid	0.25	0.457	0.342	0.312	0.255	0.251
vpb_label_right_test	0.231	0.438	0.34	0.308	0.254	0.252
vpb_label_left_train	0.203	0.364	0.289	0.274	0.225	0.071
vpb_label_left_valid	0.207	0.36	0.29	0.275	0.225	0.121
vpb_label_left_test	0.2	0.36	0.29	0.276	0.228	0.126


=============



## Vpb-bigset-1 train all(left_right)

python -m vpb_mod.model._2_fastformer_infer \
  --devices 3 \
  --base-config tutorials/asr/configs/fast-conformer_transducer_bpe.yaml \
  --hardfix-vpb \
  --hardfix-model /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo



## Model được load như sau 


def test_from_checkpoint(
    base_config: Path,
    test_manifest: Path,
    exp_dir: Path,
    exp_name: str,
    devices: int,
    precision: str,
    batch_size: int,
    nemo_path: Optional[Path] = None,
    ckpt_path: Optional[Path] = None,
):
    """
    Manual path (không dùng denoise ở đây để giữ nguyên tham chiếu; pipeline chính dùng batch + transcribe).
    """
    print("🚀 Starting test-only mode...")

    if nemo_path:
        print(f"🧠 Restoring model from .nemo: {nemo_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=str(nemo_path))
    elif ckpt_path:
        print(f"🧠 Restoring model from .ckpt: {ckpt_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(checkpoint_path=str(ckpt_path))
    else:
        raise ValueError("Must provide either --nemo or --ckpt for test-only mode.")

    model.eval()
    try:
        if hasattr(model, 'spec_augmentation') and model.spec_augmentation is not None:
            print("❗ Disabling SpecAugmentation for inference.")
            model.spec_augmentation.mask_prob = 0.0
            model.spec_augmentation = None
        if hasattr(model, 'preprocessor'):
            if hasattr(model.preprocessor, 'dither'):
                model.preprocessor.dither = 0.0
            if hasattr(model.preprocessor, 'pad_to'):
                model.preprocessor.pad_to = 0
    except Exception as e:
        print(f"⚠️ Could not disable augmentations: {e}")

    try:
        print("💡 Forcing greedy_batch decoding strategy.")
        model.change_decoding_strategy(decoder_type="greedy_batch")
        if hasattr(model, 'wer'):
            model.wer.log_prediction = False
    except Exception as e:
        print(f"⚠️ Could not set greedy decoder: {e}")

    def transcribe_audio(audio_path, model):
        audio, _ = librosa.load(audio_path, sr=16000)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(model.device)
        audio_len = torch.tensor([audio_tensor.shape[1]]).to(model.device)
        with torch.no_grad():
            logits = model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
            transcripts = model.decoding.rnnt_decoder_predictions_tensor(logits[0], logits[1])
            return transcripts[0]


    ....


## Target 
- Mình cần đóng gói Nemo
    - copy riêng các weight của các module cần thiết của nemo ra 1 chỗ 
    - chạy khởi tạo lại model từ đầu 
    - chỉ chạy phần inference 

- Sau đó là logic 
    - Convert ra ONNX để tối ưu deploy 
    - Cần convert + chạy test lại để verify chính xác performance + speed ra sao so với bản gốc 

- Thực thi
    - Thảo luận dần từng bước 
    - Cấu trúc thư mục code + data cần thiết 
    - Tác vụ từng phần cần làm và cơ bản các bước gì 