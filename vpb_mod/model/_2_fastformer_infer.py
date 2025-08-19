from __future__ import annotations
import argparse
import json
import os
import json
import librosa
import torch
from pathlib import Path
from typing import Optional
from nemo.collections import asr as nemo_asr

from jiwer import wer  # <-- Import the 'wer' function from 'jiwer'

from ._1_fastformer_trans_bpe import SIZE_PRESETS

# --------------------------------- CLI ---------------------------------
# --- (2) Ở parse_args(): thêm nhóm lựa chọn resume + flag test-only ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train / Test FastConformer-Transducer on NeMo manifests")

    # Data + tokenizer
    p.add_argument('--train-manifest', type=Path, required=False)  # <-- bỏ required để test-only không cần
    p.add_argument('--val-manifest', type=Path, default=None)
    p.add_argument('--test-manifest', type=Path, default=None)
    p.add_argument('--tokenizer-dir', type=Path, required=False)
    p.add_argument('--vocab-size', type=int, default=128)
    p.add_argument('--spe-type', type=str, default='unigram', choices=['unigram', 'bpe'])
    p.add_argument('--lowercase-text', action='store_true')

    # Base NeMo config
    p.add_argument('--base-config', type=Path, required=True)

    # Training knobs
    p.add_argument('--size', type=str, default='small', choices=list(SIZE_PRESETS.keys()))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--devices', type=int, default=1, help='-1 uses all available GPUs')
    p.add_argument('--precision', type=str, default='16', choices=['16', '32', 'bf16'])
    p.add_argument('--batch-size', type=int, default=32, help='per-GPU batch size')
    p.add_argument('--accumulate-grad-batches', type=int, default=1)
    p.add_argument('--max-duration', type=float, default=17.0)
    p.add_argument('--disable-specaug', action='store_true')

    # Logging / output
    p.add_argument('--exp-dir', type=Path, default=Path('./experiments'))
    p.add_argument('--exp-name', type=str, default='vpb_asr_fastconformer')

    # Test-only & resume
    group = p.add_mutually_exclusive_group()
    group.add_argument('--nemo', type=Path, help='Path to .nemo file to restore for test-only')
    group.add_argument('--ckpt', type=Path, help='Path to Lightning .ckpt to restore for test-only')
    p.add_argument('--test-only', action='store_true', help='Only run testing from a restored checkpoint')

    return p.parse_args()


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
    print("🚀 Starting test-only mode...")



    if nemo_path:
        print(f"🧠 Restoring model from .nemo: {nemo_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            restore_path=str(nemo_path)
        )
    elif ckpt_path:
        print(f"🧠 Restoring model from .ckpt: {ckpt_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path)
        )
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

    print("=" * 100)
    try:
        print(model.summarize(max_depth=4))
    except Exception:
        pass
    print("=" * 100)

    tokenizer = model.tokenizer

    def transcribe_audio(audio_path, model):
        audio, _ = librosa.load(audio_path, sr=16000)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(model.device)
        audio_len = torch.tensor([audio_tensor.shape[1]]).to(model.device)

        with torch.no_grad():
            logits = model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
            transcripts = model.decoding.rnnt_decoder_predictions_tensor(logits[0], logits[1])
            return transcripts[0]

    # --- Phần tính WER được bổ sung ---
    all_predictions = []
    all_references = []
    
    print("🔍 Running manual transcription and WER calculation...")
    with open(test_manifest, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line)
            # path = os.path.expanduser
            audio_path = os.path.expanduser(item['audio_filepath'])
            reference_text = item['text']

            predicted_text = transcribe_audio(audio_path, model).text

            # print(predicted_text)
            
            # Normalize texts for consistent WER calculation
            all_predictions.append(predicted_text.lower())
            all_references.append(reference_text.lower())

            # Print results every 100 samples for progress tracking
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(lines)} samples.")

                print(f"Sample {i + 1}:")
                print(f"predicted: {predicted_text}")
                print(f"reference: {reference_text}")
                print("-" * 50)
    
    # --- Calculation using `jiwer` ---
    wer_score = wer(all_references, all_predictions)
    
    print("=" * 100)
    print(f"✅ Finished testing.")
    print(f"✨ Final WER for the test set: {wer_score:.4f}")
    print("=" * 100)


# --------------------------------- BATCH ---------------------------------

# --- Sửa đổi hàm test_from_checkpoint ---
def test_batch_from_checkpoint(
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
    print("🚀 Starting test-only mode...")

    if nemo_path:
        print(f"🧠 Restoring model from .nemo: {nemo_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            restore_path=str(nemo_path)
        )
    elif ckpt_path:
        print(f"🧠 Restoring model from .ckpt: {ckpt_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path)
        )
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

    print("=" * 100)
    try:
        print(model.summarize(max_depth=4))
    except Exception:
        pass
    print("=" * 100)

    # --- Phần tính WER được bổ sung ---
    all_predictions = []
    all_references = []
    
    print("🔍 Running manual transcription and WER calculation...")
    
    # Đọc tất cả các dòng từ manifest một lần
    with open(test_manifest, 'r', encoding='utf-8') as f:
        manifest_lines = f.readlines()
        
    num_samples = len(manifest_lines)
    audio_paths = []
    reference_texts = []

    for line in manifest_lines:
        item = json.loads(line)
        audio_path = os.path.expanduser(item['audio_filepath'])
        reference_text = item['text']
        
        audio_paths.append(audio_path)
        reference_texts.append(reference_text)

    # Chia danh sách thành các batch và xử lý
    for i in range(0, num_samples, batch_size):
        batch_audio_paths = audio_paths[i:i + batch_size]
        batch_reference_texts = reference_texts[i:i + batch_size]

        # Sử dụng phương thức transcribe được tích hợp sẵn của NeMo, nó tự động xử lý batch
        predicted_texts = model.transcribe(batch_audio_paths, batch_size=batch_size)
        
        # Thêm kết quả vào danh sách tổng
        for j in range(len(predicted_texts)):
            pred_text = predicted_texts[j].text
            ref_text = batch_reference_texts[j]
            
            all_predictions.append(pred_text.lower())
            all_references.append(ref_text.lower())

            # In kết quả mẫu cho mỗi batch
            # Đây chỉ là ví dụ. Bạn có thể thay đổi cách in tùy ý.
            if j == 0 and i == 0:
                print(f"Sample 1:")
                print(f"predicted: {pred_text}")
                print(f"reference: {ref_text}")
                print("-" * 50)
            
        print(f"Processed {min(i + batch_size, num_samples)}/{num_samples} samples.")
    
    # --- Calculation using `jiwer` ---
    wer_score = wer(all_references, all_predictions)
    
    print("=" * 100)
    print(f"✅ Finished testing.")
    print(f"✨ Final WER for the test set: {wer_score:.4f}")
    print("=" * 100)



# --------------------------------- Main --------------------------------

def main():
    args = parse_args()

    # Nhánh TEST-ONLY: không cần tokenizer build lại, không cần train manifest
    if args.test_only:
        if args.test_manifest is None:
            raise ValueError("--test-only requires --test-manifest")
        test_batch_from_checkpoint(
            base_config=args.base_config,
            test_manifest=args.test_manifest.expanduser().resolve(),
            exp_dir=args.exp_dir.expanduser().resolve(),
            exp_name=args.exp_name,
            devices=args.devices,
            precision=args.precision,
            batch_size=args.batch_size,
            nemo_path=(args.nemo.expanduser().resolve() if args.nemo else None),
            ckpt_path=(args.ckpt.expanduser().resolve() if args.ckpt else None),
        )
        return
    


if __name__ == '__main__':
    main()