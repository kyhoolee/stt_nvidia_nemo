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

    # Hard-fix VPB test-suite
    p.add_argument('--hardfix-vpb', action='store_true',
                   help='Run fixed VPB test suite on a given .nemo (ignores train/val).')
    p.add_argument('--hardfix-model', type=Path, default=Path(
        "/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vietspeech/"
        "vpb_asr_fastconformer/2025-08-25_07-42-00/checkpoints/vpb_asr_fastconformer.nemo"
    ), help='Override model .nemo for hard-fix VPB suite.')
    p.add_argument('--hardfix-manifest-root', type=Path, default=Path(
        "/home/ubuntu/work/clean_dataset_vpb/manifest"
    ), help='Root folder of VPB manifests (contains standard_test, standard_test_2, manifest_vpb_right_2).')
    p.add_argument('--hardfix-outdir', type=Path, default=Path("./nemo_eval_hardfix"),
                   help='Output folder for logs and summary.tsv in hard-fix mode.')


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

    # print("=" * 100)
    # try:
    #     print(model.summarize(max_depth=4))
    # except Exception:
    #     pass
    # print("=" * 100)

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

    # print("=" * 100)
    # try:
    #     print(model.summarize(max_depth=4))
    # except Exception:
    #     pass
    # print("=" * 100)

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

        # Normalize kết quả → string
        norm_preds = [ _pred_to_text(p) for p in predicted_texts ]

        for j in range(len(norm_preds)):
            pred_text = norm_preds[j]
            ref_text = batch_reference_texts[j]

            all_predictions.append(pred_text.lower())
            all_references.append(ref_text.lower())

        # In ví dụ 1 sample đầu tiên
        if i == 0 and len(norm_preds) > 0:
            print(f"Sample 1:")
            print(f"predicted: {norm_preds[0]}")
            print(f"reference: {batch_reference_texts[0]}")
            print("-" * 50)

        print(f"Processed {min(i + batch_size, num_samples)}/{num_samples} samples.")

    # --- Calculation using `jiwer` ---
    wer_score = wer(all_references, all_predictions)
    
    print("=" * 100)
    print(f"✅ Finished testing.")
    print(f"✨ Final WER for the test set: {wer_score:.4f}")
    print("=" * 100)



# -----------------------------------------------------------------------

from datetime import datetime
from collections import OrderedDict

def _pred_to_text(x):
    """
    NeMo EncDecRNNTBPEModel.transcribe thường trả về List[str].
    Một số API khác có thể trả Hypothesis với thuộc tính .text.
    Hàm này normalize về string.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # Hypothesis-like
    txt = getattr(x, "text", None)
    if isinstance(txt, str):
        return txt
    # List 1 phần tử?
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
        return x[0]
    return str(x)


def run_hardfix_vpb_suite(
    base_config: Path,
    devices: int,
    precision: str,
    batch_size: int,
    nemo_path: Path,
    manifest_root: Path,
    outdir: Path,
):
    """
    Chạy cố định 5 bộ VPB \*_nemo.jsonl với 1 model .nemo, ghi summary.tsv.
    """
    # Danh sách manifest cố định (khớp với batch bash trước đó)
    mf = OrderedDict([
        ("standard_test_2",      manifest_root / "standard_test_2" / "test_meta_nemo.jsonl"),
        ("standard_test",        manifest_root / "standard_test"   / "test_meta_nemo.jsonl"),
        ("next_day_test_debug",  manifest_root / "standard_test"   / "next_day_test_meta_debug_nemo.jsonl"),
        ("vpb_right2_train",     manifest_root / "manifest_vpb_right_2" / "train_meta_nemo.jsonl"),
        ("vpb_right2_valid",     manifest_root / "manifest_vpb_right_2" / "valid_meta_nemo.jsonl"),
    ])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_logs = outdir / f"logs_{ts}"
    out_logs.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / f"summary_{ts}.tsv"
    with summary_path.open("w", encoding="utf-8") as sw:
        sw.write("model\tdataset\twer\tlog_path\n")

    print("==> HARD-FIX VPB SUITE")
    print(f"Model .nemo: {nemo_path}")
    print(f"Manifest root: {manifest_root}")
    print(f"Logs dir: {out_logs}")
    print(f"Summary: {summary_path}")

    # Chạy lần lượt
    for ds_name, path in mf.items():
        if not path.is_file():
            print(f"[!] MISSING manifest: {path} (skip {ds_name})")
            continue

        # exp_name cho rõ ràng
        exp_name = f"hardfix__{ds_name}__{nemo_path.stem}"
        print(f"\n--- Dataset: {ds_name}")
        print(f"    Manifest: {path}")

        # Chạy test batch → in WER; wrap để lấy WER trả về
        # Ta copy một bản rút gọn của test_batch_from_checkpoint cho phép return WER:
        wer_score = _run_single_test_return_wer(
            base_config=base_config,
            test_manifest=path,
            devices=devices,
            precision=precision,
            batch_size=batch_size,
            nemo_path=nemo_path,
            exp_name=exp_name,
            log_dir=out_logs,
        )

        # Ghi summary
        log_path = out_logs / f"{exp_name}.log"
        with summary_path.open("a", encoding="utf-8") as sw:
            sw.write(f"{nemo_path.stem}\t{ds_name}\t{wer_score if wer_score is not None else 'NA'}\t{log_path}\n")

    print("\n==> DONE HARD-FIX VPB. See summary:", summary_path)


def _run_single_test_return_wer(
    base_config: Path,
    test_manifest: Path,
    devices: int,
    precision: str,
    batch_size: int,
    nemo_path: Path,
    exp_name: str,
    log_dir: Path,
):
    """
    Chạy 1 dataset và return WER (float). Đồng thời log ra file.
    """
    log_fp = log_dir / f"{exp_name}.log"
    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        # Tái sử dụng pipeline batch (in ra console được redirect)
        # NOTE: test_batch_from_checkpoint hiện chỉ print WER, mình tính lại để return
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=str(nemo_path))
        model.eval()
        try:
            if hasattr(model, 'spec_augmentation') and model.spec_augmentation is not None:
                model.spec_augmentation.mask_prob = 0.0
                model.spec_augmentation = None
            if hasattr(model, 'preprocessor'):
                if hasattr(model.preprocessor, 'dither'):
                    model.preprocessor.dither = 0.0
                if hasattr(model.preprocessor, 'pad_to'):
                    model.preprocessor.pad_to = 0
        except Exception:
            pass
        try:
            model.change_decoding_strategy(decoder_type="greedy_batch")
            if hasattr(model, 'wer'):
                model.wer.log_prediction = False
        except Exception:
            pass

        # Đọc manifest
        with open(test_manifest, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        audio_paths = []
        ref_texts = []
        for line in lines:
            item = json.loads(line)
            audio_paths.append(os.path.expanduser(item['audio_filepath']))
            ref_texts.append(item['text'])

        # Batch transcribe
        preds = []
        for i in range(0, len(audio_paths), batch_size):
            chunk = audio_paths[i:i+batch_size]
            outs = model.transcribe(chunk, batch_size=batch_size)
            outs = [ _pred_to_text(x) for x in outs ]
            preds.extend(outs)
            print(f"Processed {min(i+batch_size, len(audio_paths))}/{len(audio_paths)} samples.")

        # WER
        from jiwer import wer as _wer
        wer_score = _wer([t.lower() for t in ref_texts], [p.lower() for p in preds])

        print("=" * 100)
        print("✅ Finished testing.")
        print(f"✨ Final WER for the test set: {wer_score:.4f}")
        print("=" * 100)
    finally:
        sys.stdout = old_stdout
        # Ghi log
        with log_fp.open("w", encoding="utf-8") as fw:
            fw.write(buf.getvalue())
    # In ra console 1 dòng tóm tắt
    print(f"[{exp_name}] WER={wer_score:.4f} | log={log_fp}")
    return float(wer_score)


# --------------------------------- Main --------------------------------

def main():
    args = parse_args()

    # ========== HARD-FIX VPB SUITE ==========
    if args.hardfix_vpb:
        run_hardfix_vpb_suite(
            base_config=args.base_config.resolve(),
            devices=args.devices,
            precision=args.precision,
            batch_size=args.batch_size,
            nemo_path=args.hardfix_model.expanduser().resolve(),
            manifest_root=args.hardfix_manifest_root.expanduser().resolve(),
            outdir=args.hardfix_outdir.expanduser().resolve(),
        )
        return

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