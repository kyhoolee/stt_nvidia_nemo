from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple

import librosa
import soundfile as sf
import torch
from nemo.collections import asr as nemo_asr
from jiwer import wer  # corpus WER

from ._1_fastformer_trans_bpe import SIZE_PRESETS

# Try optional DeepFilterNet import
_HAS_DF = True
try:
    from df import enhance as df_enhance, init_df as df_init
except Exception:
    _HAS_DF = False


# --------------------------------- CLI ---------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train / Test FastConformer-Transducer on NeMo manifests")

    # Data + tokenizer
    p.add_argument('--train-manifest', type=Path, required=False)
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

    # Noise filter (DeepFilterNet v3) options
    p.add_argument('--denoise', action='store_true', help='Bật lọc nhiễu bằng DeepFilterNet v3 trước khi ASR.')
    p.add_argument('--df-sr', type=int, default=48000, help='Sample rate cho DeepFilterNet (mặc định 48000).')
    p.add_argument('--df-cache', type=Path, default=None,
                  help='Thư mục lưu WAV tạm sau khi denoise (mặc định: auto trong exp/logs).')
    p.add_argument('--df-keep-temp', action='store_true',
                  help='Giữ lại file WAV tạm sau khi chạy. Mặc định xóa.')

    # Hard-sample options
    p.add_argument('--hard-topk', type=int, default=50, help='Số lượng mẫu tệ nhất sẽ xuất ra (0 để tắt).')
    p.add_argument('--min-words', type=int, default=4, help='Chỉ xét các mẫu có >= số từ này trong reference.')
    p.add_argument('--hard-out', type=Path, default=None, help='Đường dẫn file TSV để ghi hard samples (mặc định auto).')

    # Hard-fix VPB test-suite
    p.add_argument('--hardfix-vpb', action='store_true',
                   help='Run fixed VPB test suite on a given .nemo (ignores train/val).')
    p.add_argument('--hardfix-model', type=Path, default=Path(
        "/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vietspeech/"
        "vpb_asr_fastconformer/2025-08-25_07-42-00/checkpoints/vpb_asr_fastconformer.nemo"
    ))
    p.add_argument('--hardfix-manifest-root', type=Path, default=Path(
        "/home/ubuntu/work/clean_dataset_vpb/manifest"
    ))
    p.add_argument('--hardfix-outdir', type=Path, default=Path("./nemo_eval_hardfix"))

    # Logging / output
    p.add_argument('--exp-dir', type=Path, default=Path('./experiments'))
    p.add_argument('--exp-name', type=str, default='vpb_asr_fastconformer')

    # Test-only & resume
    group = p.add_mutually_exclusive_group()
    group.add_argument('--nemo', type=Path, help='Path to .nemo file to restore for test-only')
    group.add_argument('--ckpt', type=Path, help='Path to Lightning .ckpt to restore for test-only')
    p.add_argument('--test-only', action='store_true', help='Only run testing from a restored checkpoint')

    return p.parse_args()


# --------------------------------- Helpers ---------------------------------

def _pred_to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    txt = getattr(x, "text", None)
    if isinstance(txt, str):
        return txt
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
        return x[0]
    return str(x)


def _word_count(s: str) -> int:
    return len(s.strip().split())


def _sample_wer(ref: str, hyp: str) -> float:
    from jiwer import wer as _wer_single
    if _word_count(ref) == 0:
        return float('nan')
    return float(_wer_single([ref], [hyp]))


def _write_hard_samples_tsv(rows: List[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write("idx\taudio_path\tref_len\tpred_len\twer\treference\tprediction\n")
        for r in rows:
            ref = r['reference'].replace('\t', ' ').replace('\n', ' ')
            pred = r['prediction'].replace('\t', ' ').replace('\n', ' ')
            f.write(
                f"{r['idx']}\t{r['audio_path']}\t{r['ref_len']}\t{r['pred_len']}\t{r['wer']:.4f}\t{ref}\t{pred}\n"
            )


def _write_hard_samples_jsonl(rows: List[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _pick_device_from_flag(devices: int) -> str:
    if torch.cuda.is_available():
        if isinstance(devices, int) and devices >= 0:
            if devices < torch.cuda.device_count():
                return f"cuda:{devices}"
            else:
                print(f"[WARN] devices={devices} vượt quá số GPU ({torch.cuda.device_count()}), fallback CPU.")
                return "cpu"
        elif devices == -1:
            return "cuda:0"
        else:
            print(f"[WARN] devices={devices} không hợp lệ, fallback CPU.")
            return "cpu"
    else:
        return "cpu"


def _amp_setup(precision: str, device_str: str) -> Tuple[bool, Optional[torch.dtype]]:
    use_amp = (str(precision).strip() in {"16", "bf16"}) and device_str != "cpu"
    amp_dtype = None
    if use_amp:
        if str(precision).strip() == "16":
            amp_dtype = torch.float16
        elif str(precision).strip() == "bf16":
            amp_dtype = torch.bfloat16
    return use_amp, amp_dtype


# -------- DeepFilterNet wrapper (optional) --------

class _DFWrapper:
    def __init__(self, sr: int):
        self.sr = int(sr)
        self.model = None
        self.state = None

    def ensure_loaded(self):
        if not _HAS_DF:
            raise RuntimeError("DeepFilterNet (df) chưa được cài/không import được.")
        if self.model is None or self.state is None:
            self.model, self.state, _ = df_init()
            # đảm bảo eval mode nếu có
            try:
                self.model.eval()
            except Exception:
                pass

    def enhance_wav(self, wav: np.ndarray, in_sr: int, out_sr: int = 16000) -> np.ndarray:
        """
        Trả về ndarray float32 1 kênh, SR=out_sr (mặc định 16k).
        DF yêu cầu input dạng [C, T] (2D).
        """
        import numpy as np
        import torch

        self.ensure_loaded()

        # 1) Mono hoá -> [T]
        if wav.ndim > 1:
            # librosa.load(mono=False) -> shape [C, T]; lấy trung bình kênh
            # (nếu lỡ shape [T, C] hiếm gặp: đảo lại)
            if wav.shape[0] < wav.shape[1]:
                wav = wav.mean(axis=0)
            else:
                wav = wav.mean(axis=1)
        wav = wav.astype('float32', copy=False)

        # 2) Resample lên SR của DF (thường 48000)
        if in_sr != self.sr:
            wav = librosa.resample(wav, orig_sr=in_sr, target_sr=self.sr, res_type='polyphase')

        # 3) Chuyển sang torch và thêm trục kênh -> [1, T]
        audio_t = torch.from_numpy(wav).to(torch.float32).contiguous()
        if audio_t.ndim == 1:
            audio_t = audio_t.unsqueeze(0)  # [1, T]

        # 4) Enhance
        with torch.no_grad():
            enh_t = df_enhance(self.model, self.state, audio_t)  # expect [C, T] -> [C, T]

        # 5) Về numpy 1 kênh [T]
        if enh_t.ndim == 2 and enh_t.shape[0] == 1:
            enh = enh_t[0].detach().cpu().numpy()
        else:
            enh = enh_t.detach().cpu().numpy()
            if enh.ndim == 2:
                enh = enh.mean(axis=0)  # phòng trường hợp trả nhiều kênh

        enh = enh.astype('float32', copy=False)

        # 6) Resample về out_sr (NeMo 16k)
        if self.sr != out_sr:
            enh = librosa.resample(enh, orig_sr=self.sr, target_sr=out_sr, res_type='polyphase')

        return enh


# numpy import (placed late to avoid import if unneeded)
import numpy as np


def _maybe_build_denoised_files(
    in_paths: List[str],
    df_sr: int,
    tmp_dir: Path,
    keep_temp: bool,
) -> Tuple[List[str], List[Path]]:
    """
    Tạo WAV tạm (16k mono, PCM16) sau khi denoise. Trả:
      - out_paths: list[str] các file 16k để feed vào NeMo
      - created_files: list[Path] các file đã tạo (để cleanup)
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dfw = _DFWrapper(sr=df_sr)

    out_paths: List[str] = []
    created_files: List[Path] = []

    for idx, p in enumerate(in_paths):
        # load giữ SR gốc; mono=False để mình tự quyết mono (theo wrapper)
        y, sr = librosa.load(p, sr=None, mono=False)
        # đảm bảo float32
        if y.dtype != np.float32:
            y = y.astype(np.float32, copy=False)

        # enhance → 16k mono
        enh_16k = dfw.enhance_wav(y, in_sr=sr, out_sr=16000)

        # ghi PCM16
        out_fp = tmp_dir / f"df_{idx:07d}.wav"
        sf.write(str(out_fp), enh_16k, 16000, subtype="PCM_16")

        out_paths.append(str(out_fp))
        created_files.append(out_fp)

    # (nếu keep_temp=False, caller sẽ dọn sau khi transcribe xong)
    return out_paths, created_files


def _cleanup_files(paths: List[Path]):
    for p in paths:
        try:
            if p.is_file():
                p.unlink(missing_ok=True)
        except Exception:
            pass


# --------------------------------- TEST (manual - giữ tham chiếu) ---------------------------------

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

    all_predictions, all_references = [], []
    print("🔍 Running manual transcription and WER calculation...")
    with open(test_manifest, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            item = json.loads(line)
            audio_path = os.path.expanduser(item['audio_filepath'])
            reference_text = item['text']

            predicted_text = transcribe_audio(audio_path, model).text
            all_predictions.append(predicted_text.lower())
            all_references.append(reference_text.lower())

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(lines)} samples.")
                print(f"Sample {i + 1}:")
                print(f"predicted: {predicted_text}")
                print(f"reference: {reference_text}")
                print("-" * 50)

    wer_score = wer(all_references, all_predictions)
    print("=" * 100)
    print(f"✅ Finished testing.")
    print(f"✨ Final WER for the test set: {wer_score:.4f}")
    print("=" * 100)


# --- BATCH + denoise + hard-samples ---
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
    hard_topk: int = 50,
    min_words: int = 4,
    hard_out: Optional[Path] = None,
    denoise: bool = False,
    df_sr: int = 48000,
    df_cache: Optional[Path] = None,
    df_keep_temp: bool = False,
):
    print("🚀 Starting test-only mode...")

    if nemo_path:
        print(f"🧠 Restoring model from .nemo: {nemo_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=str(nemo_path))
    elif ckpt_path:
        print(f"🧠 Restoring model from .ckpt: {ckpt_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(checkpoint_path=str(ckpt_path))
    else:
        raise ValueError("Must provide either --nemo or --ckpt for test-only mode.")

    # Device & precision
    device_str = _pick_device_from_flag(devices)
    model.to(device_str)
    use_amp, amp_dtype = _amp_setup(precision, device_str)

    # Prepare DF cache dir
    if denoise and not _HAS_DF:
        print("[WARN] --denoise bật nhưng không import được 'df'. Tiếp tục *không* lọc nhiễu.")
        denoise = False
    if denoise:
        # ưu tiên df_cache; nếu không, tạo trong exp_dir/exp_name
        if df_cache is None:
            df_cache = (exp_dir / f"{exp_name}_dfcache") if exp_dir else Path("./_dfcache")
        df_cache = df_cache.expanduser().resolve()
        df_cache.mkdir(parents=True, exist_ok=True)
        print(f"🔊 DeepFilterNet enabled (sr={df_sr}). Temp WAV dir: {df_cache}")

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

    # --- Read manifest ---
    with open(test_manifest, 'r', encoding='utf-8') as f:
        manifest_lines = f.readlines()

    num_samples = len(manifest_lines)
    audio_paths: List[str] = []
    reference_texts: List[str] = []
    for line in manifest_lines:
        item = json.loads(line)
        audio_paths.append(os.path.expanduser(item['audio_filepath']))
        reference_texts.append(str(item['text']))

    # If denoise: build temp WAV list
    created_files: List[Path] = []
    paths_for_transcribe = audio_paths
    if denoise:
        paths_for_transcribe, created_files = _maybe_build_denoised_files(
            in_paths=audio_paths,
            df_sr=df_sr,
            tmp_dir=df_cache,
            keep_temp=df_keep_temp,
        )

    # --- Batch transcribe ---
    all_predictions: List[str] = []
    all_references: List[str] = []
    print("🔍 Running batch transcription and WER calculation...")
    for i in range(0, num_samples, batch_size):
        batch_audio_paths = paths_for_transcribe[i:i + batch_size]
        batch_reference_texts = reference_texts[i:i + batch_size]

        if use_amp and amp_dtype is not None and device_str.startswith("cuda"):
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                predicted_texts = model.transcribe(batch_audio_paths, batch_size=batch_size)
        else:
            predicted_texts = model.transcribe(batch_audio_paths, batch_size=batch_size)

        norm_preds = [_pred_to_text(p) for p in predicted_texts]

        for j in range(len(norm_preds)):
            pred_text = norm_preds[j]
            ref_text = batch_reference_texts[j]
            all_predictions.append(pred_text.lower())
            all_references.append(ref_text.lower())

        if i == 0 and len(norm_preds) > 0:
            print(f"Sample 1:")
            print(f"predicted: {norm_preds[0]}")
            print(f"reference: {batch_reference_texts[0]}")
            print("-" * 50)

        print(f"Processed {min(i + batch_size, num_samples)}/{num_samples} samples.")

    # Cleanup temp files
    if denoise and not df_keep_temp:
        _cleanup_files(created_files)

    # --- Corpus WER ---
    wer_score = wer(all_references, all_predictions)
    print("=" * 100)
    print(f"✅ Finished testing.")
    print(f"✨ Final WER for the test set: {wer_score:.4f}")
    print("=" * 100)

    # ===== Hard samples =====
    if hard_out is not None:
        hard_tsv = hard_out
        hard_jsonl = hard_out.with_suffix('.jsonl')
    else:
        hard_tsv = (exp_dir / f"{exp_name}_hard.tsv") if exp_dir else Path("./hard.tsv")
        hard_jsonl = (exp_dir / f"{exp_name}_hard.jsonl") if exp_dir else Path("./hard.jsonl")

    per_samples: List[dict] = []
    for idx, (ap, ref, pred) in enumerate(zip(audio_paths, all_references, all_predictions)):
        ref_len = _word_count(ref)
        if ref_len < int(min_words):
            continue
        s_wer = _sample_wer(ref, pred)
        if s_wer != s_wer:  # NaN
            continue
        per_samples.append({
            "idx": idx,
            "audio_path": ap,
            "ref_len": ref_len,
            "pred_len": _word_count(pred),
            "wer": float(s_wer),
            "reference": ref,
            "prediction": pred,
        })

    if hard_topk > 0 and len(per_samples) > 0:
        per_samples.sort(key=lambda x: x["wer"], reverse=True)
        hard_rows = per_samples[:hard_topk]
        _write_hard_samples_tsv(hard_rows, hard_tsv)
        _write_hard_samples_jsonl(hard_rows, hard_jsonl)
        print(f"📄 Saved hard samples (Top-{hard_topk}) to:")
        print(f"   - TSV:   {hard_tsv}")
        print(f"   - JSONL: {hard_jsonl}")
    else:
        print("ℹ️ Hard samples disabled or no eligible samples (check --hard-topk and --min-words).")

    return wer_score


# --------------------------------- HARD-FIX VPB SUITE ---------------------------------

from datetime import datetime
from collections import OrderedDict

def run_hardfix_vpb_suite(
    base_config: Path,
    devices: int,
    precision: str,
    batch_size: int,
    nemo_path: Path,
    manifest_root: Path,
    outdir: Path,
    hard_topk: int = 50,
    min_words: int = 4,
    denoise: bool = False,
    df_sr: int = 48000,
    df_cache: Optional[Path] = None,
    df_keep_temp: bool = False,
):
    """
    Chạy cố định 5 bộ VPB *_nemo.jsonl với 1 model .nemo, ghi summary.tsv.
    ĐÃ BỔ SUNG: denoise (DF v3) và hard samples cho từng dataset.
    """
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
        sw.write("model\tdataset\twer\tlog_path\thard_samples\n")

    print("==> HARD-FIX VPB SUITE")
    print(f"Model .nemo: {nemo_path}")
    print(f"Manifest root: {manifest_root}")
    print(f"Logs dir: {out_logs}")
    print(f"Summary: {summary_path}")

    for ds_name, path in mf.items():
        if not path.is_file():
            print(f"[!] MISSING manifest: {path} (skip {ds_name})")
            continue

        exp_name = f"hardfix__{ds_name}__{nemo_path.stem}"
        print(f"\n--- Dataset: {ds_name}")
        print(f"    Manifest: {path}")

        log_path = out_logs / f"{exp_name}.log"
        hard_out_path = out_logs / f"{exp_name}_hard.tsv"

        wer_score = _run_single_test_return_wer(
            base_config=base_config,
            test_manifest=path,
            devices=devices,
            precision=precision,
            batch_size=batch_size,
            nemo_path=nemo_path,
            exp_name=exp_name,
            log_dir=out_logs,
            hard_topk=hard_topk,
            min_words=min_words,
            hard_out=hard_out_path,
            denoise=denoise,
            df_sr=df_sr,
            df_cache=(df_cache if df_cache is not None else out_logs / f"{exp_name}_dfcache"),
            df_keep_temp=df_keep_temp,
        )

        with summary_path.open("a", encoding="utf-8") as sw:
            sw.write(f"{nemo_path.stem}\t{ds_name}\t{wer_score if wer_score is not None else 'NA'}\t{log_path}\t{hard_out_path}\n")

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
    hard_topk: int = 50,
    min_words: int = 4,
    hard_out: Optional[Path] = None,
    denoise: bool = False,
    df_sr: int = 48000,
    df_cache: Optional[Path] = None,
    df_keep_temp: bool = False,
):
    """
    Chạy 1 dataset và return WER (float). Đồng thời:
      - log stdout vào {log_dir}/{exp_name}.log
      - denoise (nếu bật) và lưu WAV tạm (16k)
      - xuất hard samples (TSV + JSONL)
    """
    import io, sys

    device_str = _pick_device_from_flag(devices)
    use_amp, amp_dtype = _amp_setup(precision, device_str)

    log_fp = log_dir / f"{exp_name}.log"
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    created_files: List[Path] = []
    try:
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=str(nemo_path))
        model.to(device_str)
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

        with open(test_manifest, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        audio_paths: List[str] = []
        ref_texts: List[str] = []
        for line in lines:
            item = json.loads(line)
            audio_paths.append(os.path.expanduser(item['audio_filepath']))
            ref_texts.append(str(item['text']))

        # Denoise (nếu có)
        if denoise and not _HAS_DF:
            print("[WARN] --denoise bật nhưng không import được 'df'. Tiếp tục *không* lọc nhiễu.")
            denoise = False
        if denoise:
            if df_cache is None:
                df_cache = log_dir / f"{exp_name}_dfcache"
            df_cache = df_cache.expanduser().resolve()
            df_cache.mkdir(parents=True, exist_ok=True)
            print(f"🔊 DeepFilterNet enabled (sr={df_sr}). Temp WAV dir: {df_cache}")

            paths_for_transcribe, created_files = _maybe_build_denoised_files(
                in_paths=audio_paths,
                df_sr=df_sr,
                tmp_dir=df_cache,
                keep_temp=df_keep_temp,
            )
        else:
            paths_for_transcribe = audio_paths

        # Batch transcribe
        preds: List[str] = []
        for i in range(0, len(paths_for_transcribe), batch_size):
            chunk = paths_for_transcribe[i:i+batch_size]
            if use_amp and amp_dtype is not None and device_str.startswith("cuda"):
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    outs = model.transcribe(chunk, batch_size=batch_size)
            else:
                outs = model.transcribe(chunk, batch_size=batch_size)
            outs = [_pred_to_text(x) for x in outs]
            preds.extend([o.lower() for o in outs])
            ref_texts[i:i+batch_size] = [r.lower() for r in ref_texts[i:i+batch_size]]
            print(f"Processed {min(i+batch_size, len(paths_for_transcribe))}/{len(paths_for_transcribe)} samples.")

        from jiwer import wer as _wer
        wer_score = _wer(ref_texts, preds)

        print("=" * 100)
        print("✅ Finished testing.")
        print(f"✨ Final WER for the test set: {wer_score:.4f}")
        print("=" * 100)

        # ===== Hard samples =====
        if hard_out is not None:
            hard_tsv = hard_out
            hard_jsonl = hard_out.with_suffix('.jsonl')
        else:
            hard_tsv = log_dir / f"{exp_name}_hard.tsv"
            hard_jsonl = log_dir / f"{exp_name}_hard.jsonl"

        per_samples: List[dict] = []
        for idx, (ap, ref, pred) in enumerate(zip(audio_paths, ref_texts, preds)):
            ref_len = _word_count(ref)
            if ref_len < int(min_words):
                continue
            s_wer = _sample_wer(ref, pred)
            if s_wer != s_wer:
                continue
            per_samples.append({
                "idx": idx,
                "audio_path": ap,
                "ref_len": ref_len,
                "pred_len": _word_count(pred),
                "wer": float(s_wer),
                "reference": ref,
                "prediction": pred,
            })

        if hard_topk > 0 and len(per_samples) > 0:
            per_samples.sort(key=lambda x: x["wer"], reverse=True)
            hard_rows = per_samples[:hard_topk]
            _write_hard_samples_tsv(hard_rows, hard_tsv)
            _write_hard_samples_jsonl(hard_rows, hard_jsonl)
            print(f"📄 Saved hard samples (Top-{hard_topk}) to:")
            print(f"   - TSV:   {hard_tsv}")
            print(f"   - JSONL: {hard_jsonl}")
        else:
            print("ℹ️ Hard samples disabled or no eligible samples (check --hard-topk and --min-words).")

    finally:
        sys.stdout = old_stdout
        with log_fp.open("w", encoding="utf-8") as fw:
            fw.write(buf.getvalue())
        # cleanup temp WAVs nếu có
        if denoise and not df_keep_temp:
            _cleanup_files(created_files)

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
            hard_topk=args.hard_topk,
            min_words=args.min_words,
            denoise=args.denoise,
            df_sr=args.df_sr,
            df_cache=args.df_cache.expanduser().resolve() if args.df_cache else None,
            df_keep_temp=args.df_keep_temp,
        )
        return

    # ===== TEST-ONLY =====
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
            hard_topk=args.hard_topk,
            min_words=args.min_words,
            hard_out=(args.hard_out.expanduser().resolve() if args.hard_out else None),
            denoise=args.denoise,
            df_sr=args.df_sr,
            df_cache=args.df_cache.expanduser().resolve() if args.df_cache else None,
            df_keep_temp=args.df_keep_temp,
        )
        return


if __name__ == '__main__':
    main()
