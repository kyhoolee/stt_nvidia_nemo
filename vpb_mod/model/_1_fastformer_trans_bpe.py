#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train FastConformer-Transducer (NeMo) on Vietnamese data using freshly-built NeMo manifests.

Highlights
---------
- Plugs directly into your new manifest layout (e.g., vi_small/nemo_manifests/<ds>/<ds>_{train,dev,test}.jsonl)
- Builds/loads a SentencePiece tokenizer from the training manifest text
- Applies a clean set of config overrides on top of NeMo's base YAML
- Size presets (small/medium/large) mapped to the NeMo doc table
- Sensible LR scaling by global batch size + warmup computed from epochs
- TensorBoard + optional CSV logger, exp_manager integration
- Clear comments and structure for further experiments

Example
-------
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
  --exp-dir ./experiments --exp-name vpb_asr_fastconformer

If you prefer per-dataset manifests instead of merged, just pass those paths.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple
import gc

import torch
from omegaconf import OmegaConf

# Freeze utils
from ._0_freeze_utils import (
    freeze_bottom_k_layers, freeze_bottom_ratio, freeze_top_k_layers,
    freeze_preprocessor, freeze_subsampling, freeze_pos_enc, freeze_spec_augment,
    freeze_decoder, freeze_decoder_embedding, freeze_decoder_rnn, freeze_joint,
    freeze_by_prefixes, freeze_by_regex, freeze_by_types,
    count_params, snapshot_frozen,
    FreezeScheduleCallback, _parse_stage_str, 
)

# Lightning (PyTorch Lightning v2 name-space)
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy

# NeMo
from nemo.utils import exp_manager
import nemo.collections.asr as nemo_asr

# NeMo tokenizer helper
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model


# ----------------------------- Customized Model -----------------------------
# Tùy chỉnh class để vô hiệu hóa logging dự đoán
from nemo.collections.asr.models import EncDecRNNTBPEModel
from lightning.pytorch import LightningModule
from typing import Optional, Any
import torch
from omegaconf import DictConfig

class CustomFastConformerRNNTModel(EncDecRNNTBPEModel):
    def __init__(self, cfg: DictConfig, trainer: Optional['Trainer'] = None):
        super().__init__(cfg=cfg, trainer=trainer)
        
        # Vô hiệu hóa tính năng log prediction của module WER
        # Điều này phải được thực hiện sau khi super() đã gọi và khởi tạo module WER
        self.wer.log_prediction = False


# ----------------------------- Tokenizer -----------------------------
def prepare_tokenizer(
    manifest_path: Path,
    tokenizer_dir: Path,
    vocab_size: int = 128,
    spe_type: str = "unigram",
    lowercase: bool = False,
) -> Tuple[Path, str]:
    """Create (or reuse) a SentencePiece tokenizer from manifest text.

    Returns
    -------
    tokenizer_path : Path to directory containing tokenizer.model
    tokenizer_type_cfg : str = 'bpe' (value expected by NeMo config)
    """
    print("🔧 Preparing tokenizer…")
    tokenizer_dir = tokenizer_dir.expanduser().resolve()
    corpus_dir = tokenizer_dir / "text_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    document_path = corpus_dir / "document.txt"

    # Build text corpus only once
    if not document_path.exists():
        print(f"📄 Building text corpus from manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f_in, open(document_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    if lowercase:
                        text = text.lower()
                    if text:
                        f_out.write(text + "\n")
                except Exception:
                    # Skip malformed lines silently
                    continue

    tokenizer_path = tokenizer_dir / f"tokenizer_spe_{spe_type}_v{vocab_size}"
    model_file = tokenizer_path / "tokenizer.model"

    if model_file.exists():
        print(f"⚠️ Tokenizer already exists at {tokenizer_path}, skipping creation.")
    else:
        print(f"🔠 Creating SentencePiece tokenizer at: {tokenizer_path}")
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        create_spt_model(
            data_file=str(document_path),
            vocab_size=vocab_size,
            sample_size=-1,
            do_lower_case=False,  # text already normalized above if requested
            output_dir=str(tokenizer_path),
            tokenizer_type=spe_type,  # "unigram" or "bpe"
            character_coverage=1.0,
            train_extremely_large_corpus=False,
            max_sentencepiece_length=-1,
            split_by_unicode_script=True,
            bos=False,
            eos=False,
            pad=False,
            control_symbols=None,
            user_defined_symbols=None,
            byte_fallback=False,
            split_digits=False,
            remove_extra_whitespaces=False,
        )

    print(f"✅ Tokenizer ready at: {tokenizer_path} (type=bpe)")
    return tokenizer_path, "bpe"


# --------------------------- Config helpers ---------------------------
SIZE_PRESETS = {
    # Matches the table in your NeMo YAML comment block
    "small_0": dict(d_model=176, n_heads=1, n_layers=6, pred_hidden=320, joint_hidden=320, weight_decay=0.0, xscaling=True),
    "small": dict(d_model=176, n_heads=4, n_layers=16, pred_hidden=320, joint_hidden=320, weight_decay=0.0, xscaling=True),
    "medium": dict(d_model=256, n_heads=4, n_layers=16, pred_hidden=640, joint_hidden=640, weight_decay=1e-3, xscaling=True),
    "large": dict(d_model=512, n_heads=8, n_layers=17, pred_hidden=640, joint_hidden=640, weight_decay=1e-3, xscaling=True),
}


def apply_size_preset(cfg, size: str):
    """Apply encoder / joint dimensions based on a named preset."""
    preset = SIZE_PRESETS[size]
    cfg.model.encoder.d_model = preset["d_model"]
    cfg.model.encoder.n_heads = preset["n_heads"]
    cfg.model.encoder.n_layers = preset["n_layers"]
    cfg.model.model_defaults.pred_hidden = preset["pred_hidden"]
    cfg.model.model_defaults.joint_hidden = preset["joint_hidden"]
    cfg.model.joint.jointnet.joint_hidden = preset["joint_hidden"]
    # Optional tweaks aligning with table
    cfg.model.encoder.xscaling = preset["xscaling"]
    cfg.model.optim.weight_decay = preset["weight_decay"]


def scale_optim_and_warmup(cfg, train_samples: int, epochs: int, devices: int, acc_steps: int):
    """Scale LR with global batch size and compute warmup steps.

    Notes
    -----
    - Global batch = per-GPU batch * devices * accumulate_grad_batches
    - We scale from reference global batch 256.
    """
    per_gpu = cfg.model.train_ds.batch_size
    global_bsz = per_gpu * max(devices, 1) * max(acc_steps, 1)
    base_lr = 1e-3
    cfg.model.optim.name = "adamw"
    cfg.model.optim.lr = base_lr * (global_bsz / 256)
    cfg.model.optim.sched.min_lr = cfg.model.optim.lr * 0.1  # a bit less aggressive than 0.01

    # Warmup: 5% of total steps (epochs * samples / global_bsz)
    total_steps = int(epochs * (train_samples / max(global_bsz, 1)))
    warmup_steps = max(100, int(0.05 * total_steps))
    cfg.model.optim.sched.warmup_steps = warmup_steps

    print(f"📐 Global batch size: {global_bsz}")
    print(f"📈 LR: {cfg.model.optim.lr:.6f} | warmup_steps: {warmup_steps} | total_steps~{total_steps}")


def configure_from_yaml(
    base_config: Path,
    train_manifest: Path,
    val_manifest: Optional[Path],
    test_manifest: Optional[Path],
    tokenizer_path: Path,
    tokenizer_type_cfg: str,
    size: str,
    max_duration: float,
    precision: str,
    devices: int,
    epochs: int,
    accumulate_grad_batches: int,
    disable_specaug: bool,
) -> OmegaConf:
    """Load the base YAML and apply experiment-specific overrides."""
    print("🧩 Loading + patching config…")
    cfg = OmegaConf.load(str(base_config))

    # Data
    cfg.model.sample_rate = 16000
    cfg.model.train_ds.manifest_filepath = str(train_manifest)
    if val_manifest:
        cfg.model.validation_ds.manifest_filepath = str(val_manifest)
    if test_manifest:
        cfg.model.test_ds.manifest_filepath = str(test_manifest)

    # Bucketing and duration
    cfg.model.train_ds.max_duration = float(max_duration)
    cfg.model.train_ds.bucketing_strategy = "fully_randomized"

    # Tokenizer
    cfg.model.tokenizer.dir = str(tokenizer_path)
    cfg.model.tokenizer.type = tokenizer_type_cfg  # 'bpe'

    # SpecAug (optionally disable for stability / debugging)
    if disable_specaug:
        cfg.model.spec_augment.freq_masks = 0
        cfg.model.spec_augment.time_masks = 0

    # Make validation/test a bit heavier by default
    cfg.model.validation_ds.batch_size = cfg.model.train_ds.batch_size
    cfg.model.test_ds.batch_size = cfg.model.train_ds.batch_size

    # Joint fusing can save memory
    cfg.model.joint.fuse_loss_wer = True
    cfg.model.joint.fused_batch_size = min(16, cfg.model.train_ds.batch_size)

    # Apply model size preset
    apply_size_preset(cfg, size)

    # Trainer overrides
    cfg.trainer.devices = devices
    cfg.trainer.precision = precision
    cfg.trainer.accumulate_grad_batches = accumulate_grad_batches
    # Use native DDP when multi-GPU
    if devices == -1 or (isinstance(devices, int) and devices > 1):
        cfg.trainer.strategy = OmegaConf.create({
            "_target_": "lightning.pytorch.strategies.DDPStrategy",
            "gradient_as_bucket_view": True,
        })
    else:
        cfg.trainer.strategy = "auto"


    # (tuỳ) tránh trả text từng mẫu ở dataloader test/val
    cfg.model.validation_ds.return_transcripts = False
    cfg.model.test_ds.return_transcripts = False


    # Logging via exp_manager (we still attach CSV later)
    cfg.exp_manager.create_tensorboard_logger = True
    cfg.exp_manager.create_wandb_logger = False
    cfg.exp_manager.create_checkpoint_callback = True
    cfg.exp_manager.checkpoint_callback_params.monitor = "val_wer"
    cfg.exp_manager.checkpoint_callback_params.mode = "min"
    cfg.exp_manager.checkpoint_callback_params.save_top_k = 3
    cfg.exp_manager.checkpoint_callback_params.always_save_nemo = True

    # Optim and warmup are scaled later when we know train_samples
    return cfg


# ------------------------------- I/O ---------------------------------

def count_lines(path: Path) -> int:
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    return cnt


def build_trainer(exp_dir: Path, exp_name: str, epochs: int, precision: str, devices: int) -> Trainer:
    """Create a Trainer + TensorBoard logger. exp_manager will finalize log dirs."""
    print("🚀 Building Trainer…")
    tb_logger = TensorBoardLogger(save_dir=str(exp_dir), name=exp_name, log_graph=False)

    # Chọn accelerator
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Điều kiện kích hoạt DDP (nhiều GPU)
    is_ddp = accelerator == "gpu" and (devices == -1 or (isinstance(devices, int) and devices > 1))

    # Chiến lược DDP: bật find_unused_parameters cho case freeze/schedule
    strategy = (
        DDPStrategy(
            find_unused_parameters=True,     # 👈 QUAN TRỌNG: tránh lỗi "not used in producing the loss"
            gradient_as_bucket_view=True,
            static_graph=False               # 👈 vì bạn thay đổi freeze theo epoch
        )
        if is_ddp else "auto"
    )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=epochs,
        enable_checkpointing=False,   # handled by exp_manager
        logger=False,                 # nếu muốn dùng TB logger, đổi thành logger=tb_logger
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        precision=precision,          # '16', '32', 'bf16' (Lightning v2 OK)
        strategy=strategy,
    )
    return trainer



def attach_exp_manager(trainer: Trainer, cfg: OmegaConf, exp_dir: Path, exp_name: str):
    """Attach NeMo exp_manager to create the final run directory and checkpointing."""
    print("🗂️  Setting up exp_manager…")
    # Override exp_dir/name at runtime so runs are grouped nicely
    cfg.exp_manager.exp_dir = str(exp_dir)
    cfg.exp_manager.name = exp_name
    os.environ.pop('NEMO_EXPM_VERSION', None)
    return exp_manager.exp_manager(trainer, cfg.get("exp_manager"))


def maybe_add_csv_logger(trainer: Trainer):
    """Attach a CSV logger to the same directory created by exp_manager."""
    try:
        log_dir = trainer.log_dir  # set by exp_manager
    except Exception:
        log_dir = None
    if log_dir:
        print("📝 Adding CSV Logger…")
        csv = CSVLogger(save_dir=log_dir, name="", version="")
        # PTL v2 uses `logger` (singular) or `loggers` property depending on setup
        # Ensure we append without clobbering the existing TB logger
        if hasattr(trainer, "loggers") and trainer.loggers is not None:
            trainer.loggers.append(csv)
        else:
            # Some PTL configs only expose `.logger` (single). Keep TB and CSV both.
            # In that case, we create a list and reassign (best-effort).
            current = [trainer.logger] if getattr(trainer, "logger", None) else []
            current.append(csv)
            trainer.logger = current
    else:
        print("⚠️ CSV Logger skipped: trainer.log_dir is not available.")


def log_val_metrics_to_txt(log_dir: Path, epoch: int, wer: float, loss: float):
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "val_metrics.txt", "a", encoding="utf-8") as f:
        f.write(f"Epoch {epoch:03d} | val_WER: {wer:.4f} | val_loss: {loss:.4f}\n")


# --------------------------------- CLI ---------------------------------
import json
import librosa
import torch
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional
from nemo.collections import asr as nemo_asr
from nemo.core.config import hydra_runner


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

    # ===== Fine-tune from pretrained .nemo =====
    p.add_argument('--init-from-nemo', type=Path, default=None,
                   help='Path to a .nemo checkpoint to initialize weights for fine-tuning')
    p.add_argument('--resume-from-ckpt', type=Path, default=None,
                   help='Resume training from a PTL/Nemo checkpoint .ckpt (not .nemo)')

    # Freeze / Unfreeze plan

    # --- Encoder ---
    p.add_argument("--freeze-encoder-layers", type=int, default=0,
                        help="Đóng băng K lớp dưới cùng của encoder (bottom-K).")
    p.add_argument("--freeze-encoder-ratio", type=float, default=0.0,
                        help="Đóng băng theo tỉ lệ bottom (0..1).")
    p.add_argument("--freeze-encoder-top-k", type=int, default=0,
                        help="Đóng băng K lớp trên cùng của encoder (top-K).")

    # --- Blocks / Components ---
    p.add_argument("--freeze-preprocessor", action="store_true",
                        help="Đóng băng Audio preprocessor.")
    p.add_argument("--freeze-subsampling", action="store_true",
                        help="Đóng băng encoder.pre_encode (ConvSubsampling + projection).")
    p.add_argument("--freeze-pos-enc", action="store_true",
                        help="Đóng băng positional encoding.")
    p.add_argument("--freeze-specaug", action="store_true",
                        help="Đóng băng SpecAug (thường không có params, nhưng để đồng bộ).")

    # --- Decoder & Joint ---
    p.add_argument("--freeze-decoder-all", action="store_true",
                        help="Đóng băng toàn bộ decoder.")
    p.add_argument("--freeze-decoder-embed", action="store_true",
                        help="Chỉ đóng băng embedding trong decoder.")
    p.add_argument("--freeze-decoder-rnn", action="store_true",
                        help="Chỉ đóng băng RNN trong decoder.")
    p.add_argument("--freeze-joint", action="store_true",
                        help="Đóng băng RNNT joint.")

    # --- Advanced (optional) ---
    p.add_argument("--freeze-prefix", nargs="*", default=[],
                        help="Đóng băng theo prefix tên tham số, vd: encoder.pre_encode.out decoder.prediction.embed")
    p.add_argument("--freeze-regex", nargs="*", default=[],
                        help=r"Đóng băng theo regex trên tên tham số, vd: ^encoder\.layers\.(?:0|1|2)\.")
    p.add_argument("--freeze-types", nargs="*", default=[],
                        help="Đóng băng theo kiểu lớp torch.nn, vd: LayerNorm BatchNorm1d BatchNorm2d")
    p.add_argument("--freeze-dump", action="store_true",
                        help="In tóm tắt số param trainable và vài tên đã freeze để kiểm tra.")



    # --- Advanced freeze schedule ---
    p.add_argument(
        "--stage", action="append", default=[],
        help=(
            "Định nghĩa 1 stage freeze/unfreeze áp dụng từ epoch e trở đi. "
            "Ví dụ: "
            "  --stage 'e=0,enc_bottom_k=12,pre=1,subs=1,pos=1,dec_all=1,joint=1' "
            "  --stage 'e=6,enc_bottom_k=6' "
            "  --stage 'e=10,dec_all=0,dec_rnn=0' "
            "Hỗ trợ: e, enc_bottom_k, enc_bottom_ratio, enc_top_k, pre, subs, pos, spec, "
            "dec_all, dec_embed, dec_rnn, joint, prefix, regex, types"
        )
    )

    p.add_argument("--freeze-dump-stages", action="store_true", help="In summary trainable params sau khi áp dụng stage.")

                

    # Optim / Sched overrides when fine-tuning
    p.add_argument('--lr', type=float, default=None, help='Override LR (AdamW) for FT')
    p.add_argument('--min-lr', type=float, default=None, help='Override scheduler min_lr for FT')
    p.add_argument('--warmup-steps', type=int, default=None, help='Override warmup_steps')
    p.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm')

    # RNNT decoding / FastEmit tweak
    p.add_argument('--fastemit-lambda', type=float, default=0.0,
                   help='Set FastEmit lambda for RNNT loss (0.0 → off; small ~0.001–0.01 for latency bias)')



    # Logging / output
    p.add_argument('--exp-dir', type=Path, default=Path('./experiments'))
    p.add_argument('--exp-name', type=str, default='vpb_asr_fastconformer')

    # Test-only & resume
    group = p.add_mutually_exclusive_group()
    group.add_argument('--nemo', type=Path, help='Path to .nemo file to restore for test-only')
    group.add_argument('--ckpt', type=Path, help='Path to Lightning .ckpt to restore for test-only')
    p.add_argument('--test-only', action='store_true', help='Only run testing from a restored checkpoint')

    return p.parse_args()


from jiwer import wer  # <-- Import the 'wer' function from 'jiwer'

def _setup_test_data_for(model, test_manifest, batch_size):
    """
    Setup the test data loader for the model.
    """
    cfg = OmegaConf.create({'manifest_filepath': str(test_manifest), 'batch_size': batch_size, 'num_workers': 0})
    model.setup_test_data(cfg)

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

    cfg = OmegaConf.load(str(base_config))
    trainer = build_trainer(
        exp_dir=exp_dir, exp_name=exp_name, epochs=1, precision=precision, devices=devices
    )
    attach_exp_manager(trainer, cfg, exp_dir=exp_dir, exp_name=exp_name)
    maybe_add_csv_logger(trainer)

    if nemo_path:
        print(f"🧠 Restoring model from .nemo: {nemo_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            restore_path=str(nemo_path), trainer=trainer
        )
    elif ckpt_path:
        print(f"🧠 Restoring model from .ckpt: {ckpt_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path), trainer=trainer
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





def build_or_restore_model_for_train(
    args,
    cfg: OmegaConf,
    trainer: Trainer,
):
    """
    - Nếu --init-from-nemo: restore model từ .nemo, merge cfg mới vào model.cfg,
      ép lại manifest và re-setup dataloaders/optimizer đúng cách (giữ nguyên interpolation ${model.*}).
    - Nếu không: khởi tạo model mới theo cfg hiện tại.
    Trả về (model, effective_cfg)
    """
    # 1) Ghi FastEmit vào cfg (để đồng bộ config lưu log/ckpt)
    if args.fastemit_lambda is not None and args.fastemit_lambda > 0.0:
        try:
            if not hasattr(cfg.model, "rnnt_loss"):
                cfg.model.rnnt_loss = OmegaConf.create({})
            if not hasattr(cfg.model.rnnt_loss, "warprnnt_numba_kwargs"):
                cfg.model.rnnt_loss.warprnnt_numba_kwargs = OmegaConf.create({})
            cfg.model.rnnt_loss.warprnnt_numba_kwargs.fastemit_lambda = float(args.fastemit_lambda)
            print(f"⏩ Set FastEmit lambda (cfg) = {args.fastemit_lambda}")
        except Exception as e:
            print(f"⚠️ Could not set fastemit_lambda in cfg: {e}")

    # 2) Ghi gradient clip vào cfg.trainer (Lightning đọc giá trị này)
    if args.grad_clip is not None:
        try:
            cfg.trainer.gradient_clip_val = float(args.grad_clip)
            cfg.trainer.gradient_clip_algorithm = "norm"
            print(f"✂️  Grad clip = {args.grad_clip}")
        except Exception:
            pass

    # 3) Nhánh fine-tune từ .nemo
    if args.init_from_nemo is not None:
        print(f"[FT] Restoring model from .nemo: {args.init_from_nemo}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            restore_path=str(args.init_from_nemo),
            trainer=trainer,
            map_location=("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # 3a) Merge cfg mới (CLI) vào model.cfg (giữ root 'model' để interpolation hoạt động)
        try:
            model.cfg = OmegaConf.merge(model.cfg, cfg.model)
        except Exception as e:
            print(f"⚠️ OmegaConf.merge(model.cfg, cfg.model) failed: {e}")

        # 3b) Ép lại manifest filepath trên model.cfg (để chắc chắn trỏ đúng VPB)
        try:
            if getattr(cfg.model, "train_ds", None) and getattr(cfg.model.train_ds, "manifest_filepath", None):
                model.cfg.train_ds.manifest_filepath = str(cfg.model.train_ds.manifest_filepath)
            if getattr(cfg.model, "validation_ds", None) and getattr(cfg.model.validation_ds, "manifest_filepath", None):
                model.cfg.validation_ds.manifest_filepath = str(cfg.model.validation_ds.manifest_filepath)
            if getattr(cfg.model, "test_ds", None) and getattr(cfg.model.test_ds, "manifest_filepath", None):
                model.cfg.test_ds.manifest_filepath = str(cfg.model.test_ds.manifest_filepath)
        except Exception as e:
            print(f"⚠️ Could not set manifest filepaths on model.cfg: {e}")

        # 3c) Re-setup dataloaders BẰNG model.cfg.* (không truyền subtree rời rạc)
        try:
            model.setup_training_data(model.cfg.train_ds)
            if getattr(model.cfg, "validation_ds", None):
                model.setup_validation_data(model.cfg.validation_ds)
            if getattr(model.cfg, "test_ds", None):
                model.setup_test_data(model.cfg.test_ds)
        except Exception as e:
            print(f"⚠️ setup_*_data failed: {e}")

        # 3d) Áp FastEmit trực tiếp lên module loss (do loss đã init từ checkpoint cũ)
        try:
            if args.fastemit_lambda and hasattr(model, "rnnt_loss"):
                model.rnnt_loss.warprnnt_numba_kwargs["fastemit_lambda"] = float(args.fastemit_lambda)
                # Giữ đồng bộ trong model.cfg (không bắt buộc nhưng tốt)
                if not hasattr(model.cfg, "rnnt_loss"):
                    model.cfg.rnnt_loss = OmegaConf.create({})
                if not hasattr(model.cfg.rnnt_loss, "warprnnt_numba_kwargs"):
                    model.cfg.rnnt_loss.warprnnt_numba_kwargs = OmegaConf.create({})
                model.cfg.rnnt_loss.warprnnt_numba_kwargs.fastemit_lambda = float(args.fastemit_lambda)
                print(f"⏩ Applied FastEmit lambda on loss module = {args.fastemit_lambda}")
        except Exception as e:
            print(f"⚠️ Could not apply fastemit_lambda on loss module: {e}")

        # 3e) Re-setup optimization theo model.cfg.optim (nhận override CLI nếu có)
        try:
            if args.lr is not None:
                model.cfg.optim.lr = float(args.lr)
            if args.min_lr is not None:
                model.cfg.optim.sched.min_lr = float(args.min_lr)
            if args.warmup_steps is not None:
                model.cfg.optim.sched.warmup_steps = int(args.warmup_steps)
            model.setup_optimization(model.cfg.optim)
            print(
                f"🛠️  Optim set: lr={model.cfg.optim.lr} | "
                f"min_lr={model.cfg.optim.sched.min_lr} | "
                f"warmup={model.cfg.optim.sched.warmup_steps}"
            )
        except Exception as e:
            print(f"⚠️ setup_optimization failed: {e}")

        # 3f) Tắt log prediction ở WER (đỡ spam + tiết kiệm I/O)
        if getattr(model, "wer", None) is not None:
            try:
                model.wer.log_prediction = False
            except Exception:
                pass

        # 3g) Sanity: in manifest đang dùng
        print("[SANITY] train manifest:", getattr(model.cfg.train_ds, "manifest_filepath", None))
        print("[SANITY] val   manifest:", getattr(model.cfg.validation_ds, "manifest_filepath", None))
        print("[SANITY] test  manifest:", getattr(model.cfg.test_ds, "manifest_filepath", None))

        return model, cfg

    # 4) Nhánh khởi tạo mới
    else:
        model = CustomFastConformerRNNTModel(cfg=cfg.model, trainer=trainer)
        return model, cfg





# --------------------------------- Main --------------------------------

def main():
    args = parse_args()

    # Nhánh TEST-ONLY: không cần tokenizer build lại, không cần train manifest
    if args.test_only:
        if args.test_manifest is None:
            raise ValueError("--test-only requires --test-manifest")
        test_from_checkpoint(
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
    # Nhánh TRAIN: cần build tokenizer, train manifest
    elif args.train_manifest is None:
        raise ValueError("--train-manifest is required for training")
    elif args.tokenizer_dir is None:
        raise ValueError("--tokenizer-dir is required for training")
    # Nhánh TRAIN: tiếp tục train
    else:
        main_train(args)

    

def main_train(args):
    # -------------------- Paths & basic stats --------------------
    train_manifest = args.train_manifest.expanduser().resolve()
    val_manifest   = args.val_manifest.expanduser().resolve() if args.val_manifest else None
    test_manifest  = args.test_manifest.expanduser().resolve() if args.test_manifest else None

    print("📊 Counting training samples…")
    train_samples = count_lines(train_manifest)
    print(f"📊 Number of training samples: {train_samples}")

    # -------------------- Tokenizer --------------------
    tokenizer_path, tokenizer_type_cfg = prepare_tokenizer(
        manifest_path=train_manifest,
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
        spe_type=args.spe_type,
        lowercase=args.lowercase_text,
    )

    # -------------------- Load + patch NeMo YAML --------------------
    cfg = configure_from_yaml(
        base_config=args.base_config,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        test_manifest=test_manifest,
        tokenizer_path=tokenizer_path,
        tokenizer_type_cfg=tokenizer_type_cfg,
        size=args.size,
        max_duration=args.max_duration,
        precision=args.precision,
        devices=args.devices,
        epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        disable_specaug=args.disable_specaug,
    )

    # Per-GPU batch size từ CLI
    cfg.model.train_ds.batch_size = int(args.batch_size)

    # -------------------- Optim scale & warmup --------------------
    scale_optim_and_warmup(
        cfg,
        train_samples=train_samples,
        epochs=args.epochs,
        devices=args.devices if isinstance(args.devices, int) else 1,
        acc_steps=args.accumulate_grad_batches,
    )

    # Nếu FT từ .nemo mà user KHÔNG chỉ định --lr, hạ LR mặc định xuống mức an toàn
    if getattr(args, "init_from_nemo", None) is not None and args.init_from_nemo and args.lr is None:
        cfg.model.optim.lr = min(cfg.model.optim.lr, 1e-4)
        cfg.model.optim.sched.min_lr = cfg.model.optim.lr * 0.1
        print(f"🔧 FT override LR → {cfg.model.optim.lr:.6f} (no --lr provided)")

    # Gradient clipping qua cfg.trainer (Lightning sẽ đọc)
    if getattr(args, "grad_clip", None) is not None:
        try:
            cfg.trainer.gradient_clip_val = float(args.grad_clip)
            cfg.trainer.gradient_clip_algorithm = "norm"
            print(f"✂️  Grad clip = {args.grad_clip}")
        except Exception:
            pass

    # -------------------- Trainer & loggers --------------------
    exp_dir = args.exp_dir.expanduser().resolve()
    exp_name = args.exp_name

    trainer = build_trainer(
        exp_dir=exp_dir,
        exp_name=exp_name,
        epochs=args.epochs,
        precision=args.precision,
        devices=args.devices,
    )

    attach_exp_manager(trainer, cfg, exp_dir=exp_dir, exp_name=exp_name)
    maybe_add_csv_logger(trainer)

    # -------------------- Build/Restore model for TRAIN --------------------
    print("🧠 Initializing / Restoring model…")
    model, cfg = build_or_restore_model_for_train(args, cfg, trainer)

    # -------------------- Freeze plan at start --------------------
    frozen = 0

    # 3.1 Encoder (bottom-first ưu tiên, rồi top-K)
    if getattr(args, "freeze_encoder_layers", 0) > 0:
        frozen = freeze_bottom_k_layers(model, int(args.freeze_encoder_layers))
    elif getattr(args, "freeze_encoder_ratio", 0.0) > 0.0:
        frozen = freeze_bottom_ratio(model, float(args.freeze_encoder_ratio))
    if getattr(args, "freeze_encoder_top_k", 0) > 0:
        # có thể freeze cả bottom lẫn top (tuỳ ý)
        _ = freeze_top_k_layers(model, int(args.freeze_encoder_top_k))

    if frozen > 0:
        print(f"❄️  Frozen bottom {frozen} encoder layer(s).")
    topk = getattr(args, "freeze_encoder_top_k", 0)
    if topk and topk > 0:
        print(f"❄️  Frozen TOP {topk} encoder layer(s).")

    # 3.2 Blocks / Components
    if getattr(args, "freeze_preprocessor", False):
        n = freeze_preprocessor(model)
        if n: print(f"❄️  Frozen preprocessor ({n} params).")

    if getattr(args, "freeze_subsampling", False):
        n = freeze_subsampling(model)
        if n: print(f"❄️  Frozen subsampling ({n} params).")

    if getattr(args, "freeze_pos_enc", False):
        n = freeze_pos_enc(model)
        if n: print(f"❄️  Frozen pos_enc ({n} params).")

    if getattr(args, "freeze_specaug", False):
        n = freeze_spec_augment(model)
        if n: print(f"❄️  Frozen spec_augmentation ({n} params).")

    # 3.3 Decoder & Joint
    if getattr(args, "freeze_decoder_all", False):
        n = freeze_decoder(model)
        if n: print(f"❄️  Frozen decoder (all, {n} params).")
    else:
        if getattr(args, "freeze_decoder_embed", False):
            n = freeze_decoder_embedding(model)
            if n: print(f"❄️  Frozen decoder.embed ({n} params).")
        if getattr(args, "freeze_decoder_rnn", False):
            n = freeze_decoder_rnn(model)
            if n: print(f"❄️  Frozen decoder.rnn ({n} params).")

    if getattr(args, "freeze_joint", False):
        n = freeze_joint(model)
        if n: print(f"❄️  Frozen joint ({n} params).")

    # 3.4 Advanced: prefix / regex / types
    pref_list = getattr(args, "freeze_prefix", []) or []
    if pref_list:
        affected = freeze_by_prefixes(model, pref_list)
        # chỉ in những prefix có ảnh hưởng >0
        affected = {k:v for k,v in affected.items() if v > 0}
        if affected:
            print(f"❄️  Frozen by prefixes: {affected}")

    regex_list = getattr(args, "freeze_regex", []) or []
    for pat in regex_list:
        n = freeze_by_regex(model, pat)
        if n > 0:
            print(f"❄️  Frozen by regex '{pat}' ({n} params).")

    type_list = getattr(args, "freeze_types", []) or []
    if type_list:
        # Hỗ trợ các tên: LayerNorm, BatchNorm1d, BatchNorm2d, BatchNorm3d, InstanceNorm1d/2d/3d
        n = freeze_by_types(model, tuple(type_list))  # freeze_utils chấp nhận tên dạng list[str]
        if n > 0:
            print(f"❄️  Frozen by types {type_list} ({n} params).")

    # 3.5 (Optional) Dump summary
    if getattr(args, "freeze_dump", False):
        tr, tot = count_params(model)
        print(f"Trainable params: {tr:,} / Total: {tot:,}")
        print("Frozen sample names:", snapshot_frozen(model, topk=15))





    # -------------------- Freeze plan (single-run schedule) --------------------

    # 3.b. Parse các --stage thành lịch
    stages = []
    for s in getattr(args, "stage", []) or []:
        stages.append(_parse_stage_str(s))

    # 3.c. Gắn callback lịch
    callbacks = getattr(trainer, "callbacks", [])
    sched_cb = FreezeScheduleCallback(stages=stages, dump=getattr(args, "freeze_dump_stages", False))
    if isinstance(callbacks, list):
        callbacks.append(sched_cb)
        trainer.callbacks = callbacks
    else:
        trainer.callbacks = [sched_cb]



    # -------------------- Memory hygiene --------------------
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 100)
    try:
        print(model.summarize(max_depth=4))
    except Exception:
        pass
    print("=" * 100)

    # -------------------- Train (with optional resume) --------------------
    print("🏋️ Starting training…")
    fit_kwargs = {}
    if getattr(args, "resume_from_ckpt", None):
        fit_kwargs["ckpt_path"] = str(args.resume_from_ckpt)
        print(f"↩️  Resuming from ckpt: {args.resume_from_ckpt}")

    # Với PL v2, callbacks thường nên truyền vào Trainer(...), nhưng nếu đã append ở trên thì fit() bình thường:
    trainer.fit(model, **fit_kwargs)

    # -------------------- Log last val metrics --------------------
    try:
        from pathlib import Path as _Path
        log_dir = _Path(trainer.log_dir) if getattr(trainer, 'log_dir', None) else exp_dir / exp_name
        val_wer  = float(trainer.callback_metrics.get('val_wer', 0.0))
        val_loss = float(trainer.callback_metrics.get('val_loss', 0.0))
        log_val_metrics_to_txt(log_dir, trainer.current_epoch or 0, val_wer, val_loss)
    except Exception:
        pass

    # -------------------- Test (optional) --------------------
    if test_manifest is not None:
        print("🔍 Running test…")
        trainer.test(model)


if __name__ == '__main__':
    main()
