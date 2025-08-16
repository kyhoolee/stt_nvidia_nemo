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

    # Accelerator auto: PTL decides CPU/GPU based on availability
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=epochs,
        enable_checkpointing=False,  # handled by exp_manager
        logger=False,  # we use exp_manager + CSV logger
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        precision=precision,
        strategy=DDPStrategy(gradient_as_bucket_view=True) if (accelerator == "gpu" and (devices == -1 or (isinstance(devices, int) and devices > 1))) else "auto",
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FastConformer-Transducer on NeMo manifests")

    # Data + tokenizer
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path, default=None)
    p.add_argument('--test-manifest', type=Path, default=None)
    p.add_argument('--tokenizer-dir', type=Path, required=True)
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

    return p.parse_args()


# --------------------------------- Main --------------------------------

def main():
    args = parse_args()

    train_manifest = args.train_manifest.expanduser().resolve()
    val_manifest = args.val_manifest.expanduser().resolve() if args.val_manifest else None
    test_manifest = args.test_manifest.expanduser().resolve() if args.test_manifest else None

    # Count training samples for LR/warmup scaling
    print("📊 Counting training samples…")
    train_samples = count_lines(train_manifest)
    print(f"📊 Number of training samples: {train_samples}")

    # Prepare tokenizer
    tokenizer_path, tokenizer_type_cfg = prepare_tokenizer(
        manifest_path=train_manifest,
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
        spe_type=args.spe_type,
        lowercase=args.lowercase_text,
    )

    # Load + patch NeMo config
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

    # Set per-GPU batch size override from CLI
    cfg.model.train_ds.batch_size = int(args.batch_size)

    # Scale LR and warmup using actual global batch size and epochs
    scale_optim_and_warmup(
        cfg,
        train_samples=train_samples,
        epochs=args.epochs,
        devices=args.devices if isinstance(args.devices, int) else 1,
        acc_steps=args.accumulate_grad_batches,
    )

    # Build Trainer + loggers
    exp_dir = args.exp_dir.expanduser().resolve()
    exp_name = args.exp_name

    trainer = build_trainer(
        exp_dir=exp_dir,
        exp_name=exp_name,
        epochs=args.epochs,
        precision=args.precision,
        devices=args.devices,
    )

    # Attach NeMo exp_manager (creates final run directory, ckpt callbacks, etc.)
    attach_exp_manager(trainer, cfg, exp_dir=exp_dir, exp_name=exp_name)

    # Add CSV logger alongside TB
    maybe_add_csv_logger(trainer)

    # Instantiate model
    print("🧠 Initializing model…")
    # model = nemo_asr.models.EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)
    model = CustomFastConformerRNNTModel(cfg=cfg.model, trainer=trainer)


    # Clean memory a bit before training starts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 100)
    try:
        print(model.summarize(max_depth=4))
    except Exception:
        # summarize() can fail on some PTL/Nemo combos; not critical
        pass

    # Train
    print("🏋️ Starting training…")
    trainer.fit(model)

    # Log last val metrics, if available
    try:
        log_dir = Path(trainer.log_dir) if getattr(trainer, 'log_dir', None) else exp_dir / exp_name
        val_wer = float(trainer.callback_metrics.get('val_wer', 0.0))
        val_loss = float(trainer.callback_metrics.get('val_loss', 0.0))
        log_val_metrics_to_txt(log_dir, trainer.current_epoch or 0, val_wer, val_loss)
    except Exception:
        pass

    # Test
    if args.test_manifest is not None:
        print("🔍 Running test…")
        trainer.test(model)


if __name__ == '__main__':
    main()
