#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer a trained FastConformer-Transducer model (NeMo) from a checkpoint.

This script demonstrates how to:
- Load a trained model from a .nemo or .ckpt checkpoint file.
- Configure a new test/validation dataset dynamically.
- Run inference (test) on the new data to get WER and other metrics.

This is a standalone script and does not require the training pipeline.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import gc
import torch
from omegaconf import OmegaConf

# Lightning (PyTorch Lightning v2 name-space)
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy

# NeMo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# ----------------------------- CUSTOM CLASS (to avoid prediction logging) -----------------------------
class CustomFastConformerRNNTModel(EncDecRNNTBPEModel):
    def __init__(self, cfg, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)
        # Disable prediction logging to save memory and speed up
        self.wer.log_prediction = False

# --------------------------------- MAIN INFERENCE SCRIPT ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained ASR model checkpoint.")
    parser.add_argument('--checkpoint-path', type=Path, required=True,
                        help='Path to the .nemo or .ckpt model file.')
    parser.add_argument('--manifest-path', type=Path, required=True,
                        help='Path to the manifest file for the dataset to be evaluated (e.g., dev.jsonl).')
    parser.add_argument('--base-config', type=Path, required=True,
                        help='Path to the base YAML config used for training.')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use for inference. -1 uses all available GPUs.')
    parser.add_argument('--precision', type=str, default='16', choices=['16', '32', 'bf16'],
                        help='Precision for inference.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size per GPU for inference.')
    parser.add_argument('--size', type=str, default='small_0', 
                        choices=["small_0", "small", "medium", "large"], 
                        help="Model size preset used for training.")
    # Thêm dòng này để định nghĩa tham số --vocab-size
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='Vocabulary size used to train the tokenizer.')

    args = parser.parse_args()

    # Expand user and resolve paths
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve()
    base_config = args.base_config.expanduser().resolve()

    # --- Step 1: Initialize Trainer ---
    print("🚀 Initializing Trainer...")
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=10,
        strategy=DDPStrategy(gradient_as_bucket_view=True) if (torch.cuda.is_available() and (args.devices == -1 or (isinstance(args.devices, int) and args.devices > 1))) else "auto",
    )

    # --- Step 2: Load Model ---
    print(f"🧠 Loading model from checkpoint: {checkpoint_path}")
    try:
        # For .nemo files, this is straightforward
        if checkpoint_path.suffix == '.nemo':
            model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(checkpoint_path), trainer=trainer)
        # For .ckpt files, we need the original config
        elif checkpoint_path.suffix == '.ckpt':
            # Load the base config and patch it with the correct model size.
            cfg = OmegaConf.load(str(base_config))
            
            # This is a hard-coded patch matching your training script logic
            SIZE_PRESETS = {
                "small_0": dict(d_model=176, n_heads=1, n_layers=6, pred_hidden=320, joint_hidden=320, weight_decay=0.0, xscaling=True),
                "small": dict(d_model=176, n_heads=4, n_layers=16, pred_hidden=320, joint_hidden=320, weight_decay=0.0, xscaling=True),
                "medium": dict(d_model=256, n_heads=4, n_layers=16, pred_hidden=640, joint_hidden=640, weight_decay=1e-3, xscaling=True),
                "large": dict(d_model=512, n_heads=8, n_layers=17, pred_hidden=640, joint_hidden=640, weight_decay=1e-3, xscaling=True),
            }
            preset = SIZE_PRESETS[args.size]
            cfg.model.encoder.d_model = preset["d_model"]
            cfg.model.encoder.n_heads = preset["n_heads"]
            cfg.model.encoder.n_layers = preset["n_layers"]
            cfg.model.model_defaults.pred_hidden = preset["pred_hidden"]
            cfg.model.model_defaults.joint_hidden = preset["joint_hidden"]
            cfg.model.joint.jointnet.joint_hidden = preset["joint_hidden"]
            cfg.model.encoder.xscaling = preset["xscaling"]
            
            # We must load the tokenizer directory from training script's logic as well
            # The config needs a path to the tokenizer model file
            tokenizer_dir_path = Path("~/work/nemo_work/_1_small_vi_ds/tokenizers/lsvsc").expanduser().resolve()
            tokenizer_model_path = tokenizer_dir_path / f"tokenizer_spe_unigram_v{args.vocab_size}" / "tokenizer.model"
            
            cfg.model.tokenizer.dir = str(tokenizer_model_path.parent) # Trỏ tới thư mục chứa tokenizer.model
            cfg.model.tokenizer.type = 'bpe'
            
            # Load the model and then its state_dict
            model = CustomFastConformerRNNTModel(cfg=cfg.model, trainer=trainer)
            model.load_from_checkpoint(str(checkpoint_path), strict=False)
        else:
            raise ValueError("Unsupported file format. Use .nemo or .ckpt.")

        # Ensure the model is in evaluation mode
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            
    except Exception as e:
        logging.error(f"Failed to load the model from {checkpoint_path}. Error: {e}")
        return

    # --- Step 3: Configure and Run Inference ---
    print(f"🔍 Running inference on manifest: {manifest_path}")

    # Create a new test configuration using the provided manifest path
    infer_ds_config = OmegaConf.create({
        'manifest_filepath': str(manifest_path),
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': os.cpu_count() or 1,
        'pin_memory': True,
        'return_transcripts': False, # Save memory during test
    })
    
    # Temporarily override the model's test dataset config
    model.cfg.test_ds = infer_ds_config
    
    # Use the trainer to run the test
        # Use the trainer to run the test
    results = trainer.test(model)
    
    # NEW: In kết quả ra màn hình
    print("--------------------------------------------------")
    print("Inference Results:")
    if results and len(results) > 0:
        # Lấy dictionary kết quả đầu tiên (thường là kết quả tổng hợp)
        metrics = results[0]
        for key, value in metrics.items():
            print(f"- {key}: {value:.4f}")
    else:
        print("No results found. Something might have gone wrong with the test run.")
        
    print("--------------------------------------------------")

    print("✅ Inference complete.")
    
    print("✅ Inference complete. Check logs for results.")

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()