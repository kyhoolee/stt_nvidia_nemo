#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test FastConformer-RNNT (NeMo) on a new manifest.

Features
--------
- Restore model from .nemo (recommended) or .ckpt
- Evaluate WER/Loss via trainer.test()
- Optional: dump predictions (hyp) to JSONL for manual analysis
- Tweak batch size / devices / precision via CLI

Usage
-----
# 1) Test WER từ .nemo
python test_fastconformer_nemo.py \
  --nemo ~/work/nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo \
  --test-manifest ~/work/public_datasets/vi_small/your_new_ds/nemo_manifests/your_new_ds_test.jsonl \
  --devices 1 --precision 16 --batch-size 32

# 2) Vừa test WER vừa xuất dự đoán ra JSONL
python test_fastconformer_nemo.py \
  --nemo /path/to/model.nemo \
  --test-manifest /path/to/test.jsonl \
  --dump-jsonl ./predictions_your_new_ds.jsonl

# 3) Load từ .ckpt (ít khuyến nghị hơn .nemo)
python test_fastconformer_nemo.py \
  --ckpt /path/to/checkpoint.ckpt \
  --test-manifest /path/to/test.jsonl
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from lightning.pytorch import Trainer
import nemo.collections.asr as nemo_asr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeMo FastConformer-RNNT: Test on new manifest")

    mgroup = p.add_mutually_exclusive_group(required=True)
    mgroup.add_argument("--nemo", type=Path, help="Path to .nemo file")
    mgroup.add_argument("--ckpt", type=Path, help="Path to Lightning .ckpt")

    p.add_argument("--test-manifest", type=Path, required=True, help="NeMo manifest for testing")
    p.add_argument("--devices", type=int, default=1, help="-1 = all GPUs, else number of devices")
    p.add_argument("--precision", type=str, default="16", choices=["16", "32", "bf16"])
    p.add_argument("--batch-size", type=int, default=32, help="Per-device batch size for test loader")
    p.add_argument("--num-workers", type=int, default=4, help="num_workers for test dataloader")
    p.add_argument("--return-transcripts", action="store_true", help="Return transcripts during test step")
    p.add_argument("--dump-jsonl", type=Path, default=None, help="If set, dump predictions to this JSONL")
    p.add_argument("--no-test", action="store_true", help="Skip trainer.test (only dump predictions)")

    return p.parse_args()


def restore_model(nemo_path: Path | None, ckpt_path: Path | None) -> nemo_asr.models.EncDecRNNTBPEModel:
    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    if nemo_path is not None:
        print(f"🔁 Restoring from .nemo: {nemo_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            restore_path=str(nemo_path), map_location=map_loc, strict=False
        )
    else:
        print(f"🔁 Restoring from .ckpt: {ckpt_path}")
        model = nemo_asr.models.EncDecRNNTBPEModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path), map_location=map_loc, strict=False
        )
    model.eval()
    return model


def config_test_loader(model, test_manifest: Path, batch_size: int, num_workers: int, return_transcripts: bool):
    # chỉnh test_ds trỏ vào manifest mới
    model.cfg.test_ds.manifest_filepath = str(test_manifest)
    model.cfg.test_ds.batch_size = int(batch_size)
    model.cfg.test_ds.num_workers = int(num_workers)
    model.cfg.test_ds.return_transcripts = bool(return_transcripts)
    # tránh dump từng prediction vào logger nếu model có cấu hình như vậy
    try:
        if hasattr(model, "wer") and hasattr(model.wer, "log_prediction"):
            model.wer.log_prediction = False
    except Exception:
        pass
    # build dataloader
    model.setup_test_data(model.cfg.test_ds)


def read_manifest_audio_and_meta(path: Path) -> Tuple[List[str], List[dict]]:
    audio_paths, metas = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            audio_paths.append(item["audio_filepath"])
            metas.append({"utt_id": item.get("utt_id"), "text": item.get("text", "")})
    return audio_paths, metas


def dump_predictions(model, manifest_path: Path, out_jsonl: Path, batch_size: int, num_workers: int):
    audio_paths, metas = read_manifest_audio_and_meta(manifest_path)
    print(f"🗣️  Transcribing {len(audio_paths)} files…")
    hyps = model.transcribe(
        paths2audio_files=audio_paths,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        return_hypotheses=False,  # return plain strings
    )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as w:
        for meta, hyp in zip(metas, hyps):
            rec = {"utt_id": meta.get("utt_id"), "ref": meta.get("text"), "hyp": hyp}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✅ Saved predictions to: {out_jsonl}")


def main():
    args = parse_args()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.devices if accelerator == "gpu" else None

    model = restore_model(args.nemo, args.ckpt)

    # cấu hình test loader trỏ tới manifest mới
    test_manifest = args.test_manifest.expanduser().resolve()
    config_test_loader(
        model,
        test_manifest=test_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_transcripts=args.return_transcripts,
    )

    # chạy test (WER/Loss)
    if not args.no_test:
        print("🚀 Running test (WER/Loss)…")
        trainer = Trainer(accelerator=accelerator, devices=devices, precision=args.precision)
        test_out = trainer.test(model)
        print("📊 Test metrics:", test_out)

    # xuất dự đoán nếu yêu cầu
    if args.dump_jsonl is not None:
        dump_predictions(
            model,
            manifest_path=test_manifest,
            out_jsonl=args.dump_jsonl.expanduser().resolve(),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
