#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, sys
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

import nemo.collections.asr as nemo_asr

def disable_train_time_augs(model):
    try:
        if hasattr(model, 'spec_augmentation') and model.spec_augmentation is not None:
            # tắt hoàn toàn
            model.spec_augmentation = None
        if hasattr(model, 'preprocessor'):
            if hasattr(model.preprocessor, 'dither'):
                model.preprocessor.dither = 0.0
            if hasattr(model.preprocessor, 'pad_to'):
                model.preprocessor.pad_to = 0
    except Exception as e:
        print(f"[WARN] Không thể tắt augmentations: {e}", file=sys.stderr)

def set_greedy_decoding(model):
    try:
        # với RNNT, greedy_batch là nhanh và ổn định
        model.change_decoding_strategy(decoder_type="greedy_batch")
        if hasattr(model, 'wer'):
            model.wer.log_prediction = False
    except Exception as e:
        print(f"[WARN] Không set được greedy decoder: {e}", file=sys.stderr)

def load_model(nemo_path: Path, precision: str = "32"):
    print(f"🧠 Restore from .nemo: {nemo_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=str(nemo_path), map_location="cpu")
    model.eval()
    disable_train_time_augs(model)
    set_greedy_decoding(model)
    # chuyển device hợp lý
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    else:
        model = model.to(torch.device("cpu"))

    # set precision nếu cần
    if precision in ("16", "bf16") and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        model = model.to(dtype=amp_dtype)
    return model

def read_manifest(jsonl_path: Path):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    return items

def write_manifest(jsonl_path: Path, items, out_suffix: str):
    out_path = jsonl_path.with_suffix("")  # drop .jsonl
    out_path = Path(str(out_path) + out_suffix + ".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path

def transcribe_with_api(model, paths, batch_size: int):
    """
    Dùng API chính thống của NeMo (nhanh + ổn định).
    Trả về list[str] độ dài bằng số file.
    """
    texts = model.transcribe(
        paths2audio_files=paths,
        batch_size=batch_size,
        return_hypotheses=False,
        num_workers=0,
    )
    # NeMo đôi khi trả nested (list of list) — chuẩn hoá về list[str]
    flat = []
    for x in texts:
        if isinstance(x, list):
            flat.extend(x)
        else:
            flat.append(x)
    return flat

def transcribe_fallback_manual(model, paths):
    """
    Fallback: đọc audio, đưa thẳng qua forward + decoder.
    Chạy chậm hơn nhưng chắc ăn khi version lệch.
    """
    device = next(model.parameters()).device
    out = []
    for p in tqdm(paths, desc="[fallback] decoding"):
        # Để nguyên sample rate; NeMo preprocessor sẽ tự resample nếu config có.
        wav, sr = sf.read(p, dtype='float32', always_2d=False)
        if wav.ndim > 1:  # nếu stereo -> lấy kênh 0
            wav = wav[:, 0]
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)
        wav_len = torch.tensor([wav_tensor.shape[1]], device=device, dtype=torch.int64)

        with torch.no_grad():
            logits = model.forward(input_signal=wav_tensor, input_signal_length=wav_len)
            # RNNT: (logits, out_len) thường ở logits[0], logits[1]
            hyps = model.decoding.rnnt_decoder_predictions_tensor(logits[0], logits[1])
        out.append(hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0]))
    return out

def process_one_manifest(model, jsonl_path: Path, batch_size: int, out_suffix: str, overwrite_if_exists: bool):
    print(f"\n📄 Manifest: {jsonl_path}")
    items = read_manifest(jsonl_path)

    audio_paths = []
    for it in items:
        p = os.path.expanduser(it["audio_filepath"])
        audio_paths.append(p)

    # Nếu tất cả item đã có 'model_text' và không muốn ghi đè thì bỏ qua
    if not overwrite_if_exists and all("model_text" in it and str(it["model_text"]).strip() != "" for it in items):
        out_path = write_manifest(jsonl_path, items, out_suffix)
        print(f"✔️ Đã có sẵn model_text cho tất cả. Ghi sao chép ra: {out_path}")
        return out_path

    # Chạy decode
    try:
        preds = transcribe_with_api(model, audio_paths, batch_size=batch_size)
    except Exception as e:
        print(f"[WARN] transcribe() lỗi ({e}). Dùng fallback thủ công…", file=sys.stderr)
        preds = transcribe_fallback_manual(model, audio_paths)

    assert len(preds) == len(items), "Số transcript không khớp số dòng manifest!"

    # Gắn trường model_text (lower để nhất quán so sánh WER sau này)
    for it, hyp in zip(items, preds):
        it["model_text"] = (hyp or "").strip()

    out_path = write_manifest(jsonl_path, items, out_suffix)
    print(f"✅ Done: {out_path}")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Transcribe manifests and append `model_text`.")
    ap.add_argument("--nemo", type=Path, required=True, help=".nemo checkpoint path")
    ap.add_argument("--manifests", type=Path, nargs="+", required=True, help="List of JSONL manifests")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--precision", type=str, default="32", choices=["32","16","bf16"])
    ap.add_argument("--out-suffix", type=str, default=".with_model", help="Suffix cho file output (không gồm .jsonl)")
    ap.add_argument("--overwrite-if-exists", action="store_true", help="Nếu đã có model_text thì vẫn ghi đè")
    args = ap.parse_args()

    model = load_model(args.nemo, precision=args.precision)

    for m in args.manifests:
        process_one_manifest(
            model=model,
            jsonl_path=m,
            batch_size=args.batch_size,
            out_suffix=args.out-suffix if hasattr(args, "out-suffix") else args.out_suffix,  # phòng IDE tự đổi tên
            overwrite_if_exists=args.overwrite_if_exists,
        )

if __name__ == "__main__":
    main()
