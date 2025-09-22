#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probe I/O of NeMo EncDecRNNTBPEModel submodules:
- preprocessor.forward
- encoder.forward
- decoder.forward (RNNTDecoder)
- joint.joint (NOT forward; avoid loss/wer branch)
It prints exact input/outputs (types, shapes) with dummy data.

Usage:
  python probe_rnnt_io.py --nemo /path/to/model.nemo
"""

import argparse
import inspect
from typing import Any
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

def ptitle(s: str):
    print("\n" + "="*100)
    print(s)
    print("="*100)

def fmt_shape(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor(dtype={x.dtype}, shape={tuple(x.shape)})"
    return str(type(x))

def walk_structure(x, indent=0, name=""):
    pad = "  " * indent
    if isinstance(x, torch.Tensor):
        print(f"{pad}{name}: Tensor[{tuple(x.shape)}], dtype={x.dtype}, device={x.device}")
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{name}: {type(x).__name__}(len={len(x)})")
        for i, v in enumerate(x):
            walk_structure(v, indent+1, name=f"[{i}]")
    elif isinstance(x, dict):
        print(f"{pad}{name}: dict(len={len(x)})")
        for k, v in x.items():
            walk_structure(v, indent+1, name=f"[{k!r}]")
    else:
        print(f"{pad}{name}: {type(x)} -> {x!r}")

def print_sig(mod, label: str):
    try:
        sig = inspect.signature(mod.forward)
        print(f"- forward signature of {label}: {sig}")
    except Exception as e:
        print(f"- (could not inspect signature of {label}): {type(e).__name__}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--sr", type=int, default=None, help="override sample rate (default: read from model)")
    args = ap.parse_args()

    # Load model
    model = EncDecRNNTBPEModel.restore_from(args.nemo).eval().to("cpu")

    # Disable augments (to mirror inference/ONNX)
    try:
        if hasattr(model, "spec_augmentation") and model.spec_augmentation is not None:
            model.spec_augmentation.mask_prob = 0.0
            model.spec_augmentation = None
        if hasattr(model, "preprocessor"):
            if hasattr(model.preprocessor, "dither"): model.preprocessor.dither = 0.0
            if hasattr(model.preprocessor, "pad_to"): model.preprocessor.pad_to = 0
    except Exception:
        pass

    # Basic config
    sr = int(getattr(model.preprocessor, "sample_rate", 16000))
    if args.sr: sr = args.sr
    n_mels = int(getattr(model.preprocessor, "features", 80))
    n_fft  = int(getattr(model.preprocessor, "n_fft", 512))
    frame_len = float(getattr(model.preprocessor, "frame_length", 0.02))
    frame_str = float(getattr(model.preprocessor, "frame_stride", 0.01))
    blank_id  = int(getattr(model.decoding, "blank_id", 0))

    # Decoder internals (prediction LSTM)
    try:
        L = model.decoder.prediction.dec_rnn.lstm.num_layers
        H = model.decoder.prediction.dec_rnn.lstm.hidden_size
    except Exception:
        L, H = None, None

    ptitle("MODEL OVERVIEW")
    print(model)
    print(f"\n[CFG] sample_rate={sr}, n_mels={n_mels}, n_fft={n_fft}, "
          f"frame_length={frame_len}, frame_stride={frame_str}, blank_id={blank_id}")
    print(f"[PRED NET] num_layers={L}, hidden_size={H}")

    # ---------- PREPROCESSOR ----------
    ptitle("PREPROCESSOR I/O")
    print_sig(model.preprocessor, "preprocessor")

    T = int(sr * 1.5)  # 1.5s
    wav = torch.randn(1, T, dtype=torch.float32)
    wav_len = torch.tensor([T], dtype=torch.int64)
    print("- input -> preprocessor: input_signal", fmt_shape(wav), ", length", fmt_shape(wav_len))

    with torch.no_grad():
        try:
            proc, proc_len = model.preprocessor(input_signal=wav, length=wav_len)
        except TypeError:
            proc, proc_len = model.preprocessor(wav, wav_len)

    print("- output <- preprocessor:")
    walk_structure(proc, name="processed_signal")
    walk_structure(proc_len, name="processed_signal_length")

    # ---------- ENCODER ----------
    ptitle("ENCODER I/O")
    print_sig(model.encoder, "encoder")

    # Expect encoder input [B, n_mels, T2]
    if proc.dim() == 3 and proc.shape[1] != n_mels and proc.shape[2] == n_mels:
        # Rare case some preproc variants output [B,T2,n_mels], fix to [B,n_mels,T2]
        proc = proc.transpose(1, 2).contiguous()

    print("- input -> encoder: processed_signal", fmt_shape(proc), ", length", fmt_shape(proc_len))

    with torch.no_grad():
        try:
            enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
        except TypeError:
            enc, enc_len = model.encoder(processed_signal=proc, length=proc_len)

    print("- raw output <- encoder:")
    walk_structure(enc, name="encoded (raw)")
    walk_structure(enc_len, name="encoded_length")

    # Normalize encoder to [B, T, D] (your model outputs [B, D, T])
    if enc.dim() == 3:
        if enc.shape[1] != 1 and enc.shape[1] != proc.shape[1]:
            enc_bt_d = enc.transpose(1, 2).contiguous()
        else:
            enc_bt_d = enc
    else:
        enc_bt_d = enc

    print("- normalized encoder (for joint):")
    walk_structure(enc_bt_d, name="encoded_bt_d [B,T,D]")

    # ---------- DECODER (RNNTDecoder) ----------
    ptitle("DECODER (RNNTDecoder) I/O")
    print_sig(model.decoder, "decoder")

    B = 2
    U = 4
    targets = torch.ones(B, U, dtype=torch.long)  # token=1
    target_length = torch.full((B,), U, dtype=torch.long)

    if L is None or H is None:
        try:
            L = model.decoder.prediction.dec_rnn.lstm.num_layers
            H = model.decoder.prediction.dec_rnn.lstm.hidden_size
        except Exception:
            L, H = 1, 512

    h0 = torch.zeros(L, B, H, dtype=torch.float32)
    c0 = torch.zeros(L, B, H, dtype=torch.float32)
    print("- input -> decoder: targets", fmt_shape(targets), ", target_length", fmt_shape(target_length),
          ", states(h0,c0)", fmt_shape(h0), fmt_shape(c0))

    with torch.no_grad():
        out = model.decoder(targets=targets, target_length=target_length, states=(h0, c0))

    print("- output <- decoder (structure):")
    walk_structure(out, name="decoder_out")

    if isinstance(out, tuple):
        if len(out) == 2:
            pred, st = out
            print("[DECODER] Looks like (pred, (h1,c1))")
            walk_structure(pred, name="pred")
            walk_structure(st,   name="states")
        elif len(out) == 3:
            pred, out_len2, st = out
            print("[DECODER] Looks like (pred, target_length_out, (h1,c1))")
            walk_structure(pred,    name="pred")
            walk_structure(out_len2,name="target_length_out")
            walk_structure(st,      name="states")
        else:
            print(f"[DECODER] tuple length={len(out)}; details above.")

    # ---------- JOINT ----------
    ptitle("JOINT I/O")
    print_sig(model.joint, "joint")

    # Build dummy inputs using normalized encoder output
    # enc_bt_d: [B,T,Denc]; pred_u must be [B,U,Hpred]
    B_enc, T_enc, D_enc = enc_bt_d.shape if enc_bt_d.dim()==3 else (1, 2, 512)
    T_take = min(2, T_enc)
    H_pred = H if H is not None else 640

    enc_t = enc_bt_d[:, :T_take, :].contiguous()          # [B, 1~2, D_enc]
    pred_u = torch.randn(B_enc, 3, H_pred, dtype=torch.float32)  # [B, U=3, H_pred]

    print("- input -> joint.joint: enc_bt_d", fmt_shape(enc_t), ", pred_u", fmt_shape(pred_u))

    with torch.no_grad():
        logits = model.joint.joint(enc_t, pred_u)  # BYPASS forward() to avoid loss/wer branch

    print("- output <- joint.joint:")
    walk_structure(logits, name="logits [B,T,U,vocab]")

    ptitle("DONE")
    print("Notes:")
    print("- JOINT: we call model.joint.joint(enc, pred) (pure joint) instead of forward() to avoid loss/wer inputs.")
    print("- ENCODER: normalized to [B,T,D]. Your model originally outputs [B,D,T].")
    print("- DECODER: see tuple structure (2-tuple vs 3-tuple) above for exact ONNX export mapping.")

if __name__ == "__main__":
    main()
