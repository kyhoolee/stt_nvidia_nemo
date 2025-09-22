#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export NeMo RNNT-BPE to ONNX (encoder, predictor, joint) + pack tokenizer assets.

Usage:
  python export_rnnt_core_onnx.py \
    --nemo /path/to/model.nemo \
    --out  ./asr_deploy \
    --opset 17 \
    --skip-preproc \
    --tokenizer-dir /path/to/tokenizer_spe_unigram_v1024

Notes:
- We DO NOT recommend exporting the preprocessor to ONNX. Compute mel at runtime.
- Encoder output is normalized to [B, T, 512].
- Predictor returns [B, U+1, H] and next states [2, L, B, H].
- Joint uses joint.joint(enc, pred) to avoid loss/WER branch.
"""

import json
import shutil
import traceback
import tarfile
from pathlib import Path
from typing import Optional

import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel


def log(msg: str): print(f"[EXPORT][INFO] {msg}")
def log_err(msg: str, e: Exception):
    print(f"[EXPORT][ERROR] {msg}: {type(e).__name__}: {e}")
    traceback.print_exc()


# ---------- Wrappers ----------
class PreprocWrap(torch.nn.Module):
    def __init__(self, preproc): super().__init__(); self.preproc = preproc
    def forward(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor):
        return self.preproc(input_signal=input_signal, length=input_signal_length)

class EncoderWrap(torch.nn.Module):
    """NeMo encoder returns [B, D, T]; normalize to [B, T, D] for ONNX/runtime."""
    def __init__(self, encoder): super().__init__(); self.encoder = encoder
    def forward(self, processed_signal: torch.Tensor, processed_signal_length: torch.Tensor):
        try:
            enc, enc_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        except TypeError:
            enc, enc_len = self.encoder(processed_signal=processed_signal, length=processed_signal_length)
        if enc.dim() == 3:
            enc = enc.transpose(1, 2).contiguous()  # [B, T, D]
        return enc, enc_len

class PredictorWrap(torch.nn.Module):
    """
    Stable ONNX signature:
      in : targets [B,U] (int64), target_length [B] (int64), states_hc [2,L,B,H] (float32)
      out: pred [B,U+1,H], next_states_hc [2,L,B,H]
    """
    def __init__(self, decoder): super().__init__(); self.decoder = decoder
    def forward(self, targets: torch.Tensor, target_length: torch.Tensor, states_hc: torch.Tensor):
        h = states_hc[0]  # [L,B,H]
        c = states_hc[1]  # [L,B,H]
        pred, _tgt_len_out, (h1, c1) = self.decoder(targets=targets, target_length=target_length, states=(h, c))
        pred = pred.transpose(1, 2).contiguous()  # [B,H,U+1] -> [B,U+1,H]
        next_states = torch.stack((h1, c1), dim=0)  # [2,L,B,H]
        return pred, next_states

class JointWrap(torch.nn.Module):
    """Use pure joint computation to get logits [B,T,U,V]."""
    def __init__(self, joint): super().__init__(); self.joint = joint
    def forward(self, enc: torch.Tensor, pred: torch.Tensor):
        return self.joint.joint(enc, pred)


# ---------- Helpers ----------
def disable_augments(m):
    try:
        if getattr(m, "spec_augmentation", None) is not None:
            m.spec_augmentation.mask_prob = 0.0
            m.spec_augmentation = None
        if hasattr(m, "preprocessor"):
            if hasattr(m.preprocessor, "dither"): m.preprocessor.dither = 0.0
            if hasattr(m.preprocessor, "pad_to"): m.preprocessor.pad_to = 0
        log("Disabled SpecAug + set dither=0, pad_to=0")
    except Exception as e:
        log_err("While disabling augments", e)

def force_all_to_cpu(module: torch.nn.Module):
    module.to("cpu")
    for mod in module.modules():
        for name, param in list(mod._parameters.items()):
            if isinstance(param, torch.Tensor):
                mod._parameters[name] = param.detach().to("cpu")
        for name, buf in list(mod._buffers.items()):
            if isinstance(buf, torch.Tensor):
                mod._buffers[name] = buf.detach().to("cpu")
    feat = getattr(getattr(module, "preprocessor", None), "featurizer", None)
    if feat is not None and isinstance(getattr(feat, "win", None), torch.Tensor):
        feat.win = feat.win.detach().to("cpu")


# ---------- Tokenizer packing ----------
def export_tokenizer_dir(tokenizer_dir: str, out_dir: Path):
    src = Path(tokenizer_dir).expanduser().resolve()
    dst = out_dir / "tokenizer"
    dst.mkdir(parents=True, exist_ok=True)
    copied_any = False
    for name in ["tokenizer.model", "tokenizer.vocab", "vocab.txt"]:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
            copied_any = True
    if copied_any:
        log(f"Copied tokenizer assets from {src} -> {dst}")
    else:
        log(f"No known tokenizer files in {src}; skipped")

def _extract_tokenizer_from_nemo(nemo_path: str, out_dir: Path) -> bool:
    try:
        with tarfile.open(nemo_path, "r:*") as tf:
            cand = None
            for m in tf.getmembers():
                if m.isfile() and m.name.lower().endswith(".model") and "tokenizer" in m.name.lower():
                    cand = m; break
            if cand is None:
                for m in tf.getmembers():
                    if m.isfile() and m.name.lower().endswith(".model"):
                        cand = m; break
            if cand is None:
                log("[WARN] No *.model found inside .nemo"); return False
            tmpdir = out_dir / "_tmp_tok"
            tmpdir.mkdir(parents=True, exist_ok=True)
            tf.extract(cand, path=tmpdir)
            src = (tmpdir / cand.name).resolve()
            dst_dir = out_dir / "tokenizer"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "tokenizer.model"
            shutil.move(str(src), str(dst))
            shutil.rmtree(tmpdir, ignore_errors=True)
            log(f"Extracted tokenizer.model from .nemo member: {cand.name}")
            return True
    except Exception as e:
        log_err("Extract tokenizer from .nemo failed", e)
        return False

def _dump_spm_pieces(model, out_dir: Path):
    try:
        tok = getattr(model, "tokenizer", None)
        sp = getattr(tok, "tokenizer", None)
        if sp is None:
            log("[WARN] No in-RAM SentencePieceProcessor; skip spm_pieces.txt"); return
        num_classes = int(getattr(getattr(model, "joint", None), "num_classes", 0))  # e.g. 1025
        piece_size = sp.get_piece_size()  # e.g. 1024
        upto = piece_size if num_classes == 0 else min(piece_size, num_classes - 1)
        lines = [sp.id_to_piece(i) for i in range(upto)]
        (out_dir / "spm_pieces.txt").write_text("\n".join(lines), encoding="utf-8")
        log(f"Dumped spm_pieces.txt with {len(lines)} entries")
    except Exception as e:
        log_err("Dumping spm_pieces.txt failed", e)

def export_tokenizer_assets(model, nemo_path: str, out_dir: Path, tokenizer_dir: Optional[str]):
    if tokenizer_dir:
        export_tokenizer_dir(tokenizer_dir, out_dir)
    else:
        # try extract from .nemo if not provided
        _extract_tokenizer_from_nemo(nemo_path, out_dir)
    _dump_spm_pieces(model, out_dir)


# ---------- Main export ----------
def export_rnnt_core_onnx(nemo_path: str, out_dir: str, opset: int = 17,
                          skip_preproc: bool = True, tokenizer_dir: Optional[str] = None):
    out = Path(out_dir); (out / "onnx").mkdir(parents=True, exist_ok=True)
    onnx_dir = out / "onnx"

    log(f"Loading NeMo model from: {nemo_path}")
    m = EncDecRNNTBPEModel.restore_from(nemo_path).eval()
    force_all_to_cpu(m)
    disable_augments(m)

    # decoder sizes
    try:
        L = int(m.decoder.prediction.dec_rnn.lstm.num_layers)
        H = int(m.decoder.prediction.dec_rnn.lstm.hidden_size)
    except Exception:
        L, H = 1, 640

    # Save config (use window_size/stride from NeMo to avoid preproc mismatch)
    cfg = {
        "sample_rate": int(getattr(m.preprocessor, "sample_rate", 16000)),
        "features":    int(getattr(m.preprocessor, "features", 80)),
        "n_fft":       int(getattr(m.preprocessor, "n_fft", 512)),
        "frame_length": float(getattr(m.preprocessor, "window_size",
                                getattr(m.preprocessor, "frame_length", 0.025))),
        "frame_stride": float(getattr(m.preprocessor, "window_stride",
                                getattr(m.preprocessor, "frame_stride", 0.01))),
        "normalize":   True,
        "blank_id":    int(getattr(m.decoding, "blank_id", 1024)),
        "vocab_size":  int(getattr(getattr(m, "joint", None), "num_classes", 1025)),
        "pred_hidden": int(H),
        "pred_num_layers": int(L),
        "tokenizer_relpath": "tokenizer/tokenizer.model",
        "tokenizer_type": "sentencepiece",
    }
    (out / "config_minimal.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    log(f"Saved config_minimal.json → {out/'config_minimal.json'}")

    # Tokenizer assets
    export_tokenizer_assets(m, nemo_path, out, tokenizer_dir)

    # PREPROCESSOR (optional)
    if not skip_preproc:
        try:
            log("Exporting preprocessor.onnx ...")
            preproc = PreprocWrap(m.preprocessor.eval().to("cpu"))
            wav = torch.randn(1, int(cfg["sample_rate"] * 3.0), dtype=torch.float32)
            wav_len = torch.tensor([wav.shape[1]], dtype=torch.int64)
            torch.onnx.export(
                preproc, (wav, wav_len), str(onnx_dir / "preprocessor.onnx"),
                input_names=["input_signal", "input_signal_length"],
                output_names=["processed_signal", "processed_signal_length"],
                dynamic_axes={
                    "input_signal": {0: "B", 1: "T"},
                    "input_signal_length": {0: "B"},
                    "processed_signal": {0: "B", 2: "T2"},
                    "processed_signal_length": {0: "B"},
                },
                opset_version=opset, do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
            )
            log("OK: preprocessor.onnx")
        except Exception as e:
            log_err("Export preprocessor.onnx failed (safe to skip; compute mel at runtime)", e)

    # ENCODER
    try:
        log("Exporting encoder.onnx ...")
        encw = EncoderWrap(m.encoder.eval().to("cpu"))
        mel = torch.randn(1, cfg["features"], 500, dtype=torch.float32)
        mel_len = torch.tensor([mel.shape[2]], dtype=torch.int64)
        torch.onnx.export(
            encw, (mel, mel_len), str(onnx_dir / "encoder.onnx"),
            input_names=["processed_signal", "processed_signal_length"],
            output_names=["encoded", "encoded_length"],
            dynamic_axes={
                "processed_signal": {0: "B", 2: "T2"},
                "processed_signal_length": {0: "B"},
                "encoded": {0: "B", 1: "T3"},
                "encoded_length": {0: "B"},
            },
            opset_version=opset, do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
        )
        log("OK: encoder.onnx")
    except Exception as e:
        log_err("Export encoder.onnx failed", e); return

    # PREDICTOR
    try:
        log("Exporting predictor.onnx ...")
        predw = PredictorWrap(m.decoder.eval().to("cpu"))
        B, U = 1, 4
        targets = torch.ones(B, U, dtype=torch.long)
        target_length = torch.full((B,), U, dtype=torch.long)
        states_hc = torch.zeros(2, L, B, H, dtype=torch.float32)
        torch.onnx.export(
            predw, (targets, target_length, states_hc), str(onnx_dir / "predictor.onnx"),
            input_names=["targets", "target_length", "states_hc"],
            output_names=["pred", "next_states_hc"],
            dynamic_axes={
                "targets": {0: "B", 1: "U"},
                "target_length": {0: "B"},
                "states_hc": {2: "B"},
                "pred": {0: "B", 1: "U"},
                "next_states_hc": {2: "B"},
            },
            opset_version=opset, do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
        )
        log("OK: predictor.onnx")
    except Exception as e:
        log_err("Export predictor.onnx failed", e); return

    # JOINT
    try:
        log("Exporting joint.onnx ...")
        jointw = JointWrap(m.joint.eval().to("cpu"))
        enc_t = torch.randn(1, 2, 512, dtype=torch.float32)
        pred_u = torch.randn(1, 3, H, dtype=torch.float32)
        torch.onnx.export(
            jointw, (enc_t, pred_u), str(onnx_dir / "joint.onnx"),
            input_names=["enc", "pred"],
            output_names=["logits"],
            dynamic_axes={
                "enc": {0: "B", 1: "T"},
                "pred": {0: "B", 1: "U"},
                "logits": {0: "B", 1: "T", 2: "U"},
            },
            opset_version=opset, do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL,
        )
        log("OK: joint.onnx")
    except Exception as e:
        log_err("Export joint.onnx failed", e); return

    files = [p.name for p in (onnx_dir).glob("*.onnx")]
    log(f"DONE. ONNX dir: {onnx_dir} ; files: {files}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-preproc", action="store_true", help="Skip exporting preprocessor.onnx (recommended)")
    ap.add_argument("--tokenizer-dir", type=str, default=None,
                    help="Folder with tokenizer.model/tokenizer.vocab/vocab.txt from training")
    args = ap.parse_args()
    export_rnnt_core_onnx(args.nemo, args.out, args.opset, args.skip_preproc, args.tokenizer_dir)

if __name__ == "__main__":
    main()
