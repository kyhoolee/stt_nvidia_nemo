#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export NeMo RNNT core to ONNX (preprocessor, encoder, predictor, joint)
- Force everything to CPU (params + buffers) to avoid device mismatch
- Wrappers keep NeMo's kwargs-only semantics internally
- torch.onnx.export receives positional tuples for inputs
- Verbose logs + --skip-preproc option
"""

import json, shutil, traceback
from pathlib import Path
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

def log(msg: str): print(f"[EXPORT][INFO] {msg}")
def log_err(msg: str, e: Exception):
    print(f"[EXPORT][ERROR] {msg}: {type(e).__name__}: {e}")
    traceback.print_exc()

# ---------- Wrappers to enforce kwargs internally ----------
class PreprocWrap(torch.nn.Module):
    def __init__(self, preproc): super().__init__(); self.preproc = preproc
    def forward(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor):
        return self.preproc(input_signal=input_signal, length=input_signal_length)

class EncoderWrap(torch.nn.Module):
    def __init__(self, encoder): super().__init__(); self.encoder = encoder
    def forward(self, processed_signal: torch.Tensor, processed_signal_length: torch.Tensor):
        try:
            enc, enc_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        except TypeError:
            enc, enc_len = self.encoder(processed_signal=processed_signal, length=processed_signal_length)
        return enc, enc_len

class PredictorWrap(torch.nn.Module):
    def __init__(self, predictor): super().__init__(); self.predictor = predictor
    def forward(self, tokens: torch.Tensor, states: torch.Tensor):
        # Predictor của NeMo thường trả về (output, next_states)
        out, next_states = self.predictor(tokens=tokens, states=states)
        return out, next_states

class JointWrap(torch.nn.Module):
    def __init__(self, joint): super().__init__(); self.joint = joint
    def forward(self, enc: torch.Tensor, pred: torch.Tensor):
        try:    return self.joint(encoder_outputs=enc, decoder_outputs=pred)
        except TypeError: return self.joint(enc, pred)

# ---------- Helpers ----------
def disable_augments(m):
    try:
        if hasattr(m, "spec_augmentation") and m.spec_augmentation is not None:
            m.spec_augmentation.mask_prob = 0.0
            m.spec_augmentation = None
        if hasattr(m, "preprocessor"):
            if hasattr(m.preprocessor, "dither"): m.preprocessor.dither = 0.0
            if hasattr(m.preprocessor, "pad_to"): m.preprocessor.pad_to = 0
        log("Disabled SpecAug + set dither=0, pad_to=0")
    except Exception as e:
        log_err("While disabling augments", e)

def force_all_to_cpu(module: torch.nn.Module):
    """Ensure ALL params/buffers (even hidden) are on CPU (handles STFT window, etc.)."""
    module.to("cpu")
    for mod in module.modules():
        # parameters
        for name, param in list(mod._parameters.items()):
            if isinstance(param, torch.Tensor):
                mod._parameters[name] = param.detach().to("cpu")
        # buffers
        for name, buf in list(mod._buffers.items()):
            if isinstance(buf, torch.Tensor):
                mod._buffers[name] = buf.detach().to("cpu")
    # special case: featurizer.win if present
    feat = getattr(getattr(module, "preprocessor", None), "featurizer", None)
    if feat is not None and hasattr(feat, "win") and isinstance(feat.win, torch.Tensor):
        feat.win = feat.win.detach().to("cpu")

# ---------- Main export ----------
def export_rnnt_core_onnx(nemo_path: str, out_dir: str, opset: int = 17, skip_preproc: bool = False):
    out = Path(out_dir); (out / "onnx").mkdir(parents=True, exist_ok=True)
    onnx_dir = out / "onnx"

    log(f"Loading NeMo model from: {nemo_path}")
    m = EncDecRNNTBPEModel.restore_from(nemo_path).eval()

    print("\n =============== MODEL =============== ")
    print(m)
    print("==============================================\n")

    # Make SURE everything sits on CPU (params + buffers)
    force_all_to_cpu(m)
    disable_augments(m)

    # Save minimal config + tokenizer
    cfg = {
        "sample_rate": int(getattr(m.preprocessor, "sample_rate", 16000)),
        "features":    int(getattr(m.preprocessor, "features", 80)),
        "n_fft":       int(getattr(m.preprocessor, "n_fft", 512)),
        "frame_length":float(getattr(m.preprocessor, "frame_length", 0.02)),
        "frame_stride":float(getattr(m.preprocessor, "frame_stride", 0.01)),
        "normalize":   True,
        "blank_id":    int(getattr(m.decoding, "blank_id", 0)),
        "vocab_size":  int(getattr(m.decoder, "vocab_size", getattr(getattr(m, "joint", None), "num_classes", 1024))),
    }
    (out / "config_minimal.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    log(f"Saved config_minimal.json → {out/'config_minimal.json'}")

    tok_path = getattr(getattr(m, "tokenizer", None), "model_path", None)
    if tok_path and Path(tok_path).is_file():
        shutil.copy2(tok_path, out / "tokenizer.model")
        log(f"Copied tokenizer.model from {tok_path}")

    # ===== PREPROCESSOR (wav -> mel) =====
    if not skip_preproc:
        try:
            log("Exporting preprocessor.onnx ...")
            preproc = PreprocWrap(m.preprocessor.eval().to("cpu"))
            wav = torch.randn(1, int(cfg["sample_rate"] * 3.0), dtype=torch.float32)  # 3s dummy
            wav_len = torch.tensor([wav.shape[1]], dtype=torch.int64)
            torch.onnx.export(
                preproc,
                (wav, wav_len),  # positional tuple
                str(onnx_dir / "preprocessor.onnx"),
                input_names=["input_signal", "input_signal_length"],
                output_names=["processed_signal", "processed_signal_length"],
                dynamic_axes={
                    "input_signal": {0: "B", 1: "T"},
                    "input_signal_length": {0: "B"},
                    "processed_signal": {0: "B", 2: "T2"},
                    "processed_signal_length": {0: "B"},
                },
                opset_version=opset,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
                use_complex_as_real=True,  # <--- THÊM VÀO ĐỂ SỬA LỖI 1
            )
            log("OK: preprocessor.onnx")
        except Exception as e:
            log_err("Export preprocessor.onnx failed (you can fallback to librosa later)", e)

    # ===== ENCODER (mel -> enc) =====
    try:
        log("Exporting encoder.onnx ...")
        encw = EncoderWrap(m.encoder.eval().to("cpu"))
        mel = torch.randn(1, cfg["features"], 500, dtype=torch.float32)
        mel_len = torch.tensor([mel.shape[2]], dtype=torch.int64)
        torch.onnx.export(
            encw,
            (mel, mel_len),
            str(onnx_dir / "encoder.onnx"),
            input_names=["processed_signal", "processed_signal_length"],
            output_names=["encoded", "encoded_length"],
            dynamic_axes={
                "processed_signal": {0: "B", 2: "T2"},
                "processed_signal_length": {0: "B"},
                "encoded": {0: "B", 1: "T3"},
                "encoded_length": {0: "B"},
            },
            opset_version=opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
        log("OK: encoder.onnx")
    except Exception as e:
        log_err("Export encoder.onnx failed", e)
        return

    # ===== PREDICTOR (tokens -> pred) =====
    try:
        log("Exporting predictor.onnx ...")
        predw = PredictorWrap(m.decoder.eval().to("cpu"))
        tokens = torch.ones(1, 4, dtype=torch.long)
        
        # <--- THÊM VÀO ĐỂ SỬA LỖI 2 --->
        d_model = getattr(m.decoder, "d_model", getattr(m.decoder, "hidden_size", 512))
        num_layers = getattr(m.decoder, "num_layers", 2)
        # State của LSTM thường là một tuple (h, c). Để export ONNX,
        # chúng thường được stack lại thành 1 tensor [num_layers*2, B, hidden_size]
        # Nếu decoder của bạn là Transformer, state có thể khác.
        # Shape này khá phổ biến cho các decoder dựa trên LSTM.
        dummy_states = torch.randn(num_layers * 2, 1, d_model, dtype=torch.float32).to("cpu")

        torch.onnx.export(
            predw,
            (tokens, dummy_states),  # <--- THAY ĐỔI
            str(onnx_dir / "predictor.onnx"),
            input_names=["tokens", "states"], # <--- THAY ĐỔI
            output_names=["pred", "next_states"], # <--- THAY ĐỔI
            dynamic_axes={
                "tokens": {0: "B", 1: "U"},
                "states": {1: "B"}, # batch_size của state là dynamic
                "pred":   {0: "B", 1: "U"},
                "next_states": {1: "B"}, # batch_size của state là dynamic
            },
            opset_version=opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
        log("OK: predictor.onnx")
    except Exception as e:
        log_err("Export predictor.onnx failed", e)
        return
        

    # ===== JOINT (enc,pred -> logits) =====
    try:
        log("Exporting joint.onnx ...")
        jointw = JointWrap(m.joint.eval().to("cpu"))
        d_model = getattr(m.decoder, "d_model", getattr(m.decoder, "hidden_size", 512))
        enc_t = torch.randn(1, 2, d_model, dtype=torch.float32)
        pred_u = torch.randn(1, 3, d_model, dtype=torch.float32)
        torch.onnx.export(
            jointw,
            (enc_t, pred_u),
            str(onnx_dir / "joint.onnx"),
            input_names=["enc", "pred"],
            output_names=["logits"],
            dynamic_axes={
                "enc":   {0: "B", 1: "T"},
                "pred":  {0: "B", 1: "U"},
                "logits":{0: "B", 1: "T", 2: "U"},
            },
            opset_version=opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
        log("OK: joint.onnx")
    except Exception as e:
        log_err("Export joint.onnx failed", e)
        return

    # Final check
    files = [p.name for p in (onnx_dir).glob("*.onnx")]
    log(f"DONE. ONNX dir: {onnx_dir} ; files: {files}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-preproc", action="store_true", help="Skip exporting preprocessor.onnx")
    args = ap.parse_args()
    export_rnnt_core_onnx(args.nemo, args.out, args.opset, args.skip_preproc)

if __name__ == "__main__":
    main()
