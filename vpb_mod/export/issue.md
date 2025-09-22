EncDecRNNTBPEModel was successfully restored from /home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo.

 =============== MODEL =============== 
EncDecRNNTBPEModel(
  (preprocessor): AudioToMelSpectrogramPreprocessor(
    (featurizer): FilterbankFeatures()
  )
  (encoder): ConformerEncoder(
    (pre_encode): ConvSubsampling(
      (out): Linear(in_features=2560, out_features=512, bias=True)
      (conv): Sequential(
        (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (4): ReLU(inplace=True)
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (7): ReLU(inplace=True)
      )
    )
    (pos_enc): RelPositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (layers): ModuleList(
      (0-16): 17 x ConformerLayer(
        (norm_feed_forward1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (feed_forward1): ConformerFeedForward(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
        (norm_conv): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (conv): ConformerConvolution(
          (pointwise_conv1): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (depthwise_conv): CausalConv1D(512, 512, kernel_size=(9,), stride=(1,), groups=512)
          (batch_norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Swish()
          (pointwise_conv2): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
        )
        (norm_self_att): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (self_attn): RelPositionMultiHeadAttention(
          (linear_q): Linear(in_features=512, out_features=512, bias=True)
          (linear_k): Linear(in_features=512, out_features=512, bias=True)
          (linear_v): Linear(in_features=512, out_features=512, bias=True)
          (linear_out): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear_pos): Linear(in_features=512, out_features=512, bias=False)
        )
        (norm_feed_forward2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (feed_forward2): ConformerFeedForward(
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm_out): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (decoder): RNNTDecoder(
    (prediction): ModuleDict(
      (embed): Embedding(1025, 640, padding_idx=1024)
      (dec_rnn): LSTMDropout(
        (lstm): LSTM(640, 640, dropout=0.2)
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
  )
  (joint): RNNTJoint(
    (pred): Linear(in_features=640, out_features=640, bias=True)
    (enc): Linear(in_features=512, out_features=640, bias=True)
    (joint_net): Sequential(
      (0): ReLU(inplace=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=640, out_features=1025, bias=True)
    )
    (_loss): RNNTLoss(
      (_loss): RNNTLossNumba()
    )
    (_wer): WER()
  )
  (loss): RNNTLoss(
    (_loss): RNNTLossNumba()
  )
  (spec_augmentation): SpectrogramAugmentation(
    (spec_augment): SpecAugment()
  )
  (wer): WER()
)
==============================================

[EXPORT][INFO] Disabled SpecAug + set dither=0, pad_to=0
[EXPORT][INFO] Saved config_minimal.json → vpb_mod/export/asr_deploy/config_minimal.json
[EXPORT][INFO] Exporting preprocessor.onnx ...
[EXPORT][ERROR] Export preprocessor.onnx failed (you can fallback to librosa later): TypeError: export() got an unexpected keyword argument 'use_complex_as_real'
[EXPORT][INFO] Exporting encoder.onnx ...
[EXPORT][INFO] OK: encoder.onnx
[EXPORT][INFO] Exporting predictor.onnx ...
[EXPORT][ERROR] Export predictor.onnx failed: TypeError: Input argument tokens has no corresponding input_type match. Existing input_types = dict_keys(['targets', 'target_length', 'states'])

=====================

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


=======================


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


=======================


Mình đang cố gắng export model ONNX cho model dạng 
nemo_asr.models.EncDecRNNTBPEModel

    "large": dict(d_model=512, n_heads=8, n_layers=17, pred_hidden=640, joint_hidden=640, weight_decay=1e-3, xscaling=True),


========================

giờ mình cần làm gì để xử lý được issue của phần preprocessing và phần decoder ? 