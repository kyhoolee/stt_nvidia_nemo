#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torchaudio
import onnxruntime as ort

# ================================
# Utils: audio loading + resample
# ================================
def load_audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


# ======================================
# Load NeMo .nemo to reuse tokenizer etc.
# ======================================
def load_nemo_asr(nemo_path: Path):
    import nemo.collections.asr as nemo_asr
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(nemo_path), map_location='cpu')
    asr_model.eval()
    # Đồng bộ tiền xử lý với bản origin:
    if hasattr(asr_model.preprocessor, 'dither'):
        asr_model.preprocessor.dither = 0.0
    if hasattr(asr_model.preprocessor, 'pad_to'):
        asr_model.preprocessor.pad_to = 0
    return asr_model


# ======================================
# ONNX I/O name resolver (flexible)
# ======================================
def get_input_names(session: ort.InferenceSession) -> List[str]:
    return [i.name for i in session.get_inputs()]

def get_output_names(session: ort.InferenceSession) -> List[str]:
    return [o.name for o in session.get_outputs()]

def prepare_encoder_io(enc_sess: ort.InferenceSession, signal_np: np.ndarray, length_np: np.ndarray) -> Dict[str, np.ndarray]:
    names = get_input_names(enc_sess)
    feed = {}
    # Common export variants
    # NeMo typically: "audio_signal", "length"
    candidates = {
        'audio_signal': signal_np,
        'signal': signal_np,
        'input_signal': signal_np,
        'waveforms': signal_np,
        'length': length_np,
        'input_length': length_np,
        'signal_length': length_np,
    }
    for n in names:
        if n in candidates:
            feed[n] = candidates[n]
        else:
            # try best-effort by suffix
            if 'length' in n.lower():
                feed[n] = length_np
            else:
                feed[n] = signal_np
    return feed

def prepare_predictor_io(pred_sess: ort.InferenceSession, token_np: np.ndarray, states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    names = get_input_names(pred_sess)
    feed = {}

    # tokens key
    token_keys = ['targets', 'y', 'tokens', 'input_ids', 'labels']
    key_token = None
    for k in token_keys:
        if k in names:
            key_token = k
            break
    if key_token is None:
        # fallback: first non-state input is token
        for n in names:
            if 'state' not in n.lower():
                key_token = n
                break
    feed[key_token] = token_np

    # states: could be states_hc or states_h + states_c (or lists per layer)
    state_names = [n for n in names if 'state' in n.lower()]
    if len(state_names) == 1:
        # packed state (e.g., "states_hc")
        feed[state_names[0]] = states['hc']
    else:
        # separate h, c
        # Try to map by contains 'h' vs 'c'
        h_key = next((n for n in state_names if 'h' in n.lower()), None)
        c_key = next((n for n in state_names if 'c' in n.lower()), None)
        if h_key is not None and c_key is not None:
            feed[h_key] = states['h']
            feed[c_key] = states['c']
        else:
            # if unknown, just broadcast packed to all
            for n in state_names:
                feed[n] = states.get(n, states['hc'])
    return feed

def parse_predictor_outputs(pred_sess: ort.InferenceSession, outs: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Heuristics: first is usually predictor embedding; the rest are states
    # We’ll infer by dims: pred_out should be [B,1,D] or [B, D]
    emb = None
    state_dict: Dict[str, np.ndarray] = {}
    for arr in outs:
        if arr.ndim >= 2 and (arr.shape[1] == 1 or arr.ndim == 2):
            if emb is None:
                emb = arr
                continue
        # Treat others as states
        # Try to decide h vs c by last letter in node name if available
        # But ort.run doesn’t give names, so we pack together as 'hc'
    if emb is None:
        emb = outs[0]
    # pack states as hc
    state_dict['hc'] = np.stack([outs[-2], outs[-1]]) if len(outs) >= 3 else outs[-1]
    # also expose h/c best-effort
    if len(outs) >= 3:
        state_dict['h'] = outs[-2]
        state_dict['c'] = outs[-1]
    else:
        # if only one state tensor, attempt split last dim/axis=0 if shaped like [2, ...]
        st = outs[-1]
        if st.shape[0] == 2:
            state_dict['h'] = st[0]
            state_dict['c'] = st[1]
    return emb, state_dict

def prepare_joint_io(joint_sess: ort.InferenceSession, enc_np: np.ndarray, pred_np: np.ndarray) -> Dict[str, np.ndarray]:
    names = get_input_names(joint_sess)
    feed = {}
    candidates = {
        'enc': enc_np, 'f': enc_np, 'encoder': enc_np, 'encoder_out': enc_np,
        'pred': pred_np, 'g': pred_np, 'predictor': pred_np, 'decoder_out': pred_np,
    }
    # If only 2 inputs, map by known names; else, use order: first->enc, second->pred
    if len(names) == 2:
        a, b = names
        feed[a] = candidates.get(a, enc_np if 'enc' in a.lower() or 'f' in a.lower() else pred_np)
        feed[b] = candidates.get(b, pred_np if 'pred' in b.lower() or 'g' in b.lower() else enc_np)
    else:
        # fallback by containment
        for n in names:
            if 'enc' in n.lower() or n.lower().endswith('f') or 'encoder' in n.lower():
                feed[n] = enc_np
            elif 'pred' in n.lower() or n.lower().endswith('g') or 'decoder' in n.lower():
                feed[n] = pred_np
            else:
                # default: enc first then pred
                feed[n] = enc_np if len([v for v in feed.values() if v is enc_np]) == 0 else pred_np
    return feed


def _as_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return default

def get_spm_id(obj, attr, default=-1):
    """
    Lấy ID từ SentencePieceProcessor / tokenizer:
    - Nếu là method (callable) => gọi rồi ép int.
    - Nếu là số / property => ép int.
    - Nếu không có => trả default.
    """
    if not hasattr(obj, attr):
        return default
    v = getattr(obj, attr)
    if callable(v):
        try:
            return _as_int(v(), default)
        except Exception:
            return default
    return _as_int(v, default)



# ======================================
# Greedy RNN-T decode (while-loop)
# ======================================
def rnnt_greedy_decode(
    encoder_out: np.ndarray,
    pred_sess: ort.InferenceSession,
    joint_sess: ort.InferenceSession,
    blank_id: int,
    start_token: int,
    num_layers: int,
    hidden_size: int,
    u_max: int = 4096,
) -> List[int]:
    """
    encoder_out: [B, T, D]
    Returns: token id list (no blanks)
    """
    B, T, D = encoder_out.shape
    assert B == 1, "This simple greedy decoder assumes batch=1."

    # Predictor initial state zeros
    h0 = np.zeros((num_layers, B, hidden_size), dtype=np.float32)
    c0 = np.zeros((num_layers, B, hidden_size), dtype=np.float32)
    states = {'hc': np.stack([h0, c0]), 'h': h0, 'c': c0}

    # Start token
    prev_token = np.array([[start_token]], dtype=np.int64)  # shape [B, 1] or [B,]

    t = 0
    u = 0
    out_ids: List[int] = []

    # while over time frames
    while t < T and u < u_max:


        # # Run predictor ONLY with current states + prev_token
        pred_feed = prepare_predictor_io(pred_sess, prev_token, states)
        pred_outs = pred_sess.run(None, pred_feed)
        pred_emb, next_states = parse_predictor_outputs(pred_sess, pred_outs)  # pred_emb: [B,1,Dp] (or [B,Dp])

        # Align dims to [B,1, Dp]
        if pred_emb.ndim == 2:
            pred_emb = pred_emb[:, None, :]

        # One frame from encoder
        enc_frame = encoder_out[:, t:t+1, :]  # [B,1,D]


        # --- TRONG HÀM rnnt_greedy_decode, ngay trước khi tạo joint_feed / gọi joint_sess.run ---

        # Normalize pred_emb to rank=3 as joint expects [B, 1, Dp] (or rank=2)
        # predictor có thể trả: [B,1,1,Dp] -> ép về [B,1,Dp]
        if pred_emb.ndim == 4 and pred_emb.shape[1] == 1 and pred_emb.shape[2] == 1:
            # giữ đúng trục time=1
            pred_emb = pred_emb[:, :1, 0, :]  # => [B,1,Dp]

        # Trường hợp khác: nếu predictor trả [B,Dp] thì thêm trục time
        elif pred_emb.ndim == 2:
            pred_emb = pred_emb[:, None, :]    # => [B,1,Dp]

        # Encoder frame cũng đảm bảo rank=3
        if enc_frame.ndim == 2:
            enc_frame = enc_frame[:, None, :]  # => [B,1,Df]
        elif enc_frame.ndim == 4 and enc_frame.shape[1] == 1 and enc_frame.shape[2] == 1:
            enc_frame = enc_frame[:, :1, 0, :] # an toàn nếu lỡ là [B,1,1,Df]




        # Joint
        joint_feed = prepare_joint_io(joint_sess, enc_frame, pred_emb)
        logits = joint_sess.run(None, joint_feed)[0]  # expect [B,1,1,V]
        if logits.ndim == 4:
            probs = logits[0, 0, 0, :]
        elif logits.ndim == 2:
            probs = logits[0, :]
        else:
            probs = logits.reshape(-1)
        k = int(np.argmax(probs))

        if k == blank_id:
            # move to next acoustic frame; DO NOT update predictor state
            t += 1
        else:
            # emit token; update predictor state; KEEP t
            out_ids.append(k)
            prev_token = np.array([[k]], dtype=np.int64)
            # critical: only update states on non-blank
            states = next_states
            u += 1

    return out_ids


# ======================================
# High-level inference wrapper
# ======================================
def infer_one_file(
    wav_path: Path,
    encoder_sess: ort.InferenceSession,
    predictor_sess: ort.InferenceSession,
    joint_sess: ort.InferenceSession,
    nemo_asr,
    use_bos_as_start: bool = True,
) -> str:
    sr_model = getattr(nemo_asr.preprocessor, 'sample_rate', 16000)
    wav, sr = load_audio(wav_path, sr_model)

    # NeMo preprocessor (Torch) -> numpy for ONNX encoder
    with torch.no_grad():
        sig = wav  # [1, T]
        length = torch.tensor([sig.shape[1]], dtype=torch.int64)
        processed_signal, processed_length = nemo_asr.preprocessor(input_signal=sig, length=length)

    # To numpy (batch first)
    signal_np = processed_signal.cpu().numpy()
    length_np = processed_length.cpu().numpy()

    # Encoder ONNX
    enc_feed = prepare_encoder_io(encoder_sess, signal_np, length_np)
    enc_outs = encoder_sess.run(None, enc_feed)
    # Take first output as encoder features (common export)
    enc = enc_outs[0]
    # Ensure shape [B,T,D]
    if enc.ndim == 2:
        enc = enc[:, None, :]

    # ================================================================

    # Tokenizer / SPM
    tokenizer = nemo_asr.tokenizer
    spm = getattr(tokenizer, "tokenizer", tokenizer)

    # Vocab size
    vocab_size = None
    if hasattr(spm, "__len__"):
        try:
            vocab_size = int(len(spm))
        except Exception:
            vocab_size = None
    if vocab_size is None and hasattr(spm, "get_piece_size"):
        try:
            vocab_size = int(spm.get_piece_size())
        except Exception:
            pass
    if vocab_size is None:
        # fallback an toàn
        vocab_size = 1024

    # blank_id: ưu tiên lấy từ tokenizer/nemo_asr; nếu không có dùng vocab_size
    blank_id = None
    for obj, name in [(tokenizer, "blank_id"), (nemo_asr, "blank_id")]:
        if blank_id is None and hasattr(obj, name):
            val = getattr(obj, name)
            blank_id = val() if callable(val) else val
    try:
        blank_id = int(blank_id) if blank_id is not None else vocab_size
    except Exception:
        blank_id = vocab_size

    # bos_id: dùng helper để an toàn với cả attr hoặc method
    bos_id = get_spm_id(spm, "bos_id", default=-1)

    # start_token: nếu bos_id hợp lệ (>=0) thì dùng BOS, ngược lại dùng BLANK
    start_token = bos_id if bos_id >= 0 else blank_id

    # ====================================================================



    # Try read decoder config for num_layers/hidden_size (fallback to commons)
    num_layers = getattr(getattr(nemo_asr, 'decoder', None), 'pred_rnn_layers', 2)
    hidden_size = getattr(getattr(nemo_asr, 'decoder', None), 'pred_rnn_dim', 640)

    ids = rnnt_greedy_decode(
        encoder_out=enc,
        pred_sess=predictor_sess,
        joint_sess=joint_sess,
        blank_id=int(blank_id),
        start_token=int(start_token),
        num_layers=int(num_layers),
        hidden_size=int(hidden_size),
        u_max=4096,
    )

    # Decode using NeMo tokenizer normalization
    text = nemo_asr.tokenizer.ids_to_text(ids)
    return text.strip()


# ======================================
# Evaluate on JSONL manifest
# Each line: {"audio_filepath": "...", "text": "..."}
# ======================================
def eval_manifest(
    manifest_path: Path,
    encoder_path: Path,
    predictor_path: Path,
    joint_path: Path,
    nemo_path: Path,
    provider: str = "CUDAExecutionProvider",
) -> None:
    # Load NeMo for tokenizer+preproc
    nemo_asr = load_nemo_asr(nemo_path)

    # ONNX sessions
    providers = [provider] if provider else ['CPUExecutionProvider']
    encoder_sess = ort.InferenceSession(str(encoder_path), providers=providers)
    predictor_sess = ort.InferenceSession(str(predictor_path), providers=providers)
    joint_sess = ort.InferenceSession(str(joint_path), providers=providers)

    # Read manifest
    items = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(obj)
    if not items:
        print("Empty manifest.")
        return

    from jiwer import wer
    refs, hyps = [], []

    t0 = time.time()
    total_audio_sec = 0.0

    for i, ex in enumerate(items, 1):
        wav = Path(ex['audio_filepath'])
        ref = ex.get('text', '').strip()
        # measure audio duration for RTF
        info = torchaudio.info(str(wav))
        total_audio_sec += float(info.num_frames) / float(info.sample_rate)

        hyp = infer_one_file(
            wav_path=wav,
            encoder_sess=encoder_sess,
            predictor_sess=predictor_sess,
            joint_sess=joint_sess,
            nemo_asr=nemo_asr,
            use_bos_as_start=True,
        )
        refs.append(ref)
        hyps.append(hyp)

        if i % 1 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] REF: {ref[:80]}")
            print(f"            HYP: {hyp[:80]}\n")

    t1 = time.time()
    decode_sec = t1 - t0
    wer_score = wer(refs, hyps)

    rtf = decode_sec / max(1e-6, total_audio_sec)
    print("========== Evaluation ==========")
    print(f"WER: {wer_score*100:.2f}%")
    print(f"RTF: {rtf:.4f} ( {decode_sec:.2f}s decode / {total_audio_sec:.2f}s audio )")


# ======================================
# CLI
# ======================================
def main():
    parser = argparse.ArgumentParser(description="ONNX RNNT Inference (fixed greedy)")
    parser.add_argument("--encoder", type=Path, required=True, help="Path to encoder.onnx")
    parser.add_argument("--predictor", type=Path, required=True, help="Path to predictor.onnx")
    parser.add_argument("--joint", type=Path, required=True, help="Path to joint.onnx")
    parser.add_argument("--nemo", type=Path, required=True, help="Path to .nemo (load tokenizer & preprocessor)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=Path, help="Single wav to transcribe")
    group.add_argument("--manifest", type=Path, help="JSONL manifest to evaluate")
    parser.add_argument("--provider", type=str, default="CUDAExecutionProvider", help="ONNX Runtime provider (CUDAExecutionProvider/CPUExecutionProvider)")
    args = parser.parse_args()

    if args.wav:
        nemo_asr = load_nemo_asr(args.nemo)
        providers = [args.provider] if args.provider else ['CPUExecutionProvider']
        encoder_sess = ort.InferenceSession(str(args.encoder), providers=providers)
        predictor_sess = ort.InferenceSession(str(args.predictor), providers=providers)
        joint_sess = ort.InferenceSession(str(args.joint), providers=providers)
        t0 = time.time()
        hyp = infer_one_file(args.wav, encoder_sess, predictor_sess, joint_sess, nemo_asr, use_bos_as_start=True)
        t1 = time.time()
        print(hyp)
        print(f"[Time] {t1 - t0:.3f}s")
    else:
        eval_manifest(args.manifest, args.encoder, args.predictor, args.joint, args.nemo, provider=args.provider)


if __name__ == "__main__":
    main()
