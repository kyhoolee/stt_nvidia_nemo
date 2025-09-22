#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Tuple

import torch
from jiwer import wer as jiwer_wer

from .nemo_like_runtime.preprocess import preprocess_paths_like_nemo
from .nemo_like_runtime.rnnt_onnx import RNNTModulesONNX, GreedyRNNTDecoder


# ---------- Tokenizer helpers ----------
def load_tokenizer(deploy_dir: Path):
    """
    Prefer SentencePiece model in asr_deploy/tokenizer/tokenizer.model.
    Fallback to spm_pieces.txt (id->piece list). If nothing found -> "none".
    """
    spm_path = deploy_dir / "tokenizer" / "tokenizer.model"
    if spm_path.is_file():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(); sp.load(str(spm_path))
        return ("spm", sp)

    pieces_path = deploy_dir / "spm_pieces.txt"
    if pieces_path.is_file():
        pieces = [line.rstrip("\n") for line in pieces_path.read_text(encoding="utf-8").splitlines()]
        return ("pieces", pieces)

    return ("none", None)

def ids_to_text(kind, tok, ids: List[int]) -> str:
    if kind == "spm":
        return tok.decode(ids)
    if kind == "pieces":
        s = "".join(tok[i] if 0 <= i < len(tok) else "" for i in ids)
        s = s.replace("▁", " ").strip()
        return " ".join(s.split())
    return " ".join(map(str, ids))

def ids_to_pieces(kind, tok, ids: List[int]) -> List[str]:
    if kind == "spm":
        return [tok.id_to_piece(int(i)) for i in ids]
    if kind == "pieces":
        return [tok[i] if 0 <= i < len(tok) else f"<unk:{i}>" for i in ids]
    return [str(i) for i in ids]


# ---------- IO ----------
def load_config(deploy_dir: Path) -> dict:
    cfg_path = deploy_dir / "config_minimal.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    return json.loads(cfg_path.read_text())

def read_manifest(path: Path) -> Tuple[List[str], List[str]]:
    audio_paths, texts = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            audio_paths.append(os.path.expanduser(obj["audio_filepath"]))
            texts.append(str(obj["text"]))
    return audio_paths, texts

def batched(xs: List, n: int):
    for i in range(0, len(xs), n):
        yield i, xs[i:i+n]


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", required=True)
    ap.add_argument("--manifest", action="append", required=True, help="Either a path or NAME=path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-u", type=int, default=256)
    ap.add_argument("--hard-topk", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="./onnx_eval_logs")

    # Debug flags
    ap.add_argument("--limit", type=int, default=None, help="only run first N samples per manifest")
    ap.add_argument("--debug", type=int, default=0, help="print detailed outputs for first N samples (forces bs=1 for those)")
    ap.add_argument("--trace", type=int, default=0, help="print T×U trace for first N samples (implies --debug)")
    args = ap.parse_args()

    deploy = Path(args.deploy).expanduser().resolve()
    onnx_dir = deploy / "onnx"
    enc_path = onnx_dir / "encoder.onnx"
    pred_path = onnx_dir / "predictor.onnx"
    joint_path = onnx_dir / "joint.onnx"

    kind, tok = load_tokenizer(deploy)
    print(f"[decoder] tokenizer={kind}")

    cfg = load_config(deploy)

    modules = RNNTModulesONNX(str(enc_path), str(pred_path), str(joint_path))
    L = int(cfg.get("pred_num_layers", 1))
    H = int(cfg.get("pred_hidden", 640))
    decoder = GreedyRNNTDecoder(modules, blank_id=int(cfg["blank_id"]),
                                vocab_size=int(cfg["vocab_size"]), L=L, H=H)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    named_manifests: List[Tuple[str, Path]] = []
    for m in args.manifest:
        if "=" in m:
            name, p = m.split("=", 1)
        else:
            name, p = Path(m).stem, m
        named_manifests.append((name, Path(p).expanduser().resolve()))

    for name, mf_path in named_manifests:
        audio_paths, refs = read_manifest(mf_path)
        if args.limit is not None:
            audio_paths = audio_paths[:args.limit]
            refs = refs[:args.limit]

        preds: List[str] = []

        # 1) DEBUG K samples (bs=1)
        K = max(args.debug, args.trace)
        if K > 0:
            K = min(K, len(audio_paths))
            print(f"========== DEBUG FIRST {K} SAMPLES ({name}) ==========")
            for i in range(K):
                apath = audio_paths[i]; ref = refs[i]
                proc_sig, proc_len = preprocess_paths_like_nemo([apath], cfg)  # [1,80,T], [1]
                enc, enc_len = modules.encode(proc_sig, proc_len)              # [1,T,512], [1]
                if args.trace > 0:
                    ids, trace = decoder.decode_one_with_trace(enc, int(enc_len[0]), max_u=args.max_u)
                else:
                    ids = decoder.decode_batch(enc, enc_len, max_u=args.max_u)[0]; trace = None

                hyp = ids_to_text(kind, tok, ids)
                pcs = ids_to_pieces(kind, tok, ids)

                print(f"\n--- sample #{i} ---")
                print(f"audio: {apath}")
                print(f"enc_len: {int(enc_len[0])}")
                print(f"REF : {ref}")
                print(f"IDS : {ids}")
                print(f"PIECES : {pcs}")
                print(f"HYP : {hyp}")
                if trace is not None:
                    print("TRACE (first 120 steps shown):")
                    for step in trace[:120]:
                        print(f"  t={step['t']:>4} u={step['u']:>4}  id={step['chosen_id']:>5}  blank={step['is_blank']}")

                preds.append(hyp.lower())

            # remove first K from remainder
            audio_paths = audio_paths[K:]
            refs = refs[K:]

        # 2) BULK
        for idx0, chunk in batched(audio_paths, args.batch_size):
            proc_sig, proc_len = preprocess_paths_like_nemo(chunk, cfg)
            enc, enc_len = modules.encode(proc_sig, proc_len)
            hyps_ids = decoder.decode_batch(enc, enc_len, max_u=args.max_u)
            chunk_texts = [ids_to_text(kind, tok, ids) for ids in hyps_ids]
            preds.extend([t.lower() for t in chunk_texts])
            done = K + min(idx0 + len(chunk), len(audio_paths))
            total = K + len(audio_paths)
            print(f"[{name}] Processed {done}/{total}")

        # WER
        if K > 0:
            # we evaluated first K already; combine refs accordingly
            all_refs = read_manifest(mf_path)[1][:K] + refs
        else:
            all_refs = refs
        all_refs = [r.lower() for r in all_refs]
        score = jiwer_wer(all_refs, preds)
        print("="*100)
        print(f"[{name}] WER = {score:.4f}  (N={len(all_refs)})")
        print("="*100)

        # dump preds
        log_path = out_dir / f"{name}_preds.tsv"
        with log_path.open("w", encoding="utf-8") as fw:
            fw.write("idx\treference\tprediction\n")
            for i, (r, p) in enumerate(zip(all_refs, preds)):
                fw.write(f"{i}\t{r}\t{p}\n")

        (out_dir / f"{name}_wer.txt").write_text(f"{score:.6f}\n", encoding="utf-8")

        # optional hard samples
        if args.hard_topk > 0:
            import jiwer
            per = [(i, jiwer.wer([r], [p])) for i, (r, p) in enumerate(zip(all_refs, preds))]
            per.sort(key=lambda x: x[1], reverse=True)
            topk = per[:args.hard_topk]
            hard_path = out_dir / f"{name}_hard.tsv"
            with hard_path.open("w", encoding="utf-8") as fw:
                fw.write("idx\twer\n")
                for i, w in topk:
                    fw.write(f"{i}\t{w:.6f}\n")
            print(f"[{name}] Saved hard samples Top-{args.hard_topk} → {hard_path}")

if __name__ == "__main__":
    main()
