#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe tokenizer/vocab/blank_id/num_classes từ EncDecRNNTBPEModel.
"""
import argparse, inspect, json
from pathlib import Path
from nemo.collections.asr.models import EncDecRNNTBPEModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    args = ap.parse_args()

    m = EncDecRNNTBPEModel.restore_from(args.nemo).eval()
    print("\n=== BASIC ===")
    print(type(m))
    print("blank_id:", getattr(m.decoding, "blank_id", None))
    print("joint.num_classes:", getattr(m.joint, "num_classes", None))

    print("\n=== TOKENIZER FIELD ON MODEL ===")
    tok = getattr(m, "tokenizer", None)
    print("m.tokenizer:", type(tok))
    if tok is not None:
        # NeMo’s SentencePieceTokenizer wrapper usually has .model_path and .tokenizer (SentencePieceProcessor)
        model_path = getattr(tok, "model_path", None)
        print("tokenizer.model_path:", model_path)
        inner = getattr(tok, "tokenizer", None)
        print("tokenizer.tokenizer:", type(inner))
        if inner is not None:
            try:
                size = inner.get_piece_size()
            except Exception:
                size = None
            print("tokenizer.piece_size:", size)
            # print a few samples
            for i in [0,1,2,3,4,100,500,1000]:
                if size is not None and i < size:
                    print(f"  id {i:4d}: {inner.id_to_piece(i)!r}")
    else:
        print("No m.tokenizer found!")

    print("\n=== DECODING STRATEGY ===")
    dec = getattr(m, "decoding", None)
    print("m.decoding:", type(dec))
    if dec is not None:
        # decoding.greedy, decoding.beam, etc may exist depending on config
        for name in dir(dec):
            if name.startswith("_"): continue
            attr = getattr(dec, name)
            if callable(attr) or isinstance(attr, (int, float, str, bool)):
                print(f"  {name}: {attr}")

    print("\n=== CONFIG SNAPSHOT ===")
    try:
        cfg = m.cfg
        print(json.dumps(cfg, indent=2, default=str)[:2000], "...\n")
    except Exception as e:
        print("Cannot print m.cfg:", e)

if __name__ == "__main__":
    main()
