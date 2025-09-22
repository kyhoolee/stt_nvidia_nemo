#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In ra I/O thực tế của các ONNX (encoder/predictor/joint) bằng onnxruntime.
"""

import sys
from pathlib import Path
import onnxruntime as ort

def dump_session(tag: str, path: Path):
    print("\n" + "="*100)
    print(f"{tag}: {path}")
    print("="*100)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("Inputs:")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"  [{i}] name={inp.name!r}, type={inp.type}, shape={inp.shape}")
    print("Outputs:")
    for i, out in enumerate(sess.get_outputs()):
        print(f"  [{i}] name={out.name!r}, type={out.type}, shape={out.shape}")

def main(deploy_dir: str):
    d = Path(deploy_dir)
    enc = d / "onnx" / "encoder.onnx"
    pred = d / "onnx" / "predictor.onnx"
    joint = d / "onnx" / "joint.onnx"
    for p in [enc, pred, joint]:
        if not p.is_file():
            print(f"!! Missing: {p}")
            return
    dump_session("ENCODER", enc)
    dump_session("PREDICTOR", pred)
    dump_session("JOINT", joint)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_onnx_io.py /path/to/asr_deploy")
        sys.exit(1)
    main(sys.argv[1])
