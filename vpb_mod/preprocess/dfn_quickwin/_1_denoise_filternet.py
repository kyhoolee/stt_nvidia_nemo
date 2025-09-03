#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, shutil
from pathlib import Path
import numpy as np
import soundfile as sf

# ---------- basic IO ----------
def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, sr

def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to: return x
    import librosa
    return librosa.resample(x, orig_sr=sr_from, target_sr=sr_to, res_type="kaiser_best").astype(np.float32)

def to_numpy_audio(y_any) -> np.ndarray:
    """Ép output về numpy CPU an toàn (kể cả khi là CUDA tensor)."""
    try:
        import torch
        if isinstance(y_any, torch.Tensor):
            t = y_any
        else:
            # cố gắng ép thành tensor để kéo về CPU
            t = torch.as_tensor(y_any)
        return t.squeeze().detach().cpu().numpy().astype(np.float32)
    except Exception:
        return np.squeeze(np.asarray(y_any)).astype(np.float32)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Enhance (denoise) hoặc copy WAV từ JSONL bằng DeepFilterNet3.")
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="DeepFilterNet3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy-only", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0 | cuda:1 | ...")
    ap.add_argument("--sr-out", type=int, default=16000, help="Sample rate output (mặc định 16k cho STT)")
    args = ap.parse_args()

    # === Force device BEFORE importing torch/df ===
    if args.device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ẩn GPU khỏi PyTorch
        os.environ["DF_DEVICE"] = "cpu"           # DeepFilterNet đọc biến này
    elif args.device.lower() != "auto":
        # cuda:0 -> chỉ để thấy đúng GPU index
        # (vd: cuda:1 => CUDA_VISIBLE_DEVICES=1)
        if args.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split("cuda:")[1]
        os.environ["DF_DEVICE"] = args.device

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = df_state = None
    enhance_fn = None
    model_device_str = "cpu"

    if not args.copy_only:
        try:
            from df import init_df, enhance as _enhance
            import torch
        except Exception:
            print("❌ Cần cài: pip install deepfilternet torch torchaudio librosa soundfile", file=sys.stderr)
            raise

        print(f"🔹 Loading model: {args.model}  |  requested_device={args.device}")
        model, df_state, _ = init_df(model_base_dir=args.model)

        # Phát hiện device thực sự (sau khi DF tự set)
        try:
            p = next(model.parameters())
            model_device_str = str(p.device)
        except Exception:
            # fallback: nếu không có parameters() thì giả định theo torch availability
            model_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"   ➜ DeepFilterNet actual device: {model_device_str}")
        enhance_fn = _enhance

    total, ok = 0, 0
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            idx = obj.get("idx")
            audio_path = obj.get("audio_path") or obj.get("audio_filepath")
            if not audio_path:
                continue

            total += 1
            if args.limit and total > args.limit:
                break

            in_wav = Path(audio_path).expanduser()
            stem = in_wav.stem
            out_wav = out_dir / f"{stem}_idx{idx}{'_denoised' if not args.copy_only else ''}.wav"

            if out_wav.exists() and not args.overwrite:
                print(f"⏭️  Skip (exists): {out_wav}")
                ok += 1
                continue

            try:
                if args.copy_only:
                    shutil.copy(in_wav, out_wav)
                else:
                    # 1) load, 2) upsample 48k cho DFN3
                    x, sr_in = load_mono(in_wav)
                    x48 = resample(x, sr_in, 48000)

                    import torch
                    device = torch.device(model_device_str)
                    xt = torch.from_numpy(x48).to(device=device, dtype=torch.float32).unsqueeze(0)

                    # 3) enhance
                    with torch.no_grad():
                        y_any = enhance_fn(model, df_state, xt)

                    # 4) về NumPy CPU an toàn
                    y48 = to_numpy_audio(y_any)

                    # 5) downsample về mong muốn
                    y_out = resample(y48, 48000, int(args.sr_out))
                    sf.write(str(out_wav), y_out, int(args.sr_out))

                ok += 1
                if ok % 25 == 0:
                    print(f"✅ Processed {ok}/{total} | last -> {out_wav}")
            except Exception as e:
                print(f"⚠️  Error: {in_wav} -> {e}", file=sys.stderr)

    print(f"🎯 Done. Success: {ok}/{total}. Output dir: {out_dir}")

if __name__ == "__main__":
    main()
