import os, torch, json
from nemo.collections.asr.models import EncDecRNNTBPEModel


def normalize_hyp(h):
    """Trả về chuỗi transcript từ nhiều dạng trả về khác nhau."""
    if isinstance(h, str):
        return h
    # NeMo Hypothesis object
    t = getattr(h, "text", None)
    if isinstance(t, str):
        return t
    # dict-like
    if isinstance(h, dict) and isinstance(h.get("text"), str):
        return h["text"]
    # list of hyps (chọn cái đầu)
    if isinstance(h, (list, tuple)) and h:
        return normalize_hyp(h[0])
    # fallback
    return str(h)

nemo_path = "../nemo_work/_1_small_vi_ds/experiments/lsvsc/vpb_asr_fastconformer/2025-08-16_16-36-49/checkpoints/vpb_asr_fastconformer.nemo"

files = [
    "/home/kylh/work/stt_chunkformer/data/10000.wav",
]
files = [os.path.abspath(p) for p in files]

print("🔁 Loading model…")
model = EncDecRNNTBPEModel.restore_from(nemo_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
model.eval()

print(f"🗣️ Transcribing {len(files)} files…")
hyps = model.transcribe(
    files,
    batch_size=16,
    num_workers=2,
    return_hypotheses=False,
)

for p, h in zip(files, hyps):
    text = normalize_hyp(h)
    print(json.dumps({"file": p, "hyp": text}, ensure_ascii=False))
