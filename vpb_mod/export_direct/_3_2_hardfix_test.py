import onnxruntime as ort
import librosa
import numpy as np
import torch

# === Load ONNX models ===
encoder_sess = ort.InferenceSession("encoder-vpb_fastconformer.onnx", providers=["CPUExecutionProvider"])
decoder_sess = ort.InferenceSession("decoder_joint-vpb_fastconformer.onnx", providers=["CPUExecutionProvider"])

# === Tokenizer (bạn cần load lại từ tokenizer đã train với nemo) ===
from nemo.collections.asr.parts.submodules.tokenizer import TokenizerSpec
tokenizer = TokenizerSpec("tokenizer.model")  # thay bằng file BPE tokenizer đã export

# === Hàm chạy encoder ===
def run_encoder(audio_path, sample_rate=16000):
    # load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    audio = np.expand_dims(audio, axis=0).astype(np.float32)  # shape (1, T)

    audio_len = np.array([audio.shape[1]], dtype=np.int64)   # length tensor
    ort_inputs = {
        "audio_signal": audio,
        "length": audio_len,
    }
    ort_outs = encoder_sess.run(None, ort_inputs)
    return ort_outs  # thường gồm (encoder_out, encoded_lengths)

# === Hàm greedy decoding với RNNT ===
def greedy_decode(encoder_out, encoder_lens, max_symbols=200):
    batch_size, time_steps, hidden_dim = encoder_out.shape
    # bắt đầu với token <blank> hoặc <sos>
    hyps = [[tokenizer.bos_id]] * batch_size
    for t in range(time_steps):
        enc_t = encoder_out[:, t, :].astype(np.float32)  # (B, H)

        # lấy token cuối cùng trong mỗi hypothesis
        last_tokens = np.array([h[-1] for h in hyps], dtype=np.int64).reshape(batch_size, 1)

        ort_inputs = {
            "encoder_output": enc_t,
            "target_tokens": last_tokens,
        }
        logits = decoder_sess.run(None, ort_inputs)[0]  # shape (B, V)
        next_tokens = np.argmax(logits, axis=-1)

        for i in range(batch_size):
            if next_tokens[i] != tokenizer.blank_id:
                hyps[i].append(int(next_tokens[i]))
    return hyps

# === Hàm transcribe ===
def transcribe(audio_path):
    encoder_out, encoder_lens = run_encoder(audio_path)
    hyps = greedy_decode(encoder_out, encoder_lens)
    texts = [tokenizer.ids_to_text(hyp[1:]) for hyp in hyps]  # bỏ BOS
    return texts[0]

# === Test thử ===
audio_path = "/home/ubuntu/work/clean_dataset_vpb/audio/archive_2/wavs/E_huongds_D_2025-06-02_H_091735_844_CLID_0813494717___000040066___right___000040926.wav"

print("Predicted:", transcribe(audio_path))
