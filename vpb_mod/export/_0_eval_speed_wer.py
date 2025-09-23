import os
import json
import time
from pathlib import Path

import torch
import torchaudio
import librosa
import pandas as pd
from tqdm import tqdm
from jiwer import wer

import nemo.collections.asr as nemo_asr


def eval_fastconformer(nemo_file_path: str, test_manifest: str, device: str = "cuda"):
    """
    Đánh giá Fast-Conformer NeMo model: WER + Speed (RTF)
    """

    # --- Load model ---
    print(f"🧠 Restoring Fast-Conformer model from {nemo_file_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        restore_path=nemo_file_path,
        map_location=torch.device(device)  # "cpu" hoặc "cuda"
    )
    model.eval()
    model.to(device)


    # --- Disable augmentation ---
    if hasattr(model, 'spec_augmentation') and model.spec_augmentation is not None:
        model.spec_augmentation = None
    if hasattr(model, 'preprocessor'):
        if hasattr(model.preprocessor, 'dither'):
            model.preprocessor.dither = 0.0
        if hasattr(model.preprocessor, 'pad_to'):
            model.preprocessor.pad_to = 0

    # --- Decode function ---
    @torch.no_grad()
    def transcribe_audio(audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        audio_len = torch.tensor([audio_tensor.shape[1]]).to(device)

        logits, logit_len = model.forward(input_signal=audio_tensor,
                                          input_signal_length=audio_len)
        pred = model.decoding.rnnt_decoder_predictions_tensor(logits, logit_len)
        return pred[0]

    # --- Loop manifest ---
    refs, hyps = [], []
    total_decode_time, total_audio_dur = 0.0, 0.0

    with open(test_manifest, "r", encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f]

    for i, item in enumerate(tqdm(manifest, desc="Decoding")):
        audio_path = item["audio_filepath"]
        ref_text = item.get("text", "").strip()

        # audio duration
        info = torchaudio.info(audio_path)
        audio_dur = info.num_frames / info.sample_rate
        total_audio_dur += audio_dur

        # decode
        start = time.time()
        hyp_text = transcribe_audio(audio_path).text
        elapsed = time.time() - start
        total_decode_time += elapsed

        refs.append(ref_text.lower())
        hyps.append(hyp_text.lower())

        if (i + 1) % 1 == 0:
            print(f"[{i+1}] ref: {ref_text}")
            print(f"    hyp: {hyp_text}")

    # --- Metrics ---
    wer_score = wer(refs, hyps)
    rtf = total_decode_time / total_audio_dur

    print("=" * 100)
    print("✅ Finished evaluation (Fast-Conformer CTC)")
    print(f"WER: {wer_score*100:.2f}%")
    print(f"RTF: {rtf:.4f} ({total_decode_time:.2f}s decode / {total_audio_dur:.2f}s audio)")
    print("=" * 100)

    # save results
    df = pd.DataFrame({"ref": refs, "hyp": hyps})
    df.to_csv("fastconformer_results.csv", index=False, encoding="utf-8")

    return wer_score, rtf

nemo_file_path = "/home/ubuntu/work/nemo_work/_1_small_vi_ds/experiments/vpb_ft/vpb_asr_fastconformer_bigset_full_sched_eqv/2025-09-17_11-50-52/checkpoints/vpb_asr_fastconformer_bigset_full_sched_eqv.nemo"
test_manifest = "/home/ubuntu/work/clean_dataset_vpb/manifest/standard_test/test_meta_nemo.jsonl"

wer_score, rtf = eval_fastconformer(nemo_file_path, test_manifest, device="cpu")

