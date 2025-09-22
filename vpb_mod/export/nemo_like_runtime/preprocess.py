# -*- coding: utf-8 -*-
"""
Audio loading + log-mel extraction khớp NeMo (HTK mel, power=2, log, per-feature normalize).
"""
from typing import List, Tuple
from pathlib import Path
import numpy as np
import torch, torchaudio
import soundfile as sf

def load_wav_mono(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)  # mono
    maxabs = float(np.max(np.abs(wav))) or 1.0
    wav = wav / maxabs
    return torch.from_numpy(wav), sr

def resample_if_needed(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=sr_in, new_freq=sr_out)

def fbank_like_nemo(
    wav: torch.Tensor,
    sr: int,
    n_mels: int,
    n_fft: int,
    frame_length: float,
    frame_stride: float,
    normalize: bool = True,
    eps: float = 1e-5,
) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # [1,T]
    win_length = int(sr * frame_length)
    hop_length = int(sr * frame_stride)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
        norm=None,
        mel_scale="htk",
    )(wav)  # [1, n_mels, T_frames]
    logmel = torch.log(mel + eps)
    if normalize:
        mean = logmel.mean(dim=[0, 2], keepdim=True)
        std  = logmel.std(dim=[0, 2], keepdim=True).clamp_min(1e-5)
        logmel = (logmel - mean) / std
    return logmel.squeeze(0)  # [n_mels, T_frames]

def batchify_mels(mels: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(mels)
    n_mels = mels[0].shape[0]
    lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    T_max = int(lengths.max().item())
    batch = torch.zeros(B, n_mels, T_max, dtype=torch.float32)
    for i, m in enumerate(mels):
        T = m.shape[1]
        batch[i, :, :T] = m
    return batch, lengths

def preprocess_paths_like_nemo(paths: List[str], cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    sr = int(cfg["sample_rate"])
    mels = []
    for p in paths:
        wav, sr_in = load_wav_mono(p)
        wav = resample_if_needed(wav, sr_in, sr)
        mel = fbank_like_nemo(
            wav, sr,
            n_mels=int(cfg["features"]),
            n_fft=int(cfg["n_fft"]),
            frame_length=float(cfg["frame_length"]),
            frame_stride=float(cfg["frame_stride"]),
            normalize=bool(cfg["normalize"]),
        )
        mels.append(mel)
    return batchify_mels(mels)
