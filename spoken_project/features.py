from __future__ import annotations

import numpy as np
import librosa 
from scipy.fft import dct


def _fix_audio(path: str, sr: int, seconds: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    target = int(sr * seconds)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    peak = np.max(np.abs(y))
    return y / peak if peak > 1e-9 else y


def _stats(matrix: np.ndarray) -> np.ndarray:
    """Fixed-dimensional utterance representation using mean and std."""
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)])


def mfcc_features(y: np.ndarray, sr: int, n_mfcc: int, n_fft: int,
                  hop: int, win: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                               hop_length=hop, win_length=win)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([_stats(mfcc), _stats(delta), _stats(delta2)])


def lfcc_features(y: np.ndarray, sr: int, n_lfcc: int, n_fft: int,
                  hop: int, win: int) -> np.ndarray:
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop,
                               win_length=win)) ** 2
    n_filters = 40
    freqs = np.linspace(0, sr / 2, power.shape[0])
    edges = np.linspace(0, sr / 2, n_filters + 2)
    bank = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        left, center, right = edges[i:i + 3]
        bank[i] = np.maximum(0, np.minimum((freqs-left)/(center-left+1e-12),
                                           (right-freqs)/(right-center+1e-12)))
    log_energy = np.log(np.maximum(bank @ power, 1e-12))
    lfcc = dct(log_energy, type=2, axis=0, norm="ortho")[:n_lfcc]
    delta = librosa.feature.delta(lfcc)
    delta2 = librosa.feature.delta(lfcc, order=2)
    return np.concatenate([_stats(lfcc), _stats(delta), _stats(delta2)])


def spectral_features(y: np.ndarray, sr: int, n_fft: int,
                      hop: int, win: int) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0, keepdims=True))
    return np.concatenate([_stats(x) for x in
                           (centroid, bandwidth, rolloff, zcr, rms, flux)])


def extract(path: str, kind: str, cfg: dict) -> np.ndarray:
    y = _fix_audio(path, cfg["sample_rate"], cfg["duration_seconds"])
    args = (y, cfg["sample_rate"])
    common = (cfg["n_fft"], cfg["hop_length"], cfg["win_length"])
    if kind == "mfcc":
        return mfcc_features(*args, cfg["n_mfcc"], *common)
    if kind == "lfcc":
        return lfcc_features(*args, cfg["n_lfcc"], *common)
    if kind == "spectral":
        return spectral_features(*args, *common)
    if kind == "combined":
        return np.concatenate([mfcc_features(*args, cfg["n_mfcc"], *common),
                               spectral_features(*args, *common)])
    raise ValueError(f"Unknown feature kind: {kind}")
