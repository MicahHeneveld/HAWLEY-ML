"""ECGFounder-compatible filtering and normalization."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt


def filter_bandpass(signal: np.ndarray, fs: int) -> np.ndarray:
    """Bandpass + notch + baseline removal matching ECGFounder util.filter_bandpass."""
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    filtered = np.zeros_like(signal, dtype=np.float64)
    b, a = iirnotch(50, 30, fs)
    for channel in range(signal.shape[0]):
        filtered[channel] = filtfilt(b, a, signal[channel])

    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for channel in range(filtered.shape[0]):
        filtered[channel] = filtfilt(b, a, filtered[channel])

    baseline = np.zeros_like(filtered)
    kernel_size = int(0.4 * fs) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    for channel in range(filtered.shape[0]):
        baseline[channel] = medfilt(filtered[channel], kernel_size=kernel_size)

    return filtered - baseline


def resample_lead(
    signal: np.ndarray,
    fs_in: int,
    fs_out: int,
) -> np.ndarray:
    """Resample a 1D signal to fs_out using linear interpolation."""
    if fs_in == fs_out:
        return signal.astype(np.float64, copy=False)
    if signal.size == 0:
        return signal.astype(np.float64, copy=False)

    duration = signal.size / fs_in
    x_old = np.linspace(0.0, duration, num=signal.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=int(round(duration * fs_out)), endpoint=False)
    return np.interp(x_new, x_old, signal).astype(np.float64)


def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 1:
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return (signal - np.mean(signal, axis=-1, keepdims=True)) / (
        np.std(signal, axis=-1, keepdims=True) + 1e-8
    )


def prepare_single_lead_window(
    samples_uv: np.ndarray,
    fs_in: int,
    fs_out: int = 500,
    window_samples_out: int = 5000,
) -> np.ndarray:
    """Filter, resample, truncate/pad to ECGFounder single-lead shape (5000,)."""
    lead = samples_uv.astype(np.float64)
    lead = filter_bandpass(lead[np.newaxis, :], fs_in)[0]
    lead = resample_lead(lead, fs_in, fs_out)
    if lead.size >= window_samples_out:
        lead = lead[:window_samples_out]
    else:
        lead = np.pad(lead, (0, window_samples_out - lead.size))
    return z_score_normalize(lead)
