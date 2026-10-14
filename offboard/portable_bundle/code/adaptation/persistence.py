"""Temporal smoothing and sustained-alert logic for reconstruction error streams."""

from __future__ import annotations

import numpy as np


def causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Causal (trailing) mean; index i uses values[max(0, i - window + 1) : i + 1]."""
    if window <= 1 or values.size == 0:
        return np.asarray(values, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    cumsum = np.concatenate(([0.0], np.cumsum(arr)))
    out = np.empty_like(arr)
    for i in range(arr.size):
        start = max(0, i - window + 1)
        count = i - start + 1
        out[i] = (cumsum[i + 1] - cumsum[start]) / count
    return out


def sustained_alert_mask(above_threshold: np.ndarray, consecutive: int) -> np.ndarray:
    """True at index i when `consecutive` trailing beats were all above threshold."""
    if consecutive <= 1:
        return np.asarray(above_threshold, dtype=bool)
    flags = np.asarray(above_threshold, dtype=bool)
    out = np.zeros(flags.shape, dtype=bool)
    if consecutive <= 0 or flags.size == 0:
        return out
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= consecutive:
            out[i] = True
    return out
