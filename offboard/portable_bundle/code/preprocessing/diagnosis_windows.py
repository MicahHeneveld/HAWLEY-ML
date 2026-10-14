"""Weak-label 10 s windows for ECGFounder narrow-head fine-tune."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.export_archive import ExportArchive
from datasets.sbec import read_sbec_file
from preprocessing.ecgfounder_prep import prepare_single_lead_window
from preprocessing.qc import beats_in_window, load_beats, window_passes_qc
from preprocessing.tag_schema import LAYER_C_REJECT, has_sedentary_context, normalize_tags

# Weak wellness/context proxy — NOT clinical pathology labels.
CONTEXT_NORMAL_TAG = "resting_hr"
CONTEXT_ALERT_TAGS = frozenset(
    {"elevated_hr", "cognitive_stress", "arousal_spike", "recovering"}
)


@dataclass(frozen=True)
class DiagnosisWindow:
    recording_id: str
    chunk_index: int
    window_index: int
    label: int  # 0 = normal proxy, 1 = context-alert proxy
    samples_500hz: np.ndarray


def window_weak_label(beats) -> int | None:
    """Return 0/1 or None if ambiguous."""
    if not beats:
        return None
    tags_union: set[str] = set()
    resting = 0
    alert = 0
    for beat in beats:
        tags = normalize_tags(beat.tags)
        tags_union |= tags
        if CONTEXT_NORMAL_TAG in tags:
            resting += 1
        if tags & CONTEXT_ALERT_TAGS:
            alert += 1
    if alert > 0:
        return 1
    if resting >= max(1, len(beats) // 2):
        return 0
    return None


def iter_diagnosis_windows(
    archive: ExportArchive,
    *,
    window_seconds: float = 10.0,
    stride_seconds: float = 10.0,
    target_fs: int = 500,
    max_windows: int | None = None,
    seed: int = 42,
) -> list[DiagnosisWindow]:
    """10 s ECGFounder windows on sedentary, signal-clean chunks only."""
    layer_c_exclude = tuple(LAYER_C_REJECT)
    windows: list[DiagnosisWindow] = []
    target_samples = int(window_seconds * target_fs)

    for recording in archive.list_recordings():
        for chunk in recording.chunks:
            if chunk.beats_path is None:
                continue
            sample_rate, samples = read_sbec_file(chunk.bin_path)
            beats = load_beats(chunk.beats_path)
            if not beats:
                continue

            window_samples_in = int(window_seconds * sample_rate)
            stride_samples_in = max(1, int(stride_seconds * sample_rate))
            window_index = 0
            for start in range(0, max(1, samples.size - window_samples_in + 1), stride_samples_in):
                end = start + window_samples_in
                if end > samples.size:
                    break
                chunk_beats = beats_in_window(beats, start, end)
                if not chunk_beats:
                    continue
                if not all(has_sedentary_context(b.tags) for b in chunk_beats):
                    continue
                if not window_passes_qc(
                    chunk_beats,
                    layer_a="sedentary_stationary",
                    layer_c_include=None,
                    layer_c_exclude=layer_c_exclude,
                    layer_a_exclude=(),
                    min_layer_a_fraction=0.5,
                ):
                    continue
                label = window_weak_label(chunk_beats)
                if label is None:
                    continue

                prepared = prepare_single_lead_window(
                    samples[start:end],
                    sample_rate,
                    fs_out=target_fs,
                    window_samples_out=target_samples,
                )
                windows.append(
                    DiagnosisWindow(
                        recording_id=recording.recording_id,
                        chunk_index=chunk.chunk_index,
                        window_index=window_index,
                        label=label,
                        samples_500hz=prepared,
                    )
                )
                window_index += 1
                if max_windows and len(windows) >= max_windows:
                    return windows

    if max_windows and len(windows) > max_windows:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(windows), max_windows, replace=False)
        windows = [windows[i] for i in sorted(pick.tolist())]
    return windows


def beat_diagnosis_window(
    samples: np.ndarray,
    sample_rate: int,
    beat_sample_offset: int,
    *,
    window_seconds: float = 10.0,
    target_fs: int = 500,
) -> np.ndarray:
    """10 s ECGFounder vector centered on beat R-peak."""
    target_samples = int(window_seconds * target_fs)
    half_native = int(window_seconds * sample_rate / 2)
    start = beat_sample_offset - half_native
    end = beat_sample_offset + half_native
    if start < 0 or end > samples.size:
        return np.zeros(target_samples, dtype=np.float32)
    return prepare_single_lead_window(
        samples[start:end],
        sample_rate,
        fs_out=target_fs,
        window_samples_out=target_samples,
    ).astype(np.float32)
