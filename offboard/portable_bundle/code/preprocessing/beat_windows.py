"""Extract beat-aligned 1 s windows for personal baseline training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.export_archive import ExportArchive
from datasets.sbec import read_sbec_file
from preprocessing.ecgfounder_prep import prepare_single_lead_window
from preprocessing.qc import load_beats, parse_tags


@dataclass(frozen=True)
class BeatWindow:
    recording_id: str
    chunk_index: int
    beat_index: int
    sample_offset: int
    started_at: str
    split: str
    samples_500hz: np.ndarray
    tags: frozenset[str] = frozenset()


def beat_passes_qc(
    tags: frozenset[str],
    *,
    layer_a: str,
    layer_c_include: str | None,
    layer_c_exclude: tuple[str, ...],
    layer_a_exclude: tuple[str, ...],
) -> bool:
    if not match_layer_a(tags, layer_a):
        return False
    if layer_c_include and layer_c_include not in tags:
        return False
    if any(tag in tags for tag in layer_c_exclude):
        return False
    if any(tag in tags for tag in layer_a_exclude):
        return False
    return True


def beat_passes_signal_qc(
    tags: frozenset[str],
    *,
    layer_c_include: str | None,
    layer_c_exclude: tuple[str, ...],
) -> bool:
    if layer_c_include and layer_c_include not in tags:
        return False
    if any(tag in tags for tag in layer_c_exclude):
        return False
    return True


def iter_beat_windows(
    archive: ExportArchive,
    *,
    train_recording_ids: set[str] | None,
    train_window_fraction: float | None,
    window_seconds: float,
    target_fs: int,
    layer_a: str,
    layer_c_include: str | None,
    layer_c_exclude: tuple[str, ...],
    layer_a_exclude: tuple[str, ...] = (),
    max_train_windows: int | None = None,
    max_eval_windows: int | None = None,
    seed: int = 42,
) -> list[BeatWindow]:
    """Build beat windows; split by recording ids or random window fraction."""
    windows: list[BeatWindow] = []
    target_samples = int(window_seconds * target_fs)
    half_width_500 = target_samples // 2

    for recording in archive.list_recordings():
        for chunk in recording.chunks:
            if chunk.beats_path is None:
                continue
            sample_rate, samples = read_sbec_file(chunk.bin_path)
            beats = load_beats(chunk.beats_path)
            if not beats:
                continue

            half_width_native = int(window_seconds * sample_rate / 2)
            for beat in beats:
                if not beat_passes_qc(
                    beat.tags,
                    layer_a=layer_a,
                    layer_c_include=layer_c_include,
                    layer_c_exclude=layer_c_exclude,
                    layer_a_exclude=layer_a_exclude,
                ):
                    continue

                start = beat.sample_offset - half_width_native
                end = beat.sample_offset + half_width_native
                if start < 0 or end > samples.size:
                    continue

                prepared = prepare_single_lead_window(
                    samples[start:end],
                    sample_rate,
                    fs_out=target_fs,
                    window_samples_out=target_samples,
                )
                if train_recording_ids is not None:
                    split = "train" if recording.recording_id in train_recording_ids else "eval"
                else:
                    split = "pool"
                windows.append(
                    BeatWindow(
                        recording_id=recording.recording_id,
                        chunk_index=chunk.chunk_index,
                        beat_index=beat.beat_index,
                        sample_offset=beat.sample_offset,
                        started_at=chunk.started_at,
                        split=split,
                        samples_500hz=prepared,
                        tags=beat.tags,
                    )
                )

    if train_recording_ids is None and train_window_fraction is not None:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(windows))
        rng.shuffle(indices)
        cutoff = max(1, min(int(len(indices) * train_window_fraction), len(indices) - 1))
        train_idx = set(indices[:cutoff].tolist())
        relabeled: list[BeatWindow] = []
        for i, window in enumerate(windows):
            split = "train" if i in train_idx else "eval"
            relabeled.append(
                BeatWindow(
                    recording_id=window.recording_id,
                    chunk_index=window.chunk_index,
                    beat_index=window.beat_index,
                    sample_offset=window.sample_offset,
                    started_at=window.started_at,
                    split=split,
                    samples_500hz=window.samples_500hz,
                    tags=window.tags,
                )
            )
        windows = relabeled

    if max_train_windows or max_eval_windows:
        train = [w for w in windows if w.split == "train"]
        eval_ = [w for w in windows if w.split == "eval"]
        if max_train_windows and len(train) > max_train_windows:
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(len(train), max_train_windows, replace=False).tolist())
            train = [w for i, w in enumerate(train) if i in keep]
        if max_eval_windows and len(eval_) > max_eval_windows:
            rng = np.random.default_rng(seed + 1)
            keep = set(rng.choice(len(eval_), max_eval_windows, replace=False).tolist())
            eval_ = [w for i, w in enumerate(eval_) if i in keep]
        windows = train + eval_

    return windows


def iter_all_signal_clean_beats(
    archive: ExportArchive,
    *,
    window_seconds: float,
    target_fs: int,
    layer_c_include: str | None,
    layer_c_exclude: tuple[str, ...],
) -> list[BeatWindow]:
    """All beats passing Layer C QC (any activity context)."""
    windows: list[BeatWindow] = []
    target_samples = int(window_seconds * target_fs)
    half_width_native_fn = lambda sr: int(window_seconds * sr / 2)

    for recording in archive.list_recordings():
        for chunk in recording.chunks:
            if chunk.beats_path is None:
                continue
            sample_rate, samples = read_sbec_file(chunk.bin_path)
            beats = load_beats(chunk.beats_path)
            if not beats:
                continue
            half_width_native = half_width_native_fn(sample_rate)
            for beat in beats:
                if not beat_passes_signal_qc(
                    beat.tags,
                    layer_c_include=layer_c_include,
                    layer_c_exclude=layer_c_exclude,
                ):
                    continue
                start = beat.sample_offset - half_width_native
                end = beat.sample_offset + half_width_native
                if start < 0 or end > samples.size:
                    continue
                prepared = prepare_single_lead_window(
                    samples[start:end],
                    sample_rate,
                    fs_out=target_fs,
                    window_samples_out=target_samples,
                )
                windows.append(
                    BeatWindow(
                        recording_id=recording.recording_id,
                        chunk_index=chunk.chunk_index,
                        beat_index=beat.beat_index,
                        sample_offset=beat.sample_offset,
                        started_at=chunk.started_at,
                        split="score",
                        samples_500hz=prepared,
                        tags=beat.tags,
                    )
                )
    return windows


def iter_all_qc_beats(
    archive: ExportArchive,
    *,
    window_seconds: float,
    target_fs: int,
    layer_a: str,
    layer_c_include: str | None,
    layer_c_exclude: tuple[str, ...],
    layer_a_exclude: tuple[str, ...] = (),
    seed: int = 42,
) -> list[BeatWindow]:
    """All QC-passing beat windows in the export (no train/eval split or subsampling)."""
    del seed  # reserved for future deterministic subsampling hooks
    return iter_beat_windows(
        archive,
        train_recording_ids=None,
        train_window_fraction=None,
        window_seconds=window_seconds,
        target_fs=target_fs,
        layer_a=layer_a,
        layer_c_include=layer_c_include,
        layer_c_exclude=layer_c_exclude,
        layer_a_exclude=layer_a_exclude,
        max_train_windows=None,
        max_eval_windows=None,
        seed=42,
    )
