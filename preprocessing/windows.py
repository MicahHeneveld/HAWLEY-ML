"""Extract QC-filtered 10 s windows from Sandbox chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets.export_archive import ExportArchive, RecordingRecord
from datasets.sbec import read_sbec_file
from preprocessing.ecgfounder_prep import prepare_single_lead_window
from preprocessing.qc import beats_in_window, load_beats, window_passes_qc


@dataclass(frozen=True)
class EcgWindow:
    recording_id: str
    chunk_index: int
    window_index: int
    started_at: str
    split: str
    samples_500hz: np.ndarray


def iter_clean_rest_windows(
    archive: ExportArchive,
    *,
    train_recording_ids: set[str],
    window_seconds: float,
    stride_seconds: float,
    target_fs: int,
    layer_a: str,
    layer_c_include: str | None,
    layer_c_exclude: tuple[str, ...],
    layer_a_exclude: tuple[str, ...] = (),
    min_layer_a_fraction: float = 0.5,
    max_train_windows: int | None = None,
    max_eval_windows: int | None = None,
) -> list[EcgWindow]:
    windows: list[EcgWindow] = []
    train_count = 0
    eval_count = 0
    target_samples = int(window_seconds * target_fs)

    for recording in archive.list_recordings():
        split = "train" if recording.recording_id in train_recording_ids else "eval"
        for chunk in recording.chunks:
            if split == "train" and max_train_windows and train_count >= max_train_windows:
                continue
            if split == "eval" and max_eval_windows and eval_count >= max_eval_windows:
                continue
            if (
                max_train_windows
                and max_eval_windows
                and train_count >= max_train_windows
                and eval_count >= max_eval_windows
            ):
                return windows
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
                if not window_passes_qc(
                    chunk_beats,
                    layer_a=layer_a,
                    layer_c_include=layer_c_include,
                    layer_c_exclude=layer_c_exclude,
                    layer_a_exclude=layer_a_exclude,
                    min_layer_a_fraction=min_layer_a_fraction,
                ):
                    continue

                prepared = prepare_single_lead_window(
                    samples[start:end],
                    sample_rate,
                    fs_out=target_fs,
                    window_samples_out=target_samples,
                )
                windows.append(
                    EcgWindow(
                        recording_id=recording.recording_id,
                        chunk_index=chunk.chunk_index,
                        window_index=window_index,
                        started_at=chunk.started_at,
                        split=split,
                        samples_500hz=prepared,
                    )
                )
                window_index += 1
                if split == "train":
                    train_count += 1
                    if max_train_windows and train_count >= max_train_windows:
                        break
                else:
                    eval_count += 1
                    if max_eval_windows and eval_count >= max_eval_windows:
                        break
    return windows


def split_recordings_by_time(
    recordings: list[RecordingRecord],
    train_fraction: float,
) -> tuple[set[str], set[str]]:
    if not recordings:
        return set(), set()
    cutoff = int(len(recordings) * train_fraction)
    cutoff = max(1, min(cutoff, len(recordings) - 1))
    train_ids = {rec.recording_id for rec in recordings[:cutoff]}
    eval_ids = {rec.recording_id for rec in recordings[cutoff:]}
    return train_ids, eval_ids


def split_recordings_random(
    recordings: list[RecordingRecord],
    train_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    ids = [rec.recording_id for rec in recordings]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cutoff = max(1, min(int(len(ids) * train_fraction), len(ids) - 1))
    return set(ids[:cutoff]), set(ids[cutoff:])
