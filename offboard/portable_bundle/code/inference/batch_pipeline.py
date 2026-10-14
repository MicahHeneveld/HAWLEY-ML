"""Batch scoring with activity-segment model routing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from adaptation.artifacts import load_personal_baseline_artifact
from adaptation.persistence import causal_moving_average, sustained_alert_mask
from datasets.export_archive import ExportArchive
from datasets.sbec import read_sbec_file
from evaluation.phase0_feasibility import reconstruction_errors
from fusion.pipeline import BeatFusionResult, fuse_personal_beat
from models.diagnosis_narrow import (
    DiagnosisNarrowUnavailable,
    load_diagnosis_narrow,
    predict_context_suspicious,
)
from preprocessing.beat_windows import BeatWindow, iter_all_signal_clean_beats
from preprocessing.diagnosis_windows import beat_diagnosis_window
from preprocessing.tag_schema import resolve_activity_segment


@dataclass
class ScoredBeat:
    recording_id: str
    chunk_index: int
    beat_index: int
    started_at: str
    activity_segment: str | None
    reconstruction_error: float
    smoothed_error: float
    instant_above: bool
    sustained_above: bool
    alert_tier: str
    qc_pass: bool
    qc_veto_reason: str | None
    model_slot: str
    diagnosis_suspicious: bool | None = None
    diagnosis_probability: float | None = None


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_segment_models(segment_artifacts: dict[str, str]) -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    for name, path in segment_artifacts.items():
        model, config, bundle = load_personal_baseline_artifact(Path(path))
        persistence = bundle.get("persistence") or {}
        loaded[name] = {
            "model": model,
            "config": config,
            "bundle": bundle,
            "threshold": float(bundle["metrics"]["alert_threshold"]),
            "smoothing_window": int(
                persistence.get("smoothing_window", config.get("alert_smoothing_window", 5))
            ),
            "consecutive_beats": int(
                persistence.get("consecutive_beats", config.get("alert_consecutive_beats", 3))
            ),
        }
    return loaded


def _diagnosis_probs_for_segment_windows(
    seg_windows: list[BeatWindow],
    *,
    archive: ExportArchive,
    recording_id: str,
    diagnosis_model,
    suspicious_class: int,
) -> np.ndarray:
    by_chunk: dict[int, list[tuple[int, BeatWindow]]] = defaultdict(list)
    for idx, window in enumerate(seg_windows):
        by_chunk[window.chunk_index].append((idx, window))

    recording = next(r for r in archive.list_recordings() if r.recording_id == recording_id)
    chunk_by_index = {c.chunk_index: c for c in recording.chunks}
    probs = np.zeros(len(seg_windows), dtype=np.float32)

    for chunk_index, items in by_chunk.items():
        chunk = chunk_by_index.get(chunk_index)
        if chunk is None:
            continue
        sample_rate, samples = read_sbec_file(chunk.bin_path)
        dx_rows = [
            beat_diagnosis_window(samples, sample_rate, w.sample_offset)
            for _, w in items
        ]
        batch_probs = predict_context_suspicious(
            diagnosis_model,
            np.stack(dx_rows, axis=0),
            suspicious_class=suspicious_class,
        )
        for (local_idx, _), prob in zip(items, batch_probs):
            probs[local_idx] = float(prob)
    return probs


def score_recording_beats(
    windows: list[BeatWindow],
    segment_models: dict[str, dict],
    *,
    archive: ExportArchive | None = None,
    diagnosis_pack: dict | None = None,
) -> list[ScoredBeat]:
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: (w.chunk_index, w.beat_index))

    by_segment: dict[str, list[BeatWindow]] = defaultdict(list)
    for window in windows:
        seg = resolve_activity_segment(window.tags)
        if seg and seg in segment_models:
            by_segment[seg].append(window)

    scored: list[ScoredBeat] = []
    for seg, seg_windows in by_segment.items():
        pack = segment_models[seg]
        model = pack["model"]
        threshold = pack["threshold"]
        smoothing_window = pack["smoothing_window"]
        consecutive_beats = pack["consecutive_beats"]

        x = np.stack([w.samples_500hz for w in seg_windows], axis=0)
        errors = reconstruction_errors(model, x)
        smoothed = causal_moving_average(errors, smoothing_window)
        instant = errors > threshold
        sustained = sustained_alert_mask(smoothed > threshold, consecutive_beats)

        dx_probs: np.ndarray | None = None
        if diagnosis_pack and archive is not None and seg_windows:
            dx_probs = _diagnosis_probs_for_segment_windows(
                seg_windows,
                archive=archive,
                recording_id=seg_windows[0].recording_id,
                diagnosis_model=diagnosis_pack["model"],
                suspicious_class=int(diagnosis_pack["suspicious_class"]),
            )

        for j, (window, err, sm, inst, sus) in enumerate(
            zip(seg_windows, errors, smoothed, instant, sustained)
        ):
            dx_prob = float(dx_probs[j]) if dx_probs is not None else None
            dx_threshold = (
                float(diagnosis_pack["threshold"]) if diagnosis_pack else 0.5
            )
            dx_suspicious = (
                dx_prob >= dx_threshold if dx_prob is not None else None
            )
            fusion: BeatFusionResult = fuse_personal_beat(
                tags=window.tags,
                reconstruction_error=float(err),
                smoothed_error=float(sm),
                alert_threshold=threshold,
                sustained_personal_alert=bool(sus),
                diagnosis_suspicious=dx_suspicious,
            )
            scored.append(
                ScoredBeat(
                    recording_id=window.recording_id,
                    chunk_index=window.chunk_index,
                    beat_index=window.beat_index,
                    started_at=window.started_at,
                    activity_segment=fusion.activity_segment,
                    reconstruction_error=float(err),
                    smoothed_error=float(sm),
                    instant_above=fusion.instant_above,
                    sustained_above=fusion.sustained_above,
                    alert_tier=fusion.alert_tier,
                    qc_pass=fusion.qc_pass,
                    qc_veto_reason=fusion.qc_veto_reason,
                    model_slot=fusion.model_slot,
                    diagnosis_suspicious=fusion.diagnosis_suspicious,
                    diagnosis_probability=dx_prob,
                )
            )

    scored.sort(key=lambda row: (row.chunk_index, row.beat_index))
    return scored


def run_batch(
    *,
    export_dir: Path,
    output_dir: Path,
    segment_artifacts: dict[str, str],
    layer_c_exclude: tuple[str, ...],
    max_beats: int | None = None,
    seed: int = 42,
    diagnosis_artifact_dir: Path | None = None,
) -> dict:
    segment_models = load_segment_models(segment_artifacts)
    ref_config = next(iter(segment_models.values()))["config"]

    diagnosis_pack = None
    if diagnosis_artifact_dir is not None:
        try:
            model, bundle = load_diagnosis_narrow(diagnosis_artifact_dir)
            diagnosis_pack = {
                "model": model,
                "bundle": bundle,
                "threshold": float(bundle.get("threshold", 0.5)),
                "suspicious_class": int(bundle.get("suspicious_class_index", 1)),
            }
            print(f"Loaded diagnosis narrow head from {diagnosis_artifact_dir}")
        except DiagnosisNarrowUnavailable as error:
            print(f"Diagnosis head unavailable: {error}")

    archive = ExportArchive(export_dir)
    print("Scanning signal-clean beats (all activities)...")
    windows = iter_all_signal_clean_beats(
        archive,
        window_seconds=float(ref_config["window_seconds"]),
        target_fs=int(ref_config["target_sample_rate_hz"]),
        layer_c_include=None,
        layer_c_exclude=layer_c_exclude,
    )
    routable = [
        w for w in windows if resolve_activity_segment(w.tags) in segment_models
    ]
    print(f"Routable beats: {len(routable)} / {len(windows)} total signal-clean")
    windows = routable

    if max_beats and len(windows) > max_beats:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(windows), max_beats, replace=False)
        windows = [windows[i] for i in sorted(pick.tolist())]

    by_recording: dict[str, list[BeatWindow]] = defaultdict(list)
    for window in windows:
        by_recording[window.recording_id].append(window)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tiers: dict[str, int] = defaultdict(int)
    segment_counts: dict[str, int] = defaultdict(int)
    total_beats = 0
    sustained_alerts = 0
    fusion_tiers_with_dx = Counter()

    for recording_id in sorted(by_recording.keys()):
        scored = score_recording_beats(
            by_recording[recording_id],
            segment_models,
            archive=archive,
            diagnosis_pack=diagnosis_pack,
        )
        total_beats += len(scored)
        sustained_alerts += sum(1 for row in scored if row.sustained_above)
        for row in scored:
            summary_tiers[row.alert_tier] += 1
            segment_counts[row.activity_segment or "none"] += 1
            if row.diagnosis_suspicious is not None:
                fusion_tiers_with_dx[row.alert_tier] += 1

        jsonl_path = output_dir / f"{recording_id}_scores.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in scored:
                handle.write(json.dumps(asdict(row)) + "\n")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export_dir": str(export_dir.resolve()),
        "segment_artifacts": segment_artifacts,
        "total_beats_scored": total_beats,
        "recordings_scored": len(by_recording),
        "sustained_alert_beats": sustained_alerts,
        "tier_counts": dict(summary_tiers),
        "segment_counts": dict(segment_counts),
        "diagnosis_artifact": str(diagnosis_artifact_dir) if diagnosis_artifact_dir else None,
        "tier_counts_with_diagnosis": dict(fusion_tiers_with_dx) if diagnosis_pack else None,
    }
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_batch_legacy(
    *,
    export_dir: Path,
    artifact_dir: Path,
    output_dir: Path,
    max_beats: int | None = None,
) -> dict:
    model, config, _ = load_personal_baseline_artifact(artifact_dir)
    del model
    tags = config["require_tags"]
    return run_batch(
        export_dir=export_dir,
        output_dir=output_dir,
        segment_artifacts={"rest": str(artifact_dir)},
        layer_c_exclude=tuple(tags["layer_c_exclude"]),
        max_beats=max_beats,
        seed=int(config["random_seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score Sandbox export")
    parser.add_argument("--config", default="configs/batch_micah_full.yaml")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-dir", default=None, help="Legacy single-artifact mode")
    parser.add_argument("--max-beats", type=int, default=None)
    parser.add_argument("--skip-diagnosis", action="store_true")
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    export_dir = Path(args.export_dir or cfg["export_dir"])
    output_dir = Path(args.output_dir or cfg["output_dir"])

    if args.artifact_dir or "segment_artifacts" not in cfg:
        summary = run_batch_legacy(
            export_dir=export_dir,
            artifact_dir=Path(args.artifact_dir or cfg.get("artifact_dir", "artifacts/micah/phase1_beat")),
            output_dir=output_dir,
            max_beats=args.max_beats,
        )
    else:
        dx_dir = None
        if not args.skip_diagnosis and cfg.get("diagnosis_artifact_dir"):
            dx_dir = Path(cfg["diagnosis_artifact_dir"])
        summary = run_batch(
            export_dir=export_dir,
            output_dir=output_dir,
            segment_artifacts={k: str(v) for k, v in cfg["segment_artifacts"].items()},
            layer_c_exclude=tuple(cfg["layer_c_exclude"]),
            max_beats=args.max_beats,
            diagnosis_artifact_dir=dx_dir,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
