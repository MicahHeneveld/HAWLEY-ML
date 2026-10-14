"""Phase 1 personal baseline using beat-aligned windows (recommended path)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from adaptation.artifacts import save_personal_baseline_artifact
from adaptation.persistence import causal_moving_average, sustained_alert_mask
from adaptation.training import train_with_val_threshold
from datasets.export_archive import ExportArchive
from evaluation.phase0_feasibility import (
    evaluate_personal_baseline,
    load_config,
    reconstruction_errors,
    set_seed,
    write_report,
)
from preprocessing.beat_windows import iter_beat_windows
from preprocessing.windows import split_recordings_by_time


def persisted_false_alert_rate(
    eval_errors: np.ndarray,
    threshold: float,
    *,
    smoothing_window: int,
    consecutive_beats: int,
) -> float:
    if eval_errors.size == 0:
        return 0.0
    smoothed = causal_moving_average(eval_errors, smoothing_window)
    sustained = sustained_alert_mask(smoothed > threshold, consecutive_beats)
    return float(np.mean(sustained))


@dataclass
class Phase1BeatReport:
    generated_at: str
    config_path: str
    window_mode: str
    split_method: str
    total_beats_train: int
    total_beats_eval: int
    personal_autoencoder: dict
    go_no_go: dict


def load_windows(config: dict, archive: ExportArchive):
    tags = config["require_tags"]
    split_method = config.get("split_method", "window_random")
    kwargs = dict(
        window_seconds=float(config["window_seconds"]),
        target_fs=int(config["target_sample_rate_hz"]),
        layer_a=str(tags["layer_a"]),
        layer_c_include=tags.get("layer_c_include"),
        layer_c_exclude=tuple(tags["layer_c_exclude"]),
        layer_a_exclude=tuple(tags.get("layer_a_exclude") or []),
        max_train_windows=int(config.get("max_train_windows", 5000)),
        max_eval_windows=int(config.get("max_eval_windows", 2000)),
        seed=int(config["random_seed"]),
    )

    if split_method == "window_random":
        return iter_beat_windows(
            archive,
            train_recording_ids=None,
            train_window_fraction=float(config["train_window_fraction"]),
            **kwargs,
        )
    train_ids, _ = split_recordings_by_time(
        archive.list_recordings(),
        float(config.get("train_recording_fraction", 0.7)),
    )
    return iter_beat_windows(
        archive,
        train_recording_ids=train_ids,
        train_window_fraction=None,
        **kwargs,
    )


def run_phase1_beat(config_path: Path) -> Phase1BeatReport:
    config = load_config(config_path)
    set_seed(int(config["random_seed"]))
    archive = ExportArchive(Path(config["export_dir"]))

    print("Extracting beat-aligned rest windows...")
    windows = load_windows(config, archive)
    train_windows = [w for w in windows if w.split == "train"]
    eval_windows = [w for w in windows if w.split == "eval"]
    if not train_windows or not eval_windows:
        raise RuntimeError(
            f"Insufficient beat windows: train={len(train_windows)} eval={len(eval_windows)}"
        )

    train_x = np.stack([w.samples_500hz for w in train_windows], axis=0)
    eval_x = np.stack([w.samples_500hz for w in eval_windows], axis=0)
    print(f"Beat windows: {train_x.shape[0]} train / {eval_x.shape[0]} eval "
          f"(shape {train_x.shape[1]} samples)")

    threshold_method = config.get("threshold_method", "train_p99")
    model = None
    if threshold_method == "val_p99":
        model, personal = train_with_val_threshold(train_x, eval_x, config)
    else:
        personal = evaluate_personal_baseline(train_x, eval_x, config)

    smoothing_window = int(config.get("alert_smoothing_window", 5))
    consecutive_beats = int(config.get("alert_consecutive_beats", 3))
    if model is not None:
        eval_errors = reconstruction_errors(model, eval_x)
    else:
        eval_errors = np.array([])
    if eval_errors.size:
        personal["eval_sustained_false_alert_rate"] = persisted_false_alert_rate(
            eval_errors,
            float(personal["alert_threshold"]),
            smoothing_window=smoothing_window,
            consecutive_beats=consecutive_beats,
        )
    else:
        personal["eval_sustained_false_alert_rate"] = None

    artifact_dir = config.get("artifact_dir")
    if artifact_dir and model is not None:
        bundle_path = save_personal_baseline_artifact(
            Path(artifact_dir),
            model,
            config,
            personal,
            subject_id=str(config["subject_id"]),
            config_path=config_path,
        )
        print(f"Saved artifact bundle: {bundle_path}")
    elif artifact_dir and model is None:
        print("Artifact save skipped — only supported with val_p99 training path.")

    false_alert = personal["eval_false_alert_rate"]
    go_no_go = {
        "personal_baseline_pass": false_alert < 0.05,
        "overall_phase1_pass": false_alert < 0.05,
        "notes": {
            "approach": "Beat-aligned 1s windows + window-level split (single-subject personalization).",
            "ecgfounder": "Narrow head trained (weak proxy labels); zero-shot not used as gate.",
            "cloud": "None — runs fully offline (AWS account closed).",
            "persistence": (
                f"Sustained alert: {smoothing_window}-beat MA + {consecutive_beats} consecutive beats."
            ),
        },
    }

    return Phase1BeatReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        config_path=str(config_path),
        window_mode=str(config.get("window_mode", "beat")),
        split_method=str(config.get("split_method", "window_random")),
        total_beats_train=len(train_windows),
        total_beats_eval=len(eval_windows),
        personal_autoencoder=personal,
        go_no_go=go_no_go,
    )


def write_phase1_report(report: Phase1BeatReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase1_beat_report.json"
    md_path = output_dir / "phase1_beat_report.md"
    payload = asdict(report)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    p = report.personal_autoencoder
    g = report.go_no_go
    md = f"""# Phase 1 Beat-Aligned Personal Baseline

Generated: {report.generated_at}

## Approach
- **Window mode:** {report.window_mode} (1 s @ 500 Hz, R-peak centered)
- **Split:** {report.split_method}
- **Beats:** {report.total_beats_train} train / {report.total_beats_eval} eval

## Personal Baseline
- Eval false alert @ P{p['alert_percentile']:.0f}: **{p['eval_false_alert_rate']:.2%}**
- Alert threshold: {p['alert_threshold']:.6f}
- Eval error mean / p95: {p['eval_error_mean']:.6f} / {p['eval_error_p95']:.6f}
"""
    if p.get("eval_sustained_false_alert_rate") is not None:
        md += f"- Eval sustained false alert (smoothed + consecutive): **{p['eval_sustained_false_alert_rate']:.2%}**\n"
    md += f"""
## Go / No-Go
- Personal baseline pass: **{g['personal_baseline_pass']}**
- Overall Phase 1 pass: **{g['overall_phase1_pass']}**

## Notes
- {g['notes']['approach']}
- {g['notes']['ecgfounder']}
- {g['notes']['aws']}
- {g['notes']['persistence']}
"""
    md_path.write_text(md, encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 beat-aligned personal baseline")
    parser.add_argument("--config", default="configs/phase1_beat_micah.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    report = run_phase1_beat(config_path)
    md_path = write_phase1_report(report, Path(config["output_dir"]))
    print(f"\nPhase 1 report: {md_path}")
    print(f"False alert rate: {report.personal_autoencoder['eval_false_alert_rate']:.2%}")
    print(f"Pass: {report.go_no_go['overall_phase1_pass']}")


if __name__ == "__main__":
    main()
