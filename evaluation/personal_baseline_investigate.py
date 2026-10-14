"""Compare personal baseline strategies and pick the best path forward."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from adaptation.training import evaluate_with_val_threshold
from datasets.export_archive import ExportArchive
from evaluation.phase0_feasibility import (
    evaluate_personal_baseline,
    load_config,
    set_seed,
)
from preprocessing.windows import (
    EcgWindow,
    iter_clean_rest_windows,
    split_recordings_by_time,
    split_recordings_random,
)


@dataclass
class StrategyResult:
    name: str
    split_method: str
    threshold_method: str
    train_windows: int
    eval_windows: int
    eval_false_alert_rate: float
    alert_threshold: float
    eval_error_mean: float
    eval_error_p95: float
    notes: str


def split_recordings_random(
    recordings,
    train_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    from preprocessing.windows import split_recordings_random as _split

    return _split(recordings, train_fraction, seed)


def extract_windows(
    archive: ExportArchive,
    config: dict,
    train_ids: set[str],
) -> tuple[list[EcgWindow], list[EcgWindow]]:
    tags = config["require_tags"]
    windows = iter_clean_rest_windows(
        archive,
        train_recording_ids=train_ids,
        window_seconds=float(config["window_seconds"]),
        stride_seconds=float(config["window_stride_seconds"]),
        target_fs=int(config["target_sample_rate_hz"]),
        layer_a=str(tags["layer_a"]),
        layer_c_include=tags.get("layer_c_include"),
        layer_c_exclude=tuple(tags["layer_c_exclude"]),
        layer_a_exclude=tuple(tags.get("layer_a_exclude") or []),
        min_layer_a_fraction=float(tags.get("min_layer_a_fraction", 0.5)),
        max_train_windows=int(config["max_train_windows"]),
        max_eval_windows=int(config["max_eval_windows"]),
    )
    train_windows = [w for w in windows if w.split == "train"]
    eval_windows = [w for w in windows if w.split == "eval"]
    return train_windows, eval_windows


def run_investigation(config_path: Path) -> dict:
    config = load_config(config_path)
    set_seed(int(config["random_seed"]))
    export_dir = Path(config["export_dir"])
    archive = ExportArchive(export_dir)
    recordings = archive.list_recordings()
    fraction = float(config["train_recording_fraction"])
    seed = int(config["random_seed"])

    strategies: list[tuple[str, set[str], str]] = [
        (
            "temporal_split_train_threshold",
            split_recordings_by_time(recordings, fraction)[0],
            "train_p99",
        ),
        (
            "random_split_train_threshold",
            split_recordings_random(recordings, fraction, seed)[0],
            "train_p99",
        ),
        (
            "random_split_val_threshold",
            split_recordings_random(recordings, fraction, seed)[0],
            "val_p99",
        ),
    ]

    results: list[StrategyResult] = []
    for name, train_ids, threshold_method in strategies:
        train_windows, eval_windows = extract_windows(archive, config, train_ids)
        if not train_windows or not eval_windows:
            results.append(
                StrategyResult(
                    name=name,
                    split_method=name.split("_")[0],
                    threshold_method=threshold_method,
                    train_windows=0,
                    eval_windows=0,
                    eval_false_alert_rate=1.0,
                    alert_threshold=0.0,
                    eval_error_mean=0.0,
                    eval_error_p95=0.0,
                    notes="No QC windows found",
                )
            )
            continue

        train_x = np.stack([w.samples_500hz for w in train_windows], axis=0)
        eval_x = np.stack([w.samples_500hz for w in eval_windows], axis=0)

        if threshold_method == "val_p99":
            metrics = evaluate_with_val_threshold(train_x, eval_x, config)
        else:
            metrics = evaluate_personal_baseline(train_x, eval_x, config)

        note = ""
        if name.startswith("temporal"):
            note = "Chronological split — sensitive to drift over weeks of wear."
        elif threshold_method == "val_p99":
            note = "Random split + validation threshold — recommended in literature."
        else:
            note = "Random split; threshold from all train windows."

        results.append(
            StrategyResult(
                name=name,
                split_method="temporal" if "temporal" in name else "random",
                threshold_method=threshold_method,
                train_windows=int(metrics["train_windows"]),
                eval_windows=int(metrics["eval_windows"]),
                eval_false_alert_rate=float(metrics["eval_false_alert_rate"]),
                alert_threshold=float(metrics["alert_threshold"]),
                eval_error_mean=float(metrics["eval_error_mean"]),
                eval_error_p95=float(metrics["eval_error_p95"]),
                notes=note,
            )
        )

    best = min(results, key=lambda item: item.eval_false_alert_rate)
    recommendation = build_recommendation(results, best)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "strategies": [asdict(r) for r in results],
        "best_strategy": asdict(best),
        "recommendation": recommendation,
    }


def build_recommendation(results: list[StrategyResult], best: StrategyResult) -> dict:
    temporal = next(r for r in results if r.name.startswith("temporal"))
    random_val = next(r for r in results if r.name == "random_split_val_threshold")

    return {
        "primary_path": "personal_baseline",
        "defer": ["ecgfounder_zero_shot"],
        "winning_strategy": best.name,
        "winning_false_alert_rate": best.eval_false_alert_rate,
        "rationale": [
            "ECGFounder zero-shot failed (99.5% suspicious) — fine-tune only after personal loop works.",
            "Literature: personalized AE + subject calibration beats population models on wearables.",
            f"Temporal split false alert {temporal.eval_false_alert_rate:.1%} vs "
            f"random+val threshold {random_val.eval_false_alert_rate:.1%}.",
            "AWS S3/SageMaker deferred until personal baseline passes <5% locally.",
        ],
        "next_implementation": (
            "Adopt random recording split + validation-set threshold calibration "
            "as default in Phase 1 personal baseline pipeline."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate personal baseline strategies")
    parser.add_argument(
        "--config",
        default="configs/phase1_personal_micah.yaml",
    )
    parser.add_argument(
        "--output",
        default="reports/phase1_personal/strategy_investigation.json",
    )
    args = parser.parse_args()

    report = run_investigation(Path(args.config))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    best = report["best_strategy"]
    print(f"\nBest strategy: {best['name']}")
    print(f"False alert rate: {best['eval_false_alert_rate']:.2%}")
    print(f"Report: {output}")
    for line in report["recommendation"]["rationale"]:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
