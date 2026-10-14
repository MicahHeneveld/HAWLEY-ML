"""Train rest + walk personal baseline segments."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.personal_baseline_phase1 import run_phase1_beat, write_phase1_report
from evaluation.phase0_feasibility import load_config


DEFAULT_CONFIGS = (
    "configs/phase1_beat_micah_rest.yaml",
    "configs/phase1_beat_micah_resting_hr.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train activity-segment personal baselines")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Phase 1 config (repeatable; default: rest + walk)",
    )
    args = parser.parse_args()
    configs = args.config or list(DEFAULT_CONFIGS)

    for config_path in configs:
        path = Path(config_path)
        print(f"\n=== Training segment: {path} ===")
        report = run_phase1_beat(path)
        cfg = load_config(path)
        write_phase1_report(report, Path(cfg["output_dir"]))
        segment = cfg.get("activity_segment", path.stem)
        print(
            f"Segment {segment}: false alert {report.personal_autoencoder['eval_false_alert_rate']:.2%} "
            f"pass={report.go_no_go['overall_phase1_pass']}"
        )


if __name__ == "__main__":
    main()
