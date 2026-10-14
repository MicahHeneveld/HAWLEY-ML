"""Calibrate diagnosis narrow-head threshold on held-out eval NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from models.diagnosis_narrow import load_diagnosis_narrow, predict_context_suspicious


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    max_false_suspicious: float,
) -> dict:
    best = None
    for threshold in np.linspace(0.05, 0.95, 37):
        pred = probs >= threshold
        normal = labels == 0
        alert = labels == 1
        false_susp = float(np.mean(pred[normal])) if normal.any() else 1.0
        recall = float(np.mean(pred[alert])) if alert.any() else 0.0
        row = {
            "threshold": float(threshold),
            "false_suspicious_on_normal": false_susp,
            "alert_recall_proxy": recall,
        }
        if false_susp <= max_false_suspicious:
            if best is None or recall > best["alert_recall_proxy"]:
                best = row
    if best is None:
        # No threshold met constraint — pick lowest false suspicious rate
        best = min(
            [
                {
                    "threshold": float(t),
                    "false_suspicious_on_normal": float(np.mean(probs[labels == 0] >= t))
                    if np.any(labels == 0)
                    else 1.0,
                    "alert_recall_proxy": float(np.mean(probs[labels == 1] >= t))
                    if np.any(labels == 1)
                    else 0.0,
                }
                for t in np.linspace(0.05, 0.95, 37)
            ],
            key=lambda item: item["false_suspicious_on_normal"],
        )
        best["note"] = f"No threshold achieved <={max_false_suspicious:.0%} false suspicious; using minimum false rate."
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diagnosis_narrow_micah.yaml")
    parser.add_argument(
        "--max-false-suspicious",
        type=float,
        default=0.10,
        help="Target max false suspicious rate on normal-proxy eval beats",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Override artifact_dir from config (e.g. SageMaker output)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    artifact_dir = Path(args.artifact_dir or cfg["artifact_dir"])
    data_dir = Path(cfg["processed_dir"])
    eval_npz = np.load(data_dir / "eval.npz")

    model, bundle = load_diagnosis_narrow(artifact_dir)
    suspicious_class = int(bundle.get("suspicious_class_index", 1))
    probs = predict_context_suspicious(
        model, eval_npz["x"], suspicious_class=suspicious_class
    )
    labels = eval_npz["y"]

    at_default = float(bundle.get("threshold", 0.5))
    default_stats = {
        "threshold": at_default,
        "false_suspicious_on_normal": float(np.mean(probs[labels == 0] >= at_default)),
        "alert_recall_proxy": float(np.mean(probs[labels == 1] >= at_default)),
    }

    chosen = sweep_thresholds(
        probs, labels, max_false_suspicious=args.max_false_suspicious
    )

    bundle_path = artifact_dir / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["threshold"] = chosen["threshold"]
    bundle["threshold_default_0.5"] = default_stats
    bundle["threshold_calibration"] = {
        "max_false_suspicious_target": args.max_false_suspicious,
        "chosen": chosen,
        "eval_size": int(labels.shape[0]),
    }
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    report = {
        "artifact_dir": str(artifact_dir),
        "default_at_0.5": default_stats,
        "calibrated": chosen,
    }
    out = Path("reports/diagnosis_narrow/calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
