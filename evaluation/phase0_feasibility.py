"""Phase 0 feasibility: personal baseline + optional ECGFounder diagnosis gate."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets.export_archive import ExportArchive
from models.ecgfounder_wrapper import (
    ECGFounderUnavailable,
    find_checkpoint_dir,
    load_ecgfounder,
    load_label_names,
    predict_probabilities,
    select_weight_path,
)
from models.personal_autoencoder import PersonalAutoencoder
from preprocessing.windows import iter_clean_rest_windows, split_recordings_by_time


@dataclass
class Phase0Report:
    generated_at: str
    export_dir: str
    recordings_total: int
    recordings_train: int
    recordings_eval: int
    clean_rest_windows_train: int
    clean_rest_windows_eval: int
    personal_autoencoder: dict
    ecgfounder: dict
    go_no_go: dict


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_windows(windows, max_count: int, seed: int):
    if max_count <= 0 or len(windows) <= max_count:
        return windows
    rng = random.Random(seed)
    return rng.sample(windows, max_count)


def train_autoencoder(
    train_x: np.ndarray,
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[PersonalAutoencoder, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalAutoencoder(
        input_length=train_x.shape[1],
        latent_dim=latent_dim,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x[:, np.newaxis, :].astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.append(epoch_loss / max(1, batches))
    return model, history


@torch.no_grad()
def reconstruction_errors(model: PersonalAutoencoder, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    tensor = torch.from_numpy(x[:, np.newaxis, :].astype(np.float32)).to(device)
    recon, _ = model(tensor)
    mse = ((recon - tensor) ** 2).mean(dim=(1, 2))
    return mse.cpu().numpy()


def evaluate_personal_baseline(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    config: dict,
) -> dict:
    model, history = train_autoencoder(
        train_x,
        latent_dim=int(config["ae_latent_dim"]),
        epochs=int(config["ae_epochs"]),
        batch_size=int(config["ae_batch_size"]),
        learning_rate=float(config["ae_learning_rate"]),
    )
    train_errors = reconstruction_errors(model, train_x)
    eval_errors = reconstruction_errors(model, eval_x)
    threshold = float(np.percentile(train_errors, config["ae_alert_percentile"]))
    false_alert_rate = float(np.mean(eval_errors > threshold)) if eval_errors.size else 0.0
    return {
        "train_windows": int(train_x.shape[0]),
        "eval_windows": int(eval_x.shape[0]),
        "train_error_mean": float(np.mean(train_errors)),
        "train_error_p95": float(np.percentile(train_errors, 95)),
        "eval_error_mean": float(np.mean(eval_errors)),
        "eval_error_p95": float(np.percentile(eval_errors, 95)),
        "alert_threshold": threshold,
        "alert_percentile": float(config["ae_alert_percentile"]),
        "eval_false_alert_rate": false_alert_rate,
        "final_train_loss": history[-1] if history else None,
    }


def evaluate_ecgfounder(
    eval_x: np.ndarray,
    config: dict,
) -> dict:
    checkpoint_dir = Path(config["ecgfounder_checkpoint_dir"])
    if find_checkpoint_dir(checkpoint_dir) is None:
        return {
            "status": "skipped",
            "reason": f"No checkpoint in {checkpoint_dir}",
        }

    try:
        model = load_ecgfounder(checkpoint_dir)
    except ECGFounderUnavailable as error:
        return {"status": "skipped", "reason": str(error)}

    label_names = load_label_names()
    probs = predict_probabilities(model, eval_x, batch_size=32)
    threshold = float(config["ecgfounder_threshold"])
    normal_keywords = tuple(config.get("normal_label_keywords") or ["NORMAL", "SINUS"])

    if label_names:
        normal_indices = [
            idx
            for idx, name in enumerate(label_names)
            if any(keyword in name.upper() for keyword in normal_keywords)
        ]
        pathology_indices = [idx for idx in range(len(label_names)) if idx not in normal_indices]
        normal_strength = probs[:, normal_indices].sum(axis=1) if normal_indices else np.zeros(len(probs))
        pathology_strength = (
            probs[:, pathology_indices].max(axis=1) if pathology_indices else probs.max(axis=1)
        )
        top_idx = probs.argmax(axis=1)
        top_labels = [label_names[idx] for idx in top_idx]
        suspicious = pathology_strength >= threshold
        top_label_counts = {
            label: int(count)
            for label, count in sorted(
                ((label, top_labels.count(label)) for label in set(top_labels)),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        }
    else:
        normal_strength = np.zeros(len(probs))
        pathology_strength = probs.max(axis=1)
        top_idx = probs.argmax(axis=1)
        suspicious = pathology_strength >= threshold
        top_label_counts = {}

    return {
        "status": "ran",
        "eval_windows": int(eval_x.shape[0]),
        "threshold": threshold,
        "weight_file": str(select_weight_path(checkpoint_dir)),
        "normal_label_count": len(normal_indices) if label_names else 0,
        "normal_strength_mean": float(np.mean(normal_strength)),
        "pathology_strength_mean": float(np.mean(pathology_strength)),
        "pathology_strength_p95": float(np.percentile(pathology_strength, 95)),
        "suspicious_rate": float(np.mean(suspicious)),
        "top_predicted_labels": top_label_counts,
        "note": (
            "Suspicious = max pathology-class probability >= threshold on clean rest windows."
        ),
        "normal_label_keywords": list(normal_keywords),
    }


def make_go_no_go(personal: dict, ecgfounder: dict) -> dict:
    personal_ok = personal["eval_false_alert_rate"] < 0.05
    ecgfounder_ok = True
    ecgfounder_note = "ECGFounder not run — download checkpoint to complete diagnosis gate."

    if ecgfounder.get("status") == "ran":
        ecgfounder_ok = ecgfounder["suspicious_rate"] < 0.10
        ecgfounder_note = (
            "Diagnosis gate interim proxy suspicious_rate should be < 10% on clean rest."
        )
    elif ecgfounder.get("status") == "skipped":
        ecgfounder_ok = None

    overall = personal_ok and (ecgfounder_ok is not False)
    return {
        "personal_baseline_pass": personal_ok,
        "ecgfounder_pass": ecgfounder_ok,
        "overall_phase0_pass": overall,
        "notes": {
            "personal": "Target eval false alert rate < 5% at 99th percentile threshold.",
            "ecgfounder": ecgfounder_note,
        },
    }


def run_phase0(config_path: Path, skip_ecgfounder: bool = False) -> Phase0Report:
    config = load_config(config_path)
    set_seed(int(config["random_seed"]))

    export_dir = Path(config["export_dir"])
    archive = ExportArchive(export_dir)
    recordings = archive.list_recordings()
    train_ids, eval_ids = split_recordings_by_time(
        recordings,
        float(config["train_recording_fraction"]),
    )

    print("Extracting clean rest windows...")
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

    if not train_windows or not eval_windows:
        raise RuntimeError(
            "No QC-passing windows found for train/eval split. "
            "Relax QC tags or verify export contents."
        )

    train_x = np.stack([w.samples_500hz for w in train_windows], axis=0)
    eval_x = np.stack([w.samples_500hz for w in eval_windows], axis=0)

    print(f"Train windows: {train_x.shape[0]}, Eval windows: {eval_x.shape[0]}")
    print("Training personal autoencoder...")
    personal = evaluate_personal_baseline(train_x, eval_x, config)

    if skip_ecgfounder or config.get("skip_ecgfounder"):
        ecgfounder = {"status": "skipped", "reason": "config or --skip-ecgfounder"}
    else:
        print("Evaluating ECGFounder diagnosis gate (optional)...")
        ecgfounder = evaluate_ecgfounder(eval_x, config)

    report = Phase0Report(
        generated_at=datetime.now(timezone.utc).isoformat(),
        export_dir=str(export_dir),
        recordings_total=len(recordings),
        recordings_train=len(train_ids),
        recordings_eval=len(eval_ids),
        clean_rest_windows_train=len(train_windows),
        clean_rest_windows_eval=len(eval_windows),
        personal_autoencoder=personal,
        ecgfounder=ecgfounder,
        go_no_go=make_go_no_go(personal, ecgfounder),
    )
    return report


def write_report(report: Phase0Report, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase0_report.json"
    md_path = output_dir / "phase0_report.md"

    payload = asdict(report)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    personal = report.personal_autoencoder
    ecg = report.ecgfounder
    gng = report.go_no_go
    md = f"""# Phase 0 Feasibility Report

Generated: {report.generated_at}

## Dataset
- Export: `{report.export_dir}`
- Recordings: {report.recordings_total} ({report.recordings_train} train / {report.recordings_eval} eval)
- Clean rest windows: {report.clean_rest_windows_train} train / {report.clean_rest_windows_eval} eval

## Personal Baseline (Option A)
- Eval false alert rate @ P{personal['alert_percentile']:.0f}: **{personal['eval_false_alert_rate']:.2%}
- Alert threshold (train reconstruction MSE): {personal['alert_threshold']:.6f}
- Eval error mean / p95: {personal['eval_error_mean']:.6f} / {personal['eval_error_p95']:.6f}

## Diagnosis Gate (Option B — ECGFounder)
- Status: {ecg.get('status')}
"""
    if ecg.get("status") == "ran":
        md += f"- Suspicious rate (pathology prob >= {ecg['threshold']}): **{ecg['suspicious_rate']:.2%}**\n"
        md += f"- Pathology strength mean / p95: {ecg['pathology_strength_mean']:.4f} / {ecg['pathology_strength_p95']:.4f}\n"
        md += f"- Normal strength mean: {ecg['normal_strength_mean']:.4f}\n"
        if ecg.get("top_predicted_labels"):
            md += "- Top predicted labels:\n"
            for label, count in ecg["top_predicted_labels"].items():
                md += f"  - {label}: {count}\n"
    else:
        md += f"- Reason: {ecg.get('reason', 'n/a')}\n"

    md += f"""
## Go / No-Go
- Personal baseline pass: **{gng['personal_baseline_pass']}**
- ECGFounder pass: **{gng['ecgfounder_pass']}**
- Overall Phase 0 pass: **{gng['overall_phase0_pass']}**

### Notes
- Personal: {gng['notes']['personal']}
- ECGFounder: {gng['notes']['ecgfounder']}
"""
    md_path.write_text(md, encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HAWLEY ML Phase 0 feasibility")
    parser.add_argument(
        "--config",
        default="configs/phase0_micah.yaml",
        help="Path to Phase 0 config YAML",
    )
    parser.add_argument(
        "--skip-ecgfounder",
        action="store_true",
        help="Skip optional ECGFounder diagnosis gate",
    )
    args = parser.parse_args()
    config = load_config(Path(args.config))
    report = run_phase0(Path(args.config), skip_ecgfounder=args.skip_ecgfounder)
    output_dir = Path(config["output_dir"])
    md_path = write_report(report, output_dir)
    print(f"\nPhase 0 report written to {md_path}")


if __name__ == "__main__":
    main()
