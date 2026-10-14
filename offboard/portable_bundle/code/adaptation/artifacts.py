"""Save and load personal baseline model bundles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from models.personal_autoencoder import PersonalAutoencoder


def save_personal_baseline_artifact(
    artifact_dir: Path,
    model: PersonalAutoencoder,
    config: dict,
    metrics: dict,
    *,
    subject_id: str,
    config_path: Path | None = None,
) -> Path:
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    input_length = int(model.output_length)
    latent_dim = int(config["ae_latent_dim"])
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_length": input_length,
        "latent_dim": latent_dim,
    }
    torch.save(checkpoint, artifact_dir / "personal_ae.pt")

    snapshot_path = artifact_dir / "config.snapshot.yaml"
    snapshot_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    bundle = {
        "artifact_version": "1.0",
        "subject_id": subject_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path else None,
        "metrics": metrics,
        "model": {
            "input_length": input_length,
            "latent_dim": latent_dim,
            "checkpoint_file": "personal_ae.pt",
        },
        "persistence": {
            "smoothing_window": int(config.get("alert_smoothing_window", 5)),
            "consecutive_beats": int(config.get("alert_consecutive_beats", 3)),
        },
        "activity_segment": config.get("activity_segment", "default"),
    }
    bundle_path = artifact_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def load_personal_baseline_artifact(
    artifact_dir: Path,
) -> tuple[PersonalAutoencoder, dict, dict]:
    artifact_dir = artifact_dir.resolve()
    bundle_path = artifact_dir / "bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing bundle.json in {artifact_dir}")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    snapshot_path = artifact_dir / "config.snapshot.yaml"
    config = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    ckpt_path = artifact_dir / bundle["model"]["checkpoint_file"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PersonalAutoencoder(
        input_length=int(checkpoint["input_length"]),
        latent_dim=int(checkpoint["latent_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config, bundle
