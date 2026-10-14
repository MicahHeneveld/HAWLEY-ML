"""Train personal baseline models and calibration metrics."""

from __future__ import annotations

import numpy as np

from evaluation.phase0_feasibility import reconstruction_errors, train_autoencoder
from models.personal_autoencoder import PersonalAutoencoder


def train_with_val_threshold(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    config: dict,
    *,
    val_fraction: float | None = None,
) -> tuple[PersonalAutoencoder, dict]:
    """Train on a train slice; threshold from held-out validation slice of train_x."""
    val_fraction = float(
        val_fraction if val_fraction is not None else config.get("threshold_val_fraction", 0.2)
    )
    rng = np.random.default_rng(int(config["random_seed"]))
    indices = np.arange(train_x.shape[0])
    rng.shuffle(indices)
    val_size = max(1, int(train_x.shape[0] * val_fraction))
    val_idx = indices[:val_size]
    fit_idx = indices[val_size:]
    fit_x = train_x[fit_idx]
    val_x = train_x[val_idx]

    model, history = train_autoencoder(
        fit_x,
        latent_dim=int(config["ae_latent_dim"]),
        epochs=int(config["ae_epochs"]),
        batch_size=int(config["ae_batch_size"]),
        learning_rate=float(config["ae_learning_rate"]),
    )
    val_errors = reconstruction_errors(model, val_x)
    eval_errors = reconstruction_errors(model, eval_x)
    threshold = float(np.percentile(val_errors, config["ae_alert_percentile"]))
    false_alert_rate = float(np.mean(eval_errors > threshold)) if eval_errors.size else 0.0

    metrics = {
        "train_windows": int(train_x.shape[0]),
        "eval_windows": int(eval_x.shape[0]),
        "fit_windows": int(fit_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "train_error_mean": float(np.mean(reconstruction_errors(model, fit_x))),
        "train_error_p95": float(np.percentile(reconstruction_errors(model, fit_x), 95)),
        "val_error_mean": float(np.mean(val_errors)),
        "eval_error_mean": float(np.mean(eval_errors)),
        "eval_error_p95": float(np.percentile(eval_errors, 95)),
        "alert_threshold": threshold,
        "alert_percentile": float(config["ae_alert_percentile"]),
        "eval_false_alert_rate": false_alert_rate,
        "final_train_loss": history[-1] if history else None,
        "threshold_method": "val_p99",
    }
    return model, metrics


def evaluate_with_val_threshold(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    config: dict,
    *,
    val_fraction: float = 0.2,
) -> dict:
    _, metrics = train_with_val_threshold(
        train_x, eval_x, config, val_fraction=val_fraction
    )
    return metrics
