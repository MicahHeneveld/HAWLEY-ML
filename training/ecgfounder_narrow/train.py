"""Fine-tune ECGFounder 1-lead with 2-class narrow head (linear probe)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.ecgfounder_wrapper import find_checkpoint_dir, select_weight_path

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "ECGFounder"


def _add_vendor() -> None:
    if VENDOR.is_dir() and str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def train_loop(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    base_checkpoint: Path,
    artifact_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    linear_probe: bool,
) -> dict:
    _add_vendor()
    from finetune_model import ft_1lead_ECGFounder  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ft_1lead_ECGFounder(
        device, str(base_checkpoint), n_classes=2, linear_prob=linear_probe
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x[:, np.newaxis, :].astype(np.float32)),
            torch.from_numpy(train_y.astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.append(epoch_loss / max(1, batches))

    model.eval()
    with torch.no_grad():
        ex = torch.from_numpy(eval_x[:, np.newaxis, :].astype(np.float32)).to(device)
        logits = model(ex)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = logits.argmax(dim=-1).cpu().numpy()
        acc = float(np.mean(pred == eval_y))
        alert_mask = eval_y == 1
        suspicious_rate = float(np.mean(pred[alert_mask] == 1)) if alert_mask.any() else 0.0
        false_suspicious_on_normal = (
            float(np.mean(pred[eval_y == 0] == 1)) if np.any(eval_y == 0) else 0.0
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "diagnosis_narrow.pt"
    torch.save({"state_dict": model.state_dict()}, artifact_dir / ckpt_name)

    metrics = {
        "eval_accuracy": acc,
        "eval_false_suspicious_on_normal": false_suspicious_on_normal,
        "eval_alert_recall_proxy": suspicious_rate,
        "final_train_loss": history[-1] if history else None,
        "train_size": int(train_x.shape[0]),
        "eval_size": int(eval_x.shape[0]),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diagnosis_narrow_micah.yaml")
    parser.add_argument("--data-dir", default=None, help="Override processed NPZ dir")
    parser.add_argument("--output-dir", default=None, help="Override artifact dir")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir or cfg["processed_dir"])
    artifact_dir = Path(args.output_dir or cfg["artifact_dir"])
    ckpt_cfg = Path(cfg["ecgfounder_base_checkpoint"])
    if ckpt_cfg.is_dir():
        resolved = find_checkpoint_dir(ckpt_cfg)
        if not resolved:
            raise SystemExit(f"No weights in {ckpt_cfg}")
        base_ckpt = select_weight_path(resolved, single_lead=True)
    elif ckpt_cfg.is_file():
        base_ckpt = ckpt_cfg
    else:
        raise SystemExit(
            f"Base ECGFounder weights not found: {ckpt_cfg}. "
            "Download 1_lead_ECGFounder.pth from Hugging Face into checkpoint/"
        )

    train_npz = np.load(data_dir / "train.npz")
    eval_npz = np.load(data_dir / "eval.npz")
    metrics = train_loop(
        train_x=train_npz["x"],
        train_y=train_npz["y"],
        eval_x=eval_npz["x"],
        eval_y=eval_npz["y"],
        base_checkpoint=base_ckpt,
        artifact_dir=artifact_dir,
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        linear_probe=bool(cfg.get("linear_probe", True)),
    )

    bundle = {
        "artifact_version": "1.0",
        "model_type": "ecgfounder_narrow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_classes": 2,
        "class_names": ["normal_proxy", "context_alert_proxy"],
        "suspicious_class_index": 1,
        "checkpoint_file": "diagnosis_narrow.pt",
        "base_checkpoint": str(base_ckpt.resolve()),
        "threshold": float(cfg.get("suspicious_threshold", 0.5)),
        "metrics": metrics,
        "label_note": cfg.get("label_note", ""),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (artifact_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote artifact to {artifact_dir}")


if __name__ == "__main__":
    main()
