"""Load narrow ECGFounder head and run suspicious-class inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.ecgfounder_wrapper import find_checkpoint_dir, select_weight_path

VENDOR_ECGFOUNDER = Path(__file__).resolve().parents[1] / "vendor" / "ECGFounder"


class DiagnosisNarrowUnavailable(Exception):
    pass


def _vendor_path() -> None:
    if VENDOR_ECGFOUNDER.is_dir():
        p = str(VENDOR_ECGFOUNDER)
        if p not in sys.path:
            sys.path.insert(0, p)


def load_diagnosis_narrow(artifact_dir: Path, *, device: str | None = None) -> tuple[Any, dict]:
    artifact_dir = artifact_dir.resolve()
    bundle_path = artifact_dir / "bundle.json"
    if not bundle_path.is_file():
        raise DiagnosisNarrowUnavailable(f"Missing {bundle_path}")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    ckpt_path = artifact_dir / bundle["checkpoint_file"]
    if not ckpt_path.is_file():
        raise DiagnosisNarrowUnavailable(f"Missing checkpoint {ckpt_path}")

    _vendor_path()
    try:
        from finetune_model import ft_1lead_ECGFounder  # type: ignore
    except ImportError as error:
        raise DiagnosisNarrowUnavailable(
            "vendor/ECGFounder/finetune_model.py required for narrow head"
        ) from error

    n_classes = int(bundle.get("n_classes", 2))
    base_ckpt_raw = Path(bundle["base_checkpoint"])
    if base_ckpt_raw.is_dir():
        resolved = find_checkpoint_dir(base_ckpt_raw)
        base_ckpt = select_weight_path(resolved, single_lead=True) if resolved else None
    else:
        base_ckpt = base_ckpt_raw if base_ckpt_raw.is_file() else None
    if not base_ckpt:
        raise DiagnosisNarrowUnavailable("Invalid base_checkpoint in bundle.json")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ft_1lead_ECGFounder(dev, str(base_ckpt), n_classes, linear_prob=False)
    state = torch.load(ckpt_path, map_location=dev, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, bundle


@torch.no_grad()
def predict_context_suspicious(
    model: Any,
    windows: np.ndarray,
    *,
    batch_size: int = 16,
    suspicious_class: int = 1,
) -> np.ndarray:
    """Probability of weak context-alert class (index suspicious_class)."""
    if windows.size == 0:
        return np.array([], dtype=np.float32)
    probs_out: list[np.ndarray] = []
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size]
        tensor = torch.from_numpy(batch[:, np.newaxis, :].astype(np.float32))
        if next(model.parameters()).is_cuda:
            tensor = tensor.cuda()
        logits = model(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        prob = torch.softmax(logits, dim=-1)[:, suspicious_class]
        probs_out.append(prob.cpu().numpy())
    return np.concatenate(probs_out, axis=0)
