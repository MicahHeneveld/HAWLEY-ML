"""Optional ECGFounder zero-shot wrapper (requires checkpoint download)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

VENDOR_ECGFOUNDER = Path(__file__).resolve().parents[1] / "vendor" / "ECGFounder"
TASKS_FILE = VENDOR_ECGFOUNDER / "tasks.txt"


class ECGFounderUnavailable(Exception):
    pass


def _ensure_ecgfounder_on_path() -> None:
    if VENDOR_ECGFOUNDER.is_dir():
        vendor_path = str(VENDOR_ECGFOUNDER)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)


def load_label_names() -> list[str]:
    if not TASKS_FILE.exists():
        return []
    return [line.strip() for line in TASKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_checkpoint_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    weight_files = list(path.glob("*.pth")) + list(path.glob("*.pt"))
    return path if weight_files else None


def select_weight_path(checkpoint_dir: Path, *, single_lead: bool = True) -> Path:
    """Pick the correct ECGFounder checkpoint (1-lead for Polar H10)."""
    weight_files = sorted(checkpoint_dir.glob("*.pth")) + sorted(checkpoint_dir.glob("*.pt"))
    if not weight_files:
        raise ECGFounderUnavailable(f"No .pth/.pt files in {checkpoint_dir}")

    if single_lead:
        for path in weight_files:
            if "1_lead" in path.name or "1lead" in path.name.lower():
                return path
    else:
        for path in weight_files:
            if "12_lead" in path.name or "12lead" in path.name.lower():
                return path

    return weight_files[0]


def load_ecgfounder(checkpoint_dir: Path, *, single_lead: bool = True) -> Any:
    """Load ECGFounder model if checkpoint and upstream code are available."""
    resolved_dir = find_checkpoint_dir(checkpoint_dir)
    if resolved_dir is None:
        raise ECGFounderUnavailable(
            f"No ECGFounder weights found in {checkpoint_dir}. "
            "Download 1_lead_ECGFounder.pth from "
            "https://huggingface.co/PKUDigitalHealth/ECGFounder into ./checkpoint/"
        )

    weight_path = select_weight_path(resolved_dir, single_lead=single_lead)

    _ensure_ecgfounder_on_path()
    try:
        from net1d import Net1D  # type: ignore
    except ImportError as error:
        raise ECGFounderUnavailable(
            "ECGFounder net1d module not found. Expected vendor/ECGFounder "
            "(run: git clone https://github.com/PKUDigitalHealth/ECGFounder.git "
            "vendor/ECGFounder), or run Phase 0 with --skip-ecgfounder."
        ) from error

    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=150,
        use_bn=False,
        use_do=False,
        return_features=False,
        verbose=False,
    )
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_probabilities(
    model: Any, windows: np.ndarray, batch_size: int = 32
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size]
        tensor = torch.from_numpy(batch[:, np.newaxis, :].astype(np.float32))
        logits = model(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0)
