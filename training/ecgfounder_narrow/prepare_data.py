"""Prepare weak-label NPZ for narrow ECGFounder training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from datasets.export_archive import ExportArchive
from preprocessing.diagnosis_windows import iter_diagnosis_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diagnosis narrow train/eval NPZ")
    parser.add_argument("--config", default="configs/diagnosis_narrow_micah.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    export_dir = Path(cfg["export_dir"])
    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = ExportArchive(export_dir)
    windows = iter_diagnosis_windows(
        archive,
        window_seconds=float(cfg["window_seconds"]),
        stride_seconds=float(cfg["window_stride_seconds"]),
        target_fs=int(cfg["target_sample_rate_hz"]),
        max_windows=int(cfg.get("max_windows", 8000)),
        seed=int(cfg["random_seed"]),
    )
    if len(windows) < 100:
        raise SystemExit(f"Too few diagnosis windows: {len(windows)}")

    x = np.stack([w.samples_500hz for w in windows], axis=0).astype(np.float32)
    y = np.array([w.label for w in windows], dtype=np.int64)

    rng = np.random.default_rng(int(cfg["random_seed"]))
    idx = np.arange(len(windows))
    rng.shuffle(idx)
    split = int(len(idx) * float(cfg.get("train_fraction", 0.8)))
    train_idx, eval_idx = idx[:split], idx[split:]

    np.savez_compressed(out_dir / "train.npz", x=x[train_idx], y=y[train_idx])
    np.savez_compressed(out_dir / "eval.npz", x=x[eval_idx], y=y[eval_idx])

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export_dir": str(export_dir),
        "total_windows": len(windows),
        "train_size": int(train_idx.size),
        "eval_size": int(eval_idx.size),
        "label_balance": {
            "alert": int(np.sum(y == 1)),
            "normal": int(np.sum(y == 0)),
        },
        "label_note": cfg.get("label_note", ""),
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
