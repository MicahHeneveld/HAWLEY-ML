"""Create a portable offline bundle from local artifacts + inference code."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline inference bundle (no AWS)")
    parser.add_argument(
        "--output-dir",
        default="offboard/portable_bundle",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/micah",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="checkpoint/1_lead_ECGFounder.pth",
        help="ECGFounder base weights required to load the diagnosis heads offline",
    )
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # Preserve the artifacts/micah/... layout so the bundled configs
    # (which reference artifacts/micah/<name>) resolve when run from the
    # bundle root.
    art_src = Path(args.artifacts_root)
    art_dst = root / art_src
    if art_src.is_dir():
        art_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(art_src, art_dst)

    # Base ECGFounder checkpoint. The diagnosis loader resolves
    # base_checkpoint by filename under <repo_root>/checkpoint, where
    # repo_root == the bundle's code/ dir at runtime, so place it there.
    base_ckpt = Path(args.base_checkpoint)
    checkpoint_included = False
    if base_ckpt.is_file():
        ckpt_dst = root / "code" / "checkpoint"
        ckpt_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_ckpt, ckpt_dst / base_ckpt.name)
        checkpoint_included = True
    else:
        print(f"WARNING: base checkpoint not found at {base_ckpt} — diagnosis heads will not load offline")

    code_dirs = ["adaptation", "fusion", "models", "preprocessing", "inference", "datasets"]
    for name in code_dirs:
        src = Path(name)
        if src.is_dir():
            shutil.copytree(
                src,
                root / "code" / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

    vendor = Path("vendor/ECGFounder")
    if vendor.is_dir():
        shutil.copytree(
            vendor,
            root / "code" / "vendor" / "ECGFounder",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    for extra in ("reports/coverage", "reports/diagnosis_narrow"):
        src = Path(extra)
        if src.is_dir():
            shutil.copytree(src, root / extra.replace("/", "_"))

    configs = Path("configs")
    if configs.is_dir():
        shutil.copytree(configs, root / "configs")

    shutil.copy2("requirements.txt", root / "requirements.txt")
    readme = root / "README_OFFLINE.md"
    readme.write_text(
        """# HAWLEY ML — offline bundle

Built before AWS account closure. Fully self-contained; no AWS required.

- `artifacts/` — personal AE (rest, resting_hr), local diagnosis head
  (`diagnosis_narrow`), and SageMaker-trained head (`diagnosis_narrow_sagemaker`),
  each with its `.pt` weights + calibrated `bundle.json`.
- `code/checkpoint/1_lead_ECGFounder.pth` — base ECGFounder weights required to
  load either diagnosis head.
- `code/` — batch scoring + fusion (set PYTHONPATH=code)
- `code/vendor/ECGFounder/` — vendored model code for diagnosis load
- `configs/` — batch_micah_full.yaml (points at diagnosis_narrow_sagemaker)

```bash
pip install -r requirements.txt
set PYTHONPATH=code            # Windows;  export PYTHONPATH=code on *nix
python -m inference.batch_pipeline --config configs/batch_micah_full.yaml
```

Verify weights are present: see `bundle_manifest.json` -> `weight_files`.
""",
        encoding="utf-8",
    )

    weight_files = sorted(
        str(p.relative_to(root).as_posix())
        for p in root.rglob("*.pt")
    ) + sorted(
        str(p.relative_to(root).as_posix())
        for p in root.rglob("*.pth")
    )
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts_root": str(art_src.resolve()),
        "base_checkpoint_included": checkpoint_included,
        "artifact_dirs": [
            str(p.parent.relative_to(art_dst).as_posix())
            for p in art_dst.rglob("bundle.json")
        ]
        if art_dst.is_dir()
        else [],
        "weight_files": weight_files,
    }
    (root / "bundle_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Portable bundle written to {root}")
    print(f"  base checkpoint included: {checkpoint_included}")
    print(f"  weight files: {len(weight_files)}")
    for w in weight_files:
        print(f"    {w}")


if __name__ == "__main__":
    main()
