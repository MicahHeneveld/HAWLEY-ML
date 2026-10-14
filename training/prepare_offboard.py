"""Local offline prep: calibrate diagnosis, rebuild portable bundle, write status.

No cloud dependency — AWS account is closed. This just chains the local
calibration + bundle build and records what was produced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare offline multi-model bundle")
    parser.add_argument("--skip-calibrate", action="store_true")
    parser.add_argument("--skip-bundle", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    root = Path(__file__).resolve().parents[1]

    if not args.skip_calibrate:
        run([py, "-m", "evaluation.calibrate_diagnosis"])

    if not args.skip_bundle:
        run([py, "-m", "training.build_offboard_bundle"])

    checklist = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_steps_done": [
            "calibrate_diagnosis" if not args.skip_calibrate else None,
            "build_offboard_bundle" if not args.skip_bundle else None,
        ],
        "artifacts_expected": [
            "artifacts/micah/phase1_beat",
            "artifacts/micah/phase1_beat_resting_hr",
            "artifacts/micah/diagnosis_narrow",
            "artifacts/micah/diagnosis_narrow_sagemaker",
        ],
        "portable_bundle": str((root / "offboard/portable_bundle").resolve()),
        "cloud": "none (AWS account closed; runs fully offline)",
    }
    checklist["local_steps_done"] = [s for s in checklist["local_steps_done"] if s]
    out = root / "offboard" / "OFFBOARD_STATUS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
