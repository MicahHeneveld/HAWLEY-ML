"""Corpus coverage report — recordings, tags, QC pass rates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from datasets.export_archive import ExportArchive
from preprocessing.qc import load_beats
from preprocessing.tag_schema import (
    LAYER_C_REJECT,
    has_sedentary_context,
    qc_veto,
    resolve_activity_segment,
)


@dataclass
class RecordingCoverage:
    recording_id: str
    chunk_count: int
    beat_count: int
    signal_clean_beats: int
    routable_beats: int
    dominant_layer_a: str | None
    veto_counts: dict[str, int]


def run_coverage(export_dir: Path) -> dict:
    archive = ExportArchive(export_dir)
    recordings_out: list[RecordingCoverage] = []
    totals = Counter()

    for recording in archive.list_recordings():
        veto_counts: Counter[str] = Counter()
        beats = 0
        clean = 0
        routable = 0
        layer_a = Counter()

        for chunk in recording.chunks:
            if not chunk.beats_path:
                continue
            for beat in load_beats(chunk.beats_path):
                beats += 1
                tags = beat.tags
                if has_sedentary_context(tags):
                    layer_a["sedentary"] += 1
                elif "walking" in tags:
                    layer_a["walking"] += 1
                elif "cycling" in tags:
                    layer_a["cycling"] += 1
                else:
                    layer_a["other"] += 1

                ok, reason = qc_veto(tags)
                if ok:
                    clean += 1
                elif reason:
                    veto_counts[reason] += 1

                if ok and resolve_activity_segment(tags):
                    routable += 1

        dom = layer_a.most_common(1)[0][0] if layer_a else None
        recordings_out.append(
            RecordingCoverage(
                recording_id=recording.recording_id,
                chunk_count=len(recording.chunks),
                beat_count=beats,
                signal_clean_beats=clean,
                routable_beats=routable,
                dominant_layer_a=dom,
                veto_counts=dict(veto_counts),
            )
        )
        totals["beats"] += beats
        totals["signal_clean"] += clean
        totals["routable"] += routable
        if routable > 0:
            totals["recordings_with_routable"] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "export_dir": str(export_dir.resolve()),
        "totals": dict(totals),
        "layer_c_reject_tags": sorted(LAYER_C_REJECT),
        "recordings": [asdict(r) for r in recordings_out],
        "notes": [
            "Routable = passes Layer C QC and maps to rest/resting_hr/walk/cycling segment.",
            "Many exports use stationary_activity; sedentary QC still requires clean Layer C.",
            "Recordings with motion_artifact on all beats contribute 0 routable beats.",
        ],
    }


def write_report(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_coverage.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Dataset coverage",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        f"- Total beats: **{payload['totals'].get('beats', 0)}**",
        f"- Signal-clean (Layer C): **{payload['totals'].get('signal_clean', 0)}**",
        f"- Routable to personal AE: **{payload['totals'].get('routable', 0)}**",
        f"- Recordings with routable beats: **{payload['totals'].get('recordings_with_routable', 0)}**",
        "",
        "## Per recording",
        "",
        "| Recording | Chunks | Beats | Clean | Routable | Dominant A |",
        "|-----------|--------|-------|-------|----------|------------|",
    ]
    for row in payload["recordings"]:
        if row["beat_count"] == 0:
            continue
        lines.append(
            f"| {row['recording_id']} | {row['chunk_count']} | {row['beat_count']} | "
            f"{row['signal_clean_beats']} | {row['routable_beats']} | {row['dominant_layer_a']} |"
        )
    (output_dir / "dataset_coverage.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tag/QC coverage report")
    parser.add_argument("--export-dir", default="exports/2026-07-16T162705Z")
    parser.add_argument("--output-dir", default="reports/coverage")
    args = parser.parse_args()
    payload = run_coverage(Path(args.export_dir))
    write_report(payload, Path(args.output_dir))
    print(json.dumps(payload["totals"], indent=2))


if __name__ == "__main__":
    main()
