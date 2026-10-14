"""Build dataset manifests from local Sandbox exports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from datasets.export_archive import ExportArchive


@dataclass
class ManifestChunk:
    recording_id: str
    chunk_index: int
    started_at: str
    ended_at: str
    beat_count: int
    ecg_local: str
    beats_local: str | None
    status: str


@dataclass
class ManifestRecording:
    recording_id: str
    subject_id: str
    started_at: str
    ended_at: str | None
    status: str
    chunk_count: int
    chunks: list[ManifestChunk]


@dataclass
class DatasetManifest:
    manifest_version: str
    subject_id: str
    source: str
    export_dir: str
    generated_at: str
    recordings: list[ManifestRecording]

    def to_dict(self) -> dict:
        return asdict(self)


def build_manifest(export_dir: Path, subject_id: str | None = None) -> DatasetManifest:
    export_dir = export_dir.resolve()
    archive = ExportArchive(export_dir)
    recordings_out: list[ManifestRecording] = []
    resolved_subject = subject_id or ""

    for recording in archive.list_recordings():
        resolved_subject = resolved_subject or recording.subject_id
        chunks: list[ManifestChunk] = []
        for chunk in recording.chunks:
            chunks.append(
                ManifestChunk(
                    recording_id=chunk.recording_id,
                    chunk_index=chunk.chunk_index,
                    started_at=chunk.started_at,
                    ended_at=chunk.ended_at,
                    beat_count=chunk.beat_count,
                    ecg_local=str(chunk.bin_path.relative_to(export_dir)),
                    beats_local=(
                        str(chunk.beats_path.relative_to(export_dir))
                        if chunk.beats_path
                        else None
                    ),
                    status=chunk.status,
                )
            )
        recordings_out.append(
            ManifestRecording(
                recording_id=recording.recording_id,
                subject_id=recording.subject_id,
                started_at=recording.started_at,
                ended_at=recording.ended_at,
                status=recording.status,
                chunk_count=len(chunks),
                chunks=chunks,
            )
        )

    return DatasetManifest(
        manifest_version="1.0",
        subject_id=resolved_subject,
        source="smashbeat_sandbox",
        export_dir=str(export_dir),
        generated_at=datetime.now(timezone.utc).isoformat(),
        recordings=recordings_out,
    )


def write_manifest(export_dir: Path, output_path: Path, subject_id: str | None = None) -> Path:
    manifest = build_manifest(export_dir, subject_id=subject_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return output_path
