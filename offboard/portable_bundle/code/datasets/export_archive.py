"""Load Micah Sandbox exports produced by SmashBeat export_sandbox.py."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ChunkRecord:
    recording_id: str
    chunk_index: int
    started_at: str
    ended_at: str
    beat_count: int
    bin_path: Path
    beats_path: Path | None
    status: str


@dataclass(frozen=True)
class RecordingRecord:
    recording_id: str
    subject_id: str
    started_at: str
    ended_at: str | None
    status: str
    chunks: tuple[ChunkRecord, ...]


class ExportArchive:
    def __init__(self, export_root: Path):
        self.export_root = export_root.resolve()
        if not self.export_root.is_dir():
            raise FileNotFoundError(f"Export root not found: {self.export_root}")

    def list_recordings(self) -> list[RecordingRecord]:
        recordings: list[RecordingRecord] = []
        for recording_dir in sorted(self.export_root.glob("SBX-*")):
            if not recording_dir.is_dir():
                continue
            meta_path = recording_dir / "recording.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            chunks = tuple(self._load_chunks(recording_dir, meta["id"]))
            recordings.append(
                RecordingRecord(
                    recording_id=meta["id"],
                    subject_id=meta.get("subject_id", ""),
                    started_at=meta["started_at"],
                    ended_at=meta.get("ended_at"),
                    status=meta.get("status", ""),
                    chunks=chunks,
                )
            )
        recordings.sort(key=lambda item: item.started_at)
        return recordings

    def _load_chunks(self, recording_dir: Path, recording_id: str) -> Iterator[ChunkRecord]:
        chunks_csv = recording_dir / "chunks.csv"
        if not chunks_csv.exists():
            return

        with chunks_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "ok":
                    continue
                bin_local = row.get("bin_local", "")
                beats_local = row.get("beats_local", "")
                bin_path = recording_dir / bin_local if bin_local else recording_dir / "missing.bin"
                beats_path = (
                    recording_dir / beats_local if beats_local else None
                )
                if not bin_path.exists():
                    continue
                yield ChunkRecord(
                    recording_id=recording_id,
                    chunk_index=int(row["chunk_index"]),
                    started_at=row["started_at"],
                    ended_at=row["ended_at"],
                    beat_count=int(row.get("beat_count") or 0),
                    bin_path=bin_path,
                    beats_path=beats_path if beats_path and beats_path.exists() else None,
                    status=row["status"],
                )
