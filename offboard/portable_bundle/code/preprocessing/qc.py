"""Beat metadata loading and Sandbox tag QC gates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import zstandard as zstd

_TAG_SPLIT = re.compile(r"[\[\],]")


def parse_tags(raw_tags) -> frozenset[str]:
    if raw_tags is None:
        return frozenset()
    if isinstance(raw_tags, list):
        return frozenset(str(tag).strip() for tag in raw_tags if str(tag).strip())
    text = str(raw_tags).strip()
    if not text:
        return frozenset()
    parts = [part.strip() for part in _TAG_SPLIT.split(text) if part.strip()]
    return frozenset(parts)


@dataclass(frozen=True)
class BeatRecord:
    beat_index: int
    sample_offset: int
    tags: frozenset[str]
    rr_ms: int | None
    hr_bpm: float | None


def load_beats(path: Path) -> list[BeatRecord]:
    raw = path.read_bytes()
    if not raw:
        return []
    if path.name.endswith(".zst"):
        raw = zstd.ZstdDecompressor().decompress(raw)
    beats: list[BeatRecord] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        beats.append(
            BeatRecord(
                beat_index=int(row["beat_index"]),
                sample_offset=int(row["sample_offset"]),
                tags=parse_tags(row.get("tags")),
                rr_ms=row.get("rr_ms"),
                hr_bpm=row.get("hr_bpm"),
            )
        )
    return beats


def beats_in_window(
    beats: list[BeatRecord],
    start_sample: int,
    end_sample: int,
) -> list[BeatRecord]:
    return [
        beat
        for beat in beats
        if start_sample <= beat.sample_offset < end_sample
    ]


def window_passes_qc(
    beats: list[BeatRecord],
    *,
    layer_a: str,
    layer_c_include: str | None,
    layer_c_exclude: tuple[str, ...],
    layer_a_exclude: tuple[str, ...] = (),
    min_layer_a_fraction: float = 0.5,
) -> bool:
    if not beats:
        return False

    n = len(beats)
    layer_a_hits = sum(1 for beat in beats if layer_a in beat.tags)
    clean_hits = (
        sum(1 for beat in beats if layer_c_include in beat.tags)
        if layer_c_include
        else 0
    )
    bad_hits = sum(
        1
        for beat in beats
        for tag in layer_c_exclude
        if tag in beat.tags
    )
    context_bad_hits = sum(
        1
        for beat in beats
        if layer_a not in beat.tags
        for tag in layer_a_exclude
        if tag in beat.tags
    )

    rest_ok = layer_a_hits >= max(1, int(n * min_layer_a_fraction))
    clean_ok = (
        clean_hits >= max(1, int(n * min_layer_a_fraction))
        if layer_c_include
        else True
    )
    return rest_ok and clean_ok and bad_hits == 0 and context_bad_hits == 0
