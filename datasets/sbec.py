"""Parse SmashBeat Sandbox SBEC binary ECG chunks."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import zstandard as zstd

SBEC_MAGIC = b"SBEC"
HEADER_SIZE = 16


def parse_sbec(payload: bytes) -> tuple[int, np.ndarray]:
    """Return sample rate (Hz) and microvolt samples from SBEC bytes."""
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"SBEC payload too short: {len(payload)} bytes")
    magic = payload[:4]
    if magic != SBEC_MAGIC:
        raise ValueError(f"Invalid SBEC magic: {magic!r}")

    version, sample_rate, sample_count = struct.unpack("<iii", payload[4:16])
    if version != 1:
        raise ValueError(f"Unsupported SBEC version: {version}")

    expected = HEADER_SIZE + sample_count * 4
    if len(payload) < expected:
        raise ValueError(
            f"SBEC truncated: expected {expected} bytes, got {len(payload)}"
        )

    samples = np.frombuffer(payload, dtype="<i4", count=sample_count, offset=HEADER_SIZE)
    return int(sample_rate), samples.astype(np.float64)


def decompress_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix == ".zst" or path.name.endswith(".zst"):
        return zstd.ZstdDecompressor().decompress(raw)
    return raw


def read_sbec_file(path: Path) -> tuple[int, np.ndarray]:
    return parse_sbec(decompress_bytes(path))
