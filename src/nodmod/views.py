"""Immutable read-only view types for song, cell, and sample inspection."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ['CellView', 'SampleView', 'SongView']


@dataclass(frozen=True)
class CellView:
    """Read-only snapshot of one note cell in sequence context."""

    sequence_idx: int
    pattern_idx: int
    row: int
    channel: int
    instrument_idx: int
    period: str
    effect: str
    vol_cmd: str | None = None
    vol_val: int | None = None
    volume: int | None = None


@dataclass(frozen=True)
class SampleView:
    """Read-only snapshot of one sample slot."""

    sample_idx: int
    name: str
    length: int
    finetune: int
    volume: int
    loop_start: int
    loop_length: int


@dataclass(frozen=True)
class SongView:
    """Read-only snapshot of song-level metadata."""

    format: str
    songname: str
    artist: str
    sequence: tuple[int, ...]
    n_patterns: int
    n_channels: int
