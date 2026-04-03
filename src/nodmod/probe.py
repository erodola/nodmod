"""Lightweight module probing utilities for format/capability inspection."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

__all__ = ['ProbeResult', 'detect_format', 'probe_file']

_MOD_MAGIC_CHANNELS = {
    'M.K.': 4,
    'M!K!': 4,
    'FLT4': 4,
}


@dataclass(frozen=True)
class ProbeResult:
    """Structured result returned by :func:`probe_file`."""

    path: str
    detected_format: str | None
    supported: bool
    loader: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    metadata: dict[str, object]


def _detect_format_from_data(path: str, data: bytes) -> str | None:
    if data.startswith(b'Extended Module: '):
        return 'xm'
    if len(data) >= 48 and data[44:48] == b'SCRM':
        return 's3m'
    if len(data) >= 1084:
        return 'mod'
    lower = path.lower()
    if lower.endswith('.mod'):
        return 'mod'
    if lower.endswith('.xm'):
        return 'xm'
    if lower.endswith('.s3m'):
        return 's3m'
    return None


def detect_format(path: str) -> str | None:
    """Detect likely tracker format from lightweight file signatures."""
    with open(path, 'rb') as handle:
        data = handle.read()
    return _detect_format_from_data(path, data)


def _probe_mod(path: str, data: bytes) -> ProbeResult:
    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, object] = {}

    if len(data) < 1084:
        errors.append(f"MOD header is truncated ({len(data)} bytes, expected at least 1084).")
        return ProbeResult(path, 'mod', False, None, tuple(warnings), tuple(errors), metadata)

    magic = data[1080:1084].decode('latin-1', errors='replace')
    song_length = data[950]
    restart_raw = data[951]
    table = data[952:1080]
    max_pattern_index = max(table[:song_length]) if song_length > 0 else 0

    metadata['magic'] = magic
    metadata['song_length'] = song_length
    metadata['restart_raw'] = restart_raw
    metadata['max_pattern_index'] = max_pattern_index
    metadata['n_channels'] = _MOD_MAGIC_CHANNELS.get(magic)

    if song_length > 128:
        errors.append(f"Invalid MOD song length {song_length} (expected 0-128).")
    if magic not in _MOD_MAGIC_CHANNELS:
        errors.append(
            f"Unsupported MOD magic {magic!r}. Supported magics: {', '.join(sorted(_MOD_MAGIC_CHANNELS.keys()))}."
        )
    if song_length == 0:
        warnings.append("MOD song length is 0.")

    supported = len(errors) == 0
    return ProbeResult(
        path=path,
        detected_format='mod',
        supported=supported,
        loader='MODSong' if supported else None,
        warnings=tuple(warnings),
        errors=tuple(errors),
        metadata=metadata,
    )


def _probe_xm(path: str, data: bytes) -> ProbeResult:
    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, object] = {}

    if len(data) < 80:
        errors.append(f"XM header is truncated ({len(data)} bytes, expected at least 80).")
        return ProbeResult(path, 'xm', False, None, tuple(warnings), tuple(errors), metadata)

    if not data.startswith(b'Extended Module: '):
        errors.append("Missing XM signature 'Extended Module: '.")
        return ProbeResult(path, 'xm', False, None, tuple(warnings), tuple(errors), metadata)

    tracker_name = data[38:58].rstrip(b'\x00').decode('latin-1', errors='replace')
    metadata['tracker_name'] = tracker_name
    metadata['version'] = struct.unpack_from('<H', data, 58)[0]
    header_size = struct.unpack_from('<I', data, 60)[0]
    metadata['header_size'] = header_size

    if len(data) < 80:
        errors.append("XM file is truncated before song header fields.")
    else:
        metadata['song_length'] = struct.unpack_from('<H', data, 64)[0]
        metadata['restart_position'] = struct.unpack_from('<H', data, 66)[0]
        metadata['n_channels'] = struct.unpack_from('<H', data, 68)[0]
        metadata['n_patterns'] = struct.unpack_from('<H', data, 70)[0]
        metadata['n_instruments'] = struct.unpack_from('<H', data, 72)[0]
        metadata['default_speed'] = struct.unpack_from('<H', data, 76)[0]
        metadata['default_tempo'] = struct.unpack_from('<H', data, 78)[0]

    supported = len(errors) == 0
    return ProbeResult(
        path=path,
        detected_format='xm',
        supported=supported,
        loader='XMSong' if supported else None,
        warnings=tuple(warnings),
        errors=tuple(errors),
        metadata=metadata,
    )


def _probe_s3m(path: str, data: bytes) -> ProbeResult:
    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, object] = {}

    if len(data) < 96:
        errors.append(f"S3M header is truncated ({len(data)} bytes, expected at least 96).")
        return ProbeResult(path, 's3m', False, None, tuple(warnings), tuple(errors), metadata)

    sig1 = data[28]
    song_type = data[29]
    sig = data[44:48]
    order_count, instrument_count, pattern_count, _flags, _tracker, sample_type = struct.unpack_from('<6H', data, 32)

    metadata['sig1'] = sig1
    metadata['song_type'] = song_type
    metadata['signature'] = sig.decode('latin-1', errors='replace')
    metadata['order_count'] = order_count
    metadata['instrument_count'] = instrument_count
    metadata['pattern_count'] = pattern_count
    metadata['sample_type'] = sample_type

    if sig1 != 0x1A or song_type != 0x10:
        errors.append(f"Not an S3M module (sig1={sig1:#04x}, type={song_type:#04x}).")
    if sig != b'SCRM':
        errors.append(f"Missing S3M signature 'SCRM' (found {sig!r}).")
    if sample_type not in (1, 2):
        errors.append(f"Unsupported S3M sample type {sample_type}.")
    if instrument_count > 99:
        errors.append(f"Too many instruments {instrument_count} (supported up to 99).")

    table_offset = 96 + order_count
    table_size = 2 * instrument_count + 2 * pattern_count
    if len(data) < table_offset + table_size:
        errors.append("Truncated S3M parapointer tables.")
    else:
        instrument_ptr_offset = 96 + order_count
        if instrument_count:
            fmt = '<' + 'H' * instrument_count
            instrument_ptrs = struct.unpack_from(fmt, data, instrument_ptr_offset)
        else:
            instrument_ptrs = []

        for inst_idx, para in enumerate(instrument_ptrs, start=1):
            inst_offset = para << 4
            if inst_offset + 80 > len(data):
                errors.append(f"Invalid S3M instrument pointer at slot {inst_idx}.")
                continue
            inst_type = data[inst_offset]
            if 2 <= inst_type <= 7:
                errors.append(f"Adlib S3M instruments are unsupported (instrument {inst_idx}).")
                continue
            if inst_type not in (0, 1):
                errors.append(f"Unsupported S3M instrument type {inst_type} at instrument {inst_idx}.")
                continue
            if inst_type == 1:
                pack = data[inst_offset + 30]
                flags = data[inst_offset + 31]
                if pack != 0:
                    errors.append(f"Packed S3M samples are unsupported (instrument {inst_idx}).")
                if flags & 0x02:
                    errors.append(f"Stereo S3M samples are unsupported (instrument {inst_idx}).")

    supported = len(errors) == 0
    return ProbeResult(
        path=path,
        detected_format='s3m',
        supported=supported,
        loader='S3MSong' if supported else None,
        warnings=tuple(warnings),
        errors=tuple(errors),
        metadata=metadata,
    )


def probe_file(path: str) -> ProbeResult:
    """Probe module format, support, and lightweight metadata without full song loading."""
    resolved = os.fspath(path)
    with open(resolved, 'rb') as handle:
        data = handle.read()

    detected = _detect_format_from_data(resolved, data)
    if detected == 'mod':
        return _probe_mod(resolved, data)
    if detected == 'xm':
        return _probe_xm(resolved, data)
    if detected == 's3m':
        return _probe_s3m(resolved, data)
    return ProbeResult(
        path=resolved,
        detected_format=None,
        supported=False,
        loader=None,
        warnings=(),
        errors=("Unrecognized tracker module format.",),
        metadata={},
    )
