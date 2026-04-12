from __future__ import annotations

import os
import struct
import tempfile

from nodmod import S3MSong
from .test_helpers import assert_true, pick_files


def _build_s3m_with_pattern() -> bytes:
    header = bytearray()
    title = b"Pattern Test\x00" + b"\x00" * (28 - len("Pattern Test") - 1)
    header += title
    header += bytes([0x1A, 0x10])
    header += struct.pack("<H", 0)
    header += struct.pack("<6H", 2, 0, 2, 0, 0x1320, 2)
    header += b"SCRM"
    header += bytes([64, 6, 125, 0xB0, 0, 0])
    header += b"\x00" * 8
    header += struct.pack("<H", 0)
    header += bytes([0, 255, 255, 9] + [255] * 28)
    header += bytes([0, 0xFF])
    header += struct.pack("<2H", 0x0030, 0x0040)

    data = bytearray(header)
    pattern0_offset = 0x300
    pattern1_offset = 0x400
    if len(data) > pattern0_offset:
        raise AssertionError("Synthetic S3M header unexpectedly exceeded first pattern offset")
    data += b"\x00" * (pattern0_offset - len(data))

    packed = bytearray()
    packed += bytes([0xE0, 0x40, 0x01, 40, 0x01, 0x06, 0x00])
    packed += bytes([0xA3, 0xFE, 0x02, 0x02, 0x09, 0x00])
    packed += bytes([0x00] * 62)
    pattern0 = struct.pack("<H", len(packed) + 2) + packed
    data += pattern0

    if len(data) > pattern1_offset:
        raise AssertionError("Synthetic S3M pattern unexpectedly exceeded second pattern offset")
    data += b"\x00" * (pattern1_offset - len(data))
    data += struct.pack("<H", 2)  # empty pattern with no packed data beyond the length field
    return bytes(data)


def test_s3m_pattern_decode_synthetic() -> None:
    payload = _build_s3m_with_pattern()
    with tempfile.NamedTemporaryFile(suffix=".s3m", delete=False) as tmp:
        tmp.write(payload)
        path = tmp.name
    try:
        song = S3MSong()
        song.load(path, verbose=False)
        assert_true(song.pattern_seq == [0], "Synthetic S3M pattern sequence mismatch")
        note0 = song.get_note(0, 0, 0)
        assert_true(note0.period == "C-4", "Synthetic S3M row 0 note mismatch")
        assert_true(note0.instrument_idx == 1, "Synthetic S3M row 0 instrument mismatch")
        assert_true(note0.volume == 40, "Synthetic S3M row 0 volume mismatch")
        assert_true(note0.effect == "A06", "Synthetic S3M row 0 effect mismatch")

        note1 = song.get_note(0, 1, 1)
        assert_true(note1.period == "off", "Synthetic S3M row 1 note-off mismatch")
        assert_true(note1.instrument_idx == 2, "Synthetic S3M row 1 instrument mismatch")
        assert_true(note1.effect == "B09", "Synthetic S3M row 1 effect mismatch")

        assert_true(song.get_note(0, 0, 1).is_empty(), "Synthetic S3M untouched channel should stay empty")
    finally:
        os.remove(path)


def test_s3m_pattern_decode_real_files() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "music"))
    assert_true(os.path.isdir(root), "Repo music/ directory not found for S3M pattern tests")
    files = sorted(pick_files(root, ".s3m", 3))
    assert_true(len(files) > 0, "No S3M files found in repo music/ for pattern tests")
    for path in files:
        song = S3MSong()
        song.load(path, verbose=False)
        if not song.pattern_seq:
            continue
        pat_idx = song.pattern_seq[0]
        pat = song.patterns[pat_idx]
        assert_true(pat.n_rows == 64, "Real S3M pattern row count mismatch")
        assert_true(pat.n_channels == song.n_channels, "Real S3M pattern channel count mismatch")
        populated = False
        for channel in range(pat.n_channels):
            for row in range(pat.n_rows):
                note = pat.data[channel][row]
                if not note.is_empty():
                    populated = True
                    break
            if populated:
                break
        assert_true(populated, "Real S3M first pattern should contain at least one event")


if __name__ == "__main__":
    test_s3m_pattern_decode_synthetic()
    test_s3m_pattern_decode_real_files()
    print("OK: test_s3m_pattern_decode.py")
