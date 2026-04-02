import os
import struct
import tempfile

from nodmod import S3MSong
from nodmod.types import S3MNote, S3MSample
from test_helpers import assert_true, assert_raises_msg, pick_files


def test_s3m_basic_types() -> None:
    note = S3MNote(1, "off", "A06", 40)
    assert_true("===" in repr(note), "S3MNote repr should show note-off")
    assert_true(not note.is_empty(), "S3MNote should not be empty")

    sample = S3MSample()
    assert_true(sample.c2spd == 8363, "S3MSample default c2spd mismatch")
    assert_true(sample.length == 0, "S3MSample should start empty")


def test_s3m_song_defaults() -> None:
    song = S3MSong()
    assert_true(song.file_extension == "s3m", "S3MSong extension mismatch")
    assert_true(song.n_channels == 16, "S3MSong default channels mismatch")
    assert_true(len(song.patterns) == 1, "S3MSong should start with one pattern")
    assert_true(song.pattern_seq == [0], "S3MSong should start with one sequence entry")
    assert_true(song.patterns[0].n_rows == 64, "S3MSong pattern row count mismatch")
    assert_true(song.patterns[0].n_channels == 16, "S3MSong pattern channel count mismatch")
    assert_true(all(isinstance(song.patterns[0].data[c][0], S3MNote) for c in range(song.n_channels)), "Pattern cells should use S3MNote")
    assert_true(len(song.samples) == song.MAX_SAMPLES, "S3MSong sample bank size mismatch")
    assert_true(song.get_sample(1).c2spd == 8363, "S3MSong sample bank default mismatch")
    assert_true(song.raw_channel_slots == list(range(16)), "S3MSong raw channel slots mismatch")
    assert_true(song.compact_to_raw_channel == list(range(16)), "S3MSong compact channel map mismatch")


def test_s3m_basic_editing() -> None:
    song = S3MSong()
    song.set_note(0, 0, 0, 1, "C-4", volume=32)
    note = song.get_note(0, 0, 0)
    assert_true(note.instrument_idx == 1, "S3MSong note instrument mismatch")
    assert_true(note.period == "C-4", "S3MSong note period mismatch")
    assert_true(note.volume == 32, "S3MSong note volume mismatch")

    song.set_ticks_per_row(0, 0, 1, 3)
    song.set_bpm(0, 0, 2, 150)
    assert_true(song.get_note(0, 1, 0).effect == "A03", "S3MSong speed effect mismatch")
    assert_true(song.get_note(0, 2, 0).effect == "T96", "S3MSong tempo effect mismatch")

    song.clear_pattern(0)
    assert_true(song.get_note(0, 0, 0).is_empty(), "clear_pattern should empty notes")


def test_s3m_sequence_limits() -> None:
    song = S3MSong()
    song.patterns = []
    song.pattern_seq = []
    for _ in range(256):
        song.add_pattern()
    assert_raises_msg(ValueError, "Pattern sequence too long", song.add_pattern)


def _build_s3m_header_bytes() -> bytes:
    title = b"Header Test\x00" + b"\x00" * (28 - len("Header Test") - 1)
    header = bytearray()
    header += title
    header += bytes([0x1A, 0x10])
    header += struct.pack("<H", 0)
    header += struct.pack("<6H", 5, 0, 3, 0x0080, 0x1320, 2)
    header += b"SCRM"
    header += bytes([64, 6, 125, 0xB0, 0, 252])
    header += b"ABCDEFGH"
    header += struct.pack("<H", 0)
    header += bytes([0, 8, 1, 9, 255, 255, 255, 255] + [255] * 24)
    header += bytes([0, 0xFE, 2, 0xFF, 1])
    header += struct.pack("<3H", 0x0030, 0x0040, 0x0050)
    header += bytes(range(32))
    data = bytearray(header)
    for offset in (0x300, 0x400, 0x500):
        if len(data) > offset:
            raise AssertionError("Synthetic S3M header fixture exceeded a declared pattern offset")
        data += b"\x00" * (offset - len(data))
        data += struct.pack("<H", 2)
    return bytes(data)


def test_s3m_header_load_synthetic() -> None:
    payload = _build_s3m_header_bytes()
    with tempfile.NamedTemporaryFile(suffix=".s3m", delete=False) as tmp:
        tmp.write(payload)
        path = tmp.name
    try:
        song = S3MSong()
        song.load(path, verbose=False)
        assert_true(song.songname == "Header Test", "S3M synthetic title mismatch")
        assert_true(song.order_count == 5, "S3M synthetic order_count mismatch")
        assert_true(song.instrument_count == 0, "S3M synthetic instrument_count mismatch")
        assert_true(song.pattern_count == 3, "S3M synthetic pattern_count mismatch")
        assert_true(song.order_list_raw == [0, 0xFE, 2, 0xFF, 1], "S3M raw order list mismatch")
        assert_true(song.pattern_seq == [0, 2], "S3M pattern sequence should skip marker/end orders")
        assert_true(song.instrument_offsets == [], "S3M instrument offsets mismatch")
        assert_true(song.pattern_offsets == [0x300, 0x400, 0x500], "S3M pattern offsets mismatch")
        assert_true(song.default_panning == list(range(32)), "S3M default panning mismatch")
        assert_true(song.n_channels == 4, "S3M compact channel count mismatch")
        assert_true(song.raw_channel_slots == [0, 1, 2, 3], "S3M raw channel slot mapping mismatch")
        assert_true(song.reserved2 == b"ABCDEFGH", "S3M reserved header bytes mismatch")
    finally:
        os.remove(path)


def test_s3m_header_load_real_files() -> None:
    root = r"G:\My Drive\Moduli"
    if not os.path.isdir(root):
        return
    files = sorted(pick_files(root, ".s3m", 3))
    assert_true(len(files) > 0, "No S3M files found for header tests")
    for path in files:
        song = S3MSong()
        song.load(path, verbose=False)
        assert_true(song.song_type == 0x10, "Real S3M song type mismatch")
        assert_true(song.sample_type in (1, 2), "Real S3M sample type mismatch")
        assert_true(song.order_count >= len(song.pattern_seq), "Real S3M order count mismatch")
        assert_true(song.pattern_count == len(song.patterns), "Real S3M pattern placeholder mismatch")
        assert_true(song.n_channels >= 1, "Real S3M channel count mismatch")
        assert_true(len(song.channel_settings) == 32, "Real S3M channel settings length mismatch")
        if song.default_pan_flag == 252:
            assert_true(len(song.default_panning) == 32, "Real S3M panning table mismatch")
        else:
            assert_true(song.default_panning == [], "Real S3M panning table should be absent")


if __name__ == '__main__':
    test_s3m_basic_types()
    test_s3m_song_defaults()
    test_s3m_basic_editing()
    test_s3m_sequence_limits()
    test_s3m_header_load_synthetic()
    test_s3m_header_load_real_files()
    print("OK: test_s3m_basic.py")