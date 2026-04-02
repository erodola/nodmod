from nodmod import S3MSong
from nodmod.types import S3MNote, S3MSample
from test_helpers import assert_true, assert_raises_msg


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


if __name__ == '__main__':
    test_s3m_basic_types()
    test_s3m_song_defaults()
    test_s3m_basic_editing()
    test_s3m_sequence_limits()
    print("OK: test_s3m_basic.py")