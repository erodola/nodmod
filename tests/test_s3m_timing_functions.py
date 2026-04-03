from __future__ import annotations

from nodmod import S3MSong, Song
from .test_helpers import assert_true


def test_s3m_speed_and_tempo_timestamps() -> None:
    song = S3MSong()
    song.n_channels = 1
    song.initial_speed = 6
    song.initial_tempo = 125
    song.set_ticks_per_row(0, 0, 0, 3)
    song.set_bpm(0, 0, 1, 150)

    ts = song.timestamp()[0]
    assert_true(ts[0][1] == 3, "S3M timestamp should reflect Axx speed changes")
    assert_true(ts[0][2] == 125, "S3M timestamp should preserve initial tempo on row 0")
    assert_true(ts[1][1] == 3, "S3M timestamp should carry forward speed")
    assert_true(ts[1][2] == 150, "S3M timestamp should reflect Txx tempo changes")

    row1_delta = ts[1][0] - ts[0][0]
    expected = Song.get_tick_duration(150) * 3
    assert_true(abs(row1_delta - Song.get_tick_duration(125) * 3) < 1e-9, "S3M row 0 duration mismatch")
    row2_delta = ts[2][0] - ts[1][0]
    assert_true(abs(row2_delta - expected) < 1e-9, "S3M row 1 duration mismatch")


def test_s3m_pattern_break_and_order_jump() -> None:
    song = S3MSong()
    song.n_channels = 1
    song.add_pattern()
    song.set_sequence([0, 1])
    song.set_effect(0, 0, 1, "C03")
    ts = song.timestamp()
    assert_true(len(ts[0]) == 2, "S3M pattern break should truncate the current pattern")
    assert_true(len(ts[1]) == 61, "S3M pattern break should continue from the requested row")

    song2 = S3MSong()
    song2.n_channels = 1
    song2.add_pattern()
    song2.set_sequence([0, 1])
    song2.set_effect(0, 0, 1, "B01")
    ts2 = song2.timestamp()
    assert_true(len(ts2[0]) == 2, "S3M order jump should truncate the current pattern")
    assert_true(len(ts2[1]) == 64, "S3M order jump should restart the destination pattern at row 0")


def test_s3m_pattern_delay_and_effective_rows() -> None:
    song = S3MSong()
    song.n_channels = 1
    song.set_effect(0, 0, 0, "SE2")
    ts = song.timestamp()[0]
    row1_delta = ts[1][0] - ts[0][0]
    expected = 3 * Song.get_tick_duration(song.initial_tempo) * song.initial_speed
    assert_true(abs(row1_delta - expected) < 1e-9, "S3M pattern delay should extend row duration")
    assert_true(song.get_effective_row_count(0) == 66, "S3M effective row count should include pattern delay")


if __name__ == "__main__":
    test_s3m_speed_and_tempo_timestamps()
    test_s3m_pattern_break_and_order_jump()
    test_s3m_pattern_delay_and_effective_rows()
    print("OK: test_s3m_timing_functions.py")