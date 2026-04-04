from nodmod import MODSong, S3MSong, XMSong
from .test_helpers import assert_raises_msg, assert_true


def _flatten_timestamps(stamps: list[list[tuple[float, int, int]]]) -> list[tuple[float, int, int]]:
    flat: list[tuple[float, int, int]] = []
    for pat_rows in stamps:
        flat.extend(pat_rows)
    return flat


def test_mod_playback_rows_linear_equivalence() -> None:
    song = MODSong()
    rows = list(song.iter_playback_rows())
    stamps = _flatten_timestamps(song.timestamp())

    assert_true(len(rows) == 64, "linear MOD song should visit 64 rows")
    assert_true(len(stamps) == len(rows), "MOD playback rows should align with timestamp row count")
    assert_true(rows[0].visit_idx == 0 and rows[0].sequence_idx == 0 and rows[0].row == 0, "MOD first playback row mismatch")
    assert_true(rows[0].start_sec == 0.0 and rows[0].end_sec > 0.0, "MOD first row timing bounds mismatch")

    for idx, row in enumerate(rows):
        ts_sec, ts_speed, ts_tempo = stamps[idx]
        assert_true(abs(row.end_sec - ts_sec) < 1e-12, "MOD playback/timestamp end time mismatch")
        assert_true(row.speed == ts_speed and row.tempo == ts_tempo, "MOD playback/timestamp speed-tempo mismatch")
        if idx > 0:
            assert_true(abs(row.start_sec - rows[idx - 1].end_sec) < 1e-12, "MOD row boundaries must be contiguous")


def test_mod_playback_rows_break_and_loop() -> None:
    song = MODSong()
    song.add_pattern()
    song.set_sequence([0, 1])
    song.set_effect(0, 0, 1, "D03")
    rows = list(song.iter_playback_rows())
    assert_true(len(rows) == 63, "MOD Dxx should truncate current pattern and continue from requested row")
    assert_true(rows[2].sequence_idx == 1 and rows[2].row == 3, "MOD Dxx destination row mismatch")

    loop_song = MODSong()
    loop_song.set_effect(0, 0, 0, "E60")
    loop_song.set_effect(0, 0, 1, "E62")
    loop_rows = list(loop_song.iter_playback_rows())
    assert_true(len(loop_rows) > 64, "MOD E6x loop should replay rows")
    assert_true(
        [loop_rows[i].row for i in range(6)] == [0, 1, 0, 1, 0, 1],
        "MOD E6x loop visit ordering mismatch",
    )


def test_xm_playback_rows_linear_equivalence() -> None:
    song = XMSong()
    song.n_channels = 1
    rows = list(song.iter_playback_rows())
    stamps = _flatten_timestamps(song.timestamp())

    assert_true(len(rows) == 64, "linear XM song should visit 64 rows")
    assert_true(len(stamps) == len(rows), "XM playback rows should align with timestamp row count")
    for idx, row in enumerate(rows):
        ts_sec, ts_speed, ts_tempo = stamps[idx]
        assert_true(abs(row.end_sec - ts_sec) < 1e-12, "XM playback/timestamp end time mismatch")
        assert_true(row.speed == ts_speed and row.tempo == ts_tempo, "XM playback/timestamp speed-tempo mismatch")


def test_s3m_playback_rows_linear_equivalence() -> None:
    song = S3MSong()
    song.n_channels = 1
    rows = list(song.iter_playback_rows())
    stamps = _flatten_timestamps(song.timestamp())

    assert_true(len(rows) == 64, "linear S3M song should visit 64 rows")
    assert_true(len(stamps) == len(rows), "S3M playback rows should align with timestamp row count")
    for idx, row in enumerate(rows):
        ts_sec, ts_speed, ts_tempo = stamps[idx]
        assert_true(abs(row.start_sec - ts_sec) < 1e-12, "S3M playback/timestamp start time mismatch")
        assert_true(row.speed == ts_speed and row.tempo == ts_tempo, "S3M playback/timestamp speed-tempo mismatch")


def test_playback_rows_max_steps_guard() -> None:
    song = S3MSong()
    song.n_channels = 1
    song.add_pattern()
    song.set_sequence([0, 1])
    song.set_effect(0, 0, 0, "B01")
    song.set_effect(1, 0, 0, "B00")
    assert_raises_msg(
        RuntimeError,
        "max_steps",
        lambda: list(song.iter_playback_rows(max_steps=20)),
    )


if __name__ == "__main__":
    test_mod_playback_rows_linear_equivalence()
    test_mod_playback_rows_break_and_loop()
    test_xm_playback_rows_linear_equivalence()
    test_s3m_playback_rows_linear_equivalence()
    test_playback_rows_max_steps_guard()
    print("OK: test_playback_rows_api.py")
