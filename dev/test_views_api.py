import array

from nodmod import MODSong, S3MSong, XMSong
from nodmod.types import XMSample
from test_helpers import assert_true


def test_cell_views_order_and_snapshot() -> None:
    song = MODSong()
    song.set_note(0, 2, 1, 3, "C-4", "F06")

    cells = list(song.iter_cells(sequence_only=True))
    assert_true(len(cells) == 64 * 4 * len(song.pattern_seq), "iter_cells should cover full sequence pattern grid")

    assert_true(cells[0].sequence_idx == 0 and cells[0].row == 0 and cells[0].channel == 0, "first cell order mismatch")
    assert_true(cells[1].sequence_idx == 0 and cells[1].row == 0 and cells[1].channel == 1, "row/channel ordering mismatch")
    assert_true(cells[4].sequence_idx == 0 and cells[4].row == 1 and cells[4].channel == 0, "row-major ordering mismatch")

    target = [c for c in cells if c.row == 1 and c.channel == 2 and c.sequence_idx == 0][0]
    assert_true(target.instrument_idx == 3 and target.period == "C-4" and target.effect == "F06", "cell view payload mismatch")

    song.set_note(0, 2, 1, 7, "E-4", "A01")
    assert_true(target.instrument_idx == 3 and target.period == "C-4", "cell snapshots must stay immutable")


def test_song_view_snapshot() -> None:
    song = MODSong()
    summary = song.view()
    assert_true(summary.format == "mod", "SongView format mismatch")
    assert_true(summary.sequence == tuple(song.pattern_seq), "SongView sequence mismatch")
    song.set_songname("changed")
    assert_true(summary.songname != song.songname, "SongView should be an immutable snapshot")


def test_mod_sample_views() -> None:
    song = MODSong()
    song.set_sample_pcm_i8(1, bytes([0, 1, 2]))
    song.set_sample_name(1, "Kick")
    song.set_sample_loop_bytes(1, 1, 2)

    used = list(song.iter_samples(include_empty=False))
    assert_true(len(used) == 1, "include_empty=False should skip empty MOD slots")
    assert_true(used[0].sample_idx == 1 and used[0].name == "Kick", "MOD SampleView metadata mismatch")
    assert_true(used[0].length == 3 and used[0].loop_start == 1 and used[0].loop_length == 2, "MOD SampleView loop mismatch")

    all_slots = list(song.iter_samples(include_empty=True))
    assert_true(len(all_slots) == 31, "include_empty=True should expose full MOD sample bank")


def test_xm_sample_views() -> None:
    song = XMSong()
    inst1 = song.new_instrument("Lead")
    smp1 = XMSample()
    smp1.waveform = array.array("b", [1, -1, 2])
    smp1.name = "Lead A"
    song.add_sample(inst1, smp1)

    inst2 = song.new_instrument("Pad")
    smp2 = XMSample()
    smp2.name = "Pad Empty"
    song.add_sample(inst2, smp2)

    all_samples = list(song.iter_samples(include_empty=True))
    used_samples = list(song.iter_samples(include_empty=False))

    assert_true(len(all_samples) == 2, "XM sample iteration should flatten instrument samples")
    assert_true(len(used_samples) == 1, "XM include_empty=False should skip empty waveforms")
    assert_true(all_samples[0].sample_idx == 1 and all_samples[1].sample_idx == 2, "XM flattened sample indices mismatch")


def test_s3m_sample_views() -> None:
    song = S3MSong()
    assert_true(len(list(song.iter_samples(include_empty=False))) == 0, "fresh S3M should have no non-empty sample views")
    sample = song.get_sample(1)
    sample.waveform = array.array("b", [1, 2, 3, 4])
    sample.name = "Bass"
    rows = list(song.iter_samples(include_empty=False))
    assert_true(len(rows) == 1 and rows[0].sample_idx == 1 and rows[0].name == "Bass", "S3M SampleView mismatch")


if __name__ == "__main__":
    test_cell_views_order_and_snapshot()
    test_song_view_snapshot()
    test_mod_sample_views()
    test_xm_sample_views()
    test_s3m_sample_views()
    print("OK: test_views_api.py")
