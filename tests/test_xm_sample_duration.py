from __future__ import annotations

import array
import math

from nodmod import Song
from nodmod import XMSong
from nodmod.types import XMSample
from .test_helpers import assert_raises_msg, assert_true


def _make_song_with_one_sample(length: int = 8363) -> tuple[XMSong, int]:
    song = XMSong()
    inst_idx = song.new_instrument("dur")
    smp = XMSample()
    smp.waveform = array.array("b", [0] * length)
    song.add_sample(inst_idx, smp)
    return song, inst_idx


def test_xm_sample_duration_default_reference() -> None:
    song, inst_idx = _make_song_with_one_sample(8363)
    dur = song.get_sample_duration(inst_idx, 1)
    assert_true(abs(dur - 1.0) < 1e-9, "XM default sample duration should be 1s for 8363 samples")


def test_xm_sample_duration_note_and_tuning() -> None:
    song, inst_idx = _make_song_with_one_sample(8363)
    smp = song.get_sample(inst_idx, 1)

    dur_c4 = song.get_sample_duration(inst_idx, 1, period="C-4")
    dur_c5 = song.get_sample_duration(inst_idx, 1, period="C-5")
    assert_true(dur_c5 < dur_c4, "Higher reference note should shorten duration")
    assert_true(abs((dur_c5 / dur_c4) - 0.5) < 1e-9, "One octave up should halve duration")

    smp.relative_note = 12
    dur_rel_up = song.get_sample_duration(inst_idx, 1, period="C-4")
    assert_true(abs((dur_rel_up / dur_c4) - 0.5) < 1e-9, "relative_note +12 should halve duration")

    smp.relative_note = -12
    dur_rel_down = song.get_sample_duration(inst_idx, 1, period="C-4")
    assert_true(abs((dur_rel_down / dur_c4) - 2.0) < 1e-9, "relative_note -12 should double duration")

    smp.relative_note = 0
    smp.finetune = 64  # +0.5 semitone
    dur_fine = song.get_sample_duration(inst_idx, 1, period="C-4")
    expected_ratio = 2.0 ** (-(64.0 / 128.0) / 12.0)
    assert_true(abs((dur_fine / dur_c4) - expected_ratio) < 1e-9, "XM finetune duration ratio mismatch")


def test_xm_sample_duration_sample_rate_override() -> None:
    song, inst_idx = _make_song_with_one_sample(8000)
    dur = song.get_sample_duration(inst_idx, 1, sample_rate=4000)
    assert_true(abs(dur - 2.0) < 1e-12, "Explicit sample_rate override should define duration directly")


def test_xm_sample_duration_errors() -> None:
    song_ok, inst_ok = _make_song_with_one_sample(8)
    assert_raises_msg(ValueError, "sample_rate", song_ok.get_sample_duration, inst_ok, 1, "C-4", 0)
    assert_raises_msg(ValueError, "sample_rate", song_ok.get_sample_duration, inst_ok, 1, "C-4", -1)
    assert_raises_msg(ValueError, "Invalid note format", song_ok.get_sample_duration, inst_ok, 1, "BAD")
    assert_raises_msg(IndexError, "Invalid instrument index", song_ok.get_sample_duration, 0, 1)
    assert_raises_msg(IndexError, "Invalid sample index", song_ok.get_sample_duration, inst_ok, 2)

    song_empty = XMSong()
    inst_empty = song_empty.new_instrument("empty")
    song_empty.add_sample(inst_empty, XMSample())
    assert_raises_msg(ValueError, "has no waveform data", song_empty.get_sample_duration, inst_empty, 1)


def test_xm_duration_formula_alignment() -> None:
    song, inst_idx = _make_song_with_one_sample(1000)
    smp = song.get_sample(inst_idx, 1)
    smp.relative_note = -2
    smp.finetune = -64

    ref_idx = Song.note_to_index("F#4")
    semitone_offset = (ref_idx - Song.note_to_index("C-4")) + smp.relative_note + (smp.finetune / 128.0)
    expected_rate = 8363.0 * (2.0 ** (semitone_offset / 12.0))
    expected_dur = len(smp.waveform) / expected_rate

    observed = song.get_sample_duration(inst_idx, 1, period="F#4")
    assert_true(math.isclose(observed, expected_dur, rel_tol=1e-12), "XM duration formula mismatch")


if __name__ == "__main__":
    test_xm_sample_duration_default_reference()
    test_xm_sample_duration_note_and_tuning()
    test_xm_sample_duration_sample_rate_override()
    test_xm_sample_duration_errors()
    test_xm_duration_formula_alignment()
    print("OK: test_xm_sample_duration.py")
