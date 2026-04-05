import array
import os
from nodmod import XMSong
from nodmod.types import XMSample
from .test_helpers import assert_true, with_temp_wav



def assert_raises_msg(exc_type, msg_substr: str, func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        if msg_substr not in str(exc):
            raise AssertionError(f"Expected message to contain '{msg_substr}', got '{exc}'") from exc
        return
    except Exception as exc:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")



def _note_idx(note: str) -> int:
    note = note.strip().upper()
    period_seq = ['C-', 'C#', 'D-', 'D#', 'E-', 'F-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']
    pitch = note[:2]
    octave = int(note[2])
    return (octave - 1) * 12 + period_seq.index(pitch)


def test_xm_instrument_copy() -> None:
    src = XMSong()
    i1 = src.new_instrument("One")
    s = XMSample()
    s.waveform = array.array('b', [1, 2, 3])
    src.add_sample(i1, s)
    src.get_instrument(i1).set_sample_map([1] * 96)

    dst = XMSong()
    new_idx = dst.copy_instrument_from(src, i1)
    assert_true(new_idx == 1, "copy_instrument_from should return new index")

    mapping = dst.copy_instruments_from(src, [1])
    assert_true(mapping == {1: 2}, "copy_instruments_from mapping mismatch")


def test_xm_instrument_sample_ops(tmp_dir: str) -> None:
    song = XMSong()
    inst_idx = song.new_instrument("Inst")

    smp = XMSample()
    smp.waveform = array.array('b', [1, 2, 3])
    song.add_sample(inst_idx, smp)

    inst = song.get_instrument(inst_idx)
    inst.set_sample_map([1] * 96)
    inst.set_sample_for_note("C-4", 1)

    wav_in = os.path.join(tmp_dir, "xm_sample_in.wav")
    wav_out = os.path.join(tmp_dir, "xm_sample_out.wav")
    with_temp_wav(wav_in)
    sidx = song.load_sample(inst_idx, wav_in)
    song.save_sample(inst_idx, sidx, wav_out)
    assert_true(os.path.isfile(wav_out), "XM save_sample should create wav")

    song.remove_sample(inst_idx, 1)
    song.set_instrument_sample(inst_idx, smp)
    assert_true(song.get_instrument_sample(inst_idx) is not None, "get_instrument_sample should return sample")



def test_xm_instrument_helpers(tmp_dir: str) -> None:
    song = XMSong()
    inst_idx = song.new_instrument("Env")

    smp = XMSample()
    smp.waveform = array.array('b', [1, 2, 3, 4])
    song.add_sample(inst_idx, smp)

    # Song-level envelope helpers
    song.set_volume_envelope(inst_idx, [(0, 64), (10, 32)], sustain=1, loop=(0, 1))
    inst = song.get_instrument(inst_idx)
    assert_true(len(inst.volume_envelope) == 2, "volume envelope should be set")
    assert_true(inst.volume_type & 0x01, "volume envelope should be enabled")

    song.set_panning_envelope(inst_idx, [(0, 32), (10, 48)], sustain=0)
    assert_true(len(inst.panning_envelope) == 2, "panning envelope should be set")
    assert_true(inst.panning_type & 0x01, "panning envelope should be enabled")

    song.clear_volume_envelope(inst_idx)
    assert_true(inst.volume_envelope == [], "volume envelope should be cleared")
    assert_true(inst.volume_type == 0, "volume envelope flags should be cleared")

    song.clear_panning_envelope(inst_idx)
    assert_true(inst.panning_envelope == [], "panning envelope should be cleared")
    assert_true(inst.panning_type == 0, "panning envelope flags should be cleared")

    # Sample map helpers (song-level and instrument-level)
    inst.set_sample_map([1] * 96)
    inst.clear_sample_map()
    assert_true(inst.sample_map == [], "clear_sample_map should empty map")

    inst.set_sample_map([1] * 96)
    song.set_sample_for_note(inst_idx, "C-4", 1)
    assert_true(inst.get_sample_map()[_note_idx("C-4")] == 1, "song-level set_sample_for_note failed")

    # Duplicate instrument/sample
    dup_inst = song.duplicate_instrument(inst_idx)
    assert_true(dup_inst == 2, "duplicate_instrument should create a new instrument")
    assert_true(len(song.get_instrument(dup_inst).samples) == 1, "duplicated instrument should keep samples")

    dup_sample = song.duplicate_sample(inst_idx, 1)
    assert_true(dup_sample == 2, "duplicate_sample should return new sample index")

    # Sample loop helper
    song.set_sample_loop(inst_idx, 1, start=2, length=3, loop_type=1)
    smp2 = song.get_sample(inst_idx, 1)
    assert_true(smp2.repeat_point == 2 and smp2.repeat_len == 3, "set_sample_loop should set points")
    assert_true(smp2.loop_type == 1, "set_sample_loop should set loop type")

    # Raw floats import (8-bit and 16-bit)
    sidx8 = song.load_sample_from_raw(inst_idx, [0.0, 1.0, -1.0], sample_width=1)
    smp8 = song.get_sample(inst_idx, sidx8)
    assert_true(smp8.is_16bit is False, "8-bit raw import should be 8-bit")

    sidx16 = song.load_sample_from_raw(inst_idx, [0.0, 0.5, -0.5], sample_width=2)
    smp16 = song.get_sample(inst_idx, sidx16)
    assert_true(smp16.is_16bit is True, "16-bit raw import should be 16-bit")



def test_xm_instrument_edge_cases() -> None:
    song = XMSong()
    inst_idx = song.new_instrument("Edge")
    smp = XMSample()
    smp.waveform = array.array('b', [1, 2, 3])
    song.add_sample(inst_idx, smp)

    # Invalid note format
    assert_raises_msg(ValueError, "Invalid note name", song.set_sample_for_note, inst_idx, "H-4", 1)
    # Invalid note index range
    assert_raises_msg(ValueError, "Invalid note format", song.set_sample_for_note, inst_idx, "C4", 1)
    assert_raises_msg(IndexError, "Invalid note index", song.set_sample_for_note, inst_idx, 999, 1)
    # Invalid sample index
    assert_raises_msg(IndexError, "Invalid sample index", song.set_sample_for_note, inst_idx, "C-4", 9)

    # Invalid loop type
    assert_raises_msg(ValueError, "Invalid loop_type", song.set_sample_loop, inst_idx, 1, 0, 1, 9)

    # Invalid raw sample width
    assert_raises_msg(ValueError, "sample_width", song.load_sample_from_raw, inst_idx, b"\x00", 3)



def test_xm_validation_helpers() -> None:
    song = XMSong()
    inst_idx = song.new_instrument("Val")
    smp = XMSample()
    smp.waveform = array.array('b', [1, 2, 3, 4])
    smp.loop_type = 1
    smp.repeat_point = 3
    smp.repeat_len = 2
    song.add_sample(inst_idx, smp)

    assert_raises_msg(ValueError, "Loop end", song.validate_sample_loop, inst_idx, 1)
    smp.repeat_point = 1
    smp.repeat_len = 2
    song.validate_sample_loop(inst_idx, 1)

    inst = song.get_instrument(inst_idx)
    inst.set_volume_envelope([(0, 65)])
    assert_raises_msg(ValueError, "value", inst.validate_volume_envelope)
    inst.set_volume_envelope([(1, 10), (0, 20)])
    assert_raises_msg(ValueError, "non-decreasing", inst.validate_volume_envelope)



def test_xm_sample_helpers() -> None:
    song = XMSong()
    inst_idx = song.new_instrument("H")
    smp = XMSample()
    smp.waveform = array.array('b', [1, 2, 3])
    song.add_sample(inst_idx, smp)

    song.set_sample_name(inst_idx, 1, "S")
    song.set_sample_volume(inst_idx, 1, 40)
    song.set_sample_finetune(inst_idx, 1, -5)
    song.set_sample_panning(inst_idx, 1, 200)
    song.set_sample_relative_note(inst_idx, 1, -2)
    song.set_instrument_name(inst_idx, "Inst")
    song.set_instrument_fadeout(inst_idx, 1234)
    song.set_instrument_vibrato(inst_idx, 1, 2, 3, 4)

    smp2 = song.get_sample(inst_idx, 1)
    inst = song.get_instrument(inst_idx)
    assert_true(smp2.name == "S", "set_sample_name failed")
    assert_true(smp2.volume == 40, "set_sample_volume failed")
    assert_true(smp2.finetune == -5, "set_sample_finetune failed")
    assert_true(smp2.panning == 200, "set_sample_panning failed")
    assert_true(smp2.relative_note == -2, "set_sample_relative_note failed")
    assert_true(inst.name == "Inst", "set_instrument_name failed")
    assert_true(inst.volume_fadeout == 1234, "set_instrument_fadeout failed")
    assert_true(inst.vibrato_type == 1 and inst.vibrato_sweep == 2, "set_instrument_vibrato failed")

    assert_raises_msg(ValueError, "volume", song.set_sample_volume, inst_idx, 1, 999)
    assert_raises_msg(ValueError, "finetune", song.set_sample_finetune, inst_idx, 1, 999)
    assert_raises_msg(ValueError, "panning", song.set_sample_panning, inst_idx, 1, 999)
    assert_raises_msg(ValueError, "relative note", song.set_sample_relative_note, inst_idx, 1, 999)
    assert_raises_msg(ValueError, "fadeout", song.set_instrument_fadeout, inst_idx, 999999)


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_xm_instrument_copy()
    test_xm_instrument_sample_ops(tmp_dir)
    test_xm_instrument_helpers(tmp_dir)
    test_xm_instrument_edge_cases()
    test_xm_validation_helpers()
    test_xm_sample_helpers()
    print("OK: test_xm_instruments.py")
