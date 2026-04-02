import array
import os
import random
from nodmod import MODSong, XMSong
from nodmod.types import XMNote, XMSample
from test_helpers import assert_true, assert_raises, assert_raises_msg


def test_xm_basic_ops() -> None:
    song = XMSong()
    song.default_tempo = 121
    song.n_channels = 2
    song.add_pattern(12)
    inst_idx = song.new_instrument("Test")

    for i, note in enumerate(["C-4", "C#4", "D-4"]):
        song.set_note(0, 0, i, inst_idx, note, effect="")
        song.set_note(0, 1, i, inst_idx, note, effect="", vol_cmd="v", vol_val=40)

    note = song.get_note(0, 1, 0)
    assert_true(isinstance(note, XMNote), "get_note should return XMNote")


def test_xm_patterns_edge_cases() -> None:
    song = XMSong()
    song.n_channels = 1
    assert_raises(ValueError, song.add_pattern, 0)
    assert_raises(ValueError, song.add_pattern, 257)

    song.add_pattern(1)
    song.add_pattern(256)


def test_xm_sample_loop_edges() -> None:
    song = XMSong()
    inst_idx = song.new_instrument("Loop")
    smp = XMSample()
    smp.waveform = array.array('b', [1, 2, 3, 4])
    smp.repeat_point = 0
    smp.repeat_len = 4
    smp.loop_type = 1
    song.add_sample(inst_idx, smp)
    song.get_instrument(inst_idx).set_sample_map([1] * 96)


def test_xm_random_edit_stress() -> None:
    rng = random.Random(0)
    song = XMSong()
    song.n_channels = 4
    song.add_pattern(64)
    inst_idx = song.new_instrument("Rand")
    smp = XMSample()
    smp.waveform = array.array('b', [1, 2, 3, 4])
    song.add_sample(inst_idx, smp)
    song.get_instrument(inst_idx).set_sample_map([1] * 96)

    notes = ["C-4", "C#4", "D-4", "D#4", "E-4", "F-4", "F#4", "G-4", "", "off"]
    snapshot = {}
    for r in range(64):
        for c in range(4):
            if rng.random() < 0.4:
                note = rng.choice(notes)
                eff = rng.choice(["", "F06", "A01", "B00", "E60"])
                if rng.random() < 0.5:
                    vol = rng.randint(0, 64)
                    song.set_note(0, c, r, inst_idx, note, eff, vol_cmd="v", vol_val=vol)
                    snapshot[(r, c)] = (note, eff, "v", vol)
                else:
                    song.set_note(0, c, r, inst_idx, note, eff)
                    snapshot[(r, c)] = (note, eff, "", -1)

    for (r, c), (note, eff, vcmd, vval) in snapshot.items():
        n = song.get_note(0, r, c)
        assert_true(n.period == note, "XM random period mismatch")
        assert_true(n.effect == eff, "XM random effect mismatch")
        assert_true(getattr(n, "vol_cmd", "") == vcmd, "XM random vol_cmd mismatch")
        assert_true(getattr(n, "vol_val", -1) == vval, "XM random vol_val mismatch")


def test_xm_instrument_map_stress() -> None:
    song = XMSong()
    inst_idx = song.new_instrument("Map")
    for i in range(3):
        smp = XMSample()
        smp.waveform = array.array('b', [i, i + 1, i + 2, i + 3])
        song.add_sample(inst_idx, smp)

    inst = song.get_instrument(inst_idx)
    map_1based = [(i % 3) + 1 for i in range(96)]
    inst.set_sample_map(map_1based)
    assert_true(inst.get_sample_map()[0] == 1, "sample map start mismatch")
    assert_true(inst.get_sample_map()[1] == 2, "sample map cycle mismatch")


def test_pattern_sequence_stress() -> None:
    mod = MODSong()
    mod.add_pattern()
    mod.add_pattern()
    mod.pattern_seq = [2, 1, 0]
    _ = mod.get_effective_row_count(0)

    xm = XMSong()
    xm.n_channels = 1
    xm.add_pattern(4)
    xm.add_pattern(4)
    xm.add_pattern(4)
    xm.pattern_seq = [2, 0, 1]
    xm.new_instrument("Inst")
    xm.set_note(0, 0, 0, 1, "C-4", "")


def test_xm_misc(tmp_dir: str) -> None:
    song = XMSong()
    song.n_channels = 1
    song.add_pattern(4)
    song.new_instrument("Inst")

    assert_true(song.uses_linear_frequency in (True, False), "uses_linear_frequency should be bool")

    ascii_path = os.path.join(tmp_dir, "xm_ascii.txt")
    song.save_ascii(ascii_path, verbose=False)
    assert_true(os.path.isfile(ascii_path), "XM save_ascii should create file")

    xm_path = os.path.join(tmp_dir, "xm_test.xm")
    song.save(xm_path, verbose=False)
    assert_true(os.path.isfile(xm_path), "XM save should create file")

    stamps = song.timestamp()
    assert_true(isinstance(stamps, list) and len(stamps) > 0, "XM timestamp should return data")
    n_rows = song.get_effective_row_count(0)
    assert_true(isinstance(n_rows, int) and n_rows >= 1, "XM get_effective_row_count should return rows")

    clone = song.copy()
    assert_true(isinstance(clone, XMSong), "copy should return XMSong")



def test_xm_channel_ops() -> None:
    song = XMSong()
    song.n_channels = 2
    song.add_pattern(4)

    song.add_channel(1)
    assert_true(song.n_channels == 3, "add_channel should increase channel count")
    assert_true(song.patterns[0].n_channels == 3, "pattern channel count should update")

    song.clear_channel(1)
    note = song.get_note(0, 0, 1)
    assert_true(note.is_empty(), "clear_channel should empty notes")

    song.remove_channel(1)
    assert_true(song.n_channels == 2, "remove_channel should decrease channel count")

    assert_raises_msg(ValueError, "channel count", song.add_channel, 0)
    assert_raises_msg(IndexError, "Invalid channel index", song.clear_channel, 9)
    assert_raises_msg(IndexError, "Invalid channel index", song.remove_channel, 9)
    song.remove_channel(0)
    assert_raises_msg(ValueError, "last channel", song.remove_channel, 0)



def test_xm_mute_channel_global_effects() -> None:
    song = XMSong()
    song.n_channels = 1
    song.add_pattern(4)
    song.set_effect(0, 0, 0, "F06")
    song.set_effect(0, 0, 1, "B01")
    song.set_effect(0, 0, 2, "D02")
    song.set_effect(0, 0, 3, "E60")
    song.mute_channel(0)
    assert_true(song.get_note(0, 0, 0).effect == "F06", "XM mute_channel should preserve Fxx")
    assert_true(song.get_note(0, 1, 0).effect == "B01", "XM mute_channel should preserve Bxx")
    assert_true(song.get_note(0, 2, 0).effect == "D02", "XM mute_channel should preserve Dxx")
    assert_true(song.get_note(0, 3, 0).effect == "E60", "XM mute_channel should preserve Exx")



def test_xm_pattern_ops() -> None:
    song = XMSong()
    song.n_channels = 1

    sequence_len = len(song.pattern_seq)
    song.add_pattern(4)
    song.insert_pattern(0, after=False)
    assert_true(len(song.pattern_seq) == sequence_len + 2, "insert_pattern should insert into sequence")

    song.set_note(0, 0, 0, 0, "C-4", "")

    song.resize_pattern(0, 2)
    assert_true(song.patterns[song.pattern_seq[0]].n_rows == 2, "resize_pattern should shrink")
    song.resize_pattern(0, 4)
    assert_true(song.patterns[song.pattern_seq[0]].n_rows == 4, "resize_pattern should extend")
    song.clear_pattern(0)
    assert_true(song.get_note(0, 0, 0).is_empty(), "clear_pattern should empty notes")

    song.remove_pattern(0)
    assert_true(len(song.pattern_seq) == sequence_len + 1, "remove_pattern should remove from sequence")

    song.add_pattern(4)
    song.set_sequence([1, 0])
    assert_true(song.pattern_seq == [1, 0], "set_sequence should update sequence")
    assert_raises_msg(ValueError, "Pattern sequence too long", song.set_sequence, [0] * 257)
    assert_raises(IndexError, song.set_sequence, [99])

    song.add_pattern(4)
    song.add_to_sequence(1)
    assert_true(song.pattern_seq[-1] == 1, "add_to_sequence should append")
    song.add_to_sequence(0, sequence_position=0)
    assert_true(song.pattern_seq[0] == 0, "add_to_sequence should insert at pos")
    song.set_sequence([0] * 256)
    assert_raises_msg(ValueError, "Pattern sequence too long", song.add_to_sequence, 0, None)


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_xm_basic_ops()
    test_xm_patterns_edge_cases()
    test_xm_sample_loop_edges()
    test_xm_random_edit_stress()
    test_xm_instrument_map_stress()
    test_pattern_sequence_stress()
    test_xm_misc(tmp_dir)
    test_xm_channel_ops()
    test_xm_mute_channel_global_effects()
    test_xm_pattern_ops()
    print("OK: test_xm_basic.py")
