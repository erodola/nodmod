import os
import random
import shutil
from nodmod import MODSong
from .test_helpers import assert_true, assert_raises, assert_raises_msg


def test_mod_note_helpers() -> None:
    note_raw = bytearray([0x13, 0x58, 0x2B, 0x01])
    assert_true(MODSong.get_sample_from_note(note_raw) == 18, "get_sample_from_note decoded wrong sample")
    assert_true(MODSong.get_period_from_note(note_raw) == "C-4", "get_period_from_note decoded wrong period")
    assert_true(MODSong.get_effect_from_note(note_raw) == (0x0B, 0x01), "get_effect_from_note decoded wrong effect")


def test_mod_basic_ops() -> None:
    song = MODSong()

    song.set_note(0, 0, 0, 1, "C-4", "F06")
    note = song.get_note(0, 0, 0)
    assert_true(note.instrument_idx == 1, "MOD set_note instrument")
    assert_true(note.period == "C-4", "MOD set_note period")
    assert_true(note.effect == "F06", "MOD set_note effect")

    song.set_effect(0, 0, 0, "B01")
    note = song.get_note(0, 0, 0)
    assert_true(note.effect == "B01", "MOD set_effect failed")

    song.clear_channel(0)
    assert_true(song.get_note(0, 0, 0).is_empty(), "clear_channel should empty notes")

    song.set_effect(0, 0, 0, "F06")
    song.mute_channel(0)
    note = song.get_note(0, 0, 0)
    assert_true(note.effect == "F06", "mute_channel should preserve global effect")

    song.add_pattern()
    assert_true(len(song.patterns) >= 2, "add_pattern failed")
    song.duplicate_pattern(0)
    assert_true(len(song.patterns) >= 3, "duplicate_pattern failed")

    song.remove_patterns_after(0)
    assert_true(len(song.pattern_seq) == 1, "remove_patterns_after failed")

    song.add_pattern()
    song.remove_pattern(0)
    assert_true(len(song.pattern_seq) == 1, "remove_pattern failed")

    song.keep_pattern(0)
    assert_true(len(song.pattern_seq) == 1, "keep_pattern failed")

    song.remove_all_patterns(sequence_only=True)
    assert_true(len(song.pattern_seq) == 0, "remove_all_patterns (seq only) failed")


def test_mod_row_count_and_duration() -> None:
    song = MODSong()
    song.set_effect(0, 0, 0, "E60")
    song.set_effect(0, 0, 1, "E62")
    n_rows = song.get_effective_row_count(0)
    assert_true(n_rows >= 64, "get_effective_row_count should be >= original rows")


def test_mod_effect_setters() -> None:
    song = MODSong()

    song.set_bpm(0, 0, 0, 125)
    song.set_ticks_per_row(0, 0, 1, 6)
    song.set_portamento(0, 0, 2, 5)
    song.set_tone_portamento(0, 0, 3, 3)
    song.set_tone_portamento_slide(0, 0, 4, 2)
    song.set_tremolo(0, 0, 5, 3, 4)
    song.set_vibrato(0, 0, 6, 3, 4)
    song.set_vibrato_slide(0, 0, 7, 2)
    song.set_volume(0, 0, 8, 32)
    song.set_volume_slide(0, 0, 9, 2)


def test_mod_misc(tmp_dir: str) -> None:
    song = MODSong()
    ascii_path = os.path.join(tmp_dir, "mod_ascii.txt")
    song.save_ascii(ascii_path, verbose=False)
    assert_true(os.path.isfile(ascii_path), "save_ascii should create file")

    mod_path = os.path.join(tmp_dir, "mod_test.mod")
    song.save(mod_path, verbose=False)
    assert_true(os.path.isfile(mod_path), "save should create mod")

    stamps = song.timestamp()
    assert_true(len(stamps) > 0, "timestamp should return data")
    duration = song.get_song_duration()
    assert_true(duration > 0, "get_song_duration should be > 0")

    wav_out = os.path.join(tmp_dir, "mod_render.wav")
    has_renderer = shutil.which("openmpt123") or shutil.which("ffmpeg")
    try:
        song.render(wav_out, verbose=False)
        assert_true(has_renderer is not None, "render should not succeed without renderer")
        assert_true(os.path.isfile(wav_out), "render should create wav")
    except FileNotFoundError:
        assert_true(has_renderer is None, "render raised but renderer exists")

    clone = song.copy()
    assert_true(isinstance(clone, MODSong), "copy should return MODSong")


def test_mod_save_excludes_unreferenced_patterns(tmp_dir: str) -> None:
    song = MODSong()
    song.set_note(0, 0, 0, 1, "C-4", "")
    song.add_pattern()
    song.add_pattern()
    song.patterns[2].data[0][0] = song.patterns[2].data[0][0].__class__(2, "D-4", "B01")
    song.set_sequence([0])

    assert_true(len(song.patterns) == 3, "test setup should keep unreferenced patterns in the pool")

    mod_path = os.path.join(tmp_dir, "mod_unreferenced_patterns.mod")
    song.save(mod_path, verbose=False)

    loaded = MODSong()
    loaded.load(mod_path, verbose=False)

    assert_true(loaded.pattern_seq == [0], "reloaded MOD sequence should keep only referenced patterns")
    assert_true(len(loaded.patterns) == 1, "reloaded MOD should exclude unreferenced patterns")
    assert_true(loaded.get_note(0, 0, 0).period == "C-4", "reloaded referenced pattern should be preserved")


def test_mod_edge_cases() -> None:
    song = MODSong()

    assert_raises(IndexError, song.get_note, 0, -1, 0)
    assert_raises(IndexError, song.get_note, 0, 0, -1)

    assert_raises(IndexError, song.get_sample, 0)
    assert_raises(IndexError, song.get_sample, 32)


def test_mod_multi_channel_patterns() -> None:
    song = MODSong()
    for c in range(4):
        song.set_note(0, c, 0, c + 1, "C-4", "")
    for c in range(4):
        note = song.get_note(0, 0, c)
        assert_true(note.instrument_idx == c + 1, "multi-channel note mismatch")


def test_mod_random_edit_stress() -> None:
    rng = random.Random(0)
    song = MODSong()
    snapshot = {}
    for r in range(64):
        for c in range(4):
            if rng.random() < 0.3:
                inst = rng.randint(1, 4)
                note = rng.choice(["C-4", "D-4", "E-4", "F-4", "", "G-4"])
                effect = rng.choice(["", "F06", "A01", "B00"])
                song.set_note(0, c, r, inst, note, effect)
                snapshot[(r, c)] = (inst, note, effect)
    # verify snapshot
    for (r, c), (inst, note, effect) in snapshot.items():
        n = song.get_note(0, r, c)
        assert_true(n.instrument_idx == inst, "random stress instrument mismatch")
        assert_true(n.period == note, "random stress period mismatch")
        assert_true(n.effect == effect, "random stress effect mismatch")



def test_mod_channel_ops() -> None:
    song = MODSong()
    assert_raises_msg(NotImplementedError, "fixed 4 channels", song.add_channel, 1)
    assert_raises_msg(NotImplementedError, "fixed 4 channels", song.remove_channel, 0)
    assert_raises_msg(ValueError, "channel count", song.add_channel, 0)



def test_mod_mute_channel_global_effects() -> None:
    song = MODSong()
    song.set_effect(0, 0, 0, "F06")
    song.set_effect(0, 0, 1, "B01")
    song.set_effect(0, 0, 2, "D02")
    song.set_effect(0, 0, 3, "C20")
    song.set_effect(0, 0, 4, "E60")
    song.mute_channel(0)
    assert_true(song.get_note(0, 0, 0).effect == "F06", "mute_channel should preserve Fxx")
    assert_true(song.get_note(0, 1, 0).effect == "B01", "mute_channel should preserve Bxx")
    assert_true(song.get_note(0, 2, 0).effect == "D02", "mute_channel should preserve Dxx")
    assert_true(song.get_note(0, 3, 0).effect == "C20", "mute_channel should preserve Cxx")
    assert_true(song.get_note(0, 4, 0).effect == "E60", "mute_channel should preserve Exx")



def test_mod_pattern_ops() -> None:
    song = MODSong()
    assert_raises_msg(ValueError, "fixed 64 rows", song.resize_pattern, 0, 32)
    song.insert_pattern(0, after=True)
    assert_true(len(song.pattern_seq) == 2, "insert_pattern should insert into sequence")
    song.remove_pattern(0)
    assert_true(len(song.pattern_seq) == 1, "remove_pattern should remove from sequence")

    song.add_pattern()
    song.set_sequence([1, 0, 1])
    assert_true(song.pattern_seq == [1, 0, 1], "set_sequence should update sequence")
    assert_raises_msg(ValueError, "Pattern sequence too long", song.set_sequence, [0] * 129)
    assert_raises(IndexError, song.set_sequence, [99])

    song.add_pattern()
    song.add_to_sequence(1)
    assert_true(song.pattern_seq[-1] == 1, "add_to_sequence should append")
    song.add_to_sequence(0, sequence_position=0)
    assert_true(song.pattern_seq[0] == 0, "add_to_sequence should insert at pos")
    song.set_sequence([0] * 128)
    assert_raises_msg(ValueError, "Pattern sequence too long", song.add_to_sequence, 0, None)



def test_mod_validation_helpers() -> None:
    song = MODSong()
    idx, smp = song.load_sample_from_raw([0.0, 0.1, -0.1, 0.0])
    smp.repeat_point = 3
    smp.repeat_len = 4
    assert_raises_msg(ValueError, "Loop end", song.validate_sample_loop, idx)



def test_mod_sample_helpers() -> None:
    song = MODSong()
    idx, _smp = song.load_sample_from_raw([0.0, 0.1, -0.1, 0.0])
    song.set_sample_name(idx, "S")
    song.set_sample_volume(idx, 32)
    song.set_sample_finetune(idx, 3)
    song.set_sample_loop(idx, 1, 2)
    smp = song.get_sample(idx)
    assert_true(smp.name == "S", "set_sample_name failed")
    assert_true(smp.volume == 32, "set_sample_volume failed")
    assert_true(smp.finetune == 3, "set_sample_finetune failed")
    assert_true(smp.repeat_point == 1 and smp.repeat_len == 2, "set_sample_loop failed")
    assert_raises_msg(ValueError, "volume", song.set_sample_volume, idx, 999)
    assert_raises_msg(ValueError, "finetune", song.set_sample_finetune, idx, 99)


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_mod_note_helpers()
    test_mod_basic_ops()
    test_mod_row_count_and_duration()
    test_mod_effect_setters()
    test_mod_misc(tmp_dir)
    test_mod_save_excludes_unreferenced_patterns(tmp_dir)
    test_mod_edge_cases()
    test_mod_multi_channel_patterns()
    test_mod_random_edit_stress()
    test_mod_channel_ops()
    test_mod_mute_channel_global_effects()
    test_mod_pattern_ops()
    test_mod_validation_helpers()
    test_mod_sample_helpers()
    print("OK: test_mod_basic.py")
