import array
import tempfile
import unittest
from unittest import mock

from nodmod import MODSong
from nodmod import Note
from nodmod import Sample
from nodmod import Song
from nodmod import XMSong


class SongPlanTests(unittest.TestCase):
    def test_song_utility_helpers_and_repr(self):
        mod = MODSong()

        self.assertEqual(Song.parse_effect("A05"), ("A", 0x05))
        self.assertEqual(Song.parse_effect("E60"), ("E6", 0x0))
        self.assertEqual(Song.parse_effect("X2F"), ("X2", 0xF))
        self.assertEqual(Song.parse_effect(""), ("", None))

        self.assertEqual(Song.get_bpm("F7D"), 125)
        self.assertIsNone(Song.get_bpm("F06"))
        self.assertEqual(Song.get_ticks_per_row("F06"), 6)
        self.assertIsNone(Song.get_ticks_per_row("F7D"))

        self.assertTrue(Song.note_in_range("C-4", "B-3", "D-4"))
        self.assertFalse(Song.note_in_range("F-5", "C-4", "E-5"))
        self.assertEqual(Song.transpose_note("C-4", 2), "D-4")
        self.assertEqual(Song.transpose_note("B-3", 1), "C-4")
        self.assertEqual(MODSong.note_to_period("C-4"), 856)
        self.assertEqual(MODSong.period_to_note(856), "C-4")

        info = mod.get_song_info()
        self.assertEqual(info["format"], "mod")
        self.assertEqual(info["n_channels"], 4)
        self.assertEqual(info["n_patterns"], 1)
        self.assertEqual(info["sequence_length"], 1)
        self.assertIn("MODSong", repr(mod))
        self.assertEqual(str(mod), repr(mod))

    def test_mod_effect_setters_cover_normal_and_extended_ranges(self):
        song = MODSong()

        song.set_arpeggio(0, 0, 0, 3, 7)
        song.set_panning(0, 1, 0, 255)
        song.set_sample_offset(0, 2, 0, 0xAB)
        song.set_position_jump(0, 3, 0, 0x12)
        song.set_pattern_break(0, 0, 1, 37)
        song.set_fine_portamento(0, 1, 1, 9)
        song.set_fine_portamento(0, 2, 1, -10)
        song.set_glissando(0, 3, 1, True)
        song.set_vibrato_waveform(0, 0, 2, 2)
        song.set_finetune(0, 1, 2, 15)
        song.set_pattern_loop(0, 2, 2, 0)
        song.set_tremolo_waveform(0, 3, 2, 3)
        song.set_retrigger(0, 0, 3, 7)
        song.set_fine_volume_slide(0, 1, 3, 5)
        song.set_fine_volume_slide(0, 2, 3, -4)
        song.set_note_cut(0, 3, 3, 9)
        song.set_note_delay(0, 0, 4, 6)
        song.set_pattern_delay(0, 1, 4, 2)

        self.assertEqual(song.get_note(0, 0, 0).effect, "037")
        self.assertEqual(song.get_note(0, 0, 1).effect, "8FF")
        self.assertEqual(song.get_note(0, 0, 2).effect, "9AB")
        self.assertEqual(song.get_note(0, 0, 3).effect, "B12")
        self.assertEqual(song.get_note(0, 1, 0).effect, "D37")
        self.assertEqual(song.get_note(0, 1, 1).effect, "E19")
        self.assertEqual(song.get_note(0, 1, 2).effect, "E2A")
        self.assertEqual(song.get_note(0, 1, 3).effect, "E31")
        self.assertEqual(song.get_note(0, 2, 0).effect, "E42")
        self.assertEqual(song.get_note(0, 2, 1).effect, "E5F")
        self.assertEqual(song.get_note(0, 2, 2).effect, "E60")
        self.assertEqual(song.get_note(0, 2, 3).effect, "E73")
        self.assertEqual(song.get_note(0, 3, 0).effect, "E97")
        self.assertEqual(song.get_note(0, 3, 1).effect, "EA5")
        self.assertEqual(song.get_note(0, 3, 2).effect, "EB4")
        self.assertEqual(song.get_note(0, 3, 3).effect, "EC9")
        self.assertEqual(song.get_note(0, 4, 0).effect, "ED6")
        self.assertEqual(song.get_note(0, 4, 1).effect, "EE2")

    def test_xm_effect_and_song_level_setters(self):
        song = XMSong()

        self.assertEqual(len(song.patterns), 1)
        self.assertEqual(song.pattern_seq, [0])

        song.set_default_speed(3)
        song.set_default_tempo(140)
        song.set_song_restart(0)
        song.set_linear_frequency(True)
        song.set_n_channels(12)

        self.assertEqual(song.default_speed, 3)
        self.assertEqual(song.default_tempo, 140)
        self.assertTrue(song.uses_linear_frequency)
        self.assertEqual(song.n_channels, 12)
        self.assertEqual(song.patterns[0].n_channels, 12)

        with self.assertRaises(ValueError):
            song.set_default_speed(0)
        with self.assertRaises(ValueError):
            song.set_default_tempo(31)
        with self.assertRaises(IndexError):
            song.set_song_restart(1)

        song.set_arpeggio(0, 0, 0, 1, 2)
        song.set_panning(0, 1, 0, 0x80)
        song.set_sample_offset(0, 2, 0, 0x40)
        song.set_position_jump(0, 3, 0, 0x04)
        song.set_volume_slide(0, 4, 0, 5)
        song.set_portamento(0, 5, 0, -6)
        song.set_tone_portamento(0, 6, 0, 0x20)
        song.set_tone_portamento_slide(0, 7, 0, -3)
        song.set_vibrato(0, 8, 0, 2, 9)
        song.set_vibrato_slide(0, 9, 0, 4)
        song.set_tremolo(0, 10, 0, 3, 8)
        song.set_pattern_break(0, 11, 0, 45)
        song.set_fine_portamento(0, 0, 1, 7)
        song.set_global_volume(0, 1, 1, 64)
        song.set_global_volume_slide(0, 2, 1, -2)
        song.set_key_off(0, 3, 1, 32)
        song.set_envelope_position(0, 4, 1, 0x7F)
        song.set_panning_slide(0, 5, 1, 3)
        song.set_retrigger_volume_slide(0, 6, 1, 0xA, 0x3)
        song.set_tremor(0, 7, 1, 4, 5)
        song.set_extra_fine_portamento(0, 8, 1, -9)

        self.assertEqual(song.get_note(0, 0, 0).effect, "012")
        self.assertEqual(song.get_note(0, 0, 1).effect, "880")
        self.assertEqual(song.get_note(0, 0, 2).effect, "940")
        self.assertEqual(song.get_note(0, 0, 3).effect, "B04")
        self.assertEqual(song.get_note(0, 0, 4).effect, "A50")
        self.assertEqual(song.get_note(0, 0, 5).effect, "206")
        self.assertEqual(song.get_note(0, 0, 6).effect, "320")
        self.assertEqual(song.get_note(0, 0, 7).effect, "503")
        self.assertEqual(song.get_note(0, 0, 8).effect, "429")
        self.assertEqual(song.get_note(0, 0, 9).effect, "640")
        self.assertEqual(song.get_note(0, 0, 10).effect, "738")
        self.assertEqual(song.get_note(0, 0, 11).effect, "D45")
        self.assertEqual(song.get_note(0, 1, 0).effect, "E17")
        self.assertEqual(song.get_note(0, 1, 1).effect, "G40")
        self.assertEqual(song.get_note(0, 1, 2).effect, "H02")
        self.assertEqual(song.get_note(0, 1, 3).effect, "K20")
        self.assertEqual(song.get_note(0, 1, 4).effect, "L7F")
        self.assertEqual(song.get_note(0, 1, 5).effect, "P30")
        self.assertEqual(song.get_note(0, 1, 6).effect, "RA3")
        self.assertEqual(song.get_note(0, 1, 7).effect, "T45")
        self.assertEqual(song.get_note(0, 1, 8).effect, "X29")

    def test_pattern_row_and_note_operations_preserve_structure(self):
        song = MODSong()
        song.set_effect(0, 0, 0, "B02")
        song.set_effect(0, 1, 0, "C20")
        song.set_effect(0, 2, 0, "D12")
        song.set_effect(0, 3, 0, "E60")
        song.set_note(0, 0, 0, 1, "C-4")
        song.set_note(0, 1, 0, 2, "D-4")
        song.set_note(0, 2, 0, 3, "E-4")
        song.set_note(0, 3, 0, 4, "F-4")

        self.assertEqual(song.get_note(0, 0, 0).effect, "B02")
        self.assertEqual(song.get_note(0, 0, 1).effect, "C20")
        self.assertEqual(song.get_note(0, 0, 2).effect, "D12")
        self.assertEqual(song.get_note(0, 0, 3).effect, "E60")

        song.clear_note(0, 0, 0)
        self.assertTrue(song.get_note(0, 0, 0).is_empty())

        song.set_note(0, 0, 1, 1, "C-4", "A0F")
        song.set_note(0, 1, 1, 2, "D-4", "B01")
        song.copy_row(0, 1, 0, 2)
        self.assertEqual(repr(song.get_note(0, 2, 0)), repr(song.get_note(0, 1, 0)))
        self.assertEqual(repr(song.get_note(0, 2, 1)), repr(song.get_note(0, 1, 1)))

        song.clear_row(0, 2)
        for channel in range(song.CHANNELS):
            self.assertTrue(song.get_note(0, 2, channel).is_empty())

        song.set_note(0, 0, 3, 5, "G-4")
        song.shift_pattern(0, 2)
        self.assertTrue(song.get_note(0, 0, 0).is_empty())
        self.assertEqual(song.get_note(0, 5, 0).period, "G-4")

        song.set_note(0, 2, 1, 6, "A-4", "A01")
        song.copy_channel_data(0, 1, 0, 2)
        self.assertEqual(repr(song.get_note(0, 2, 2)), repr(song.get_note(0, 2, 1)))

        song.swap_channels(1, 2)
        self.assertEqual(song.get_note(0, 2, 1).period, "D-4")

        self.assertFalse(song.is_pattern_empty(0))
        song.clear_pattern(0)
        self.assertTrue(song.is_pattern_empty(0))

    def test_pattern_copy_and_used_references(self):
        src = MODSong()
        src.set_note(0, 0, 0, 7, "C-4", "A0F")

        dst = MODSong()
        new_pattern = dst.copy_pattern_from(src, 0)
        self.assertEqual(new_pattern, 1)
        self.assertEqual(dst.pattern_seq, [0, 1])
        self.assertEqual(dst.patterns[1].data[0][0].period, "C-4")

        dst.patterns.append(dst.patterns[0].__class__(64, 4))
        dst.patterns[2].data[1][1] = Note(11, "D-4", "")
        self.assertEqual(dst.get_used_patterns(), [0, 1])
        self.assertEqual(dst.get_used_samples(), [7, 11])

        xm = XMSong()
        xm.new_instrument("lead")
        xm.new_instrument("bass")
        xm.set_note(0, 0, 0, 1, "C-4")
        xm.patterns.append(xm.patterns[0].__class__(64, xm.n_channels))
        for channel in range(xm.n_channels):
            for row in range(64):
                xm.patterns[1].data[channel][row] = type(xm.get_note(0, 0, 0))()
        xm.patterns[1].data[0][1].instrument_idx = 2
        self.assertEqual(xm.get_used_instruments(), [1, 2])

    def test_mod_sample_count_stays_in_sync(self):
        song = MODSong()

        smp = Sample()
        smp.waveform = array.array("b", [1, -1, 2, -2])
        song.set_sample(1, smp)
        self.assertEqual(song.n_actual_samples, 1)

        song.copy_sample_from(song, 1, 2)
        self.assertEqual(song.n_actual_samples, 2)

        song.remove_sample(2)
        self.assertEqual(song.n_actual_samples, 1)

        song.keep_sample(1)
        self.assertEqual(song.n_actual_samples, 1)

        song.remove_sample(1)
        self.assertEqual(song.n_actual_samples, 0)

    def test_xm_mute_preserves_global_effects_consistently(self):
        song = XMSong()
        song.set_volume(0, 0, 0, 32)
        song.set_bpm(0, 0, 1, 125)
        song.set_global_volume(0, 0, 2, 64)
        song.set_panning_slide(0, 0, 3, 3)
        song.set_note(0, 0, 4, 2, "C-4", "A0F", "v", 48)

        song.mute_channel(0)

        self.assertEqual(song.get_note(0, 0, 0).effect, "C20")
        self.assertEqual(song.get_note(0, 1, 0).effect, "F7D")
        self.assertEqual(song.get_note(0, 2, 0).effect, "G40")
        self.assertEqual(song.get_note(0, 3, 0).effect, "")
        self.assertTrue(song.get_note(0, 4, 0).is_empty())

    def test_render_accepts_configurable_channel_count(self):
        song = MODSong()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = f"{tmpdir}/render.wav"

            def fake_save(path, verbose=False):
                with open(path, "wb") as file:
                    file.write(b"mod")

            song.save = fake_save

            with mock.patch("subprocess.run") as run_mock, mock.patch("shutil.move"):
                song.render(out_file, verbose=False, cleanup=False, channels=2)

            args = run_mock.call_args.args[0]
            self.assertEqual(args[:5], ["openmpt123", out_file[:-4] + ".mod", "-q", "--channels", "2"])


if __name__ == "__main__":
    unittest.main()