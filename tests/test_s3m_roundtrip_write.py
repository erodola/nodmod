from __future__ import annotations

import array
import os
import tempfile

from nodmod import S3MSong
from nodmod.types import S3MSample
from .test_helpers import assert_true, compare_s3m_songs, pick_files


def _make_generated_song() -> S3MSong:
    song = S3MSong()
    song.songname = "Generated S3M"
    song.n_channels = 2
    song.global_volume = 48
    song.initial_speed = 3
    song.initial_tempo = 150
    song.master_volume = 0xB0
    song.default_pan_flag = 252
    song.default_panning = [0x20] * 32

    sample1 = S3MSample()
    sample1.instrument_type = 1
    sample1.name = "Lead"
    sample1.filename = "LEAD.SMP"
    sample1.volume = 32
    sample1.c2spd = 8363
    sample1.waveform = array.array('b', [-128, -64, 0, 64, 127])
    sample1.repeat_point = 1
    sample1.repeat_len = 3
    song.set_sample(1, sample1)

    sample2 = S3MSample()
    sample2.instrument_type = 1
    sample2.name = "Bass16"
    sample2.filename = "BASS.SMP"
    sample2.volume = 40
    sample2.c2spd = 9000
    sample2.is_16bit = True
    sample2.waveform = array.array('h', [-32768, -1000, 0, 1000, 32767])
    song.set_sample(2, sample2)

    song.set_note(0, 0, 0, 1, "C-4", effect="A03", volume=20)
    song.set_note(0, 1, 1, 2, "D#4", effect="T96", volume=30)
    song.set_note(0, 0, 2, 0, "off", effect="B00")

    song.add_pattern()
    song.set_note(1, 1, 0, 2, "F-5", effect="C12", volume=40)
    song.set_sequence([0, 1])
    return song


def test_s3m_generated_roundtrip() -> None:
    song = _make_generated_song()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "generated.s3m")
        song.save(path, verbose=False)
        assert_true(os.path.isfile(path), "Generated S3M save should create file")

        loaded = S3MSong()
        loaded.load(path, verbose=False)
        compare_s3m_songs(song, loaded)


def test_s3m_corpus_semantic_roundtrip() -> None:
    root = r"G:\My Drive\Moduli"
    if not os.path.isdir(root):
        return
    files = sorted(pick_files(root, ".s3m", 5))
    assert_true(len(files) > 0, "No S3M files found for roundtrip tests")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, source in enumerate(files, start=1):
            song = S3MSong()
            song.load(source, verbose=False)
            out_path = os.path.join(tmp_dir, f"roundtrip_{idx}.s3m")
            song.save(out_path, verbose=False)
            reloaded = S3MSong()
            reloaded.load(out_path, verbose=False)
            compare_s3m_songs(song, reloaded)


if __name__ == "__main__":
    test_s3m_generated_roundtrip()
    test_s3m_corpus_semantic_roundtrip()
    print("OK: test_s3m_roundtrip_write.py")