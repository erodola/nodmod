from __future__ import annotations

from nodmod import MODSong, Sample


def test_mod_sample_api_load_list_set_copy() -> None:
    song = MODSong()

    raw = [0.0, 0.5, -0.5, 1.0, -1.0]
    idx, smp = song.load_sample_from_raw(raw)
    assert idx > 0
    assert len(smp.waveform) == len(raw)

    samples = song.list_samples()
    assert len(samples) == 31
    assert samples[idx - 1] is smp

    new_smp = Sample()
    new_smp.waveform = smp.waveform
    song.set_sample(idx, new_smp)
    assert song.get_sample(idx).waveform == new_smp.waveform

    song2 = MODSong()
    dst_idx = song2.copy_sample_from(song, idx, None)
    assert dst_idx > 0
    assert len(song2.get_sample(dst_idx).waveform) == len(smp.waveform)


def test_mod_sample_api_copy_sample_from_explicit_destination() -> None:
    src = MODSong()
    idx, smp = src.load_sample_from_raw([0.25, -0.25, 0.0])
    assert idx == 1

    dst = MODSong()
    dst_idx = dst.copy_sample_from(src, idx, 5)
    assert dst_idx == 5
    assert len(dst.get_sample(5).waveform) == len(smp.waveform)

