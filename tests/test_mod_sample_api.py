from __future__ import annotations

from nodmod import MODSong, Sample


def main() -> int:
    song = MODSong()

    # add sample from raw
    raw = [0.0, 0.5, -0.5, 1.0, -1.0]
    idx, smp = song.load_sample_from_raw(raw)
    assert idx > 0
    assert len(smp.waveform) == len(raw)

    # list samples
    samples = song.list_samples()
    assert len(samples) == 31

    # set sample
    new_smp = Sample()
    new_smp.waveform = smp.waveform
    song.set_sample(idx, new_smp)
    assert song.get_sample(idx).waveform == new_smp.waveform

    # copy sample between songs
    song2 = MODSong()
    dst_idx = song2.copy_sample_from(song, idx, None)
    assert dst_idx > 0
    assert len(song2.get_sample(dst_idx).waveform) == len(smp.waveform)

    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_mod_sample_api.py")
    raise SystemExit(rc)

