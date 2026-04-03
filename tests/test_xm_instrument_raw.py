from __future__ import annotations

from nodmod import XMSong


def main() -> int:
    song = XMSong()
    inst_idx = song.new_instrument("raw-test")

    # 8-bit raw sample
    raw8 = bytes([0, 1, 255, 128])  # signed: 0,1,-1,-128
    s_idx8 = song.load_sample_from_raw(inst_idx, raw8, sample_width=1)
    smp8 = song.get_sample(inst_idx, s_idx8)
    assert smp8.is_16bit is False
    assert len(smp8.waveform) == 4

    # 16-bit raw sample (little-endian)
    raw16 = (0).to_bytes(2, 'little', signed=True) + (1000).to_bytes(2, 'little', signed=True)
    s_idx16 = song.load_sample_from_raw(inst_idx, raw16, sample_width=2)
    smp16 = song.get_sample(inst_idx, s_idx16)
    assert smp16.is_16bit is True
    assert len(smp16.waveform) == 2

    # map sample across range
    song.set_sample_map_range(inst_idx, s_idx16, "C-2", "C-3")
    inst = song.get_instrument(inst_idx)
    assert inst.sample_map[12] == s_idx16 - 1
    assert inst.sample_map[24] == s_idx16 - 1

    # map sample across all notes
    song.set_sample_map_all(inst_idx, s_idx8)
    assert all(v == s_idx8 - 1 for v in inst.sample_map)

    # copy sample to another song
    song2 = XMSong()
    inst2 = song2.new_instrument("dst")
    new_idx = song2.copy_sample_from(song, inst_idx, s_idx8, inst2)
    smp2 = song2.get_sample(inst2, new_idx)
    assert smp2.is_16bit is False
    assert len(smp2.waveform) == len(smp8.waveform)

    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_xm_instrument_raw.py")
    raise SystemExit(rc)

