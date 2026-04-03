from __future__ import annotations

import array

from nodmod import XMSong
from nodmod.types import XMSample


def build_source_song() -> XMSong:
    src = XMSong()
    inst_idx = src.new_instrument("SrcInst")
    inst = src.get_instrument(inst_idx)

    # Envelopes with flags
    inst.set_volume_envelope([(0, 0), (10, 64)], sustain=1, loop=(0, 1), enabled=True, raw_type=0x07)
    inst.set_panning_envelope([(0, 32), (10, 32)], sustain=0, loop=(0, 1), enabled=True, raw_type=0x05)

    # Sample
    smp = XMSample()
    smp.name = "S1"
    smp.waveform = array.array('b', [1, 2, 3, 4])
    src.add_sample(inst_idx, smp)

    # Sample map
    inst.set_sample_map([1] * 96)

    return src


def test_copy_instrument_from():
    src = build_source_song()
    dst = XMSong()

    new_idx = dst.copy_instrument_from(src, 1)
    assert new_idx == 1

    inst = dst.get_instrument(new_idx)
    assert inst.name == "SrcInst"
    assert inst.volume_type == 0x07
    assert inst.panning_type == 0x05
    assert len(inst.samples) == 1
    assert inst.samples[0].name == "S1"
    assert inst.get_sample_map()[0] == 1


def test_copy_instruments_from():
    src = build_source_song()
    # add a second instrument
    inst2 = src.new_instrument("Second")
    smp2 = XMSample()
    smp2.waveform = array.array('b', [9])
    src.add_sample(inst2, smp2)
    src.get_instrument(inst2).set_sample_map([1] * 96)

    dst = XMSong()
    mapping = dst.copy_instruments_from(src, [1, 2])

    assert mapping == {1: 1, 2: 2}
    assert dst.get_instrument(1).name == "SrcInst"
    assert dst.get_instrument(2).name == "Second"


def main() -> int:
    test_copy_instrument_from()
    test_copy_instruments_from()
    print("OK: XM instrument copy tests passed")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_xm_copy_instrument.py")
    raise SystemExit(rc)
