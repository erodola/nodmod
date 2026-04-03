from __future__ import annotations

import array

from nodmod import XMSong
from nodmod.types import XMSample


def test_instrument_crud():
    song = XMSong()
    idx = song.new_instrument("Piano")
    assert idx == 1
    inst = song.get_instrument(1)
    assert inst.name == "Piano"
    assert len(song.list_instruments()) == 1


def test_sample_add_get_remove_and_map():
    song = XMSong()
    inst_idx = song.new_instrument("Test")

    s1 = XMSample()
    s1.waveform = array.array('b', [0, 1, 2])
    s2 = XMSample()
    s2.waveform = array.array('b', [3, 4, 5])

    s1_idx = song.add_sample(inst_idx, s1)
    s2_idx = song.add_sample(inst_idx, s2)

    assert s1_idx == 1
    assert s2_idx == 2

    # sample map should be valid (1-based) and default to sample 1
    inst = song.get_instrument(inst_idx)
    assert inst.get_sample_map()[0] == 1

    inst.set_sample_for_note(0, 2)
    assert inst.get_sample_map()[0] == 2

    song.remove_sample(inst_idx, 1)
    inst = song.get_instrument(inst_idx)
    # after removing sample 1, old sample 2 becomes sample 1
    assert inst.get_sample_map()[0] == 1
    assert len(inst.samples) == 1


def test_set_instrument_sample():
    song = XMSong()
    inst_idx = song.new_instrument("OneShot")

    s = XMSample()
    s.waveform = array.array('b', [1, 2, 3])
    song.set_instrument_sample(inst_idx, s)

    inst = song.get_instrument(inst_idx)
    assert len(inst.samples) == 1
    assert inst.get_sample_map()[10] == 1

    s_out = song.get_instrument_sample(inst_idx)
    assert s_out.waveform.tolist() == [1, 2, 3]


def test_envelope_setters():
    song = XMSong()
    inst_idx = song.new_instrument("Env")
    inst = song.get_instrument(inst_idx)

    inst.set_volume_envelope([(0, 0), (10, 64)], sustain=1, loop=(0, 1), enabled=True)
    inst.set_panning_envelope([(0, 32), (10, 32)], sustain=0, loop=None, enabled=True)

    assert len(inst.volume_envelope) == 2
    assert inst.volume_type != 0
    assert len(inst.panning_envelope) == 2
    assert inst.panning_type != 0


def main() -> int:
    test_instrument_crud()
    test_sample_add_get_remove_and_map()
    test_set_instrument_sample()
    test_envelope_setters()
    print("OK: XM instrument/sample API tests passed")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_xm_instrument_api.py")
    raise SystemExit(rc)
