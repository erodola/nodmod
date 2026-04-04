import array

from nodmod import MODSong, S3MSong, XMSong
from nodmod.types import XMSample
from .test_helpers import assert_raises_msg, assert_true


def _make_xm_with_two_instruments() -> tuple[XMSong, int, int]:
    song = XMSong()
    song.n_channels = 1

    inst1 = song.new_instrument("Lead")
    smp1 = XMSample()
    smp1.waveform = array.array("b", [1, -1, 1, -1])
    song.add_sample(inst1, smp1)

    inst2 = song.new_instrument("Pad")
    smp2a = XMSample()
    smp2a.waveform = array.array("b", [2, -2, 2, -2])
    smp2b = XMSample()
    smp2b.waveform = array.array("b", [3, -3, 3, -3])
    song.add_sample(inst2, smp2a)
    song.add_sample(inst2, smp2b)
    song.get_instrument(inst2).set_sample_map([1] * 96)
    return song, inst1, inst2


def test_mod_used_samples_scope_and_order() -> None:
    song = MODSong()
    song.add_pattern()
    song.set_sequence([0, 1])

    song.set_note(0, 0, 0, 1, "C-4", "B01")
    song.set_note(0, 0, 1, 5, "D-4", "")
    song.set_note(1, 0, 0, 2, "E-4", "")

    assert_true(song.get_used_samples(scope="sequence", order="sorted") == [1, 2, 5], "MOD sequence sorted mismatch")
    assert_true(song.get_used_samples(scope="sequence", order="first_use") == [1, 5, 2], "MOD sequence first_use mismatch")
    assert_true(song.get_used_samples(scope="reachable", order="sorted") == [1, 2], "MOD reachable sorted mismatch")
    assert_true(song.get_used_samples(scope="reachable", order="first_use") == [1, 2], "MOD reachable first_use mismatch")


def test_xm_used_resources_scope() -> None:
    song, inst1, inst2 = _make_xm_with_two_instruments()
    song.add_pattern(64)
    song.set_sequence([0, 1])

    song.set_note(0, 0, 0, inst1, "C-4", "B01")
    song.set_note(0, 0, 1, inst2, "D-4", "")
    song.set_note(1, 0, 0, inst1, "E-4", "")

    assert_true(song.get_used_instruments(scope="sequence", order="sorted") == [inst1, inst2], "XM sequence instruments mismatch")
    assert_true(song.get_used_instruments(scope="reachable", order="sorted") == [inst1], "XM reachable instruments mismatch")
    assert_true(song.get_used_samples(scope="sequence", order="sorted") == [1, 2], "XM sequence samples mismatch")
    assert_true(song.get_used_samples(scope="reachable", order="sorted") == [1], "XM reachable samples mismatch")


def test_s3m_used_samples_scope() -> None:
    song = S3MSong()
    song.n_channels = 1
    song.add_pattern()
    song.set_sequence([0, 1])

    song.set_note(0, 0, 0, 1, "C-4", "B01", 20)
    song.set_note(0, 0, 1, 7, "D-4", "", 30)
    song.set_note(1, 0, 0, 2, "E-4", "", 40)

    assert_true(song.get_used_samples(scope="sequence", order="sorted") == [1, 2, 7], "S3M sequence sorted mismatch")
    assert_true(song.get_used_samples(scope="reachable", order="sorted") == [1, 2], "S3M reachable sorted mismatch")


def test_used_resources_argument_validation() -> None:
    mod = MODSong()
    assert_raises_msg(ValueError, "Invalid scope", mod.get_used_samples, scope="all")
    assert_raises_msg(ValueError, "Invalid order", mod.get_used_samples, order="encountered")


if __name__ == "__main__":
    test_mod_used_samples_scope_and_order()
    test_xm_used_resources_scope()
    test_s3m_used_samples_scope()
    test_used_resources_argument_validation()
    print("OK: test_used_resources_scope.py")
