import array

from nodmod import MODSong
from test_helpers import assert_true, assert_raises_msg


def test_pcm_i8_bytes_roundtrip() -> None:
    song = MODSong()
    payload = bytes([0x00, 0x01, 0x7F, 0x80, 0xFF])
    song.set_sample_pcm_i8(1, payload)
    out = song.get_sample_pcm_i8(1)
    assert_true(out == payload, "PCM i8 bytes should roundtrip exactly")


def test_pcm_i8_supported_input_types() -> None:
    song = MODSong()
    payload = bytes([0x00, 0x7F, 0x80, 0xFF])

    song.set_sample_pcm_i8(1, bytearray(payload))
    assert_true(song.get_sample_pcm_i8(1) == payload, "bytearray input should be accepted")

    song.set_sample_pcm_i8(1, memoryview(payload))
    assert_true(song.get_sample_pcm_i8(1) == payload, "memoryview input should be accepted")

    song.set_sample_pcm_i8(1, array.array("b", [0, 127, -128, -1]))
    assert_true(song.get_sample_pcm_i8(1) == payload, "array('b') input should be accepted")

    song.set_sample_pcm_i8(1, array.array("B", [0, 127, 128, 255]))
    assert_true(song.get_sample_pcm_i8(1) == payload, "array('B') input should be accepted")


def test_pcm_i8_reset_meta_and_loop_bytes() -> None:
    song = MODSong()
    song.set_sample_name(1, "Kick")
    song.set_sample_volume(1, 40)
    song.set_sample_finetune(1, 7)
    song.set_sample_loop(1, 3, 5)

    song.set_sample_pcm_i8(1, bytes([0, 1, 2, 3]), reset_meta=False)
    smp = song.get_sample(1)
    assert_true(smp.name == "Kick", "reset_meta=False should preserve sample metadata")
    assert_true(smp.volume == 40 and smp.finetune == 7, "reset_meta=False should keep volume/finetune")
    assert_true(smp.repeat_point == 3 and smp.repeat_len == 5, "reset_meta=False should keep loop metadata")

    song.set_sample_pcm_i8(1, bytes([0, 1]), reset_meta=True)
    smp = song.get_sample(1)
    assert_true(smp.name == "", "reset_meta=True should reset sample name")
    assert_true(smp.volume == 64 and smp.finetune == 0, "reset_meta=True should reset volume/finetune")
    assert_true(smp.repeat_point == 0 and smp.repeat_len == 0, "reset_meta=True should reset loop metadata")

    song.set_sample_loop_bytes(1, 10, 20)
    assert_true(smp.repeat_point == 10 and smp.repeat_len == 20, "set_sample_loop_bytes should use byte units")


def test_pcm_i8_validation() -> None:
    song = MODSong()
    assert_raises_msg(TypeError, "Invalid pcm_i8 type", song.set_sample_pcm_i8, 1, [0, 1, 2])
    assert_raises_msg(TypeError, "typecode", song.set_sample_pcm_i8, 1, array.array("h", [1, 2]))
    assert_raises_msg(ValueError, "expected >= 0", song.set_sample_loop_bytes, 1, -1, 2)
    assert_raises_msg(ValueError, "expected >= 0", song.set_sample_loop_bytes, 1, 1, -2)


if __name__ == "__main__":
    test_pcm_i8_bytes_roundtrip()
    test_pcm_i8_supported_input_types()
    test_pcm_i8_reset_meta_and_loop_bytes()
    test_pcm_i8_validation()
    print("OK: test_mod_pcm_helpers.py")
