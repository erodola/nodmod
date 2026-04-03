from nodmod import (
    decode_mod_effect,
    encode_mod_effect,
    is_speed_effect,
    is_tempo_effect,
    merge_xy,
    split_xy,
)
from .test_helpers import assert_true, assert_raises_msg


def test_effect_codec_roundtrip() -> None:
    pairs = [
        ("0", 0x12),
        ("B", 0x01),
        ("C", 0x40),
        ("D", 0x23),
        ("E", 0x6F),
        ("F", 0x06),
        ("F", 0x7D),
    ]
    for command, arg in pairs:
        encoded = encode_mod_effect(command, arg)
        decoded = decode_mod_effect(encoded)
        assert_true(decoded.command == command, f"command mismatch for {encoded}")
        assert_true(decoded.arg == arg, f"arg mismatch for {encoded}")
        assert_true(decoded.raw == encoded, f"raw canonical mismatch for {encoded}")
        assert_true(decode_mod_effect(encoded).raw == encode_mod_effect(command, arg), "decode/encode should roundtrip")


def test_effect_codec_extended_fields() -> None:
    info = decode_mod_effect("E6F")
    assert_true(info.is_extended, "E effects should be marked as extended")
    assert_true(info.extended_cmd == "E6", "extended command should expose high nibble")
    assert_true((info.x, info.y) == (6, 15), "nibble decomposition should match")


def test_effect_codec_speed_tempo_helpers() -> None:
    assert_true(is_speed_effect("F01"), "F01 should be speed")
    assert_true(is_speed_effect("F1F"), "F1F should be speed")
    assert_true(not is_speed_effect("F20"), "F20 should not be speed")
    assert_true(is_tempo_effect("F20"), "F20 should be tempo")
    assert_true(is_tempo_effect("FFF"), "FFF should be tempo")
    assert_true(not is_tempo_effect("F00"), "F00 should not be tempo")


def test_effect_codec_xy_helpers() -> None:
    assert_true(split_xy(0xAF) == (10, 15), "split_xy should split nibbles")
    assert_true(merge_xy(10, 15) == 0xAF, "merge_xy should merge nibbles")
    assert_raises_msg(ValueError, "expected 0-255", split_xy, 256)
    assert_raises_msg(ValueError, "expected 0-15", merge_xy, 16, 0)
    assert_raises_msg(ValueError, "expected 0-15", merge_xy, 0, 16)


def test_effect_codec_validation() -> None:
    assert_raises_msg(ValueError, "expected 3 characters", decode_mod_effect, "")
    assert_raises_msg(ValueError, "expected 3 characters", decode_mod_effect, "F0")
    assert_raises_msg(ValueError, "command", decode_mod_effect, "G00")
    assert_raises_msg(ValueError, "argument", decode_mod_effect, "FGG")
    assert_raises_msg(ValueError, "one hex character", encode_mod_effect, "FG", 1)
    assert_raises_msg(ValueError, "expected 0-255", encode_mod_effect, "F", 256)


if __name__ == "__main__":
    test_effect_codec_roundtrip()
    test_effect_codec_extended_fields()
    test_effect_codec_speed_tempo_helpers()
    test_effect_codec_xy_helpers()
    test_effect_codec_validation()
    print("OK: test_effect_codec.py")
