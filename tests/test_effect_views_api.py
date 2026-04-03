from nodmod import MODSong
from .test_helpers import assert_true


def test_effect_views_non_empty_filtering() -> None:
    song = MODSong()
    song.set_effect(0, 0, 0, "F06")

    all_effects = list(song.iter_effects(include_empty=True))
    used_effects = list(song.iter_effects(include_empty=False))

    assert_true(len(all_effects) == 64 * 4, "include_empty=True should expose every cell effect slot")
    assert_true(len(used_effects) == 1, "include_empty=False should keep only non-empty effects")
    assert_true(used_effects[0].raw == "F06", "non-empty effect payload mismatch")


def test_effect_views_decoded_payload() -> None:
    song = MODSong()
    song.set_effect(0, 1, 2, "E6F")

    effects = list(song.iter_effects(include_empty=False, decoded=True))
    target = [e for e in effects if e.row == 2 and e.channel == 1][0]
    assert_true(target.command == "E", "decoded command mismatch")
    assert_true(target.arg == 0x6F, "decoded arg mismatch")
    assert_true(target.x == 0x6 and target.y == 0xF, "decoded nibble mismatch")
    assert_true(target.extended_cmd == "E6", "decoded extended command mismatch")


def test_effect_views_coordinate_order() -> None:
    song = MODSong()
    song.add_pattern()
    song.set_sequence([1, 0])
    song.set_effect(0, 2, 1, "A0F")
    song.set_effect(1, 3, 0, "B01")

    effects = list(song.iter_effects(include_empty=False))
    assert_true(len(effects) == 2, "expected two non-empty effects")
    assert_true(
        effects[0].sequence_idx == 0 and effects[0].pattern_idx == 1 and effects[0].row == 1 and effects[0].channel == 2,
        "first effect traversal order mismatch",
    )
    assert_true(
        effects[1].sequence_idx == 1 and effects[1].pattern_idx == 0 and effects[1].row == 0 and effects[1].channel == 3,
        "second effect traversal order mismatch",
    )


def test_effect_views_malformed_decode_is_graceful() -> None:
    song = MODSong()
    song.set_effect(0, 0, 0, "ZZZ")

    effects = list(song.iter_effects(include_empty=False, decoded=True))
    assert_true(len(effects) == 1, "malformed effect should still be yielded")
    assert_true(effects[0].raw == "ZZZ", "malformed effect raw payload mismatch")
    assert_true(
        effects[0].command is None and effects[0].arg is None and effects[0].x is None and effects[0].y is None,
        "malformed effect decode should not raise and should produce None fields",
    )
    assert_true(effects[0].extended_cmd is None, "malformed effect extended command should be None")


if __name__ == "__main__":
    test_effect_views_non_empty_filtering()
    test_effect_views_decoded_payload()
    test_effect_views_coordinate_order()
    test_effect_views_malformed_decode_is_graceful()
    print("OK: test_effect_views_api.py")
