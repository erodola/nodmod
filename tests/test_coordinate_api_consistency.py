from nodmod import MODSong, S3MSong, XMSong
from .test_helpers import assert_true


def _capture_exc(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentional test helper
        return type(exc), str(exc)
    return None


def _assert_same_exception(legacy_fn, legacy_args, rc_fn, rc_args, label: str) -> None:
    legacy_exc = _capture_exc(legacy_fn, *legacy_args)
    rc_exc = _capture_exc(rc_fn, *rc_args)
    assert_true(legacy_exc is not None, f"{label}: legacy call should raise")
    assert_true(rc_exc is not None, f"{label}: rc call should raise")
    assert_true(legacy_exc[0] is rc_exc[0], f"{label}: exception type mismatch")
    assert_true(legacy_exc[1] == rc_exc[1], f"{label}: exception message mismatch")


def test_mod_coordinate_wrappers() -> None:
    song = MODSong()

    song.set_note(0, 1, 2, 3, "C-4", "F06")
    note_legacy = song.get_note(0, 2, 1)
    note_rc = song.get_note_rc(0, 2, 1)
    assert_true(note_legacy is note_rc, "MOD get_note_rc should match get_note")

    song.set_note_rc(0, 4, 2, 5, "D-4", "A01")
    note = song.get_note(0, 4, 2)
    assert_true(note.instrument_idx == 5 and note.period == "D-4" and note.effect == "A01", "MOD set_note_rc mismatch")

    song.set_effect_rc(0, 4, 2, "B01")
    assert_true(song.get_note(0, 4, 2).effect == "B01", "MOD set_effect_rc mismatch")

    song.clear_note_rc(0, 4, 2)
    assert_true(song.get_note(0, 4, 2).is_empty(), "MOD clear_note_rc mismatch")

    _assert_same_exception(
        song.set_note,
        (0, 0, 64, 1, "C-4", ""),
        song.set_note_rc,
        (0, 64, 0, 1, "C-4", ""),
        "MOD set_note row bounds",
    )
    _assert_same_exception(
        song.set_effect,
        (0, -1, 0, "F06"),
        song.set_effect_rc,
        (0, 0, -1, "F06"),
        "MOD set_effect channel bounds",
    )


def test_xm_coordinate_wrappers() -> None:
    song = XMSong()
    song.set_n_channels(2)

    song.set_note_rc(0, 0, 1, 2, "E-4", "F06", vol_cmd="v", vol_val=24)
    note = song.get_note(0, 0, 1)
    assert_true(note.instrument_idx == 2 and note.period == "E-4", "XM set_note_rc note mismatch")
    assert_true(note.effect == "F06", "XM set_note_rc effect mismatch")
    assert_true(note.vol_cmd == "v" and note.vol_val == 24, "XM set_note_rc volume mismatch")

    _assert_same_exception(
        song.set_note,
        (0, 99, 0, 1, "C-4", "", None, None),
        song.set_note_rc,
        (0, 0, 99, 1, "C-4", "", None, None),
        "XM set_note channel bounds",
    )


def test_s3m_coordinate_wrappers() -> None:
    song = S3MSong()

    song.set_note_rc(0, 0, 0, 1, "G-4", "A03", 33)
    note = song.get_note(0, 0, 0)
    assert_true(note.instrument_idx == 1 and note.period == "G-4", "S3M set_note_rc note mismatch")
    assert_true(note.effect == "A03" and note.volume == 33, "S3M set_note_rc effect/volume mismatch")

    _assert_same_exception(
        song.set_note,
        (0, 0, 64, 1, "C-4", "", None),
        song.set_note_rc,
        (0, 64, 0, 1, "C-4", "", None),
        "S3M set_note row bounds",
    )


if __name__ == "__main__":
    test_mod_coordinate_wrappers()
    test_xm_coordinate_wrappers()
    test_s3m_coordinate_wrappers()
    print("OK: test_coordinate_api_consistency.py")
