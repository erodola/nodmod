from __future__ import annotations

from nodmod import MODSong, XMSong
from nodmod.types import XMNote


def test_mod_write_and_get_note():
    song = MODSong()
    song.set_note(0, 0, 0, 1, "C-4", "F06")
    note = song.get_note(0, 0, 0)
    assert note.instrument_idx == 1
    assert note.period == "C-4"
    assert note.effect == "F06"


def test_xm_write_and_get_note_with_volume():
    song = XMSong()
    song.add_pattern(n_rows=64)
    song.set_note(0, 0, 0, 1, "C-4", "F06", vol_cmd="v", vol_val=40)
    note = song.get_note(0, 0, 0)
    assert isinstance(note, XMNote)
    assert note.instrument_idx == 1
    assert note.period == "C-4"
    assert note.effect == "F06"
    assert note.vol_cmd == "v"
    assert note.vol_val == 40


def test_xm_write_preserves_volume_when_not_specified():
    song = XMSong()
    song.add_pattern(n_rows=64)
    song.set_note(0, 0, 0, 1, "C-4", "F06", vol_cmd="v", vol_val=40)
    song.set_note(0, 0, 0, 1, "D-4", "F06")
    note = song.get_note(0, 0, 0)
    assert note.period == "D-4"
    assert note.vol_cmd == "v"
    assert note.vol_val == 40


def main() -> int:
    test_mod_write_and_get_note()
    test_xm_write_and_get_note_with_volume()
    test_xm_write_preserves_volume_when_not_specified()
    print("OK: note editing tests passed")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_note_editing.py")
    raise SystemExit(rc)
