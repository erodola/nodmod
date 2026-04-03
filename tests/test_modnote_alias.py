from __future__ import annotations

from nodmod.types import Note, MODNote


def test_modnote_alias():
    assert MODNote is Note


def test_modnote_instance():
    n = MODNote(1, "C-4", "F06")
    assert isinstance(n, Note)
    assert n.instrument_idx == 1
    assert n.period == "C-4"
    assert n.effect == "F06"


def main() -> int:
    test_modnote_alias()
    test_modnote_instance()
    print("OK: MODNote alias tests passed")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_modnote_alias.py")
    raise SystemExit(rc)
