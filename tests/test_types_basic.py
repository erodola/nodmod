from nodmod.types import MODNote, Note, S3MNote, XMNote
from .test_helpers import assert_true


def test_types_basic() -> None:
    n = Note(1, "C-4", "F06")
    assert_true(repr(n).startswith("C-4"), "Note repr should include period")
    assert_true(not n.is_empty(), "Note should not be empty")

    mn = MODNote(2, "D-4", "A01")
    assert_true(isinstance(mn, Note), "MODNote should alias Note")

    xn = XMNote(1, "off", "", "v", 40)
    assert_true("===" in repr(xn), "XMNote repr should show note-off")
    assert_true(not xn.is_empty(), "XMNote should not be empty")

    sn = S3MNote(1, "off", "A03", 40)
    assert_true("===" in repr(sn), "S3MNote repr should show note-off")
    assert_true(not sn.is_empty(), "S3MNote should not be empty")


if __name__ == '__main__':
    test_types_basic()
    print("OK: test_types_basic.py")
