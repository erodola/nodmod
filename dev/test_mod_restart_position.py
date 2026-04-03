import os

from nodmod import MODSong
from test_helpers import assert_true, assert_raises_msg


def test_restart_default_values() -> None:
    song = MODSong()
    assert_true(song.get_restart_position(raw=True) == 127, "default raw restart should be 127")
    assert_true(song.restart_position is None, "default normalized restart should be None")
    info = song.get_song_info()
    assert_true(info.get("restart_position") is None, "song info should include normalized restart position")


def test_restart_explicit_roundtrip(tmp_dir: str) -> None:
    src = MODSong()
    src.set_restart_position(5)
    path = os.path.join(tmp_dir, "restart_explicit.mod")
    src.save(path, verbose=False)

    loaded = MODSong()
    loaded.load(path, verbose=False)
    assert_true(loaded.get_restart_position(raw=True) == 5, "explicit restart raw byte should persist")
    assert_true(loaded.restart_position == 5, "explicit restart normalized value should persist")


def test_restart_raw_edge_roundtrip(tmp_dir: str) -> None:
    src = MODSong()
    src.set_restart_position(255, raw=True)
    path = os.path.join(tmp_dir, "restart_edge.mod")
    src.save(path, verbose=False)

    loaded = MODSong()
    loaded.load(path, verbose=False)
    assert_true(loaded.get_restart_position(raw=True) == 255, "raw edge value should roundtrip")
    assert_true(loaded.restart_position == 255, "normalized view should expose unusual raw values")


def test_restart_load_existing_header_byte(tmp_dir: str) -> None:
    path = os.path.join(tmp_dir, "restart_injected.mod")
    song = MODSong()
    song.save(path, verbose=False)

    with open(path, "r+b") as handle:
        raw = bytearray(handle.read())
        raw[951] = 12
        handle.seek(0)
        handle.write(raw)
        handle.truncate()

    loaded = MODSong()
    loaded.load(path, verbose=False)
    assert_true(loaded.get_restart_position(raw=True) == 12, "load should preserve non-default header byte")
    assert_true(loaded.restart_position == 12, "normalized restart should match injected value")


def test_restart_validation() -> None:
    song = MODSong()
    assert_raises_msg(ValueError, "expected 0-255", song.set_restart_position, 256)
    assert_raises_msg(TypeError, "expected int or None", song.set_restart_position, "1")


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_restart_default_values()
    test_restart_explicit_roundtrip(tmp_dir)
    test_restart_raw_edge_roundtrip(tmp_dir)
    test_restart_load_existing_header_byte(tmp_dir)
    test_restart_validation()
    print("OK: test_mod_restart_position.py")
