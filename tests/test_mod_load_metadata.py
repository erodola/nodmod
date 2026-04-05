from __future__ import annotations

import os
import shutil

from nodmod import MODSong
from .test_helpers import assert_true


def _make_song_with_title(path: str, title: str) -> None:
    song = MODSong()
    song.songname = title
    song.save(path, verbose=False)


def _write_header_title(path: str, raw_title: bytes) -> None:
    header = raw_title[:20].ljust(20, b"\x00")
    with open(path, "r+b") as handle:
        payload = bytearray(handle.read())
        payload[:20] = header
        handle.seek(0)
        handle.write(payload)
        handle.truncate()


def test_mod_load_uses_header_title_by_default(tmp_dir: str) -> None:
    src = os.path.join(tmp_dir, "header_source.mod")
    renamed = os.path.join(tmp_dir, "Artist Name - Renamed Copy.mod")
    _make_song_with_title(src, "HeaderTitle")
    shutil.copyfile(src, renamed)

    a = MODSong()
    b = MODSong()
    a.load(src, verbose=False)
    b.load(renamed, verbose=False)

    assert_true(a.songname == "HeaderTitle", "default load should read header title")
    assert_true(b.songname == "HeaderTitle", "renamed file should still use header title")
    assert_true(a.artist == "Unknown Artist", "default artist should not come from filename")
    assert_true(b.artist == "Unknown Artist", "default artist should not come from filename")


def test_mod_load_filename_metadata_override_opt_in(tmp_dir: str) -> None:
    path = os.path.join(tmp_dir, "Some Artist - Parsed Name.mod")
    _make_song_with_title(path, "HeaderWinsByDefault")

    song = MODSong()
    song.load(path, verbose=False, metadata_from_filename=True)

    assert_true(song.artist == "Some Artist", "opt-in filename metadata should parse artist")
    assert_true(song.songname == "Parsed Name", "opt-in filename metadata should parse title")


def test_mod_load_empty_header_title_stays_empty(tmp_dir: str) -> None:
    path = os.path.join(tmp_dir, "Artist - Should Not Win.mod")
    _make_song_with_title(path, "")

    song = MODSong()
    song.load(path, verbose=False)
    assert_true(song.songname == "", "empty MOD header title should load as empty string")


def test_mod_load_header_title_latin1_fallback(tmp_dir: str) -> None:
    path = os.path.join(tmp_dir, "latin1_header.mod")
    _make_song_with_title(path, "")
    _write_header_title(path, b"Caf\xe9")

    song = MODSong()
    song.load(path, verbose=False)

    expected = "Caf" + chr(233)
    assert_true(song.songname == expected, "header decode should fallback to latin-1 when utf-8 fails")


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_mod_load_uses_header_title_by_default(tmp_dir)
    test_mod_load_filename_metadata_override_opt_in(tmp_dir)
    test_mod_load_empty_header_title_stays_empty(tmp_dir)
    test_mod_load_header_title_latin1_fallback(tmp_dir)
    print("OK: test_mod_load_metadata.py")
