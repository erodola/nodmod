import array
import os

from nodmod import MODSong, S3MSong, XMSong, load_song
from .test_helpers import assert_raises_msg, assert_true


def _make_mod(path: str) -> None:
    song = MODSong()
    song.save(path, verbose=False)


def _make_xm(path: str) -> None:
    song = XMSong()
    song.save(path, verbose=False)


def _make_s3m(path: str) -> None:
    song = S3MSong()
    sample = song.get_sample(1)
    sample.instrument_type = 1
    sample.waveform = array.array("b", [1, -1, 2, -2])
    song.save(path, verbose=False)


def test_load_song_dispatch(tmp_dir: str) -> None:
    mod_path = os.path.join(tmp_dir, "loader_valid.mod")
    xm_path = os.path.join(tmp_dir, "loader_valid.xm")
    s3m_path = os.path.join(tmp_dir, "loader_valid.s3m")
    _make_mod(mod_path)
    _make_xm(xm_path)
    _make_s3m(s3m_path)

    mod = load_song(mod_path, verbose=False)
    xm = load_song(xm_path, verbose=False)
    s3m = load_song(s3m_path, verbose=False)
    assert_true(isinstance(mod, MODSong), "load_song should dispatch MOD files to MODSong")
    assert_true(isinstance(xm, XMSong), "load_song should dispatch XM files to XMSong")
    assert_true(isinstance(s3m, S3MSong), "load_song should dispatch S3M files to S3MSong")


def test_load_song_unsupported_variant(tmp_dir: str) -> None:
    mod_path = os.path.join(tmp_dir, "loader_bad_magic.mod")
    _make_mod(mod_path)

    with open(mod_path, "r+b") as handle:
        payload = bytearray(handle.read())
        payload[1080:1084] = b"6CHN"
        handle.seek(0)
        handle.write(payload)
        handle.truncate()

    assert_raises_msg(NotImplementedError, "Unsupported MOD module", load_song, mod_path)


def test_load_song_unrecognized_and_missing(tmp_dir: str) -> None:
    bad_path = os.path.join(tmp_dir, "loader_bad.bin")
    with open(bad_path, "wb") as handle:
        handle.write(b"not a tracker module")

    assert_raises_msg(ValueError, "Unsupported or unrecognized", load_song, bad_path)
    assert_raises_msg(FileNotFoundError, "No such file", load_song, os.path.join(tmp_dir, "missing.mod"))


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_load_song_dispatch(tmp_dir)
    test_load_song_unsupported_variant(tmp_dir)
    test_load_song_unrecognized_and_missing(tmp_dir)
    print("OK: test_load_song_api.py")
