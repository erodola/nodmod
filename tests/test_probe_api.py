import array
import os
import struct

from nodmod import MODSong, S3MSong, XMSong, detect_format, probe_file
from .test_helpers import assert_true


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


def test_probe_valid_formats(tmp_dir: str) -> None:
    mod_path = os.path.join(tmp_dir, "probe_valid.mod")
    xm_path = os.path.join(tmp_dir, "probe_valid.xm")
    s3m_path = os.path.join(tmp_dir, "probe_valid.s3m")
    _make_mod(mod_path)
    _make_xm(xm_path)
    _make_s3m(s3m_path)

    mod = probe_file(mod_path)
    xm = probe_file(xm_path)
    s3m = probe_file(s3m_path)

    assert_true(mod.detected_format == "mod" and mod.supported and mod.loader == "MODSong", "valid MOD probe mismatch")
    assert_true(xm.detected_format == "xm" and xm.supported and xm.loader == "XMSong", "valid XM probe mismatch")
    assert_true(s3m.detected_format == "s3m" and s3m.supported and s3m.loader == "S3MSong", "valid S3M probe mismatch")
    assert_true(detect_format(mod_path) == "mod", "detect_format MOD mismatch")
    assert_true(detect_format(xm_path) == "xm", "detect_format XM mismatch")
    assert_true(detect_format(s3m_path) == "s3m", "detect_format S3M mismatch")


def test_probe_malformed_file(tmp_dir: str) -> None:
    bad_path = os.path.join(tmp_dir, "probe_bad.bin")
    with open(bad_path, "wb") as handle:
        handle.write(b"not a tracker module")
    res = probe_file(bad_path)
    assert_true(res.supported is False, "malformed file should be unsupported")
    assert_true(res.detected_format is None, "malformed file should have no detected format")
    assert_true(len(res.errors) > 0, "malformed file should expose structured errors")


def test_probe_unsupported_mod_magic(tmp_dir: str) -> None:
    mod_path = os.path.join(tmp_dir, "probe_bad_magic.mod")
    _make_mod(mod_path)

    with open(mod_path, "r+b") as handle:
        payload = bytearray(handle.read())
        payload[1080:1084] = b"6CHN"
        handle.seek(0)
        handle.write(payload)
        handle.truncate()

    res = probe_file(mod_path)
    assert_true(res.detected_format == "mod", "unsupported MOD variant should still detect as mod")
    assert_true(res.supported is False, "unsupported MOD magic should be unsupported")
    assert_true(any("Unsupported MOD magic" in err for err in res.errors), "unsupported MOD magic reason missing")
    assert_true(res.metadata.get("magic") == "6CHN", "probe metadata should expose MOD magic")


def test_probe_unsupported_s3m_adlib(tmp_dir: str) -> None:
    s3m_path = os.path.join(tmp_dir, "probe_bad_adlib.s3m")
    _make_s3m(s3m_path)

    with open(s3m_path, "r+b") as handle:
        payload = bytearray(handle.read())
        order_count, instrument_count, _pattern_count = struct.unpack_from("<3H", payload, 32)
        assert_true(instrument_count >= 1, "fixture should contain at least one S3M instrument")
        ptr_offset = 96 + order_count
        inst_ptr = struct.unpack_from("<H", payload, ptr_offset)[0]
        inst_offset = inst_ptr << 4
        payload[inst_offset] = 2  # Adlib instrument type
        handle.seek(0)
        handle.write(payload)
        handle.truncate()

    res = probe_file(s3m_path)
    assert_true(res.detected_format == "s3m", "unsupported S3M variant should still detect as s3m")
    assert_true(res.supported is False, "Adlib S3M should be unsupported")
    assert_true(any("Adlib S3M instruments" in err for err in res.errors), "unsupported S3M reason missing")


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_probe_valid_formats(tmp_dir)
    test_probe_malformed_file(tmp_dir)
    test_probe_unsupported_mod_magic(tmp_dir)
    test_probe_unsupported_s3m_adlib(tmp_dir)
    print("OK: test_probe_api.py")
