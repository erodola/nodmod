from __future__ import annotations

import os
import struct
import tempfile

from nodmod import S3MSong
from nodmod.types import S3MSample
from test_helpers import assert_raises_msg, assert_true, pick_files


def _build_s3m_with_one_instrument(inst_type: int, sample_type: int, flags: int, sample_bytes: bytes) -> bytes:
    header = bytearray()
    title = b"PCM Test\x00" + b"\x00" * (28 - len("PCM Test") - 1)
    header += title
    header += bytes([0x1A, 0x10])
    header += struct.pack("<H", 0)
    header += struct.pack("<6H", 1, 1, 0, 0, 0x1320, sample_type)
    header += b"SCRM"
    header += bytes([64, 6, 125, 0xB0, 0, 0])
    header += b"\x00" * 8
    header += struct.pack("<H", 0)
    header += bytes([0, 8, 1, 9] + [255] * 28)
    header += bytes([0xFF])
    header += struct.pack("<H", 0x0010)

    data = bytearray(header)
    if len(data) > 0x100:
        raise AssertionError("Synthetic S3M header unexpectedly exceeded instrument offset")
    data += b"\x00" * (0x100 - len(data))

    inst = bytearray(80)
    inst[0] = inst_type
    inst[1:13] = b"TEST.SMP   \x00"
    if inst_type == 1:
        sample_paragraph = 0x0020
        inst[13] = (sample_paragraph >> 16) & 0xFF
        struct.pack_into("<H", inst, 14, sample_paragraph & 0xFFFF)
        struct.pack_into("<I", inst, 16, len(sample_bytes))
        struct.pack_into("<I", inst, 20, 1 if len(sample_bytes) > 1 else 0)
        struct.pack_into("<I", inst, 24, len(sample_bytes))
        inst[28] = 40
        inst[29] = 0
        inst[30] = 0
        inst[31] = flags
        struct.pack_into("<I", inst, 32, 9000)
        inst[48:76] = b"SYNTH PCM\x00" + b"\x00" * (28 - len("SYNTH PCM") - 1)
        inst[76:80] = b"SCRS"
    else:
        inst[48:76] = b"ADLIB\x00" + b"\x00" * (28 - len("ADLIB") - 1)
        inst[76:80] = b"SCRI"

    data += inst
    if len(data) > 0x200:
        raise AssertionError("Synthetic S3M instrument unexpectedly exceeded sample offset")
    data += b"\x00" * (0x200 - len(data))
    data += sample_bytes
    return bytes(data)


def _load_synthetic(payload: bytes) -> S3MSong:
    with tempfile.NamedTemporaryFile(suffix=".s3m", delete=False) as tmp:
        tmp.write(payload)
        path = tmp.name
    try:
        song = S3MSong()
        song.load(path, verbose=False)
        return song
    finally:
        os.remove(path)


def test_s3m_load_pcm_sample_8bit() -> None:
    payload = _build_s3m_with_one_instrument(1, 2, 0x01, bytes([0, 128, 255]))
    song = _load_synthetic(payload)
    sample = song.get_sample(1)
    assert_true(isinstance(sample, S3MSample), "S3M sample should use S3MSample")
    assert_true(sample.name == "SYNTH PCM", "S3M sample title mismatch")
    assert_true(sample.filename.startswith("TEST.SMP"), "S3M sample filename mismatch")
    assert_true(list(sample.waveform) == [-128, 0, 127], "S3M 8-bit unsigned decode mismatch")
    assert_true(sample.repeat_point == 1, "S3M loop start mismatch")
    assert_true(sample.repeat_len == 2, "S3M loop length mismatch")
    assert_true(sample.c2spd == 9000, "S3M c2spd mismatch")


def test_s3m_load_pcm_sample_16bit() -> None:
    sample_bytes = struct.pack("<3H", 0, 32768, 65535)
    payload = _build_s3m_with_one_instrument(1, 2, 0x04, sample_bytes)
    song = _load_synthetic(payload)
    sample = song.get_sample(1)
    assert_true(sample.is_16bit, "S3M 16-bit flag mismatch")
    assert_true(sample.waveform.typecode == 'h', "S3M 16-bit waveform type mismatch")
    assert_true(list(sample.waveform) == [-32768, 0, 32767], "S3M 16-bit unsigned decode mismatch")


def test_s3m_rejects_adlib_instruments() -> None:
    payload = _build_s3m_with_one_instrument(2, 2, 0, b"")
    with tempfile.NamedTemporaryFile(suffix=".s3m", delete=False) as tmp:
        tmp.write(payload)
        path = tmp.name
    try:
        song = S3MSong()
        assert_raises_msg(NotImplementedError, "Adlib S3M instruments", song.load, path, False)
    finally:
        os.remove(path)


def test_s3m_load_real_pcm_samples() -> None:
    root = r"G:\My Drive\Moduli"
    if not os.path.isdir(root):
        return
    files = sorted(pick_files(root, ".s3m", 3))
    assert_true(len(files) > 0, "No S3M files found for sample tests")
    for path in files:
        song = S3MSong()
        song.load(path, verbose=False)
        assert_true(song.n_actual_samples > 0, "Real S3M should expose PCM samples")
        non_empty = [sample for sample in song.list_samples() if len(sample.waveform) > 0]
        assert_true(len(non_empty) > 0, "Real S3M should have at least one non-empty sample")
        assert_true(all(sample.pack == 0 for sample in non_empty), "Real S3M pack mode mismatch")
        assert_true(all(not sample.is_stereo for sample in non_empty), "Real S3M stereo flag mismatch")


if __name__ == "__main__":
    test_s3m_load_pcm_sample_8bit()
    test_s3m_load_pcm_sample_16bit()
    test_s3m_rejects_adlib_instruments()
    test_s3m_load_real_pcm_samples()
    print("OK: test_s3m_sample_api.py")