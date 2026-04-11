from __future__ import annotations

import hashlib
import tempfile
import wave
from pathlib import Path

from nodmod import MODSong, XMSong

from .test_helpers import assert_true


EXPECTED_INPUT_PARAMS = (1, 1, 8000, 4)
EXPECTED_INPUT_FRAMES_HASH = "81f9456ee0bb909b7fb7c887c21c21f9fc45430e881dd5d92450609410fc3936"
EXPECTED_SIGNED_PCM_HASH = "0ea63d7a278310ea75e36f76ce5149c9e60de62b1528408930041d763806e7c8"
EXPECTED_XM_OUT_PARAMS = (1, 1, 8000, 4)
EXPECTED_XM_OUT_FRAMES_HASH = "81f9456ee0bb909b7fb7c887c21c21f9fc45430e881dd5d92450609410fc3936"
EXPECTED_MOD_OUT_PARAMS = (1, 1, 8000, 3)
EXPECTED_MOD_OUT_FRAMES_HASH = "f0cc8cd1e723a8b77992e88701fab4957472b9fc551bb89125e3879c7bf05631"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_fixture_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(bytes([0, 64, 128, 255]))


def _read_wav(path: Path) -> tuple[tuple[int, int, int, int], bytes]:
    with wave.open(str(path), "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())
        frames = wf.readframes(wf.getnframes())
    return params, frames


def test_xm_wav_fixture_hash_stability() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        wav_in = tmp / "fixture_in.wav"
        wav_out = tmp / "xm_out.wav"

        _write_fixture_wav(wav_in)
        in_params, in_frames = _read_wav(wav_in)
        assert_true(in_params == EXPECTED_INPUT_PARAMS, "fixture WAV params drifted")
        assert_true(_sha256(in_frames) == EXPECTED_INPUT_FRAMES_HASH, "fixture WAV frame hash drifted")

        song = XMSong()
        inst_idx = song.new_instrument("HashFixture")
        sample_idx = song.load_sample(inst_idx, str(wav_in))
        loaded = song.get_sample(inst_idx, sample_idx).waveform.tobytes()
        assert_true(_sha256(loaded) == EXPECTED_SIGNED_PCM_HASH, "XM loaded waveform hash drifted")

        song.save_sample(inst_idx, sample_idx, str(wav_out), sample_rate=8000)
        out_params, out_frames = _read_wav(wav_out)
        assert_true(out_params == EXPECTED_XM_OUT_PARAMS, "XM exported WAV params drifted")
        assert_true(_sha256(out_frames) == EXPECTED_XM_OUT_FRAMES_HASH, "XM exported WAV frame hash drifted")


def test_mod_wav_fixture_hash_stability() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        wav_in = tmp / "fixture_in.wav"
        wav_out = tmp / "mod_out.wav"

        _write_fixture_wav(wav_in)
        in_params, in_frames = _read_wav(wav_in)
        assert_true(in_params == EXPECTED_INPUT_PARAMS, "fixture WAV params drifted")
        assert_true(_sha256(in_frames) == EXPECTED_INPUT_FRAMES_HASH, "fixture WAV frame hash drifted")

        song = MODSong()
        _sample_idx, sample = song.load_sample(str(wav_in), sample_idx=1)
        loaded = sample.waveform.tobytes()
        assert_true(_sha256(loaded) == EXPECTED_SIGNED_PCM_HASH, "MOD loaded waveform hash drifted")

        song.save_sample(1, str(wav_out), period="C-5", force_sample_rate=8000)
        out_params, out_frames = _read_wav(wav_out)
        assert_true(out_params == EXPECTED_MOD_OUT_PARAMS, "MOD exported WAV params drifted")
        assert_true(_sha256(out_frames) == EXPECTED_MOD_OUT_FRAMES_HASH, "MOD exported WAV frame hash drifted")

