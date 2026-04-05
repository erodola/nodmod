from __future__ import annotations

import array
import os

from nodmod import MODSong
from nodmod.types import Sample
from .test_helpers import assert_raises_msg, assert_true


def test_sample_validate_loop_rejects_invalid_bounds() -> None:
    smp = Sample()
    smp.waveform = array.array("b", [0, 1, -1, 0])
    smp.repeat_point = 3
    smp.repeat_len = 4
    assert_raises_msg(ValueError, "Loop end", smp.validate_loop)


def test_sample_sanitize_loop_coerce_rules() -> None:
    smp = Sample()
    smp.repeat_point = 9
    smp.repeat_len = 7
    smp.sanitize_loop()
    assert_true(smp.repeat_point == 0 and smp.repeat_len == 0, "empty waveform should force loop off")

    smp.waveform = array.array("b", [0, 1, -1, 0])
    smp.repeat_point = -3
    smp.repeat_len = 8
    smp.sanitize_loop()
    assert_true(smp.repeat_point == 0 and smp.repeat_len == 4, "negative start and long length should coerce in bounds")
    smp.validate_loop()

    smp.repeat_point = 3
    smp.repeat_len = 4
    smp.sanitize_loop()
    assert_true(smp.repeat_point == 0 and smp.repeat_len == 0, "degenerate <=1 loop should disable loop")
    smp.validate_loop()


def test_modsong_validate_and_sanitize_samples() -> None:
    song = MODSong()
    idx, _ = song.load_sample_from_raw([0.0, 0.1, -0.1, 0.0])
    smp = song.get_sample(idx)
    smp.repeat_point = 3
    smp.repeat_len = 4

    assert_raises_msg(ValueError, "Sample 1", song.validate_samples)
    song.sanitize_samples()
    song.validate_samples()
    assert_true(smp.repeat_point == 0 and smp.repeat_len == 0, "sanitize_samples should coerce invalid loop metadata")


def test_modsong_save_validate_samples_flag(tmp_dir: str) -> None:
    song = MODSong()
    idx, _ = song.load_sample_from_raw([0.0, 0.1, -0.1, 0.0])
    smp = song.get_sample(idx)
    smp.repeat_point = 3
    smp.repeat_len = 4

    legacy_path = os.path.join(tmp_dir, "legacy_loop.mod")
    strict_path = os.path.join(tmp_dir, "strict_loop.mod")

    song.save(legacy_path, verbose=False)
    assert_true(os.path.isfile(legacy_path), "legacy save should still work without strict validation")
    assert_raises_msg(ValueError, "Sample 1", song.save, strict_path, False, validate_samples=True)


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_sample_validate_loop_rejects_invalid_bounds()
    test_sample_sanitize_loop_coerce_rules()
    test_modsong_validate_and_sanitize_samples()
    test_modsong_save_validate_samples_flag(tmp_dir)
    print("OK: test_mod_sample_loop_safety.py")
