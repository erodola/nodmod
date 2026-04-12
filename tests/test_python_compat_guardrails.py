from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .test_helpers import assert_true


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not pythonpath else f"{SRC}{os.pathsep}{pythonpath}"
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_import_does_not_require_pydub_at_module_import_time() -> None:
    script = r"""
import builtins

real_import = builtins.__import__

def guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pydub" or name.startswith("pydub."):
        raise ModuleNotFoundError("No module named 'pydub'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guard
import nodmod
print("import-ok")
"""
    result = _run_subprocess(script)
    assert_true(
        result.returncode == 0,
        f"nodmod import should not require pydub at import time.\nstdout={result.stdout}\nstderr={result.stderr}",
    )
    assert_true("import-ok" in result.stdout, "subprocess should confirm successful import")


def test_wav_apis_raise_clear_error_when_pydub_is_missing() -> None:
    script = r"""
import builtins
from nodmod import MODSong, S3MSong, XMSong

real_import = builtins.__import__

def guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pydub" or name.startswith("pydub."):
        raise ModuleNotFoundError("No module named 'pydub'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guard

mod = MODSong()
try:
    mod.load_sample_from_raw([0.0], 1)
except ImportError as exc:
    if "WAV sample I/O requires `pydub`." not in str(exc):
        print(f"bad-mod-msg: {exc}")
        raise SystemExit(11)
else:
    print("mod did not raise ImportError")
    raise SystemExit(10)

xm = XMSong()
inst = xm.new_instrument("Guard")
try:
    xm.load_sample(inst, "not_used.wav")
except ImportError as exc:
    if "WAV sample I/O requires `pydub`." not in str(exc):
        print(f"bad-xm-msg: {exc}")
        raise SystemExit(21)
else:
    print("xm did not raise ImportError")
    raise SystemExit(20)

s3m = S3MSong()
try:
    s3m.load_sample_from_raw([0.0], 1)
except ImportError as exc:
    if "WAV sample I/O requires `pydub`." not in str(exc):
        print(f"bad-s3m-msg: {exc}")
        raise SystemExit(31)
else:
    print("s3m did not raise ImportError")
    raise SystemExit(30)

print("wav-errors-ok")
"""
    result = _run_subprocess(script)
    assert_true(
        result.returncode == 0,
        f"WAV APIs should fail with clear pydub guidance when unavailable.\nstdout={result.stdout}\nstderr={result.stderr}",
    )
    assert_true("wav-errors-ok" in result.stdout, "subprocess should confirm WAV API error guardrails")

