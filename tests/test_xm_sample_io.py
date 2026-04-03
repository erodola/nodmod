from __future__ import annotations

import os
import wave
import array

from nodmod import XMSong
from nodmod.types import XMSample


def write_test_wav(path: str) -> None:
    # 8-bit mono, 8000 Hz, 4 samples
    data = bytes([0, 64, 128, 255])
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(data)


def test_load_and_save_sample():
    song = XMSong()
    inst_idx = song.new_instrument("SampleIO")

    wav_in = os.path.join(os.getcwd(), "dev", "_xm_sample_in.wav")
    wav_out = os.path.join(os.getcwd(), "dev", "_xm_sample_out.wav")

    write_test_wav(wav_in)

    sample_idx = song.load_sample(inst_idx, wav_in)
    smp = song.get_sample(inst_idx, sample_idx)
    assert isinstance(smp, XMSample)
    assert len(smp.waveform) == 4

    song.save_sample(inst_idx, sample_idx, wav_out)
    assert os.path.isfile(wav_out)


def main() -> int:
    test_load_and_save_sample()
    print("OK: XM sample IO tests passed")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_xm_sample_io.py")
    raise SystemExit(rc)
