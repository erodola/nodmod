import array
import os
import random
import wave
from typing import List
from nodmod import MODSong, S3MSong, XMSong
from nodmod.types import S3MNote, S3MSample


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def assert_raises(exc_type, func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


def assert_raises_msg(exc_type, msg_substr: str, func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        if msg_substr not in str(exc):
            raise AssertionError(f"Expected message to contain '{msg_substr}', got '{exc}'") from exc
        return
    except Exception as exc:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")



def pick_files(root: str, ext: str, count: int, seed: int = 0) -> List[str]:
    matches: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                matches.append(os.path.join(dirpath, name))
    if not matches:
        return []
    rng = random.Random(seed)
    if count >= len(matches):
        return matches
    return rng.sample(matches, count)


def with_temp_wav(path: str, sample_width: int = 1) -> None:
    data = bytes([0, 64, 128, 255]) if sample_width == 1 else (b"\x00\x00" * 4)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(8000)
        wf.writeframes(data)


def compare_mod_songs(a: MODSong, b: MODSong) -> None:
    assert_true(a.pattern_seq == b.pattern_seq, "MOD pattern_seq mismatch")

    max_pat = max(a.pattern_seq) if a.pattern_seq else -1
    max_pat_b = max(b.pattern_seq) if b.pattern_seq else -1
    assert_true(max_pat == max_pat_b, "MOD max pattern index mismatch")

    for p in range(max_pat + 1):
        pa = a.patterns[p]
        pb = b.patterns[p]
        assert_true(len(pa.data) == len(pb.data), "MOD channel count mismatch")
        for c in range(len(pa.data)):
            assert_true(len(pa.data[c]) == len(pb.data[c]), "MOD row count mismatch")
            for r in range(len(pa.data[c])):
                na = pa.data[c][r]
                nb = pb.data[c][r]
                assert_true(na.instrument_idx == nb.instrument_idx, "MOD note instrument mismatch")
                assert_true(na.period == nb.period, "MOD note period mismatch")
                assert_true(na.effect == nb.effect, "MOD note effect mismatch")

    # Samples: compare metadata + full waveform content
    for i in range(31):
        sa = a.samples[i]
        sb = b.samples[i]
        assert_true(sa.finetune == sb.finetune, "MOD sample finetune mismatch")
        assert_true(sa.volume == sb.volume, "MOD sample volume mismatch")
        assert_true(sa.repeat_point == sb.repeat_point, "MOD sample repeat_point mismatch")
        assert_true(sa.repeat_len == sb.repeat_len, "MOD sample repeat_len mismatch")
        assert_true(sa.waveform == sb.waveform, "MOD sample waveform mismatch")


def compare_xm_songs(a: XMSong, b: XMSong) -> None:
    assert_true(a.pattern_seq == b.pattern_seq, "XM pattern_seq mismatch")
    assert_true(a.tracker_name == b.tracker_name, "XM tracker_name mismatch")
    assert_true(a.song_restart == b.song_restart, "XM song_restart mismatch")
    assert_true(a.flags == b.flags, "XM flags mismatch")
    assert_true(a.default_speed == b.default_speed, "XM speed mismatch")
    assert_true(a.default_tempo == b.default_tempo, "XM tempo mismatch")
    assert_true(a.n_channels == b.n_channels, "XM channels mismatch")

    max_pat = max(a.pattern_seq) if a.pattern_seq else -1
    max_pat_b = max(b.pattern_seq) if b.pattern_seq else -1
    assert_true(max_pat == max_pat_b, "XM max pattern index mismatch")

    for p in range(max_pat + 1):
        pa = a.patterns[p]
        pb = b.patterns[p]
        assert_true(pa.n_rows == pb.n_rows, "XM pattern rows mismatch")
        assert_true(pa.n_channels == pb.n_channels, "XM pattern channels mismatch")
        for c in range(pa.n_channels):
            for r in range(pa.n_rows):
                na = pa.data[c][r]
                nb = pb.data[c][r]
                assert_true(na.instrument_idx == nb.instrument_idx, "XM note instrument mismatch")
                assert_true(na.period == nb.period, "XM note period mismatch")
                assert_true(na.effect == nb.effect, "XM note effect mismatch")
                assert_true(getattr(na, "vol_cmd", "") == getattr(nb, "vol_cmd", ""), "XM note vol_cmd mismatch")
                assert_true(getattr(na, "vol_val", -1) == getattr(nb, "vol_val", -1), "XM note vol_val mismatch")

    assert_true(len(a.instruments) == len(b.instruments), "XM instrument count mismatch")
    for i in range(len(a.instruments)):
        ia = a.instruments[i]
        ib = b.instruments[i]
        assert_true(ia.name == ib.name, "XM instrument name mismatch")
        assert_true(ia.volume_type == ib.volume_type, "XM volume_type mismatch")
        assert_true(ia.panning_type == ib.panning_type, "XM panning_type mismatch")
        assert_true(ia.vibrato_type == ib.vibrato_type, "XM vibrato_type mismatch")
        assert_true(ia.vibrato_sweep == ib.vibrato_sweep, "XM vibrato_sweep mismatch")
        assert_true(ia.vibrato_depth == ib.vibrato_depth, "XM vibrato_depth mismatch")
        assert_true(ia.vibrato_rate == ib.vibrato_rate, "XM vibrato_rate mismatch")
        assert_true(ia.volume_fadeout == ib.volume_fadeout, "XM volume_fadeout mismatch")
        assert_true(ia.sample_map == ib.sample_map, "XM sample_map mismatch")

        assert_true(len(ia.samples) == len(ib.samples), "XM sample count mismatch")
        for s in range(len(ia.samples)):
            sa = ia.samples[s]
            sb = ib.samples[s]
            assert_true(sa.name == sb.name, "XM sample name mismatch")
            assert_true(sa.volume == sb.volume, "XM sample volume mismatch")
            assert_true(sa.finetune == sb.finetune, "XM sample finetune mismatch")
            assert_true(sa.panning == sb.panning, "XM sample panning mismatch")
            assert_true(sa.relative_note == sb.relative_note, "XM sample relative_note mismatch")
            assert_true(sa._reserved == sb._reserved, "XM sample reserved mismatch")
            assert_true(sa.loop_type == sb.loop_type, "XM sample loop_type mismatch")
            assert_true(sa.is_16bit == sb.is_16bit, "XM sample is_16bit mismatch")
            assert_true(sa.repeat_point == sb.repeat_point, "XM sample repeat_point mismatch")
            assert_true(sa.repeat_len == sb.repeat_len, "XM sample repeat_len mismatch")
            assert_true(sa.waveform == sb.waveform, "XM sample waveform mismatch")


def compare_s3m_songs(a: S3MSong, b: S3MSong) -> None:
    assert_true(a.pattern_seq == b.pattern_seq, "S3M pattern_seq mismatch")
    assert_true(a.n_channels == b.n_channels, "S3M channel count mismatch")
    assert_true(a.channel_settings == b.channel_settings, "S3M channel settings mismatch")
    assert_true(a.global_volume == b.global_volume, "S3M global volume mismatch")
    assert_true(a.initial_speed == b.initial_speed, "S3M speed mismatch")
    assert_true(a.initial_tempo == b.initial_tempo, "S3M tempo mismatch")
    assert_true(a.master_volume == b.master_volume, "S3M master volume mismatch")
    assert_true(a.sample_type == b.sample_type, "S3M sample type mismatch")
    assert_true(a.order_list_raw == b.order_list_raw, "S3M raw order list mismatch")
    assert_true(a.instrument_count == b.instrument_count, "S3M instrument count mismatch")

    assert_true(len(a.patterns) == len(b.patterns), "S3M pattern count mismatch")
    for p in range(len(a.patterns)):
        pa = a.patterns[p]
        pb = b.patterns[p]
        assert_true(pa.n_rows == pb.n_rows, "S3M pattern rows mismatch")
        assert_true(pa.n_channels == pb.n_channels, "S3M pattern channels mismatch")
        for c in range(pa.n_channels):
            for r in range(pa.n_rows):
                na = pa.data[c][r]
                nb = pb.data[c][r]
                assert_true(isinstance(na, S3MNote) and isinstance(nb, S3MNote), "S3M note type mismatch")
                assert_true(na.instrument_idx == nb.instrument_idx, "S3M note instrument mismatch")
                assert_true(na.period == nb.period, "S3M note period mismatch")
                assert_true(na.effect == nb.effect, "S3M note effect mismatch")
                assert_true(na.volume == nb.volume, "S3M note volume mismatch")

    for i in range(max(a.instrument_count, b.instrument_count)):
        sa = a.samples[i]
        sb = b.samples[i]
        assert_true(isinstance(sa, S3MSample) and isinstance(sb, S3MSample), "S3M sample type mismatch")
        assert_true(sa.instrument_type == sb.instrument_type, "S3M instrument type mismatch")
        assert_true(sa.name == sb.name, "S3M sample name mismatch")
        assert_true(sa.filename == sb.filename, "S3M sample filename mismatch")
        assert_true(sa.volume == sb.volume, "S3M sample volume mismatch")
        assert_true(sa.pack == sb.pack, "S3M sample pack mismatch")
        assert_true(sa.is_16bit == sb.is_16bit, "S3M sample 16-bit flag mismatch")
        assert_true(sa.is_stereo == sb.is_stereo, "S3M sample stereo flag mismatch")
        assert_true(sa.c2spd == sb.c2spd, "S3M sample c2spd mismatch")
        assert_true(sa.repeat_point == sb.repeat_point, "S3M sample repeat_point mismatch")
        assert_true(sa.repeat_len == sb.repeat_len, "S3M sample repeat_len mismatch")
        assert_true(sa.waveform == sb.waveform, "S3M sample waveform mismatch")


if __name__ == "__main__":
    print("OK: test_helpers.py")
