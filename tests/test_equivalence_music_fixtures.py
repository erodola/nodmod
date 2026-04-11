from __future__ import annotations

from pathlib import Path

from nodmod import MODSong, S3MSong, Song, XMSong, decode_mod_effect, detect_format, load_song
from .test_helpers import (
    assert_true,
    compare_mod_songs,
    compare_s3m_songs,
    compare_xm_songs,
)


ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = ROOT / "music"
FIXTURES = {
    "mod": MUSIC_DIR / "spice_it_up.mod",
    "xm": MUSIC_DIR / "catch_22.xm",
    "s3m": MUSIC_DIR / "celestial_lullabye.s3m",
}


def _assert_fixtures_exist() -> None:
    for path in FIXTURES.values():
        assert_true(path.is_file(), f"Missing fixture: {path}")


def _normalize_volume_fields(note) -> tuple[str | None, int | None, int | None]:
    vol_cmd = getattr(note, "vol_cmd", None)
    if vol_cmd == "":
        vol_cmd = None
    vol_val = getattr(note, "vol_val", None)
    if not isinstance(vol_val, int) or vol_val < 0:
        vol_val = None
    volume = getattr(note, "volume", None)
    if not isinstance(volume, int) or volume < 0:
        volume = None
    return vol_cmd, vol_val, volume


def _iter_pattern_entries(song, sequence_only: bool):
    if sequence_only:
        for seq_idx, pat_idx in enumerate(song.pattern_seq):
            if 0 <= pat_idx < len(song.patterns):
                yield seq_idx, pat_idx
    else:
        for pat_idx in range(len(song.patterns)):
            yield -1, pat_idx


def _manual_mod_resolve_sample(raw_sample: int, period: str, latched_sample: int) -> tuple[int, int]:
    if raw_sample > 0:
        latched_sample = raw_sample
    if period != "":
        if raw_sample > 0:
            return raw_sample, latched_sample
        return latched_sample, latched_sample
    return raw_sample, latched_sample


def _manual_mod_sequence_effective_map(song: MODSong) -> dict[tuple[int, int, int], int]:
    latched = [0] * MODSong.CHANNELS
    out: dict[tuple[int, int, int], int] = {}
    for seq_idx, pat_idx in _iter_pattern_entries(song, sequence_only=True):
        pat = song.patterns[pat_idx]
        for row in range(pat.n_rows):
            for channel in range(pat.n_channels):
                note = pat.data[channel][row]
                raw = getattr(note, "instrument_idx", 0)
                period = getattr(note, "period", "")
                effective, latched[channel] = _manual_mod_resolve_sample(raw, period, latched[channel])
                out[(seq_idx, row, channel)] = effective
    return out


def _manual_mod_used_sample_indices(song: MODSong, scope: str) -> list[int]:
    values: list[int] = []
    latched = [0] * MODSong.CHANNELS

    if scope == "sequence":
        for seq_idx, pat_idx in _iter_pattern_entries(song, sequence_only=True):
            pat = song.patterns[pat_idx]
            for row in range(pat.n_rows):
                for channel in range(pat.n_channels):
                    note = pat.data[channel][row]
                    raw = getattr(note, "instrument_idx", 0)
                    period = getattr(note, "period", "")
                    effective, latched[channel] = _manual_mod_resolve_sample(raw, period, latched[channel])
                    used = effective if period != "" else raw
                    if used > 0:
                        values.append(used)
        return values

    for played in song.iter_playback_rows():
        pat = song.patterns[played.pattern_idx]
        row = played.row
        for channel in range(pat.n_channels):
            note = pat.data[channel][row]
            raw = getattr(note, "instrument_idx", 0)
            period = getattr(note, "period", "")
            effective, latched[channel] = _manual_mod_resolve_sample(raw, period, latched[channel])
            used = effective if period != "" else raw
            if used > 0:
                values.append(used)
    return values


def _manual_iter_cells(song, sequence_only: bool):
    mod_effective = None
    if isinstance(song, MODSong) and sequence_only:
        mod_effective = _manual_mod_sequence_effective_map(song)

    for seq_idx, pat_idx in _iter_pattern_entries(song, sequence_only=sequence_only):
        pat = song.patterns[pat_idx]
        for row in range(pat.n_rows):
            for channel in range(pat.n_channels):
                note = pat.data[channel][row]
                vol_cmd, vol_val, volume = _normalize_volume_fields(note)
                instrument_idx = getattr(note, "instrument_idx", 0)
                if mod_effective is not None:
                    instrument_idx = mod_effective[(seq_idx, row, channel)]
                yield {
                    "sequence_idx": seq_idx,
                    "pattern_idx": pat_idx,
                    "row": row,
                    "channel": channel,
                    "instrument_idx": instrument_idx,
                    "period": getattr(note, "period", ""),
                    "effect": getattr(note, "effect", ""),
                    "vol_cmd": vol_cmd,
                    "vol_val": vol_val,
                    "volume": volume,
                }


def _manual_iter_rows_from_cells(song, sequence_only: bool):
    rows = []
    current = None
    current_cells = []
    for cell in _manual_iter_cells(song, sequence_only=sequence_only):
        key = (cell["sequence_idx"], cell["pattern_idx"], cell["row"])
        if current is None:
            current = key
        if key != current:
            rows.append(
                {
                    "sequence_idx": current[0],
                    "pattern_idx": current[1],
                    "row": current[2],
                    "cells": tuple(current_cells),
                }
            )
            current = key
            current_cells = []
        current_cells.append(cell)
    if current is not None:
        rows.append(
            {
                "sequence_idx": current[0],
                "pattern_idx": current[1],
                "row": current[2],
                "cells": tuple(current_cells),
            }
        )
    return rows


def _flatten_timestamp(song) -> list[tuple[float, int, int]]:
    flat: list[tuple[float, int, int]] = []
    for rows in song.timestamp():
        flat.extend(rows)
    return flat


def _first_use(values: list[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _manual_note_indices_from_scope(song, scope: str) -> list[int]:
    assert_true(scope in {"sequence", "reachable"}, f"Invalid scope {scope}")
    if isinstance(song, MODSong):
        return _manual_mod_used_sample_indices(song, scope)

    values: list[int] = []
    if scope == "sequence":
        for _, pat_idx in _iter_pattern_entries(song, sequence_only=True):
            pat = song.patterns[pat_idx]
            for row in range(pat.n_rows):
                for channel in range(pat.n_channels):
                    idx = getattr(pat.data[channel][row], "instrument_idx", 0)
                    if idx > 0:
                        values.append(idx)
        return values

    for played in song.iter_playback_rows():
        pat = song.patterns[played.pattern_idx]
        row = played.row
        for channel in range(pat.n_channels):
            idx = getattr(pat.data[channel][row], "instrument_idx", 0)
            if idx > 0:
                values.append(idx)
    return values


def _manual_xm_flat_sample_indices(song: XMSong, scope: str) -> list[int]:
    flat_by_instrument: dict[int, list[int]] = {}
    flat_idx = 1
    for inst_idx, inst in enumerate(song.instruments, start=1):
        samples = list(range(flat_idx, flat_idx + len(inst.samples)))
        flat_by_instrument[inst_idx] = samples
        flat_idx += len(inst.samples)

    out: list[int] = []
    note_instruments = _manual_note_indices_from_scope(song, scope=scope)

    if scope == "sequence":
        note_iter = []
        for _, pat_idx in _iter_pattern_entries(song, sequence_only=True):
            pat = song.patterns[pat_idx]
            for row in range(pat.n_rows):
                for channel in range(pat.n_channels):
                    note_iter.append(pat.data[channel][row])
    else:
        note_iter = []
        for played in song.iter_playback_rows():
            pat = song.patterns[played.pattern_idx]
            row = played.row
            for channel in range(pat.n_channels):
                note_iter.append(pat.data[channel][row])

    _ = note_instruments  # used for parity with manual instrument extraction
    for note in note_iter:
        inst_idx = getattr(note, "instrument_idx", 0)
        if inst_idx <= 0 or inst_idx > len(song.instruments):
            continue
        inst = song.instruments[inst_idx - 1]
        inst_flat = flat_by_instrument.get(inst_idx, [])
        if not inst_flat:
            continue
        if len(inst_flat) == 1:
            out.append(inst_flat[0])
            continue
        mapped_flat = None
        period = getattr(note, "period", "")
        if period not in {"", "off"} and len(inst.sample_map) == 96:
            try:
                note_idx = Song.note_to_index(period)
            except (TypeError, ValueError):
                note_idx = None
            if note_idx is not None:
                mapped_local = inst.sample_map[note_idx]
                if 0 <= mapped_local < len(inst_flat):
                    mapped_flat = inst_flat[mapped_local]
        if mapped_flat is not None:
            out.append(mapped_flat)
        else:
            out.extend(inst_flat)
    return out


def _load_manual(path: Path):
    fmt = detect_format(str(path))
    if fmt == "mod":
        song = MODSong()
    elif fmt == "xm":
        song = XMSong()
    elif fmt == "s3m":
        song = S3MSong()
    else:
        raise AssertionError(f"Unexpected format for fixture {path}: {fmt}")
    song.load(str(path), verbose=False)
    return song


def _assert_song_equal(a, b) -> None:
    if isinstance(a, MODSong):
        compare_mod_songs(a, b)
    elif isinstance(a, XMSong):
        compare_xm_songs(a, b)
    elif isinstance(a, S3MSong):
        compare_s3m_songs(a, b)
    else:
        raise AssertionError(f"Unexpected song type: {type(a)}")


def test_load_song_equivalent_to_manual_dispatch() -> None:
    _assert_fixtures_exist()
    for path in FIXTURES.values():
        auto = load_song(str(path), verbose=False)
        manual = _load_manual(path)
        assert_true(type(auto) is type(manual), f"Type mismatch for {path.name}")
        _assert_song_equal(auto, manual)


def test_iter_cells_equivalent_to_direct_pattern_walk() -> None:
    _assert_fixtures_exist()
    for path in FIXTURES.values():
        song = load_song(str(path), verbose=False)
        api_cells = list(song.iter_cells(sequence_only=True))
        manual_cells = list(_manual_iter_cells(song, sequence_only=True))
        assert_true(len(api_cells) == len(manual_cells), f"iter_cells length mismatch for {path.name}")
        for api, man in zip(api_cells, manual_cells):
            assert_true(api.sequence_idx == man["sequence_idx"], "sequence_idx mismatch")
            assert_true(api.pattern_idx == man["pattern_idx"], "pattern_idx mismatch")
            assert_true(api.row == man["row"] and api.channel == man["channel"], "row/channel mismatch")
            assert_true(api.instrument_idx == man["instrument_idx"], "instrument mismatch")
            assert_true(api.period == man["period"] and api.effect == man["effect"], "period/effect mismatch")
            assert_true(api.vol_cmd == man["vol_cmd"] and api.vol_val == man["vol_val"], "volume-column mismatch")
            assert_true(api.volume == man["volume"], "S3M volume mismatch")


def test_iter_rows_equivalent_to_grouped_iter_cells() -> None:
    _assert_fixtures_exist()
    for path in FIXTURES.values():
        song = load_song(str(path), verbose=False)
        for sequence_only in (True, False):
            api_rows = list(song.iter_rows(sequence_only=sequence_only))
            man_rows = _manual_iter_rows_from_cells(song, sequence_only=sequence_only)
            assert_true(len(api_rows) == len(man_rows), f"iter_rows length mismatch for {path.name}")
            for api, man in zip(api_rows, man_rows):
                assert_true(api.sequence_idx == man["sequence_idx"], "iter_rows sequence mismatch")
                assert_true(api.pattern_idx == man["pattern_idx"], "iter_rows pattern mismatch")
                assert_true(api.row == man["row"], "iter_rows row mismatch")
                assert_true(len(api.cells) == len(man["cells"]), "iter_rows channel count mismatch")
                for api_cell, man_cell in zip(api.cells, man["cells"]):
                    assert_true(api_cell.effect == man_cell["effect"], "iter_rows cell effect mismatch")
                    assert_true(api_cell.instrument_idx == man_cell["instrument_idx"], "iter_rows cell instrument mismatch")


def test_iter_effects_equivalent_to_iter_cells_workaround() -> None:
    _assert_fixtures_exist()
    for path in FIXTURES.values():
        song = load_song(str(path), verbose=False)
        for include_empty in (True, False):
            for decoded in (True, False):
                api_effects = list(song.iter_effects(sequence_only=True, include_empty=include_empty, decoded=decoded))
                manual_effects = []
                for cell in song.iter_cells(sequence_only=True):
                    raw = cell.effect
                    if not include_empty and raw == "":
                        continue
                    command = None
                    arg = None
                    x = None
                    y = None
                    extended_cmd = None
                    if decoded and raw != "":
                        try:
                            info = decode_mod_effect(raw)
                        except (TypeError, ValueError):
                            info = None
                        if info is not None:
                            command = info.command
                            arg = info.arg
                            x = info.x
                            y = info.y
                            extended_cmd = info.extended_cmd
                    manual_effects.append((cell.sequence_idx, cell.pattern_idx, cell.row, cell.channel, raw, command, arg, x, y, extended_cmd))
                assert_true(len(api_effects) == len(manual_effects), f"iter_effects length mismatch for {path.name}")
                for api, man in zip(api_effects, manual_effects):
                    assert_true(
                        (api.sequence_idx, api.pattern_idx, api.row, api.channel, api.raw, api.command, api.arg, api.x, api.y, api.extended_cmd) == man,
                        f"iter_effects payload mismatch for {path.name}",
                    )


def test_iter_playback_rows_equivalent_to_timestamp_workflow() -> None:
    _assert_fixtures_exist()
    for path in FIXTURES.values():
        song = load_song(str(path), verbose=False)
        playback = list(song.iter_playback_rows())
        ts = _flatten_timestamp(song)
        assert_true(len(playback) == len(ts), f"playback/timestamp length mismatch for {path.name}")
        for i, row in enumerate(playback):
            assert_true(row.visit_idx == i, "visit_idx mismatch")
            assert_true(row.end_sec >= row.start_sec >= 0.0, "invalid playback timing bounds")
            if i > 0:
                assert_true(abs(row.start_sec - playback[i - 1].end_sec) < 1e-9, "non-contiguous playback row timing")
            ts_sec, ts_speed, ts_tempo = ts[i]
            assert_true(row.speed == ts_speed and row.tempo == ts_tempo, "speed/tempo mismatch")
            if song.file_extension in {"mod", "xm"}:
                assert_true(abs(row.end_sec - ts_sec) < 1e-9, "end_sec mismatch vs timestamp")
            else:
                assert_true(abs(row.start_sec - ts_sec) < 1e-9, "start_sec mismatch vs timestamp")


def test_iter_rows_reachable_equivalent_to_manual_playback_row_build() -> None:
    _assert_fixtures_exist()
    for path in FIXTURES.values():
        song = load_song(str(path), verbose=False)
        playback = list(song.iter_playback_rows())
        reachable_rows = list(song.iter_rows(reachable_only=True))
        assert_true(len(playback) == len(reachable_rows), f"reachable row count mismatch for {path.name}")
        for played, row in zip(playback, reachable_rows):
            assert_true((played.sequence_idx, played.pattern_idx, played.row) == (row.sequence_idx, row.pattern_idx, row.row), "reachable row coord mismatch")
            pat = song.patterns[played.pattern_idx]
            assert_true(len(row.cells) == pat.n_channels, "reachable row channel count mismatch")
            for channel, cell in enumerate(row.cells):
                note = pat.data[channel][played.row]
                assert_true(cell.effect == getattr(note, "effect", ""), "reachable row effect mismatch")
                assert_true(cell.period == getattr(note, "period", ""), "reachable row period mismatch")


def test_used_resources_equivalent_to_manual_scans() -> None:
    _assert_fixtures_exist()
    for path in FIXTURES.values():
        song = load_song(str(path), verbose=False)

        if isinstance(song, XMSong):
            for scope in ("sequence", "reachable"):
                manual_instruments = _first_use(_manual_note_indices_from_scope(song, scope=scope))
                api_first = song.get_used_instruments(scope=scope, order="first_use")
                api_sorted = song.get_used_instruments(scope=scope, order="sorted")
                assert_true(api_first == manual_instruments, f"XM instruments first_use mismatch for {path.name}")
                assert_true(api_sorted == sorted(manual_instruments), f"XM instruments sorted mismatch for {path.name}")

                manual_samples = _first_use(_manual_xm_flat_sample_indices(song, scope=scope))
                api_samples_first = song.get_used_samples(scope=scope, order="first_use")
                api_samples_sorted = song.get_used_samples(scope=scope, order="sorted")
                assert_true(api_samples_first == manual_samples, f"XM samples first_use mismatch for {path.name}")
                assert_true(api_samples_sorted == sorted(manual_samples), f"XM samples sorted mismatch for {path.name}")
        else:
            for scope in ("sequence", "reachable"):
                manual = _first_use(_manual_note_indices_from_scope(song, scope=scope))
                api_first = song.get_used_samples(scope=scope, order="first_use")
                api_sorted = song.get_used_samples(scope=scope, order="sorted")
                assert_true(api_first == manual, f"used_samples first_use mismatch for {path.name}")
                assert_true(api_sorted == sorted(manual), f"used_samples sorted mismatch for {path.name}")


def test_fixture_mod_roundtrip_is_byte_exact(tmp_path) -> None:
    _assert_fixtures_exist()
    source = FIXTURES["mod"]

    song = load_song(str(source), verbose=False)
    out_path = tmp_path / source.name
    song.save(str(out_path), verbose=False)

    assert_true(
        out_path.read_bytes() == source.read_bytes(),
        "MOD fixture save/load/save should be byte-exact to source fixture",
    )


def test_fixture_roundtrip_bytes_are_stable_across_formats(tmp_path) -> None:
    _assert_fixtures_exist()
    for path in FIXTURES.values():
        out1 = tmp_path / f"round1_{path.name}"
        out2 = tmp_path / f"round2_{path.name}"

        song1 = load_song(str(path), verbose=False)
        song1.save(str(out1), verbose=False)

        song2 = load_song(str(out1), verbose=False)
        song2.save(str(out2), verbose=False)

        assert_true(
            out1.read_bytes() == out2.read_bytes(),
            f"Roundtrip bytes should stabilize after one pass for {path.name}",
        )


if __name__ == "__main__":
    test_load_song_equivalent_to_manual_dispatch()
    test_iter_cells_equivalent_to_direct_pattern_walk()
    test_iter_rows_equivalent_to_grouped_iter_cells()
    test_iter_effects_equivalent_to_iter_cells_workaround()
    test_iter_playback_rows_equivalent_to_timestamp_workflow()
    test_iter_rows_reachable_equivalent_to_manual_playback_row_build()
    test_used_resources_equivalent_to_manual_scans()
    print("OK: test_equivalence_music_fixtures.py")
