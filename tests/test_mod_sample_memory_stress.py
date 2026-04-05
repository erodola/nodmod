from __future__ import annotations

import random

from nodmod import MODSong
from .test_helpers import assert_true


_PERIODS = ["", "C-2", "D#2", "F-3", "G#3", "A-4", "C-5", "D-4", "B-5"]
_EFFECTS = [
    "",
    "A01",
    "A0F",
    "C20",
    "F06",
    "F7D",
    "B00",
    "B01",
    "B02",
    "D00",
    "D16",
    "D31",
    "E60",
    "E61",
    "E62",
    "E63",
]


def _resolve_mod_cell(raw_sample: int, period: str, latched_sample: int) -> tuple[int, int]:
    if raw_sample > 0:
        latched_sample = raw_sample
    if period != "":
        if raw_sample > 0:
            return raw_sample, latched_sample
        return latched_sample, latched_sample
    return raw_sample, latched_sample


def _oracle_sequence(song: MODSong) -> tuple[dict[tuple[int, int, int], int], list[int]]:
    latched = [0] * MODSong.CHANNELS
    effective: dict[tuple[int, int, int], int] = {}
    first_use: list[int] = []
    seen: set[int] = set()

    for seq_idx, pat_idx in enumerate(song.pattern_seq):
        if pat_idx < 0 or pat_idx >= len(song.patterns):
            continue
        pat = song.patterns[pat_idx]
        for row in range(pat.n_rows):
            for channel in range(min(pat.n_channels, MODSong.CHANNELS)):
                note = pat.data[channel][row]
                raw = int(getattr(note, "instrument_idx", 0))
                period = getattr(note, "period", "")
                eff, latched[channel] = _resolve_mod_cell(raw, period, latched[channel])
                effective[(seq_idx, row, channel)] = eff
                used = eff if period != "" else raw
                if used > 0 and used not in seen:
                    seen.add(used)
                    first_use.append(used)
    return effective, first_use


def _oracle_reachable(song: MODSong) -> tuple[dict[tuple[int, int], int], list[int]]:
    latched = [0] * MODSong.CHANNELS
    effective: dict[tuple[int, int], int] = {}
    first_use: list[int] = []
    seen: set[int] = set()

    for played in song.iter_playback_rows():
        if played.pattern_idx < 0 or played.pattern_idx >= len(song.patterns):
            continue
        pat = song.patterns[played.pattern_idx]
        if played.row < 0 or played.row >= pat.n_rows:
            continue
        for channel in range(min(pat.n_channels, MODSong.CHANNELS)):
            note = pat.data[channel][played.row]
            raw = int(getattr(note, "instrument_idx", 0))
            period = getattr(note, "period", "")
            eff, latched[channel] = _resolve_mod_cell(raw, period, latched[channel])
            effective[(played.visit_idx, channel)] = eff
            used = eff if period != "" else raw
            if used > 0 and used not in seen:
                seen.add(used)
                first_use.append(used)
    return effective, first_use


def _random_cell(rng: random.Random, song: MODSong) -> tuple[int, int, int]:
    seq_idx = rng.randrange(len(song.pattern_seq))
    row = rng.randrange(MODSong.ROWS)
    channel = rng.randrange(MODSong.CHANNELS)
    return seq_idx, row, channel


def _random_mutation(rng: random.Random, song: MODSong) -> None:
    op = rng.randrange(16)
    if op == 0:
        seq, row, ch = _random_cell(rng, song)
        song.set_note(seq, ch, row, rng.randint(0, 9), rng.choice(_PERIODS), rng.choice(_EFFECTS))
        return
    if op == 1:
        seq, row, ch = _random_cell(rng, song)
        song.clear_note(seq, ch, row)
        return
    if op == 2:
        seq, row, ch = _random_cell(rng, song)
        song.set_effect(seq, ch, row, rng.choice(_EFFECTS))
        return
    if op == 3:
        seq = rng.randrange(len(song.pattern_seq))
        row = rng.randrange(MODSong.ROWS)
        song.clear_row(seq, row)
        return
    if op == 4:
        seq = rng.randrange(len(song.pattern_seq))
        song.shift_pattern(seq, rng.randint(-3, 3))
        return
    if op == 5:
        src_seq = rng.randrange(len(song.pattern_seq))
        dst_seq = rng.randrange(len(song.pattern_seq))
        src_row = rng.randrange(MODSong.ROWS)
        dst_row = rng.randrange(MODSong.ROWS)
        song.copy_row(src_seq, src_row, dst_seq, dst_row)
        return
    if op == 6:
        src_seq = rng.randrange(len(song.pattern_seq))
        dst_seq = rng.randrange(len(song.pattern_seq))
        src_ch = rng.randrange(MODSong.CHANNELS)
        dst_ch = rng.randrange(MODSong.CHANNELS)
        song.copy_channel_data(src_seq, src_ch, dst_seq, dst_ch)
        return
    if op == 7:
        ch1 = rng.randrange(MODSong.CHANNELS)
        ch2 = (ch1 + rng.randint(1, MODSong.CHANNELS - 1)) % MODSong.CHANNELS
        song.swap_channels(ch1, ch2)
        return
    if op == 8:
        if len(song.patterns) < 12:
            song.add_pattern()
        else:
            seq = rng.randrange(len(song.pattern_seq))
            song.clear_pattern(seq)
        return
    if op == 9:
        if len(song.pattern_seq) > 1:
            song.remove_pattern(rng.randrange(len(song.pattern_seq)))
        else:
            song.add_pattern()
        return
    if op == 10:
        if len(song.pattern_seq) > 0 and len(song.patterns) < 14:
            song.insert_pattern(rng.randrange(len(song.pattern_seq)), after=bool(rng.randrange(2)))
        else:
            seq = [rng.randrange(len(song.patterns)) for _ in range(rng.randint(1, max(1, len(song.patterns))))]
            song.set_sequence(seq)
        return
    if op == 11:
        if len(song.pattern_seq) > 0 and len(song.patterns) < 14:
            song.duplicate_pattern(rng.randrange(len(song.pattern_seq)))
        else:
            song.keep_pattern(rng.randrange(len(song.pattern_seq)))
        return
    if op == 12:
        seq = [rng.randrange(len(song.patterns)) for _ in range(rng.randint(1, min(18, max(1, len(song.patterns) * 2))))]
        song.set_sequence(seq)
        return
    if op == 13:
        if len(song.pattern_seq) > 1:
            song.remove_patterns_after(rng.randrange(len(song.pattern_seq)))
        else:
            song.add_to_sequence(rng.randrange(len(song.patterns)))
        return
    if op == 14:
        song.clear_channel(rng.randrange(MODSong.CHANNELS))
        return
    song.mute_channel(rng.randrange(MODSong.CHANNELS))


def test_mod_randomized_mutation_and_resolution_stress() -> None:
    rng = random.Random(934_771)
    song = MODSong()
    for _ in range(3):
        song.add_pattern()
    song.set_sequence([rng.randrange(len(song.patterns)) for _ in range(6)])

    for step in range(220):
        _ = song.get_note(0, 0, 0)
        assert_true(
            song._resolved_sequence_cache_version == song._resolution_version,
            "sequence cache should be warm after read",
        )
        seq_cache_obj = song._resolved_sequence_cells

        _ = song.get_used_samples(scope="reachable", order="first_use")
        assert_true(
            song._resolved_reachable_cache_version == song._resolution_version,
            "reachable cache should be warm after read",
        )
        reach_cache_obj = song._resolved_reachable_cells

        _random_mutation(rng, song)

        assert_true(
            song._resolved_sequence_cache_version != song._resolution_version,
            "sequence cache should be invalidated by mutation",
        )
        assert_true(
            song._resolved_reachable_cache_version != song._resolution_version,
            "reachable cache should be invalidated by mutation",
        )

        seq_oracle, seq_first_use = _oracle_sequence(song)
        assert_true(
            song.get_used_samples(scope="sequence", order="first_use") == seq_first_use,
            f"sequence first_use mismatch at step {step}",
        )
        assert_true(
            song.get_used_samples(scope="sequence", order="sorted") == sorted(seq_first_use),
            f"sequence sorted mismatch at step {step}",
        )

        assert_true(
            song._resolved_sequence_cache_version == song._resolution_version,
            f"sequence cache should rebuild lazily at step {step}",
        )
        assert_true(
            song._resolved_sequence_cells is not seq_cache_obj,
            "sequence cache object should refresh after invalidation/rebuild",
        )

        for _ in range(12):
            seq_idx, row, channel = _random_cell(rng, song)
            raw = song.get_note_raw(seq_idx, row, channel)
            resolved = song.get_note(seq_idx, row, channel)
            expected = seq_oracle[(seq_idx, row, channel)]
            assert_true(
                resolved.instrument_idx == expected,
                f"resolved get_note mismatch at step={step}, seq={seq_idx}, row={row}, ch={channel}",
            )
            assert_true(
                raw.instrument_idx == song.patterns[song.pattern_seq[seq_idx]].data[channel][row].instrument_idx,
                "raw getter mismatch",
            )

        cells_resolved = list(song.iter_cells(sequence_only=True))
        cells_raw = list(song.iter_cells(sequence_only=True, resolved=False))
        assert_true(len(cells_resolved) == len(cells_raw), "resolved/raw cell count mismatch")
        for idx in range(0, len(cells_resolved), max(1, len(cells_resolved) // 64)):
            cell_r = cells_resolved[idx]
            cell_raw = cells_raw[idx]
            exp = seq_oracle[(cell_r.sequence_idx, cell_r.row, cell_r.channel)]
            assert_true(cell_r.instrument_idx == exp, f"iter_cells resolved mismatch at step {step}")
            raw_note = song.patterns[song.pattern_seq[cell_raw.sequence_idx]].data[cell_raw.channel][cell_raw.row]
            assert_true(cell_raw.instrument_idx == raw_note.instrument_idx, "iter_cells raw mismatch")

        if step % 11 == 0:
            reach_oracle, reach_first_use = _oracle_reachable(song)
            assert_true(
                song.get_used_samples(scope="reachable", order="first_use") == reach_first_use,
                f"reachable first_use mismatch at step {step}",
            )
            assert_true(
                song.get_used_samples(scope="reachable", order="sorted") == sorted(reach_first_use),
                f"reachable sorted mismatch at step {step}",
            )

            rows = list(song.iter_rows(reachable_only=True))
            for i, row_view in enumerate(rows[: min(80, len(rows))]):
                played = song._resolved_reachable_rows[i]
                for channel, cell in enumerate(row_view.cells):
                    expected = reach_oracle.get((played.visit_idx, channel), 0)
                    assert_true(cell.instrument_idx == expected, f"reachable row mismatch at step {step}")
            assert_true(
                song._resolved_reachable_cache_version == song._resolution_version,
                f"reachable cache should be warm after reads at step {step}",
            )
            assert_true(
                song._resolved_reachable_cells is not reach_cache_obj,
                "reachable cache object should refresh after invalidation/rebuild",
            )


def test_mod_reachable_rows_random_control_flow_consistency() -> None:
    rng = random.Random(551_204)
    song = MODSong()
    for _ in range(5):
        song.add_pattern()
    song.set_sequence([rng.randrange(len(song.patterns)) for _ in range(10)])

    for seq_idx in range(len(song.pattern_seq)):
        for row in range(MODSong.ROWS):
            for channel in range(MODSong.CHANNELS):
                if rng.random() < 0.18:
                    song.set_note(seq_idx, channel, row, rng.randint(0, 8), rng.choice(_PERIODS), "")
                if rng.random() < 0.07:
                    song.set_effect(seq_idx, channel, row, rng.choice(_EFFECTS))

    reach_oracle, reach_first_use = _oracle_reachable(song)
    api_rows = list(song.iter_rows(reachable_only=True))
    api_used_first = song.get_used_samples(scope="reachable", order="first_use")
    api_used_sorted = song.get_used_samples(scope="reachable", order="sorted")

    assert_true(api_used_first == reach_first_use, "reachable first_use mismatch for random control flow")
    assert_true(api_used_sorted == sorted(reach_first_use), "reachable sorted mismatch for random control flow")
    assert_true(len(api_rows) == len(song._resolved_reachable_rows), "reachable row count mismatch")

    for i, row_view in enumerate(api_rows[: min(120, len(api_rows))]):
        played = song._resolved_reachable_rows[i]
        assert_true(row_view.sequence_idx == played.sequence_idx, "reachable row sequence mismatch")
        assert_true(row_view.row == played.row, "reachable row index mismatch")
        for channel, cell in enumerate(row_view.cells):
            expected = reach_oracle.get((played.visit_idx, channel), 0)
            assert_true(cell.instrument_idx == expected, "reachable row effective sample mismatch")


if __name__ == "__main__":
    test_mod_randomized_mutation_and_resolution_stress()
    test_mod_reachable_rows_random_control_flow_consistency()
    print("OK: test_mod_sample_memory_stress.py")
