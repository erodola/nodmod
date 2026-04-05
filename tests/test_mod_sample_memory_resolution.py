from __future__ import annotations

from nodmod import MODSong
from .test_helpers import assert_true


def test_mod_motif_sequence_resolution_and_raw_access() -> None:
    song = MODSong()
    song.set_note(0, 0, 0, 3, "D-5", "")
    song.set_note(0, 0, 1, 0, "C-5", "")
    song.set_note(0, 0, 2, 0, "D-4", "")
    song.set_note(0, 0, 3, 3, "C-4", "")

    resolved = [song.get_note(0, row, 0).instrument_idx for row in range(4)]
    raw = [song.get_note_raw(0, row, 0).instrument_idx for row in range(4)]
    assert_true(resolved == [3, 3, 3, 3], "MOD resolved motif mismatch")
    assert_true(raw == [3, 0, 0, 3], "MOD raw motif mismatch")

    cells_resolved = [c for c in song.iter_cells(sequence_only=True) if c.sequence_idx == 0 and c.channel == 0 and c.row < 4]
    cells_raw = list(song.iter_cells(sequence_only=True, resolved=False))
    cells_raw = [c for c in cells_raw if c.sequence_idx == 0 and c.channel == 0 and c.row < 4]
    assert_true([c.instrument_idx for c in cells_resolved] == [3, 3, 3, 3], "iter_cells resolved motif mismatch")
    assert_true([c.instrument_idx for c in cells_raw] == [3, 0, 0, 3], "iter_cells raw motif mismatch")


def test_mod_first_note_without_sample_resolves_to_zero() -> None:
    song = MODSong()
    song.set_note(0, 0, 0, 0, "C-5", "")
    assert_true(song.get_note(0, 0, 0).instrument_idx == 0, "first note without sample should resolve to 0")
    assert_true(song.get_note_raw(0, 0, 0).instrument_idx == 0, "raw sample should remain 0")


def test_mod_cross_pattern_carry_sequence_scope() -> None:
    song = MODSong()
    song.add_pattern()
    song.set_sequence([0, 1])

    song.set_note(0, 0, 0, 4, "C-4", "")
    song.set_note(1, 0, 0, 0, "D-4", "")
    song.set_note(1, 0, 1, 0, "E-4", "")

    assert_true(song.get_note(1, 0, 0).instrument_idx == 4, "cross-pattern carry at row 0 mismatch")
    assert_true(song.get_note(1, 1, 0).instrument_idx == 4, "cross-pattern carry at row 1 mismatch")


def test_mod_pattern_reuse_depends_on_sequence_context() -> None:
    song = MODSong()
    song.add_pattern()
    song.add_pattern()
    song.set_sequence([0, 1, 2, 1])

    song.set_note(0, 0, 0, 2, "C-4", "")
    song.set_note(2, 0, 0, 7, "C-5", "")
    song.set_note(1, 0, 0, 0, "D-4", "")

    assert_true(song.get_note(1, 0, 0).instrument_idx == 2, "first reuse context mismatch")
    assert_true(song.get_note(3, 0, 0).instrument_idx == 7, "second reuse context mismatch")


def test_mod_reachable_resolution_with_position_jump() -> None:
    song = MODSong()
    song.add_pattern()
    song.set_sequence([0, 1])

    song.set_note(0, 0, 0, 1, "C-4", "B01")
    song.set_note(0, 0, 1, 2, "E-4", "")
    song.set_note(1, 0, 0, 0, "D-4", "")

    used_reachable = song.get_used_samples(scope="reachable", order="sorted")
    assert_true(used_reachable == [1], "reachable used samples should honor control-flow and carry semantics")

    rows = list(song.iter_rows(reachable_only=True))
    target_rows = [row for row in rows if row.sequence_idx == 1 and row.row == 0]
    assert_true(len(target_rows) == 1, "expected one reachable row at sequence 1 row 0")
    assert_true(target_rows[0].cells[0].instrument_idx == 1, "reachable row effective sample mismatch")


def test_mod_resolution_cache_lifecycle() -> None:
    song = MODSong()
    assert_true(song._resolved_sequence_cache_version == -1, "sequence cache should start cold")

    _ = song.get_note(0, 0, 0)
    cold_version = song._resolved_sequence_cache_version
    cache_id = id(song._resolved_sequence_cells)
    assert_true(cold_version == song._resolution_version, "sequence cache should warm on first query")

    _ = song.get_note(0, 0, 0)
    _ = list(song.iter_cells(sequence_only=True))
    assert_true(song._resolved_sequence_cache_version == cold_version, "cache version should stay warm on reads")
    assert_true(id(song._resolved_sequence_cells) == cache_id, "cache object should be reused while clean")

    song.set_note(0, 0, 1, 5, "D-4", "")
    assert_true(song._resolved_sequence_cache_version != song._resolution_version, "mutation should invalidate cache")

    _ = song.get_note(0, 1, 0)
    assert_true(song._resolved_sequence_cache_version == song._resolution_version, "cache should rebuild lazily after mutation")


if __name__ == "__main__":
    test_mod_motif_sequence_resolution_and_raw_access()
    test_mod_first_note_without_sample_resolves_to_zero()
    test_mod_cross_pattern_carry_sequence_scope()
    test_mod_pattern_reuse_depends_on_sequence_context()
    test_mod_reachable_resolution_with_position_jump()
    test_mod_resolution_cache_lifecycle()
    print("OK: test_mod_sample_memory_resolution.py")
