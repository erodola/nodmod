from nodmod import MODSong, Note, Pattern, Song
from .test_helpers import assert_raises_msg, assert_true


class _NoPlaybackSong(Song):
    def __init__(self) -> None:
        super().__init__()
        self.patterns = [Pattern(2, 1)]
        self.pattern_seq = [0]

    @property
    def file_extension(self) -> str:
        return "nop"

    def save(self, fname: str, verbose: bool = True):  # noqa: ARG002
        raise NotImplementedError

    def timestamp(self) -> list[list[tuple[float, int, int]]]:
        return [[(0.0, 6, 125), (0.12, 6, 125)]]

    def get_effective_row_count(self, sequence_idx: int) -> int:  # noqa: ARG002
        return 2


def test_row_views_order_and_snapshot() -> None:
    song = MODSong()
    song.set_note(0, 2, 1, 3, "C-4", "F06")

    rows = list(song.iter_rows(sequence_only=True))
    assert_true(len(rows) == 64 * len(song.pattern_seq), "iter_rows should cover every row in sequence traversal")

    assert_true(rows[0].sequence_idx == 0 and rows[0].row == 0, "first row order mismatch")
    assert_true(rows[1].sequence_idx == 0 and rows[1].row == 1, "row ordering mismatch")
    assert_true(rows[0].pattern_idx == song.pattern_seq[0], "pattern index payload mismatch")

    target = [r for r in rows if r.sequence_idx == 0 and r.row == 1][0]
    assert_true(isinstance(target.cells, tuple), "RowView cells must be immutable tuple")
    assert_true(len(target.cells) == 4, "RowView channel tuple length mismatch")
    assert_true(
        target.cells[2].instrument_idx == 3 and target.cells[2].period == "C-4" and target.cells[2].effect == "F06",
        "RowView payload mismatch",
    )

    song.set_note(0, 2, 1, 7, "E-4", "A01")
    assert_true(
        target.cells[2].instrument_idx == 3 and target.cells[2].period == "C-4" and target.cells[2].effect == "F06",
        "RowView snapshots must stay immutable",
    )


def test_row_views_sequence_only_behavior() -> None:
    song = MODSong()
    song.add_pattern()
    song.patterns[1].data[0][0] = Note(7, "D-4", "A01")
    song.set_sequence([0])

    seq_rows = list(song.iter_rows(sequence_only=True))
    pool_rows = list(song.iter_rows(sequence_only=False))

    assert_true(len(seq_rows) == 64, "sequence_only=True should only traverse referenced sequence entries")
    assert_true(len(pool_rows) == 128, "sequence_only=False should traverse full pattern pool")

    pool_target = [r for r in pool_rows if r.pattern_idx == 1 and r.row == 0][0]
    assert_true(pool_target.sequence_idx == -1, "pattern-pool traversal should use sequence_idx=-1")
    assert_true(
        pool_target.cells[0].instrument_idx == 7 and pool_target.cells[0].period == "D-4",
        "pattern-pool row payload mismatch",
    )


def test_row_views_reachable_only_guard() -> None:
    song = _NoPlaybackSong()
    assert_raises_msg(
        NotImplementedError,
        "reachable_only=True",
        lambda: list(song.iter_rows(reachable_only=True)),
    )


def test_row_views_reachable_only_follows_playback() -> None:
    song = MODSong()
    song.add_pattern()
    song.set_sequence([0, 1])
    song.set_effect(0, 0, 1, "D03")

    rows = list(song.iter_rows(reachable_only=True))
    assert_true(len(rows) == 63, "reachable row traversal should follow playback truncation/jumps")
    assert_true(rows[0].sequence_idx == 0 and rows[0].row == 0, "reachable traversal first row mismatch")
    assert_true(rows[2].sequence_idx == 1 and rows[2].row == 3, "reachable traversal jump target mismatch")


if __name__ == "__main__":
    test_row_views_order_and_snapshot()
    test_row_views_sequence_only_behavior()
    test_row_views_reachable_only_guard()
    test_row_views_reachable_only_follows_playback()
    print("OK: test_row_views_api.py")
