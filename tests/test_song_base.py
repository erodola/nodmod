from nodmod import MODSong, XMSong
from .test_helpers import assert_true, assert_raises_msg


def test_song_base() -> None:
    assert_true(MODSong.note_to_index("C-4") == 36, "note_to_index C-4")
    assert_true(MODSong.index_to_note(36) == "C-4", "index_to_note 36")
    assert_raises_msg(ValueError, "Invalid note format", MODSong.note_to_index, "C4")
    assert_raises_msg(IndexError, "Invalid note index", MODSong.index_to_note, 96)
    assert_true(abs(MODSong.get_tick_duration(125) - 0.02) < 1e-6, "Tick duration mismatch")
    artist, name = MODSong.artist_songname_from_filename("Artist - Track.mod")
    assert_true(artist == "Artist" and name == "Track", "artist_songname_from_filename mismatch")

    mod = MODSong()
    mod.set_note(0, 0, 0, 1, "C-4")
    assert_raises_msg(IndexError, "Invalid sequence index", mod.set_note, -1, 0, 0, 1, "C-4")
    assert_raises_msg(IndexError, "Invalid channel index", mod.set_note, 0, -1, 0, 1, "C-4")
    assert_raises_msg(IndexError, "Invalid row index", mod.set_effect, 0, 0, -1, "F06")

    xm = XMSong()
    xm.n_channels = 1
    xm.add_pattern(4)
    assert_raises_msg(IndexError, "Invalid song restart position", xm.set_song_restart, -1)
    assert_raises_msg(IndexError, "Invalid sequence index", xm.get_effective_row_count, -1)


if __name__ == '__main__':
    test_song_base()
    print("OK: test_song_base.py")
