import os

from nodmod import MODSong, S3MSong, XMSong
from .test_helpers import assert_true


def test_mod_ascii_matches_save(tmp_dir: str) -> None:
    song = MODSong()
    song.set_note(0, 0, 0, 1, "C-4", "F06")
    text = song.to_ascii()
    out_path = os.path.join(tmp_dir, "mod_ascii_api.txt")
    song.save_ascii(out_path, verbose=False)
    with open(out_path, "r", encoding="ascii") as handle:
        saved = handle.read()
    assert_true(text == saved, "MOD save_ascii should delegate to to_ascii")
    assert_true("| C-4 01 F06 " in text, "MOD dump should include tracker cell formatting")


def test_xm_ascii_matches_save(tmp_dir: str) -> None:
    song = XMSong()
    song.set_n_channels(2)
    song.set_note(0, 0, 0, 1, "C-4", "F06", vol_cmd="v", vol_val=32)
    text = song.to_ascii(sequence_only=True, include_headers=True)
    out_path = os.path.join(tmp_dir, "xm_ascii_api.txt")
    song.save_ascii(out_path, verbose=False)
    with open(out_path, "r", encoding="ascii") as handle:
        saved = handle.read()
    assert_true(text == saved, "XM save_ascii should delegate to to_ascii with headers")
    assert_true("# Pattern 0 (unique pattern 0):" in text, "XM header should be present")


def test_ascii_determinism_and_pool_mode() -> None:
    song = S3MSong()
    song.set_note(0, 0, 0, 1, "C-4", "A03", 20)
    song.add_pattern()
    song.set_note(1, 0, 0, 2, "E-4", "T96", 30)

    seq_dump_1 = song.to_ascii(sequence_only=True, include_headers=False)
    seq_dump_2 = song.to_ascii(sequence_only=True, include_headers=False)
    pool_dump = song.to_ascii(sequence_only=False, include_headers=True)

    assert_true(seq_dump_1 == seq_dump_2, "to_ascii output should be deterministic")
    assert_true("# Pattern pool 0:" in pool_dump, "pool dump should include pool headers")
    assert_true("| C-4 01 v20 A03 " in pool_dump, "S3M dump should include note formatting")


if __name__ == "__main__":
    tmp_dir = os.path.join(os.getcwd(), "dev")
    test_mod_ascii_matches_save(tmp_dir)
    test_xm_ascii_matches_save(tmp_dir)
    test_ascii_determinism_and_pool_mode()
    print("OK: test_ascii_api.py")
