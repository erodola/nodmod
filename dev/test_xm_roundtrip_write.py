from __future__ import annotations

import os
import random
from typing import List

from nodmod import XMSong
from nodmod.types import Pattern


def find_xm_files(root: str) -> List[str]:
    matches: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".xm"):
                matches.append(os.path.join(dirpath, name))
    return matches


def copy_instruments(src: XMSong, dst: XMSong) -> None:
    for i in range(1, len(src.instruments) + 1):
        dst.copy_instrument_from(src, i)


def copy_patterns_with_set_note(src: XMSong, dst: XMSong) -> None:
    # Build destination patterns list (empty)
    dst.patterns = []
    for pat in src.patterns:
        dst.patterns.append(Pattern(pat.n_rows, pat.n_channels))

    # Temporarily set sequence to identity to address patterns by index
    dst.pattern_seq = list(range(len(dst.patterns)))

    for p_idx, pat in enumerate(src.patterns):
        for r in range(pat.n_rows):
            for c in range(pat.n_channels):
                note = pat.data[c][r]
                vol_cmd = getattr(note, "vol_cmd", None)
                vol_val = getattr(note, "vol_val", None)
                dst.set_note(
                    sequence_idx=p_idx,
                    channel=c,
                    row=r,
                    instrument_idx=note.instrument_idx,
                    period=note.period,
                    effect=note.effect,
                    vol_cmd=vol_cmd,
                    vol_val=vol_val,
                )

    # Restore the original sequence
    dst.pattern_seq = list(src.pattern_seq)


def roundtrip_random_xm(root: str, count: int = 100) -> int:
    files = find_xm_files(root)
    if not files:
        print(f"No .xm files found under: {root}")
        return 2

    rng = random.Random(0)
    sample = files if len(files) <= count else rng.sample(files, count)

    failures = 0
    skipped = 0
    for idx, src_path in enumerate(sample, start=1):
        src = XMSong()
        try:
            src.load(src_path, verbose=False)
        except Exception as exc:
            skipped += 1
            print(f"SKIP {idx}/{len(sample)}: {src_path} ({type(exc).__name__}: {exc})")
            continue

        dst = XMSong()
        dst.artist = src.artist
        dst.songname = src.songname

        dst.tracker_name = src.tracker_name
        dst.song_restart = src.song_restart
        dst.flags = src.flags
        dst.default_speed = src.default_speed
        dst.default_tempo = src.default_tempo
        dst.n_channels = src.n_channels

        copy_instruments(src, dst)
        copy_patterns_with_set_note(src, dst)

        out_path = os.path.join(os.getcwd(), "dev", f"_roundtrip_test_{idx}.xm")
        dst.save(out_path, verbose=False)

        baseline_path = os.path.join(os.getcwd(), "dev", f"_roundtrip_baseline_{idx}.xm")
        src.save(baseline_path, verbose=False)

        with open(baseline_path, "rb") as f:
            base_bytes = f.read()
        with open(out_path, "rb") as f:
            dst_bytes = f.read()

        if base_bytes != dst_bytes:
            failures += 1
            mismatch_idx = next((i for i in range(min(len(base_bytes), len(dst_bytes))) if base_bytes[i] != dst_bytes[i]), None)
            print(f"Mismatch {idx}/{len(sample)}: {src_path}")
            if mismatch_idx is not None:
                print(f"First mismatch at byte {mismatch_idx}: base={base_bytes[mismatch_idx]:02X} dst={dst_bytes[mismatch_idx]:02X}")
            else:
                print(f"Length mismatch: base={len(base_bytes)} dst={len(dst_bytes)}")
        else:
            print(f"OK {idx}/{len(sample)}: {src_path}")

    print(f"\nSummary: {len(sample)} files, {failures} mismatches, {skipped} skipped")
    return 1 if failures else 0


def main() -> int:
    root = r"G:\My Drive\Moduli"
    return roundtrip_random_xm(root, count=10**9)


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        print("OK: test_xm_roundtrip_write.py")
    raise SystemExit(rc)

