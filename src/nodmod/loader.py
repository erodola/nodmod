"""Convenience helpers for loading tracker songs with automatic format dispatch."""

from __future__ import annotations

from .modsong import MODSong
from .probe import detect_format, probe_file
from .s3msong import S3MSong
from .xmsong import XMSong

__all__ = ["load_song"]


def load_song(path: str, *, verbose: bool = False):
    """Detect file format, instantiate the matching Song subtype, and load it."""
    detected = detect_format(path)
    if detected is None:
        raise ValueError(f"Unsupported or unrecognized tracker module format: {path!r}.")

    probe = probe_file(path)
    if not probe.supported:
        if detected == "mod":
            fmt = "MOD"
        elif detected == "xm":
            fmt = "XM"
        elif detected == "s3m":
            fmt = "S3M"
        else:
            fmt = detected.upper()
        details = "; ".join(probe.errors) if probe.errors else "unsupported module variant."
        raise NotImplementedError(f"Unsupported {fmt} module: {details}")

    if detected == "mod":
        song = MODSong()
    elif detected == "xm":
        song = XMSong()
    elif detected == "s3m":
        song = S3MSong()
    else:
        raise ValueError(f"Unsupported or unrecognized tracker module format: {path!r}.")

    song.load(path, verbose=verbose)
    return song
