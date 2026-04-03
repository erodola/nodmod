"""Helpers for decoding and encoding MOD-style three-character effects."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    'EffectInfo',
    'decode_mod_effect',
    'encode_mod_effect',
    'split_xy',
    'merge_xy',
    'is_speed_effect',
    'is_tempo_effect',
]


@dataclass(frozen=True)
class EffectInfo:
    """Decoded MOD-style effect information."""

    raw: str
    command: str
    arg: int
    x: int
    y: int
    is_extended: bool
    extended_cmd: str | None


def split_xy(arg: int) -> tuple[int, int]:
    """Split an 8-bit argument into high and low nibbles."""
    if not isinstance(arg, int):
        raise TypeError(f"Invalid arg type {type(arg).__name__} (expected int).")
    if arg < 0 or arg > 255:
        raise ValueError(f"Invalid arg {arg} (expected 0-255).")
    return (arg >> 4) & 0x0F, arg & 0x0F


def merge_xy(x: int, y: int) -> int:
    """Merge high/low nibbles into one 8-bit argument."""
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("x and y must be integers.")
    if x < 0 or x > 15:
        raise ValueError(f"Invalid x nibble {x} (expected 0-15).")
    if y < 0 or y > 15:
        raise ValueError(f"Invalid y nibble {y} (expected 0-15).")
    return (x << 4) | y


def encode_mod_effect(command: str, arg: int) -> str:
    """Encode a MOD-style effect command and argument into canonical text."""
    if not isinstance(command, str):
        raise TypeError(f"Invalid command type {type(command).__name__} (expected str).")
    cmd = command.strip().upper()
    if len(cmd) != 1 or cmd[0] not in '0123456789ABCDEF':
        raise ValueError(f"Invalid MOD effect command {command!r} (expected one hex character).")
    if not isinstance(arg, int):
        raise TypeError(f"Invalid arg type {type(arg).__name__} (expected int).")
    if arg < 0 or arg > 255:
        raise ValueError(f"Invalid arg {arg} (expected 0-255).")
    return f"{cmd}{arg:02X}"


def decode_mod_effect(effect: str) -> EffectInfo:
    """Decode a MOD-style three-character effect string."""
    if not isinstance(effect, str):
        raise TypeError(f"Invalid effect type {type(effect).__name__} (expected str).")
    raw = effect.strip().upper()
    if len(raw) != 3:
        raise ValueError(f"Invalid MOD effect {effect!r} (expected 3 characters).")
    cmd = raw[0]
    if cmd not in '0123456789ABCDEF':
        raise ValueError(f"Invalid MOD effect command {cmd!r} in {effect!r}.")
    try:
        arg = int(raw[1:], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid MOD effect argument {raw[1:]!r} in {effect!r}.") from exc
    x, y = split_xy(arg)
    is_extended = cmd == 'E'
    return EffectInfo(
        raw=raw,
        command=cmd,
        arg=arg,
        x=x,
        y=y,
        is_extended=is_extended,
        extended_cmd=f"E{x:X}" if is_extended else None,
    )


def is_speed_effect(effect: str) -> bool:
    """Return True if effect is a classic Fxx speed command (1..31)."""
    info = decode_mod_effect(effect)
    return info.command == 'F' and 1 <= info.arg <= 31


def is_tempo_effect(effect: str) -> bool:
    """Return True if effect is a classic Fxx tempo command (32..255)."""
    info = decode_mod_effect(effect)
    return info.command == 'F' and info.arg >= 32
