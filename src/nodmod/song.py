"""
Abstract base class for tracker song formats.

The Song class defines the common interface for all tracker formats (MOD, XM, etc.).
Format-specific implementations are in modsong.py, xmsong.py, etc.
"""

import copy
import os
import shutil
import subprocess
from abc import ABC, abstractmethod

from .effects import decode_mod_effect
from .types import Note
from .views import CellView, EffectView, RowView, SampleView, SongView

__all__ = ['Song']


class Song(ABC):
    """
    A song is a collection of patterns played in a specific sequence, possibly with repetitions.
    Songs can be loaded from file (e.g., MOD, XM format) or composed from scratch.
    
    Note: Sample/instrument storage is format-specific:
    - MODSong stores samples directly (notes reference samples)
    - XMSong stores instruments (notes reference instruments, which contain samples)
    """

    PERIOD_SEQ = ['C-', 'C#', 'D-', 'D#', 'E-', 'F-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']
    PRESERVED_EFFECT_PREFIXES = frozenset({'B', 'C', 'D', 'E', 'F'})

    def __init__(self):
        """Initialize shared song metadata, pattern storage, and sequence state."""

        self.artist = ""
        self.songname = "new song"
        self.patterns = []
        self.pattern_seq = []  # The actual sequence of patterns making up the song.

    def __repr__(self) -> str:
        n_channels = getattr(self, 'n_channels', self.patterns[0].n_channels if self.patterns else 0)
        return (
            f"{self.__class__.__name__}(songname={self.songname!r}, "
            f"patterns={len(self.patterns)}, sequence={len(self.pattern_seq)}, channels={n_channels})"
        )

    __str__ = __repr__

    '''
    -------------------------------------
    SONG
    -------------------------------------
    '''

    def set_artist(self, artist_name: str):
        """Set the song artist metadata string."""
        self.artist = artist_name

    def set_songname(self, song_name: str):
        """Set the song title metadata string."""
        self.songname = song_name

    def view(self) -> SongView:
        """Return an immutable song-level snapshot."""
        n_channels = getattr(self, 'n_channels', self.patterns[0].n_channels if self.patterns else 0)
        return SongView(
            format=self.file_extension,
            songname=self.songname,
            artist=self.artist,
            sequence=tuple(self.pattern_seq),
            n_patterns=len(self.patterns),
            n_channels=n_channels,
        )

    def _iter_pattern_entries(self, sequence_only: bool) -> list[tuple[int, int]]:
        """Build deterministic traversal entries as (sequence_idx, pattern_idx)."""
        if sequence_only:
            return [(seq_idx, pat_idx) for seq_idx, pat_idx in enumerate(self.pattern_seq)]
        return [(-1, pat_idx) for pat_idx in range(len(self.patterns))]

    @staticmethod
    def _make_cell_view(
        *,
        sequence_idx: int,
        pattern_idx: int,
        row: int,
        channel: int,
        note,
    ) -> CellView:
        """Build one immutable cell snapshot from a concrete note object."""
        vol_cmd = getattr(note, 'vol_cmd', None)
        if vol_cmd == '':
            vol_cmd = None
        vol_val = getattr(note, 'vol_val', None)
        if not isinstance(vol_val, int) or vol_val < 0:
            vol_val = None
        volume = getattr(note, 'volume', None)
        if not isinstance(volume, int) or volume < 0:
            volume = None
        return CellView(
            sequence_idx=sequence_idx,
            pattern_idx=pattern_idx,
            row=row,
            channel=channel,
            instrument_idx=getattr(note, 'instrument_idx', 0),
            period=getattr(note, 'period', ''),
            effect=getattr(note, 'effect', ''),
            vol_cmd=vol_cmd,
            vol_val=vol_val,
            volume=volume,
        )

    def iter_cells(self, *, sequence_only: bool = True):
        """Yield immutable cell snapshots in deterministic sequence,row,channel order."""
        for sequence_idx, pattern_idx in self._iter_pattern_entries(sequence_only):
            pat = self.patterns[pattern_idx]
            for row in range(pat.n_rows):
                for channel in range(pat.n_channels):
                    note = pat.data[channel][row]
                    yield self._make_cell_view(
                        sequence_idx=sequence_idx,
                        pattern_idx=pattern_idx,
                        row=row,
                        channel=channel,
                        note=note,
                    )

    def iter_rows(self, *, sequence_only: bool = True, reachable_only: bool = False):
        """Yield immutable row snapshots in deterministic sequence,row order."""
        if reachable_only:
            try:
                for played_row in self.iter_playback_rows():
                    pat = self.patterns[played_row.pattern_idx]
                    cells = tuple(
                        self._make_cell_view(
                            sequence_idx=played_row.sequence_idx,
                            pattern_idx=played_row.pattern_idx,
                            row=played_row.row,
                            channel=channel,
                            note=pat.data[channel][played_row.row],
                        )
                        for channel in range(pat.n_channels)
                    )
                    yield RowView(
                        sequence_idx=played_row.sequence_idx,
                        pattern_idx=played_row.pattern_idx,
                        row=played_row.row,
                        cells=cells,
                    )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    "iter_rows(reachable_only=True) requires iter_playback_rows() support for this song format."
                ) from exc
            return

        for sequence_idx, pattern_idx in self._iter_pattern_entries(sequence_only):
            pat = self.patterns[pattern_idx]
            for row in range(pat.n_rows):
                cells = tuple(
                    self._make_cell_view(
                        sequence_idx=sequence_idx,
                        pattern_idx=pattern_idx,
                        row=row,
                        channel=channel,
                        note=pat.data[channel][row],
                    )
                    for channel in range(pat.n_channels)
                )
                yield RowView(
                    sequence_idx=sequence_idx,
                    pattern_idx=pattern_idx,
                    row=row,
                    cells=cells,
                )

    def iter_effects(
        self,
        *,
        sequence_only: bool = True,
        include_empty: bool = False,
        decoded: bool = True,
    ):
        """Yield immutable effect snapshots in deterministic sequence,row,channel order."""
        for cell in self.iter_cells(sequence_only=sequence_only):
            raw = cell.effect
            if not include_empty and raw == '':
                continue

            command = None
            arg = None
            x = None
            y = None
            extended_cmd = None
            if decoded and raw != '':
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

            yield EffectView(
                sequence_idx=cell.sequence_idx,
                pattern_idx=cell.pattern_idx,
                row=cell.row,
                channel=cell.channel,
                raw=raw,
                command=command,
                arg=arg,
                x=x,
                y=y,
                extended_cmd=extended_cmd,
            )

    def iter_samples(self, *, include_empty: bool = True):
        """Yield immutable sample-slot snapshots for song formats with direct sample banks."""
        if not hasattr(self, 'samples'):
            return
        sample_slots = getattr(self, 'samples')
        if not isinstance(sample_slots, list):
            return
        for sample_idx, sample in enumerate(sample_slots, start=1):
            length = len(getattr(sample, 'waveform', []))
            if not include_empty and length == 0:
                continue
            yield SampleView(
                sample_idx=sample_idx,
                name=getattr(sample, 'name', ''),
                length=length,
                finetune=getattr(sample, 'finetune', 0),
                volume=getattr(sample, 'volume', 0),
                loop_start=getattr(sample, 'repeat_point', 0),
                loop_length=getattr(sample, 'repeat_len', 0),
            )

    @staticmethod
    def _effect_text(note_or_effect) -> str:
        """Normalize a note or raw effect value into uppercase effect text."""
        if hasattr(note_or_effect, 'effect'):
            effect = note_or_effect.effect
        elif note_or_effect is None:
            effect = ''
        else:
            effect = str(note_or_effect)
        return effect.strip().upper()

    @staticmethod
    def parse_effect(effect_str) -> tuple[str, int | None]:
        """Parse textual tracker effect notation into a command prefix and numeric value.

        Standard effects such as ``F06`` are returned as ``('F', 0x06)``.
        Extended effects with embedded subcommands such as ``E6F`` or ``X12``
        keep the two-character prefix so callers can distinguish the subcommand.

        :param effect_str: A note object or effect string such as ``F06`` or ``E6F``.
        :return: A tuple ``(command, value)``. Empty effects return ``('', None)``.
        """
        effect = Song._effect_text(effect_str)
        if effect == '':
            return '', None
        if len(effect) < 2:
            raise ValueError(f"Invalid effect string {effect!r}.")
        if effect[0] in {'E', 'X'}:
            if len(effect) != 3:
                raise ValueError(f"Invalid extended effect string {effect!r}.")
            return effect[:2], int(effect[2], 16)
        return effect[0], int(effect[1:], 16)

    @staticmethod
    def get_bpm(note_or_effect) -> int | None:
        """Return the BPM encoded by a classic Fxx effect, if present.

        In classic tracker tempo semantics, ``F20`` and above represent BPM,
        while lower non-zero values represent speed instead.

        :param note_or_effect: A note object or effect string.
        :return: The BPM value, or ``None`` if the input does not encode BPM.
        """
        effect_name, value = Song.parse_effect(note_or_effect)
        if effect_name == 'F' and value is not None and value >= 32:
            return value
        return None

    @staticmethod
    def get_ticks_per_row(note_or_effect) -> int | None:
        """Return the speed encoded by a classic Fxx effect, if present.

        In classic tracker tempo semantics, ``F01`` through ``F1F`` represent
        ticks-per-row, while larger values represent BPM instead.

        :param note_or_effect: A note object or effect string.
        :return: The ticks-per-row value, or ``None`` if the input does not encode speed.
        """
        effect_name, value = Song.parse_effect(note_or_effect)
        if effect_name == 'F' and value is not None and 1 <= value <= 31:
            return value
        return None

    @staticmethod
    def get_tick_duration(bpm: int) -> float:
        """
        Returns the duration of a single tick in seconds, given the tempo in BPM.
        A tick is the smallest unit of time in a song.
        When the music is played, each row is played for a number of ticks (called 'speed').

        :param bpm: The tempo in beats per minute.
        :return: The duration of a single tick in seconds.
        """

        return 2.5 / bpm  # See the 'Classic' tempo mode at https://wiki.openmpt.org/Manual:_Song_Properties

    @staticmethod
    def note_to_index(note: str | int) -> int:
        """
        Converts a note string like C-4 or F#3 to a 0-95 index.
        """
        if isinstance(note, int):
            return note
        s = note.strip().upper()
        if len(s) != 3 or s[1] not in ("-", "#"):
            raise ValueError(f"Invalid note format {note}. Expected like C-4 or F#3.")
        pitch = s[:2]
        try:
            octave = int(s[2])
        except ValueError as exc:
            raise ValueError(f"Invalid note octave {note}. Expected a single digit octave.") from exc
        if octave < 1 or octave > 8:
            raise ValueError(f"Invalid note octave {note}. Octave must be 1-8 (maps to note index 0-95).")
        if pitch not in Song.PERIOD_SEQ:
            raise ValueError(f"Invalid note name {note}. Expected C-, C#, D-, D#, E-, F-, F#, G-, G#, A-, A#, B-.")
        note_idx = Song.PERIOD_SEQ.index(pitch)
        return (octave - 1) * 12 + note_idx

    @staticmethod
    def index_to_note(idx: int) -> str:
        """
        Converts a 0-95 note index to a note string like C-4.
        """
        if idx < 0 or idx >= 96:
            raise IndexError(f"Invalid note index {idx} (expected 0-95).")
        octave = idx // 12 + 1
        pitch = Song.PERIOD_SEQ[idx % 12]
        return f"{pitch}{octave}"


    @staticmethod
    def artist_songname_from_filename(filename: str):
        """Split a filename of the form 'artist - title.ext' into metadata fields."""
        filename = os.path.basename(filename)
        parts = filename.split(' - ')
        assert len(parts) <= 2
        if len(parts) == 1:
            artist_name = "Unknown Artist"
            song_name = parts[0].strip()
        else:
            artist_name = parts[0].strip()
            song_name = parts[1].strip()
        song_name = os.path.splitext(song_name)[0]
        # print(f"{filename} | {artist_name} | {song_name}")
        return artist_name, song_name

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """
        Returns the file extension for this song format (e.g., 'mod', 'xm').
        """
        pass

    @abstractmethod
    def save(self, fname: str, verbose: bool = True):
        """
        Saves the song to a file in its native format.

        :param fname: Complete file path.
        :param verbose: False for silent saving.
        """
        pass

    def render(self, fname: str, verbose: bool = True, cleanup: bool = True, channels: int = 1):
        """
        Renders the current song as a WAV file at 44.1kHz.
        
        NOTE: Requires openmpt123 or ffmpeg to be installed and available in PATH.

        :param fname: Complete path of the output WAV file.
        :param verbose: False for silent rendering.
        :param cleanup: True to remove the temporary module file generated for rendering.
        :param channels: Number of output channels for the render command.
        :return: None.
        """

        if channels <= 0:
            raise ValueError(f"Invalid channel count {channels} (expected >=1).")

        if verbose:
            print("Rendering as wav... ", end='', flush=True)

        if os.path.isfile(fname):
            os.remove(fname)

        noext = os.path.splitext(fname)[0]
        ext = self.file_extension
        temp_file = f"{noext}.{ext}"
        temp_wav = f"{temp_file}.wav"

        if os.path.isfile(temp_file):
            os.remove(temp_file)

        self.save(temp_file, verbose=False)

        if os.path.isfile(temp_wav):
            os.remove(temp_wav)

        try:
            subprocess.run(
                ["openmpt123", temp_file, "-q", "--channels", str(channels), "--samplerate", "44100", "--render"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            try:
                subprocess.run(
                    ["ffmpeg", "-ar", "44100", "-i", temp_file, temp_wav], 
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "Neither openmpt123 nor ffmpeg found. Install one of them to render WAV files."
                ) from e

        shutil.move(temp_wav, fname)

        if cleanup:
            os.remove(temp_file)

        if verbose:
            print("done.")

    @abstractmethod
    def timestamp(self) -> list[list[tuple[float, int, int]]]:
        """
        Annotates the time of each row in the song, taking into account the speed and bpm changes.

        :return: A list where each element is a list corresponding to pattern in the sequence.
                 Within each list, each row is a triple (timestamp [s], speed, bpm).
        """
        pass

    def iter_playback_rows(
        self,
        *,
        profile: str | None = None,  # noqa: ARG002
        exact: bool = True,  # noqa: ARG002
        max_steps: int = 250_000,  # noqa: ARG002
    ):
        """Yield playback-order rows with timing metadata when supported by a concrete format."""
        raise NotImplementedError("iter_playback_rows() is not implemented for this song format.")

    def get_song_duration(self) -> float:
        """
        Returns the duration of the song in seconds.

        :return: The song duration in seconds.
        """
        
        timestamps = self.timestamp()
        if not timestamps or not timestamps[-1]:
            return 0.0
        return timestamps[-1][-1][0]

    def get_song_info(self) -> dict:
        """
        Returns a quick song summary suitable for inspection or logging.
        """
        timestamps = []
        try:
            timestamps = self.timestamp()
        except Exception:
            timestamps = []
        first_row = timestamps[0][0] if timestamps and timestamps[0] else None
        n_channels = getattr(self, 'n_channels', self.patterns[0].n_channels if self.patterns else 0)
        info = {
            'format': self.file_extension,
            'songname': self.songname,
            'artist': self.artist,
            'n_patterns': len(self.patterns),
            'sequence_length': len(self.pattern_seq),
            'n_channels': n_channels,
            'sequence': list(self.pattern_seq),
            'speed': first_row[1] if first_row else None,
            'bpm': first_row[2] if first_row else None,
            'duration_seconds': self.get_song_duration() if timestamps else 0.0,
        }
        if hasattr(self, 'restart_position'):
            info['restart_position'] = getattr(self, 'restart_position')
        return info

    def _ascii_pattern_entries(self, sequence_only: bool) -> list[tuple[int | None, int]]:
        """Build pattern references for ASCII rendering."""
        if sequence_only:
            return [(seq_idx, pat_idx) for seq_idx, pat_idx in enumerate(self.pattern_seq)]
        return [(None, pat_idx) for pat_idx in range(len(self.patterns))]

    def to_ascii(self, *, sequence_only: bool = True, include_headers: bool = False) -> str:
        """Return a deterministic tracker-style ASCII dump of pattern note cells.

        :param sequence_only: True to dump pattern order entries, False for full pattern pool.
        :param include_headers: True to include per-pattern metadata headers.
        :return: The ASCII dump as a Python string.
        """
        lines: list[str] = []
        entries = self._ascii_pattern_entries(sequence_only)

        for sequence_idx, pattern_idx in entries:
            pat = self.patterns[pattern_idx]
            if include_headers:
                if sequence_idx is None:
                    lines.append(
                        f"# Pattern pool {pattern_idx}: {pat.n_rows} rows, {pat.n_channels} channels"
                    )
                else:
                    lines.append(
                        f"# Pattern {sequence_idx} (unique pattern {pattern_idx}): "
                        f"{pat.n_rows} rows, {pat.n_channels} channels"
                    )
            for row in range(pat.n_rows):
                row_text = ""
                for channel in range(pat.n_channels):
                    row_text += f"| {pat.data[channel][row]} "
                lines.append(f"{row_text}|")
            lines.append("")
        return "\n".join(lines)

    def save_ascii(self, fname: str, verbose: bool = True):
        """Write ``to_ascii()`` output to a file using ASCII encoding."""
        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)
        with open(fname, 'w', encoding='ascii') as file:
            file.write(self.to_ascii())
        if verbose:
            print('done.')
    
    '''
    -------------------------------------
    PATTERNS
    -------------------------------------
    '''

    @abstractmethod
    def get_effective_row_count(self, sequence_idx: int) -> int:
        """
        Returns the effective number of rows that get played in a pattern.
        Accounts for position jumps, loops, and breaks.

        TODO: do a separate version for the entire song

        :param sequence_idx: The 0-based sequence index to inspect.
        :return: The effective number of rows that gets played in the pattern.
        """
        pass


    def add_to_sequence(self, pattern_idx: int, sequence_position: int | None = None) -> None:
        """
        Inserts an existing 0-based pattern pool index into the song sequence.

        :param pattern_idx: The 0-based pattern pool index to insert.
        :param sequence_position: The 0-based insertion position in `pattern_seq`, or None to append.
        """
        if pattern_idx < 0 or pattern_idx >= len(self.patterns):
            raise IndexError(f"Invalid pattern index {pattern_idx} (expected 0-{len(self.patterns)-1}).")
        if sequence_position is None:
            self.pattern_seq.append(pattern_idx)
            return
        if sequence_position < 0 or sequence_position > len(self.pattern_seq):
            raise IndexError(f"Invalid sequence position {sequence_position} (expected 0-{len(self.pattern_seq)}).")
        self.pattern_seq = self.pattern_seq[:sequence_position] + [pattern_idx] + self.pattern_seq[sequence_position:]


    def set_sequence(self, seq: list[int]) -> None:
        """
        Sets the song sequence using 0-based pattern pool indices.
        """
        if not isinstance(seq, list):
            raise ValueError("Pattern sequence must be a list of integers.")
        for idx in seq:
            if not isinstance(idx, int):
                raise ValueError(f"Invalid pattern index {idx} (expected int).")
            if idx < 0 or idx >= len(self.patterns):
                raise IndexError(f"Invalid pattern index {idx} (expected 0-{len(self.patterns)-1}).")
        self.pattern_seq = list(seq)


    def remove_patterns_after(self, sequence_idx: int):
        """
        Removes all patterns (in the pattern sequence) after the specified one.

        :param sequence_idx: The 0-based sequence index after which all later sequence entries are removed.
        :return: None.
        """

        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        self.pattern_seq = self.pattern_seq[:sequence_idx + 1]

    def remove_pattern(self, sequence_idx: int) -> None:
        """
        Removes a specified pattern from the song sequence.

        Example:
        - The current sequence is 2, 14, 1, 0, 0, 17
        - self.remove_pattern(3)
        - The new sequence is 2, 14, 1, 0, 17

        :param sequence_idx: The 0-based sequence index to remove.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        self.pattern_seq = self.pattern_seq[:sequence_idx] + self.pattern_seq[sequence_idx + 1:]

    def remove_all_patterns(self, sequence_only: bool) -> None:
        """
        Removes all patterns from the song sequence.

        :param sequence_only: If True, only the song sequence is cleared. The patterns are kept.
        """
        self.pattern_seq = []

        if not sequence_only:
            self.patterns = []

    def keep_pattern(self, sequence_idx: int) -> None:
        """
        Removes all the other patterns different from 'pattern'.

        :param sequence_idx: The 0-based sequence index to keep.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        self.pattern_seq = [self.pattern_seq[sequence_idx]]

    def insert_pattern(self, sequence_idx: int, after: bool = True) -> int:
        """
        Inserts a copy of a pattern into the sequence, before or after the given index.

        :param sequence_idx: The 0-based sequence index to duplicate in place.
        :param after: If True, insert after; if False, insert before.
        :return: The new 0-based pattern pool index.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")
        self.patterns.append(copy.deepcopy(self.patterns[self.pattern_seq[sequence_idx]]))
        new_idx = len(self.patterns) - 1
        seq_pos = sequence_idx + 1 if after else sequence_idx
        self.pattern_seq = self.pattern_seq[:seq_pos] + [new_idx] + self.pattern_seq[seq_pos:]
        return new_idx

    def duplicate_pattern(self, sequence_idx: int) -> int:
        """
        Creates a fresh copy of the given pattern, and appends it at the end of the song sequence.

        :param sequence_idx: The 0-based sequence index to duplicate.
        :return: The new 0-based pattern pool index.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        self.patterns.append(copy.deepcopy(self.patterns[self.pattern_seq[sequence_idx]]))
        n = len(self.patterns) - 1
        self.pattern_seq.append(n)

        return n

    def copy_pattern_from(self, src_song: 'Song', src_pattern_idx: int) -> int:
        """
        Copies a source pattern pool entry from another song and appends it to this song and sequence.

        :param src_song: The source song.
        :param src_pattern_idx: The source pattern pool index.
        :return: The destination pattern pool index.
        """
        if src_pattern_idx < 0 or src_pattern_idx >= len(src_song.patterns):
            raise IndexError(f"Invalid source pattern index {src_pattern_idx} (expected 0-{len(src_song.patterns)-1}).")
        self.patterns.append(copy.deepcopy(src_song.patterns[src_pattern_idx]))
        new_idx = len(self.patterns) - 1
        self.pattern_seq.append(new_idx)
        return new_idx

    def is_pattern_empty(self, pattern: int) -> bool:
        """
        Return True if every note in the referenced sequence pattern is empty.

        This checks the concrete pattern reached through the song sequence, not a
        raw pattern-pool index.

        :param pattern: The 0-based sequence index to inspect.
        :return: True if every note in that sequence entry is empty.
        """
        pat = self._get_sequence_pattern(pattern)
        for channel in pat.data:
            for note in channel:
                if not note.is_empty():
                    return False
        return True

    def get_used_patterns(self) -> list[int]:
        """
        Return unique pattern-pool indices referenced by the song sequence.

        The result preserves first-use order, which is often more useful than a
        numerically sorted list when inspecting or rewriting sequences.

        :return: Pattern-pool indices referenced by ``pattern_seq`` in first-use order.
        """
        used = []
        seen = set()
        for pattern_idx in self.pattern_seq:
            if pattern_idx not in seen:
                seen.add(pattern_idx)
                used.append(pattern_idx)
        return used

    '''
    -------------------------------------
    NOTES
    -------------------------------------
    '''

    def get_note_rc(self, sequence_idx: int, row: int, channel: int):
        """Return a note using canonical coordinate order (sequence, row, channel)."""
        return self.get_note(sequence_idx, row, channel)

    def set_note_rc(
        self,
        sequence_idx: int,
        row: int,
        channel: int,
        sample_idx: int,
        period: str,
        effect: str = "",
    ):
        """Write a note using canonical coordinate order (sequence, row, channel)."""
        self.set_note(sequence_idx, channel, row, sample_idx, period, effect)

    def clear_note_rc(self, sequence_idx: int, row: int, channel: int):
        """Clear one note using canonical coordinate order (sequence, row, channel)."""
        self.clear_note(sequence_idx, channel, row)

    def set_note(self, sequence_idx:int, channel: int, row: int, sample_idx: int, period: str, effect: str = ""):
        """
        Writes a note in the given pattern, channel and row with the given sample.
        If no effect is given and the current note already has a speed effect, leaves it unchanged.

        :param sequence_idx: The 0-based sequence index to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param sample_idx: The sample index to write.
        :param period: The note period (pitch) to write, e.g. "C-4".
        :param effect: The note effect, e.g. "ED1".
        :return: None.
        """

        pat = self._get_sequence_pattern(sequence_idx)
        if channel < 0 or channel >= pat.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{pat.n_channels-1}).")
        if row < 0 or row >= pat.n_rows:
            raise IndexError(f"Invalid row index {row} (expected 0-{pat.n_rows-1}).")
        cur_note = pat.data[channel][row]
        if effect == '':
            effect = self._preserved_effect(cur_note.effect)

        pat.data[channel][row] = Note(sample_idx, period, effect)

    def clear_note(self, sequence_idx: int, channel: int, row: int):
        """
        Clear a single note cell completely.

        The replacement note is created using the existing note class for that
        pattern, so XM and S3M-specific note fields are reset correctly.

        :param sequence_idx: The 0-based sequence index to modify.
        :param channel: The 0-based channel index to clear.
        :param row: The 0-based row index to clear.
        """
        pat = self._get_sequence_pattern(sequence_idx)
        if channel < 0 or channel >= pat.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{pat.n_channels-1}).")
        if row < 0 or row >= pat.n_rows:
            raise IndexError(f"Invalid row index {row} (expected 0-{pat.n_rows-1}).")
        pat.data[channel][row] = type(pat.data[channel][row])()

    def clear_row(self, sequence_idx: int, row: int):
        """
        Clear a full row across all channels in a sequence pattern.

        Each cell is reset using the concrete note type stored by that pattern.

        :param sequence_idx: The 0-based sequence index to modify.
        :param row: The 0-based row index to clear.
        """
        pat = self._get_sequence_pattern(sequence_idx)
        if row < 0 or row >= pat.n_rows:
            raise IndexError(f"Invalid row index {row} (expected 0-{pat.n_rows-1}).")
        for channel in range(pat.n_channels):
            pat.data[channel][row] = type(pat.data[channel][row])()

    def copy_row(self, src_sequence_idx: int, src_row: int, dst_sequence_idx: int, dst_row: int):
        """
        Copy a full row between sequence patterns.

        Destination cells are cleared first and then overwritten for the channel
        range common to both source and destination patterns.

        :param src_sequence_idx: Source 0-based sequence index.
        :param src_row: Source 0-based row index.
        :param dst_sequence_idx: Destination 0-based sequence index.
        :param dst_row: Destination 0-based row index.
        """
        src_pat = self._get_sequence_pattern(src_sequence_idx)
        dst_pat = self._get_sequence_pattern(dst_sequence_idx)
        if src_row < 0 or src_row >= src_pat.n_rows:
            raise IndexError(f"Invalid source row index {src_row} (expected 0-{src_pat.n_rows-1}).")
        if dst_row < 0 or dst_row >= dst_pat.n_rows:
            raise IndexError(f"Invalid destination row index {dst_row} (expected 0-{dst_pat.n_rows-1}).")
        for channel in range(dst_pat.n_channels):
            dst_pat.data[channel][dst_row] = type(dst_pat.data[channel][dst_row])()
        for channel in range(min(src_pat.n_channels, dst_pat.n_channels)):
            dst_pat.data[channel][dst_row] = copy.deepcopy(src_pat.data[channel][src_row])

    def shift_pattern(self, sequence_idx: int, delta_rows: int):
        """
        Shift all rows in a sequence pattern using clipping semantics.

        Positive values shift notes downward, negative values shift them upward.
        Notes pushed outside the pattern bounds are discarded, and newly exposed
        rows are filled with empty notes of the appropriate concrete note type.

        :param sequence_idx: The 0-based sequence index to modify.
        :param delta_rows: Row offset to apply to every note in the pattern.
        """
        pat = self._get_sequence_pattern(sequence_idx)
        for channel in range(pat.n_channels):
            old_rows = pat.data[channel]
            new_rows = [type(old_rows[row])() for row in range(pat.n_rows)]
            for row, note in enumerate(old_rows):
                new_row = row + delta_rows
                if 0 <= new_row < pat.n_rows:
                    new_rows[new_row] = copy.deepcopy(note)
            pat.data[channel] = new_rows

    def copy_channel_data(self, src_sequence_idx: int, src_channel: int, dst_sequence_idx: int, dst_channel: int):
        """
        Copies one channel's note data to another channel, across the same or different patterns.
        """
        src_pat = self._get_sequence_pattern(src_sequence_idx)
        dst_pat = self._get_sequence_pattern(dst_sequence_idx)
        if src_channel < 0 or src_channel >= src_pat.n_channels:
            raise IndexError(f"Invalid source channel index {src_channel} (expected 0-{src_pat.n_channels-1}).")
        if dst_channel < 0 or dst_channel >= dst_pat.n_channels:
            raise IndexError(f"Invalid destination channel index {dst_channel} (expected 0-{dst_pat.n_channels-1}).")
        dst_pat.data[dst_channel] = [type(dst_pat.data[dst_channel][row])() for row in range(dst_pat.n_rows)]
        for row in range(min(src_pat.n_rows, dst_pat.n_rows)):
            dst_pat.data[dst_channel][row] = copy.deepcopy(src_pat.data[src_channel][row])

    def swap_channels(self, ch1: int, ch2: int):
        """
        Swaps two channel columns across every pattern in the song.
        """
        if ch1 == ch2:
            return
        for pat in self.patterns:
            if ch1 < 0 or ch1 >= pat.n_channels or ch2 < 0 or ch2 >= pat.n_channels:
                raise IndexError(f"Invalid channel indices {ch1}, {ch2} for pattern with {pat.n_channels} channels.")
            pat.data[ch1], pat.data[ch2] = pat.data[ch2], pat.data[ch1]

    '''
    -------------------------------------
    EFFECTS
    -------------------------------------
    '''

    def set_effect_rc(self, sequence_idx: int, row: int, channel: int, effect: str = ""):
        """Write an effect using canonical coordinate order (sequence, row, channel)."""
        self.set_effect(sequence_idx, channel, row, effect)

    def set_effect(self, sequence_idx: int, channel: int, row: int, effect: str = ""):
        """
        Writes a given effect in the given pattern, channel and row.
        Does not touch the period or sample, if present.

        :param sequence_idx: The 0-based sequence index to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param effect: The desired effect, e.g. "ED1".
        :return: None.
        """

        pat = self._get_sequence_pattern(sequence_idx)
        if channel < 0 or channel >= pat.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{pat.n_channels-1}).")
        if row < 0 or row >= pat.n_rows:
            raise IndexError(f"Invalid row index {row} (expected 0-{pat.n_rows-1}).")
        pat.data[channel][row].effect = effect

    @staticmethod
    def note_in_range(note_str: str, lo: str | int, hi: str | int) -> bool:
        """
        Return True if a note falls within an inclusive note range.

        All inputs may be either tracker note strings such as ``C-4`` or raw
        numeric note indices.

        :param note_str: The note to test.
        :param lo: Lower inclusive bound.
        :param hi: Upper inclusive bound.
        :return: True if ``note_str`` is within ``[lo, hi]``.
        """
        note_idx = Song.note_to_index(note_str)
        lo_idx = Song.note_to_index(lo)
        hi_idx = Song.note_to_index(hi)
        return lo_idx <= note_idx <= hi_idx

    @staticmethod
    def transpose_note(note_str: str, semitones: int) -> str:
        """
        Transposes a note string by the given number of semitones.
        """
        if note_str == '':
            return ''
        if note_str == 'off':
            return 'off'
        new_index = Song.note_to_index(note_str) + semitones
        return Song.index_to_note(new_index)

    def set_bpm(self, pattern: int, channel: int, row: int, bpm: int):
        """Write a classic Fxx tempo command at the requested pattern location."""
        if bpm < 32 or bpm > 255:
            raise ValueError(f"Invalid tempo {bpm} (expected 32-255).")
        self.set_effect(pattern, channel, row, f"F{bpm:02X}")

    def set_ticks_per_row(self, pattern: int, channel: int, row: int, ticks: int):
        """Write a classic Fxx speed command at the requested pattern location."""
        if ticks < 1 or ticks > 31:
            raise ValueError(f"Invalid ticks per row {ticks} (expected 1-31).")
        self.set_effect(pattern, channel, row, f"F{ticks:02X}")

    def set_arpeggio(self, pattern: int, channel: int, row: int, note1: int, note2: int):
        """Write a 0xy arpeggio effect using two semitone offsets."""
        if note1 < 0 or note1 > 15 or note2 < 0 or note2 > 15:
            raise ValueError("Arpeggio offsets must be in the range 0-15.")
        self.set_effect(pattern, channel, row, f"0{note1:X}{note2:X}")

    def set_panning(self, pattern: int, channel: int, row: int, panning: int):
        """Write an 8xx panning command using a 0-255 position."""
        if panning < 0 or panning > 255:
            raise ValueError(f"Invalid panning {panning} (expected 0-255).")
        self.set_effect(pattern, channel, row, f"8{panning:02X}")

    def set_sample_offset(self, pattern: int, channel: int, row: int, offset: int):
        """Write a 9xx sample-offset command."""
        if offset < 0 or offset > 255:
            raise ValueError(f"Invalid sample offset {offset} (expected 0-255).")
        self.set_effect(pattern, channel, row, f"9{offset:02X}")

    def set_position_jump(self, pattern: int, channel: int, row: int, pos: int):
        """Write a Bxx order jump command."""
        if pos < 0 or pos > 255:
            raise ValueError(f"Invalid position jump {pos} (expected 0-255).")
        self.set_effect(pattern, channel, row, f"B{pos:02X}")

    def set_pattern_break(self, pattern: int, channel: int, row: int, row_target: int):
        """Write a Dxx pattern-break command using decimal row notation."""
        if row_target < 0 or row_target > 99:
            raise ValueError(f"Invalid pattern break row {row_target} (expected 0-99).")
        tens, ones = divmod(row_target, 10)
        self.set_effect(pattern, channel, row, f"D{tens:X}{ones:X}")

    def set_portamento(self, pattern: int, channel: int, row: int, slide: int):
        """Write an up or down portamento effect depending on the slide sign."""
        if slide < -255 or slide > 255:
            raise ValueError(f"Invalid portamento slide {slide} (expected -255 to 255).")
        if slide > 0:
            self.set_effect(pattern, channel, row, f"1{slide:02X}")
        elif slide < 0:
            self.set_effect(pattern, channel, row, f"2{(-slide):02X}")

    def set_tone_portamento(self, pattern: int, channel: int, row: int, speed: int):
        """Write a 3xx tone-portamento command."""
        if speed < 0 or speed > 255:
            raise ValueError(f"Invalid tone portamento speed {speed} (expected 0-255).")
        self.set_effect(pattern, channel, row, f"3{speed:02X}")

    def set_tone_portamento_slide(self, pattern: int, channel: int, row: int, slide: int):
        """Write a 5xx tone-portamento-plus-volume-slide command."""
        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid tone portamento slide {slide} (expected -15 to 15).")
        effect_value = 0
        if slide > 0:
            effect_value = slide << 4
        elif slide < 0:
            effect_value = -slide
        self.set_effect(pattern, channel, row, f"5{effect_value:02X}")

    def set_volume(self, pattern: int, channel: int, row: int, volume: int):
        """Write a Cxx set-volume command."""
        if volume < 0 or volume > 64:
            raise ValueError(f"Invalid volume {volume} (expected 0-64).")
        self.set_effect(pattern, channel, row, f"C{volume:02X}")

    def set_volume_slide(self, pattern: int, channel: int, row: int, slide: int):
        """Write an Axy volume-slide command."""
        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid volume slide {slide} (expected -15 to 15).")
        effect_value = 0
        if slide > 0:
            effect_value = slide << 4
        elif slide < 0:
            effect_value = -slide
        self.set_effect(pattern, channel, row, f"A{effect_value:02X}")

    def set_vibrato(self, pattern: int, channel: int, row: int, speed: int, depth: int):
        """Write a 4xy vibrato command."""
        if speed < 0 or speed > 15:
            raise ValueError(f"Invalid vibrato speed {speed} (expected 0-15).")
        if depth < 0 or depth > 15:
            raise ValueError(f"Invalid vibrato depth {depth} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"4{(speed << 4) | depth:02X}")

    def set_vibrato_slide(self, pattern: int, channel: int, row: int, slide: int):
        """Write a 6xy vibrato-plus-volume-slide command."""
        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid vibrato slide {slide} (expected -15 to 15).")
        effect_value = 0
        if slide > 0:
            effect_value = slide << 4
        elif slide < 0:
            effect_value = -slide
        self.set_effect(pattern, channel, row, f"6{effect_value:02X}")

    def set_tremolo(self, pattern: int, channel: int, row: int, speed: int, depth: int):
        """Write a 7xy tremolo command."""
        if speed < 0 or speed > 15:
            raise ValueError(f"Invalid tremolo speed {speed} (expected 0-15).")
        if depth < 0 or depth > 15:
            raise ValueError(f"Invalid tremolo depth {depth} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"7{(speed << 4) | depth:02X}")

    def set_fine_portamento(self, pattern: int, channel: int, row: int, slide: int):
        """Write an E1x or E2x fine-portamento command."""
        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid fine portamento {slide} (expected -15 to 15).")
        if slide > 0:
            self.set_effect(pattern, channel, row, f"E1{slide:X}")
        elif slide < 0:
            self.set_effect(pattern, channel, row, f"E2{-slide:X}")
        else:
            self.set_effect(pattern, channel, row, "E10")

    def set_glissando(self, pattern: int, channel: int, row: int, on: bool):
        """Enable or disable glissando control with E3x."""
        self.set_effect(pattern, channel, row, f"E3{1 if on else 0}")

    def set_vibrato_waveform(self, pattern: int, channel: int, row: int, wave: int):
        """Select the vibrato waveform using E4x."""
        if wave < 0 or wave > 7:
            raise ValueError(f"Invalid vibrato waveform {wave} (expected 0-7).")
        self.set_effect(pattern, channel, row, f"E4{wave:X}")

    def set_finetune(self, pattern: int, channel: int, row: int, finetune: int):
        """Write an E5x finetune command."""
        if finetune < 0 or finetune > 15:
            raise ValueError(f"Invalid finetune {finetune} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"E5{finetune:X}")

    def set_pattern_loop(self, pattern: int, channel: int, row: int, count: int):
        """Write an E6x pattern-loop command."""
        if count < 0 or count > 15:
            raise ValueError(f"Invalid pattern loop count {count} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"E6{count:X}")

    def set_tremolo_waveform(self, pattern: int, channel: int, row: int, wave: int):
        """Select the tremolo waveform using E7x."""
        if wave < 0 or wave > 7:
            raise ValueError(f"Invalid tremolo waveform {wave} (expected 0-7).")
        self.set_effect(pattern, channel, row, f"E7{wave:X}")

    def set_retrigger(self, pattern: int, channel: int, row: int, interval: int):
        """Write an E9x retrigger-note command."""
        if interval < 0 or interval > 15:
            raise ValueError(f"Invalid retrigger interval {interval} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"E9{interval:X}")

    def set_fine_volume_slide(self, pattern: int, channel: int, row: int, slide: int):
        """Write an EAx or EBx fine-volume-slide command."""
        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid fine volume slide {slide} (expected -15 to 15).")
        if slide > 0:
            self.set_effect(pattern, channel, row, f"EA{slide:X}")
        elif slide < 0:
            self.set_effect(pattern, channel, row, f"EB{-slide:X}")
        else:
            self.set_effect(pattern, channel, row, "EA0")

    def set_note_cut(self, pattern: int, channel: int, row: int, tick: int):
        """Write an ECx note-cut command."""
        if tick < 0 or tick > 15:
            raise ValueError(f"Invalid note cut tick {tick} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"EC{tick:X}")

    def set_note_delay(self, pattern: int, channel: int, row: int, tick: int):
        """Write an EDx note-delay command."""
        if tick < 0 or tick > 15:
            raise ValueError(f"Invalid note delay tick {tick} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"ED{tick:X}")

    def set_pattern_delay(self, pattern: int, channel: int, row: int, rows: int):
        """Write an EEx pattern-delay command."""
        if rows < 0 or rows > 15:
            raise ValueError(f"Invalid pattern delay {rows} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"EE{rows:X}")

    def _get_sequence_pattern(self, sequence_idx: int):
        """Return the concrete pattern referenced by a sequence position.

        :param sequence_idx: 0-based index into ``pattern_seq``.
        :return: The concrete pattern object referenced by that sequence entry.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")
        return self.patterns[self.pattern_seq[sequence_idx]]

    def _preserved_effect(self, effect: str) -> str:
        """Return structural effects that should survive note overwrite operations."""
        effect = self._effect_text(effect)
        if effect and effect[0] in self.PRESERVED_EFFECT_PREFIXES:
            return effect
        return ''
