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

from .types import Note

__all__ = ['Song']


class Song(ABC):
    """
    A song is a collection of patterns played in a specific sequence, possibly with repetitions.
    In addition, songs also store the samples that are used to play the notes.
    Songs can be loaded from file (e.g., MOD format) or composed from scratch.
    """

    PERIOD_SEQ = ['C-', 'C#', 'D-', 'D#', 'E-', 'F-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']

    def __init__(self):

        self.artist = ""
        self.songname = "new song"
        self.patterns = []
        self.pattern_seq = []  # The actual sequence of patterns making up the song.

        # To be consistent with the module standard, we always store as many samples
        # as the maximum allowed, possibly with empty slots within the sample bank.
        self.samples = []
        self.n_actual_samples = 0  # The number of non-empty samples present in the song.

    '''
    -------------------------------------
    SONG
    -------------------------------------
    '''

    def set_artist(self, artist_name: str):
        self.artist = artist_name

    def set_songname(self, song_name: str):
        self.songname = song_name

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
    def artist_songname_from_filename(filename: str):
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
    def save_to_file(self, fname: str, verbose: bool = True):
        """
        Saves the song to a file in its native format.

        :param fname: Complete file path.
        :param verbose: False for silent saving.
        """
        pass

    def render_as_wav(self, fname: str, verbose: bool = True, cleanup: bool = True):
        """
        Renders the current song as a WAV file.
        Note: Requires openmpt123 or ffmpeg to be installed and available in PATH.

        :param fname: Complete path of the output WAV file.
        :param verbose: False for silent rendering.
        :param cleanup: True to remove the temporary module file generated for rendering.
        :return: None.
        """

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

        self.save_to_file(temp_file, verbose=False)

        if os.path.isfile(temp_wav):
            os.remove(temp_wav)

        try:
            subprocess.run(
                ["openmpt123", temp_file, "-q", "--channels", "1", "--samplerate", "44100", "--render"],
                check=True
            )
        except FileNotFoundError:
            try:
                subprocess.run(["ffmpeg", "-i", temp_file, temp_wav], check=True)
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
    def timestamp(self) -> list[list[float, int, int]]:
        """
        Annotates the time of each row in the song, taking into account the speed and bpm changes.

        :return: A list where each element is a list corresponding to pattern in the sequence.
                 Within each list, each row is a triple (timestamp [s], speed, bpm).
        """
        pass

    def get_song_duration(self) -> float:
        """
        Returns the duration of the song in seconds.

        :return: The song duration in seconds.
        """
        
        return self.timestamp()[-1][-1][0]
    
    '''
    -------------------------------------
    PATTERNS
    -------------------------------------
    '''

    @abstractmethod
    def get_effective_row_count(self, pattern: int) -> int:
        """
        Returns the effective number of rows that get played in a pattern.
        Accounts for position jumps, loops, and breaks.

        TODO: do a separate version for the entire song

        :param pattern: The pattern index (within the song sequence).
        :return: The effective number of rows that gets played in the pattern.
        """
        pass

    @abstractmethod
    def get_pattern_duration(self, pattern: int) -> float:
        """
        Returns the duration of a pattern in seconds.

        :param pattern: The pattern index (within the song sequence).
        :return: The pattern duration in seconds.
        """
        pass

    def remove_patterns_after(self, pattern: int):
        """
        Removes all patterns (in the pattern sequence) after the specified one.

        :param pattern: The pattern index (within the song sequence) to remove all patterns after.
        :return: None.
        """

        if pattern < 0 or pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        self.pattern_seq = self.pattern_seq[:pattern + 1]

    def remove_pattern_from_seq(self, pattern: int) -> None:
        """
        Removes a specified pattern from the song sequence.

        Example:
        - The current sequence is 2, 14, 1, 0, 0, 17
        - self.remove_pattern_from_seq(3)
        - The new sequence is 2, 14, 1, 0, 17

        :param pattern: The pattern index (within the song sequence) to be removed.
        """
        if pattern < 0 or pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        self.pattern_seq = self.pattern_seq[:pattern] + self.pattern_seq[pattern + 1:]

    def remove_all_patterns(self, sequence_only: bool) -> None:
        """
        Removes all patterns from the song sequence.

        :param sequence_only: If True, only the song sequence is cleared. The patterns are kept.
        """
        self.pattern_seq = []

        if not sequence_only:
            self.patterns = []

    def keep_pattern_from_seq(self, pattern: int) -> None:
        """
        Removes all the other patterns different from 'pattern'.

        :param pattern: The pattern index (within the song sequence) to be kept.
        """
        if pattern < 0 or pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        self.pattern_seq = [self.pattern_seq[pattern]]

    def duplicate_pattern(self, pattern: int) -> int:
        """
        Creates a fresh copy of the given pattern, and appends it at the end of the song sequence.

        :param pattern: The pattern index (within the song sequence) to be duplicated.
        :return: The index of the new pattern.
        """
        if pattern < 0 or pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        self.patterns.append(copy.deepcopy(self.patterns[self.pattern_seq[pattern]]))
        n = len(self.patterns) - 1
        self.pattern_seq.append(n)

        return n

    '''
    -------------------------------------
    NOTES
    -------------------------------------
    '''

    def write_note(self, pattern:int, channel: int, row: int, sample: int, period: str, effect: str = ""):
        """
        Writes a note in the given pattern, channel and row with the given sample.
        If no effect is given and the current note already has a speed effect, leaves it unchanged.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param sample: The sample index to write.
        :param period: The note period (pitch) to write, e.g. "C-4".
        :param effect: The note effect, e.g. "ED1".
        :return: None.
        """

        cur_efx = self.patterns[self.pattern_seq[pattern]].data[channel][row].effect
        if effect == '' and cur_efx != '' and cur_efx[0] == 'F':
            effect = cur_efx

        self.patterns[self.pattern_seq[pattern]].data[channel][row] = (
            Note(sample, period, effect))

    '''
    -------------------------------------
    EFFECTS
    -------------------------------------
    '''

    def write_effect(self, pattern: int, channel: int, row: int, effect: str = ""):
        """
        Writes a given effect in the given pattern, channel and row.
        Does not touch the period or sample, if present.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param effect: The desired effect, e.g. "ED1".
        :return: None.
        """

        self.patterns[self.pattern_seq[pattern]].data[channel][row].effect \
            = effect
