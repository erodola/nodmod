import copy
import array
import os
from abc import ABC, abstractmethod


class Pattern:
    """
    A pattern is a page of notes, and is part of a song.
    It is made of a number of channels of same length; each channel stores a note per row.
    """

    def __init__(self, n_rows: int, n_channels: int):
        self.n_rows = n_rows
        self.n_channels = n_channels

        # use it as data[channel][row]
        self.data = [[Note() for _ in range(n_rows)] for _ in range(n_channels)]


class Sample:
    """
    A sample is a digitized soundwave plus some additional attributes.
    Samples are played as notes in a song.
    """

    def __init__(self):
        self.name = ""

        # MOD attributes
        self.finetune = 0
        self.volume = 64
        self.repeat_point = 0
        self.repeat_len = 0

        self.waveform = array.array('b')  # signed integers (8-bit audio)

        # Tells which sample pitch corresponds to true G (Sol).
        # Can be estimated, e.g., with MODSong.tune_sample().
        self.tune = ''


class Note:
    """
    A note is a sample that is played at a specific pitch (period).
    Every note can carry a particular effect.
    """

    def __init__(self, sample_idx: int = 0, period: str = '', effect: str = ''):
        self.sample_idx = sample_idx
        self.period = period
        self.effect = effect

    def __str__(self):
        s = ''
        if self.period == '':
            s += '--- '
        else:
            s += self.period + ' '
        if self.sample_idx == 0:
            s += '-- '
        else:
            s += f"{self.sample_idx:02d}" + ' '
        if self.effect == '':
            s += '---'
        else:
            s += self.effect
        return s
    
    def is_empty(self) -> bool:
        return self.sample_idx == 0 and self.period == '' and self.effect == ''


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

    @abstractmethod
    def annotate_time(self) -> list[list[float]]:
        """
        Annotates the time of each row in the song, taking into account the speed and bpm changes.

        TODO: do a separate version for individual patterns and the entire song

        :return: A list where each element is a list corresponding to pattern in the sequence.
                 Within each list, each row is annotated with a time in seconds.
        """
        pass

    @abstractmethod
    def get_song_duration(self) -> float:
        """
        Returns the duration of the song in seconds.

        :return: The song duration in seconds.
        """
        pass
    
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

    def remove_pattern_from_seq(self, pattern: int):
        """
        Removes a specified pattern from the song sequence.

        Example:
        - The current sequence is 2, 14, 1, 0, 0, 17
        - self.remove_pattern_from_seq(3)
        - The new sequence is 2, 14, 1, 0, 17

        :param pattern: The pattern index (within the song sequence) to be removed.
        :return: None.
        """

        if pattern < 0 or pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        self.pattern_seq = self.pattern_seq[:pattern] + self.pattern_seq[pattern + 1:]

    def keep_pattern_from_seq(self, pattern: int):
        """
        Removes all the other patterns different from 'pattern'.

        :param pattern: The pattern index (within the song sequence) to be kept.
        :return: None.
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
