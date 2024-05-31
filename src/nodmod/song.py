import binascii
import random
import string
import struct
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

        self.current_sample = 0
        self.current_pattern = 0
        self.current_channel = 0
        self.current_row = 0

    def set_artist(self, artist_name: str):
        self.artist = artist_name

    def set_songname(self, song_name: str):
        self.songname = song_name

    @staticmethod
    def get_tick_duration(bpm: int) -> float:
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

    def use_pattern(self, pattern_idx: int):
        """
        Sets the pattern to use for writing.

        :param pattern_idx: The index of the pattern to use.
        :return: None.
        """

        if pattern_idx < 0 or pattern_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern_idx}")
        self.current_pattern = pattern_idx

    def write_note(self, period: str, effect: str = ""):
        """
        Writes a note in the current pattern, channel and row with the current sample.
        If no effect is given and the current note already has a speed effect, leaves it unchanged.

        :param period: The note period (pitch) to write, e.g. "C-4".
        :param effect: The note effect, e.g. "ED1".
        :return: None.
        """

        cur_efx = self.patterns[self.pattern_seq[self.current_pattern]].data[self.current_channel][self.current_row].effect
        if effect == '' and cur_efx != '' and cur_efx[0] == 'F':
            effect = cur_efx

        self.patterns[self.pattern_seq[self.current_pattern]].data[self.current_channel][self.current_row] = (
            Note(self.current_sample, period, effect))

    def write_effect(self, effect: str = ""):
        """
        Writes a given effect in the current pattern, channel and row.
        Does not touch the period or sample, if present.

        :param effect: The desired effect, e.g. "ED1".
        :return: None.
        """

        self.patterns[self.pattern_seq[self.current_pattern]].data[self.current_channel][self.current_row].effect \
            = effect
