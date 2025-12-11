"""
Data types for tracker music formats (MOD, XM, etc.)

This module contains all the data classes used across different tracker formats:
- Pattern, Note, XMNote: musical content
- Sample: audio waveform data
- Instrument, EnvelopePoint: XM instrument containers and envelopes
"""

import array

__all__ = ['Pattern', 'Sample', 'EnvelopePoint', 'Instrument', 'Note', 'XMNote']


class Note:
    """
    A note is a sample that is played at a specific pitch (period), possibly with envelopes.
    Every note can be modified by effects.
    This base class is used for MOD files.
    """

    def __init__(self, instrument_idx: int = 0, period: str = '', effect: str = ''):

        # note that for mod files, instrument and sample are synonymous
        self.instrument_idx = instrument_idx

        self.period = period
        self.effect = effect

    def __repr__(self):
        s = ''
        if self.period == '':
            s += '--- '
        else:
            s += self.period + ' '
        if self.instrument_idx == 0:
            s += '-- '
        else:
            s += f"{self.instrument_idx:02d}" + ' '
        if self.effect == '':
            s += '---'
        else:
            s += self.effect
        return s
    
    def is_empty(self) -> bool:
        return self.instrument_idx == 0 and self.period == '' and self.effect == ''


class XMNote(Note):
    """
    Extended note class for XM files.
    XM notes have an additional volume column with its own command and value.
    
    Volume column commands:
        'v' - Set volume (0-64)
        'd' - Volume slide down
        'c' - Volume slide up  
        'b' - Fine volume slide down
        'a' - Fine volume slide up
        'u' - Vibrato speed
        'h' - Vibrato depth
        'p' - Set panning position
        'l' - Panning slide left
        'r' - Panning slide right
        'g' - Tone portamento
    """

    def __init__(self, instrument_idx: int = 0, period: str = '', effect: str = '',
                 vol_cmd: str = '', vol_val: int = -1):
        super().__init__(instrument_idx, period, effect)
        
        # XM-specific: volume column command and value
        # vol_cmd is a single character (v, d, c, b, a, u, h, p, l, r, g) or empty
        # vol_val is 0-64 for most commands, or -1 if no volume column data
        self.vol_cmd = vol_cmd
        self.vol_val = vol_val

    def __repr__(self):
        s = ''
        # Period
        if self.period == '':
            s += '--- '
        elif self.period == 'off':
            s += '=== '
        else:
            s += self.period + ' '
        # Instrument
        if self.instrument_idx == 0:
            s += '-- '
        else:
            s += f"{self.instrument_idx:02d} "
        # Volume column
        if self.vol_cmd == '':
            s += '-- '
        else:
            s += f"{self.vol_cmd}{self.vol_val:02d} " if self.vol_val >= 0 else '-- '
        # Effect
        if self.effect == '':
            s += '---'
        else:
            s += self.effect
        return s
    
    def is_empty(self) -> bool:
        return self.instrument_idx == 0 and self.period == '' and \
               self.effect == '' and self.vol_cmd == ''


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

    def __len__(self) -> int:
        return self.n_rows


class Sample:
    """
    A sample is a digitized soundwave plus some additional attributes.
    Samples are played as notes in a song.
    
    For MOD files: Samples are referenced directly by notes.
    For XM files: Samples are contained within Instruments.
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


class EnvelopePoint:
    """A single point in a volume or panning envelope."""
    def __init__(self, frame: int = 0, value: int = 0):
        self.frame = frame  # X-coordinate (0-65535, but FT2 only supports 0-255)
        self.value = value  # Y-coordinate (0-64)


class Instrument:
    """
    An instrument is a container for samples, used in XM files.
    
    In MOD files, there is no concept of instruments - notes reference samples directly.
    In XM files, notes reference instruments, and each instrument can contain:
      - 0 samples (stub instrument, often used just for storing a name/comment)
      - 1 sample (simple instrument)
      - Multiple samples (e.g., different samples for different note ranges)
    
    The instrument also contains envelope and other playback settings.
    """

    def __init__(self):
        self.name = ""
        
        # List of Sample objects belonging to this instrument
        self.samples: list[Sample] = []
        
        # Sample-to-note mapping: which sample index to use for each note (0-95)
        # In XM, this maps MIDI-like note numbers to sample indices within self.samples
        # If empty, all notes use sample index 0 (or no sample if samples list is empty)
        self.sample_map: list[int] = []
        
        # Volume envelope
        self.volume_envelope: list[EnvelopePoint] = []
        self.volume_sustain_point: int = 0
        self.volume_loop_start: int = 0
        self.volume_loop_end: int = 0
        self.volume_type: int = 0  # bit 0: On, bit 1: Sustain, bit 2: Loop
        
        # Panning envelope
        self.panning_envelope: list[EnvelopePoint] = []
        self.panning_sustain_point: int = 0
        self.panning_loop_start: int = 0
        self.panning_loop_end: int = 0
        self.panning_type: int = 0  # bit 0: On, bit 1: Sustain, bit 2: Loop
        
        # Vibrato settings (auto-vibrato applied to all notes in this instrument)
        self.vibrato_type: int = 0
        self.vibrato_sweep: int = 0
        self.vibrato_depth: int = 0
        self.vibrato_rate: int = 0
        
        # Volume fadeout (0-65535, applied after note release)
        self.volume_fadeout: int = 0

    @property
    def n_samples(self) -> int:
        """Return the number of samples in this instrument."""
        return len(self.samples)
    
    def is_empty(self) -> bool:
        """Return True if this instrument has no samples (stub instrument)."""
        return len(self.samples) == 0
