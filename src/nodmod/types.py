"""
Data types for tracker music formats (MOD, XM, etc.)

This module contains all the data classes used across different tracker formats:
- Pattern, Note, XMNote: musical content
- Sample: audio waveform data
- Instrument, EnvelopePoint: XM instrument containers and envelopes
"""

import array

__all__ = ['Pattern', 'Sample', 'XMSample', 'EnvelopePoint', 'Instrument', 'Note', 'XMNote']


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

    def __repr__(self):
        s = ''
        for row in range(self.n_rows):
            for channel in range(self.n_channels):
                s += str(self.data[channel][row]) + ' '
                if channel < self.n_channels - 1:
                    s += '| '
            s += '\n'
        return s


class Sample:
    """
    A sample is a digitized soundwave plus some additional attributes.
    This base class is used for MOD files where samples are referenced directly by notes.
    """

    # Loop type constants (shared with XMSample)
    LOOP_NONE = 0
    LOOP_FORWARD = 1
    LOOP_PINGPONG = 2

    def __init__(self):
        self.name = ""

        # MOD sample attributes
        self.finetune = 0      # 0-15 (MOD uses 4-bit unsigned finetune)
        self.volume = 64       # 0-64
        self.repeat_point = 0  # Loop start (in samples)
        self.repeat_len = 0    # Loop length (in samples, 0 or 1 = no loop)

        # Waveform data: signed 8-bit samples (MOD is always 8-bit)
        self.waveform = array.array('b')

        # Tells which sample pitch corresponds to true G (Sol).
        # Can be estimated, e.g., with MODSong.tune_sample().
        self.tune = ''
    
    @property
    def length(self) -> int:
        """Return sample length in samples (not bytes)."""
        return len(self.waveform)


class XMSample(Sample):
    """
    Extended sample class for XM files.
    XM samples have additional attributes: panning, relative note, 16-bit support,
    and a different finetune range.
    """

    def __init__(self):
        super().__init__()
        
        # Override finetune semantics: XM uses signed -128 to +127 (±127 = one half-tone)
        self.finetune = 0
        
        # XM-specific attributes
        self.loop_type = Sample.LOOP_NONE  # 0=none, 1=forward, 2=ping-pong
        self.panning = 128       # 0-255 (128 = center)
        self.relative_note = 0   # Signed: -96 to +95, 0 = C-4 plays as C-4
        self.is_16bit = False    # True for 16-bit samples
        
        # Internal: reserved byte for byte-perfect round-trip (users can ignore, default 0 is correct)
        self._reserved: int = 0
        
        # Waveform: 8-bit uses 'b' (signed byte), 16-bit uses 'h' (signed short)
        # Default to 8-bit, changed when loading 16-bit samples
        self.waveform = array.array('b')


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
        
        # --- Internal fields for byte-perfect round-trip (users can ignore these) ---
        # Instrument type byte: officially "always 0", default 0 is correct for new instruments
        self._type: int = 0
        # Original header size: 0 means "use standard size" (29 for empty, 263 for non-empty)
        self._header_size: int = 0
        
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
