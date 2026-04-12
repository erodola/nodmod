"""
Data types for tracker music formats (MOD, XM, etc.)

This module contains all the data classes used across different tracker formats:
- Pattern, Note, MODNote, XMNote: musical content
- Sample: audio waveform data
- Instrument, EnvelopePoint: XM instrument containers and envelopes
"""

import array

__all__ = ['Pattern', 'Sample', 'XMSample', 'S3MSample', 'EnvelopePoint', 'Instrument', 'Note', 'MODNote', 'XMNote', 'S3MNote']


class Note:
    """
    A note is a sample that is played at a specific pitch (period), possibly with envelopes.
    Every note can be modified by effects.
    This base class is used for MOD files.
    """

    def __init__(self, instrument_idx: int = 0, period: str = '', effect: str = ''):
        """Create a basic tracker note with sample/instrument, pitch, and effect."""

        # note that for mod files, instrument and sample are synonymous
        self.instrument_idx = instrument_idx

        self.period = period
        self.effect = effect

    def __repr__(self):
        """Return a compact tracker-style textual representation of the note."""
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
        """Return True if the note contains no pitch, instrument, or effect."""
        return self.instrument_idx == 0 and self.period == '' and self.effect == ''


# MOD-specific alias for clarity (MOD notes are identical to the base Note).
MODNote = Note


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
        """Create an XM note with an optional volume-column command.

        XM stores note effects in two places: the main effect column and the
        dedicated volume column. ``vol_cmd`` should be one of the supported XM
        volume commands such as ``'v'`` or ``'p'``; ``vol_val`` stores the
        command parameter and defaults to ``-1`` when no volume-column data is set.

        :param instrument_idx: 1-based instrument index, or 0 for none.
        :param period: Note text such as ``C-5`` or ``off``.
        :param effect: Main effect-column text.
        :param vol_cmd: XM volume-column command, or ``''`` for none.
        :param vol_val: XM volume-column value, or ``-1`` for none.
        """
        super().__init__(instrument_idx, period, effect)
        
        # XM-specific: volume column command and value
        # vol_cmd is a single character (v, d, c, b, a, u, h, p, l, r, g) or empty
        # vol_val is 0-64 for most commands, or -1 if no volume column data
        self.vol_cmd = vol_cmd
        self.vol_val = vol_val

    def __repr__(self):
        """Return a compact textual representation including the XM volume column."""
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
        """Return True if the XM note has no note, instrument, volume, or effect data."""
        return self.instrument_idx == 0 and self.period == '' and \
               self.effect == '' and self.vol_cmd == ''


class S3MNote(Note):
    """
    Note class for S3M files.
    S3M notes add a simple volume column without XM's extra volume commands.
    """

    def __init__(self, instrument_idx: int = 0, period: str = '', effect: str = '', volume: int = -1):
        """Create an S3M note with an optional 0-64 volume-column value.

        S3M has a simple numeric volume column rather than XM's extra volume
        command language. A negative value indicates that the volume column is empty.

        :param instrument_idx: 1-based sample index, or 0 for none.
        :param period: Note text such as ``C-5`` or ``off``.
        :param effect: Main effect-column text.
        :param volume: Volume-column value in ``0..64``, or ``-1`` for none.
        """
        super().__init__(instrument_idx, period, effect)
        self.volume = volume

    def __repr__(self):
        """Return a compact textual representation including the S3M volume column."""
        s = ''
        if self.period == '':
            s += '--- '
        elif self.period == 'off':
            s += '=== '
        else:
            s += self.period + ' '
        if self.instrument_idx == 0:
            s += '-- '
        else:
            s += f"{self.instrument_idx:02d} "
        if self.volume < 0:
            s += '-- '
        else:
            s += f"v{self.volume:02d} "
        if self.effect == '':
            s += '---'
        else:
            s += self.effect
        return s

    def is_empty(self) -> bool:
        """Return True if the S3M note has no note, instrument, volume, or effect data."""
        return self.instrument_idx == 0 and self.period == '' and self.effect == '' and self.volume < 0


class Pattern:
    """
    A pattern is a page of notes, and is part of a song.
    It is made of a number of channels of same length; each channel stores a note per row.
    """

    def __init__(self, n_rows: int, n_channels: int):
        """Create a pattern grid with the requested row and channel dimensions."""
        self.n_rows = n_rows
        self.n_channels = n_channels

        # use it as data[channel][row]
        self.data = [[Note() for _ in range(n_rows)] for _ in range(n_channels)]

    def __len__(self) -> int:
        """Return the number of rows in the pattern."""
        return self.n_rows

    def __repr__(self):
        """Render the pattern as tracker-style text, one row per line."""
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
        """Create an empty sample with MOD-compatible defaults."""
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

    def validate_loop(self) -> None:
        """Validate loop metadata against waveform bounds.

        Validation is strict and canonical:
        - loop start/length must be non-negative
        - empty waveforms must have loop metadata disabled (`0,0`)
        - loop start must be inside waveform bounds for active loops
        - loop end must not exceed waveform length
        - loop lengths `<= 1` are considered disabled and must be canonicalized
          to `repeat_point=0`, `repeat_len=0`
        """
        start = int(self.repeat_point)
        length = int(self.repeat_len)
        n = len(self.waveform)

        if start < 0:
            raise ValueError(f"Loop start {start} cannot be negative.")
        if length < 0:
            raise ValueError(f"Loop length {length} cannot be negative.")

        if n == 0:
            if start != 0 or length != 0:
                raise ValueError(
                    "Empty waveform requires loop metadata to be disabled "
                    "(repeat_point=0, repeat_len=0)."
                )
            return

        if length <= 1:
            if start != 0 or length != 0:
                raise ValueError(
                    "Loop length <= 1 is treated as no loop; "
                    "use repeat_point=0 and repeat_len=0."
                )
            return

        if start >= n:
            raise ValueError(f"Loop start {start} is outside sample length {n}.")
        if start + length > n:
            raise ValueError(f"Loop end {start + length} exceeds sample length {n}.")

    def sanitize_loop(self, *, mode: str = "coerce") -> None:
        """Normalize loop metadata to safe in-bounds values.

        Supported modes:
        - `coerce` (default): clamp and disable invalid loops according to
          nodmod's MOD-compatible safety rules.
        """
        if mode != "coerce":
            raise ValueError(f"Invalid sanitize mode {mode!r} (expected 'coerce').")

        start = int(self.repeat_point)
        length = int(self.repeat_len)
        n = len(self.waveform)

        start = max(0, start)
        length = max(0, length)

        if n == 0:
            self.repeat_point = 0
            self.repeat_len = 0
            return

        if start >= n:
            self.repeat_point = 0
            self.repeat_len = 0
            return

        max_len = n - start
        if length > max_len:
            length = max_len

        if length <= 1:
            self.repeat_point = 0
            self.repeat_len = 0
            return

        self.repeat_point = start
        self.repeat_len = length


class XMSample(Sample):
    """
    Extended sample class for XM files.
    XM samples have additional attributes: panning, relative note, 16-bit support,
    and a different finetune range.
    """

    def __init__(self):
        """Create an XM sample with panning, relative-note, and loop metadata."""
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


class S3MSample(Sample):
    """
    Sample class for S3M PCM instruments.
    """

    def __init__(self):
        """Create an S3M PCM sample/instrument record with serialization metadata."""
        super().__init__()
        self.instrument_type = 0
        self.filename = ""
        self.c2spd = 8363
        self.pack = 0
        self.flags = 0
        self.is_16bit = False
        self.is_stereo = False
        # Instrument-header byte 29 ("dsk" in old docs), preserved as-is.
        self.dsk = 0
        self.sample_offset = 0
        self._internal: bytes = b'\x00' * 12
        self._signature = "SCRS"

    @property
    def _reserved_byte(self) -> int:
        """Backward-compatible alias for the S3M instrument-header dsk byte."""
        return self.dsk

    @_reserved_byte.setter
    def _reserved_byte(self, value: int) -> None:
        self.dsk = int(value) & 0xFF


class EnvelopePoint:
    """A single point in a volume or panning envelope."""
    def __init__(self, frame: int = 0, value: int = 0):
        """Create an envelope point at the given frame and value."""
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
        """Create an empty XM instrument with default envelopes and playback settings."""
        self.name = ""
        
        # --- Internal fields for byte-perfect round-trip (users can ignore these) ---
        # Instrument type byte: officially "always 0", default 0 is correct for new instruments
        self._type: int = 0
        # Original header size: 0 means "use standard size" (29 for empty, 263 for non-empty)
        self._header_size: int = 0
        
        # List of Sample objects belonging to this instrument
        self.samples: list[Sample] = []
        
        # Internal 0-based sample_map: which sample slot to use for each note index 0-95.
        # Public APIs expose 1-based sample indices and convert at the boundary.
        # If empty, there is no public sample map configured yet.
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

    def set_sample_map(self, map96: list[int]):
        """
        Set the public 96-entry note-to-sample map using 1-based sample indices.

        The map is indexed by note number ``0..95`` (``C-1`` through ``B-8``).
        Values passed to this method are public 1-based sample indices; the
        instrument converts them internally to XM's stored 0-based representation.

        :param map96: A 96-entry list of 1-based sample indices.
        """
        if len(map96) != 96:
            raise ValueError(f"Sample map must have 96 entries (got {len(map96)}).")
        n_samples = len(self.samples)
        if n_samples == 0:
            # No samples: keep an empty map
            self.sample_map = []
            return
        for v in map96:
            if v < 1 or v > n_samples:
                raise IndexError(f"Invalid sample index {v} (expected 1-{n_samples}).")
        # Store as the internal 0-based sample_map representation.
        self.sample_map = [v - 1 for v in map96]

    def get_sample_map(self) -> list[int]:
        """
        Return the public 96-entry note-to-sample map using 1-based sample indices.

        :return: A 96-entry list of 1-based sample indices, or an empty list if no map is configured.
        """
        if not self.sample_map:
            return []
        return [v + 1 for v in self.sample_map]

    @staticmethod
    def _note_str_to_idx(note: str | int) -> int:
        """Convert a note name or numeric note index into a 0-based note index."""
        from .song import Song

        return Song.note_to_index(note)

    def clear_sample_map(self) -> None:
        """Clear the public note-to-sample mapping for the instrument."""
        self.sample_map = []

    def set_sample_for_note(self, note: str | int, sample_idx: int):
        """
        Set the sample mapped to one note in the instrument.

        ``note`` may be either a tracker note string such as ``C-4`` or a raw
        note index in ``0..95``. ``sample_idx`` uses the public 1-based sample numbering.

        :param note: Note string or numeric note index.
        :param sample_idx: 1-based sample index within this instrument.
        """
        note_idx = self._note_str_to_idx(note)
        if note_idx < 0 or note_idx >= 96:
            raise IndexError(f"Invalid note index {note} (expected 0-95).")
        n_samples = len(self.samples)
        if n_samples == 0:
            raise ValueError("Instrument has no samples.")
        if sample_idx < 1 or sample_idx > n_samples:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-{n_samples}).")
        if not self.sample_map or len(self.sample_map) != 96:
            self.sample_map = [0] * 96
        self.sample_map[note_idx] = sample_idx - 1

    def _normalize_envelope_points(self, points: list[EnvelopePoint | tuple[int, int]]) -> list[EnvelopePoint]:
        """Convert mixed envelope-point input into copied EnvelopePoint instances."""
        norm: list[EnvelopePoint] = []
        for p in points:
            if isinstance(p, EnvelopePoint):
                norm.append(EnvelopePoint(p.frame, p.value))
            else:
                frame, value = p
                norm.append(EnvelopePoint(frame, value))
        return norm

    def set_volume_envelope(
        self,
        points: list[EnvelopePoint | tuple[int, int]],
        sustain: int | None = None,
        loop: tuple[int, int] | None = None,
        enabled: bool = True,
        sustain_enabled: bool | None = None,
        loop_enabled: bool | None = None,
        raw_type: int | None = None,
    ):
        """Set the volume envelope points and control flags.

        The convenience arguments ``enabled``, ``sustain_enabled``, and
        ``loop_enabled`` are used to derive the XM envelope type bits unless
        ``raw_type`` is provided explicitly, in which case the raw bitfield wins.

        :param points: Envelope points as ``EnvelopePoint`` objects or ``(frame, value)`` tuples.
        :param sustain: Sustain-point index, or ``None`` for no sustain point.
        :param loop: Optional ``(start, end)`` loop-point indices.
        :param enabled: Whether the envelope should be marked enabled.
        :param sustain_enabled: Explicit sustain-flag override.
        :param loop_enabled: Explicit loop-flag override.
        :param raw_type: Optional raw XM envelope-type bitfield.
        """
        self.volume_envelope = self._normalize_envelope_points(points)
        self.volume_sustain_point = sustain if sustain is not None else 0
        if loop is not None:
            self.volume_loop_start, self.volume_loop_end = loop
        else:
            self.volume_loop_start = 0
            self.volume_loop_end = 0

        if sustain_enabled is None:
            sustain_enabled = sustain is not None
        if loop_enabled is None:
            loop_enabled = loop is not None

        if raw_type is not None:
            self.volume_type = raw_type
        else:
            self.volume_type = 0
            if enabled and self.volume_envelope:
                self.volume_type |= 0x01
            if sustain_enabled:
                self.volume_type |= 0x02
            if loop_enabled:
                self.volume_type |= 0x04

    def set_panning_envelope(
        self,
        points: list[EnvelopePoint | tuple[int, int]],
        sustain: int | None = None,
        loop: tuple[int, int] | None = None,
        enabled: bool = True,
        sustain_enabled: bool | None = None,
        loop_enabled: bool | None = None,
        raw_type: int | None = None,
    ):
        """Set the panning envelope points and control flags.

        The convenience arguments ``enabled``, ``sustain_enabled``, and
        ``loop_enabled`` are used to derive the XM envelope type bits unless
        ``raw_type`` is provided explicitly, in which case the raw bitfield wins.

        :param points: Envelope points as ``EnvelopePoint`` objects or ``(frame, value)`` tuples.
        :param sustain: Sustain-point index, or ``None`` for no sustain point.
        :param loop: Optional ``(start, end)`` loop-point indices.
        :param enabled: Whether the envelope should be marked enabled.
        :param sustain_enabled: Explicit sustain-flag override.
        :param loop_enabled: Explicit loop-flag override.
        :param raw_type: Optional raw XM envelope-type bitfield.
        """
        self.panning_envelope = self._normalize_envelope_points(points)
        self.panning_sustain_point = sustain if sustain is not None else 0
        if loop is not None:
            self.panning_loop_start, self.panning_loop_end = loop
        else:
            self.panning_loop_start = 0
            self.panning_loop_end = 0

        if sustain_enabled is None:
            sustain_enabled = sustain is not None
        if loop_enabled is None:
            loop_enabled = loop is not None

        if raw_type is not None:
            self.panning_type = raw_type
        else:
            self.panning_type = 0
            if enabled and self.panning_envelope:
                self.panning_type |= 0x01
            if sustain_enabled:
                self.panning_type |= 0x02
            if loop_enabled:
                self.panning_type |= 0x04


    def _validate_envelope(self, points: list[EnvelopePoint], sustain: int, loop_start: int, loop_end: int, name: str) -> None:
        """Validate one XM envelope against point-count and index constraints."""
        if len(points) > 12:
            raise ValueError(f"{name} envelope has {len(points)} points (XM supports up to 12).")
        last_frame = -1
        for p in points:
            if p.frame < 0:
                raise ValueError(f"{name} envelope frame {p.frame} is negative.")
            if p.value < 0 or p.value > 64:
                raise ValueError(f"{name} envelope value {p.value} out of range (0-64).")
            if p.frame < last_frame:
                raise ValueError(f"{name} envelope frames must be non-decreasing.")
            last_frame = p.frame
        if points:
            max_idx = len(points) - 1
            if sustain < 0 or sustain > max_idx:
                raise ValueError(f"{name} sustain point {sustain} out of range (0-{max_idx}).")
            if loop_start < 0 or loop_start > max_idx or loop_end < 0 or loop_end > max_idx:
                raise ValueError(f"{name} loop points out of range (0-{max_idx}).")
            if loop_start > loop_end:
                raise ValueError(f"{name} loop start {loop_start} greater than loop end {loop_end}.")

    def validate_volume_envelope(self) -> None:
        """Validate the instrument's volume envelope configuration."""
        self._validate_envelope(self.volume_envelope, self.volume_sustain_point, self.volume_loop_start, self.volume_loop_end, "volume")

    def validate_panning_envelope(self) -> None:
        """Validate the instrument's panning envelope configuration."""
        self._validate_envelope(self.panning_envelope, self.panning_sustain_point, self.panning_loop_start, self.panning_loop_end, "panning")

    def validate_envelopes(self) -> None:
        """Validate both instrument envelopes."""
        self.validate_volume_envelope()
        self.validate_panning_envelope()
