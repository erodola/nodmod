"""Support for loading, editing, and saving FastTracker 2 XM modules."""

import array
import copy
import warnings
import pydub  # needed for loading WAV samples

from nodmod import Song
from nodmod import XMSample
from nodmod import Sample
from nodmod import Instrument
from nodmod import EnvelopePoint
from nodmod import Pattern
from nodmod import XMNote
from .views import PlaybackRowView, SampleView


class XMSong(Song):
    """
    XM (FastTracker 2 Extended Module) song format.
    
    Unlike MOD files where notes reference samples directly, XM files use instruments.
    Each instrument can contain 0, 1, or many samples. Notes in XM files reference
    instruments by index, and the instrument determines which sample(s) to play.
    """

    PRESERVED_EFFECT_PREFIXES = Song.PRESERVED_EFFECT_PREFIXES | frozenset({'G', 'H'})
    
    @property
    def file_extension(self) -> str:
        """File extension used when saving XM songs."""
        return 'xm'
    
    @property
    def uses_linear_frequency(self) -> bool:
        """True if using linear frequency table, False for Amiga frequency table."""
        return (self.flags & 0x01) == 1

    @property
    def n_channels(self) -> int:
        """Number of channels exposed by the song."""
        return self._n_channels

    @n_channels.setter
    def n_channels(self, n: int) -> None:
        """Resize all patterns to the requested XM channel count."""
        if n < 1 or n > 32:
            raise ValueError(f"Invalid channel count {n} (expected 1-32).")
        self._n_channels = n
        for pat in self.patterns:
            if pat.n_channels < n:
                for _ in range(n - pat.n_channels):
                    pat.data.append([XMNote() for _ in range(pat.n_rows)])
            elif pat.n_channels > n:
                pat.data = pat.data[:n]
            pat.n_channels = n
    
    def _note_str_to_idx(self, note: str | int) -> int:
        """Convert a note string or index into a normalized numeric note index."""
        return Song.note_to_index(note)

    def __init__(self):
        """Create an empty XM song with one default pattern and XM header defaults."""
        super().__init__()
        
        # XM-specific: instruments list (notes reference instruments, not samples directly)
        # Each Instrument can contain multiple Sample objects
        self.instruments: list[Instrument] = []
        self.n_instruments = 0  # Number from header (includes empty instruments)
        
        # XM header fields (needed for file re-saving)
        self.tracker_name = ""     # Name of tracker that created the file
        self.song_restart = 0      # Pattern position to restart from when song loops
        self.flags = 0             # bit 0: 0 = Amiga frequency table, 1 = Linear frequency table
        self.default_speed = 6     # Default ticks per row
        self.default_tempo = 125   # Default BPM
        self.n_channels = 8        # Number of channels (XM supports 2-32)
        self.add_pattern()

    def set_default_speed(self, speed: int) -> None:
        """Set the XM default speed in ticks per row.

        This is the value used at song start until a later ``Fxx`` effect changes it.

        :param speed: Default ticks-per-row value in ``1..31``.
        """
        if speed < 1 or speed > 31:
            raise ValueError(f"Invalid default speed {speed} (expected 1-31).")
        self.default_speed = speed

    def set_default_tempo(self, bpm: int) -> None:
        """Set the XM default tempo in BPM.

        This is the starting BPM used until a later ``Fxx`` effect changes it.

        :param bpm: Default BPM value in ``32..255``.
        """
        if bpm < 32 or bpm > 255:
            raise ValueError(f"Invalid default tempo {bpm} (expected 32-255).")
        self.default_tempo = bpm

    def set_song_restart(self, song_restart_position: int) -> None:
        """Set the XM restart position used when the order list loops.

        :param song_restart_position: 0-based sequence position used as the restart point.
        """
        if song_restart_position < 0 or song_restart_position >= len(self.pattern_seq):
            raise IndexError(
                f"Invalid song restart position {song_restart_position} (expected 0-{len(self.pattern_seq)-1})."
            )
        self.song_restart = song_restart_position

    def set_linear_frequency(self, on: bool) -> None:
        """Enable or disable XM linear-frequency mode.

        XM supports two pitch models: Amiga periods and FastTracker 2 linear
        frequencies. This flag only affects XM playback and serialization.

        :param on: True for linear-frequency mode, False for Amiga mode.
        """
        if on:
            self.flags |= 0x01
        else:
            self.flags &= ~0x01

    def set_n_channels(self, n: int) -> None:
        """Convenience wrapper around the XM channel-count property.

        Resizing the channel count also resizes every pattern in the song.

        :param n: New XM channel count in ``1..32``.
        """
        self.n_channels = n
        
    def copy(self) -> 'XMSong':
        """
        Creates a deep copy of this song.
        
        :return: A new XMSong instance with all data copied.
        """
        new_song = XMSong()
        
        # Base Song attributes
        new_song.artist = self.artist
        new_song.songname = self.songname
        new_song.patterns = copy.deepcopy(self.patterns)
        new_song.pattern_seq = copy.deepcopy(self.pattern_seq)
        
        # XM-specific attributes
        new_song.instruments = copy.deepcopy(self.instruments)
        new_song.n_instruments = self.n_instruments
        new_song.tracker_name = self.tracker_name
        new_song.song_restart = self.song_restart
        new_song.flags = self.flags
        new_song.default_speed = self.default_speed
        new_song.default_tempo = self.default_tempo
        new_song.n_channels = self.n_channels
        
        return new_song

    '''
    -------------------------------------
    IMPORT AND EXPORT
    -------------------------------------
    '''

    def save_ascii(self, fname: str, verbose: bool = True):
        """
        Writes the song as readable text with ASCII encoding.
        
        Format per note: | period instrument volume effect |
        Example: | C-5 01 v64 A00 | --- -- -- --- |
        
        :param fname: Complete file path.
        :param verbose: False for silent saving.
        :return: None.
        """
        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)

        with open(fname, 'w', encoding='ascii') as file:
            file.write(self.to_ascii(sequence_only=True, include_headers=True))

        if verbose:
            print('done.')

    def save(self, fname: str, verbose: bool = True):
        """
        Saves the song as a standard XM file.
        
        :param fname: Complete file path.
        :param verbose: False for silent saving.
        :return: None.
        """
        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)

        data = bytearray()

        def str_to_bytes_padded(s: str, max_len: int, encoding: str = 'latin-1', pad: int = 0x00) -> bytes:
            """Convert string to bytes, truncating or padding as needed."""
            r = bytes(s, encoding)
            if len(r) > max_len:
                r = r[:max_len]
            else:
                r += bytes([pad] * (max_len - len(r)))
            return r

        def period_to_note_byte(period: str) -> int:
            """Convert period string (e.g., 'C-5', 'off', '') to XM note byte."""
            if period == '':
                return 0
            if period == 'off':
                return 97
            # Parse note like "C-5", "C#4", etc.
            # XM notes: 1=C-1, 2=C#1, ..., 12=B-1, 13=C-2, ...
            note_name = period[:2]  # e.g., "C-", "C#"
            octave = int(period[2:])  # e.g., 5
            note_idx = self.PERIOD_SEQ.index(note_name)  # 0-based: C-=0, C#=1, ..., B-=11
            return note_idx + (octave - 1) * 12 + 1

        def volume_to_byte(vol_cmd: str, vol_val: int) -> int:
            """Convert volume command and value to XM volume byte."""
            if vol_cmd == '':
                return 0
            if vol_cmd == 'v':
                # Set volume: 0x10-0x50 for values 0-64
                if vol_val <= 15:
                    return 0x10 + vol_val
                elif vol_val <= 31:
                    return 0x20 + (vol_val - 16)
                elif vol_val <= 47:
                    return 0x30 + (vol_val - 32)
                elif vol_val <= 63:
                    return 0x40 + (vol_val - 48)
                else:  # 64
                    return 0x50
            elif vol_cmd == 'd':  # volume slide down
                return 0x60 + vol_val
            elif vol_cmd == 'c':  # volume slide up
                return 0x70 + vol_val
            elif vol_cmd == 'b':  # fine volume slide down
                return 0x80 + vol_val
            elif vol_cmd == 'a':  # fine volume slide up
                return 0x90 + vol_val
            elif vol_cmd == 'u':  # vibrato speed
                return 0xA0 + vol_val
            elif vol_cmd == 'h':  # vibrato depth
                return 0xB0 + vol_val
            elif vol_cmd == 'p':  # set panning
                return 0xC0 + (vol_val // 4)
            elif vol_cmd == 'l':  # panning slide left
                return 0xD0 + vol_val
            elif vol_cmd == 'r':  # panning slide right
                return 0xE0 + vol_val
            elif vol_cmd == 'g':  # tone portamento
                return 0xF0 + vol_val
            return 0

        def effect_to_bytes(effect: str) -> tuple[int, int]:
            """Convert effect string (e.g., 'A00', 'E12') to (effect_type, effect_param) bytes."""
            if effect == '':
                return 0, 0
            
            effect_char = effect[0]
            param_str = effect[1:] if len(effect) > 1 else '00'
            
            # Map effect character to effect type byte
            if effect_char.isdigit() or effect_char in 'ABCDF':
                effect_type = int(effect_char, 16)
            elif effect_char == 'E':
                effect_type = 0x0E
            elif effect_char == 'G':
                effect_type = 0x10
            elif effect_char == 'H':
                effect_type = 0x11
            elif effect_char == 'L':
                effect_type = 0x15
            elif effect_char == 'P':
                effect_type = 0x19
            elif effect_char == 'R':
                effect_type = 0x1B
            elif effect_char == 'K':
                effect_type = 0x14
            elif effect_char == 'T':
                effect_type = 0x1D
            elif effect_char == 'X':
                effect_type = 0x21
            else:
                effect_type = 0
            
            # Parse parameter
            effect_param = int(param_str, 16) if param_str else 0
            
            return effect_type, effect_param

        # ----------------------------
        # Write XM header (60 bytes fixed + variable part)
        # ----------------------------

        # ID text (17 bytes)
        data += b'Extended Module: '
        
        # Module name (20 bytes, space-padded)
        data += str_to_bytes_padded(self.songname, 20, pad=0x20)
        
        # Always 0x1A
        data += bytes([0x1A])
        
        # Tracker name (20 bytes, space-padded)
        data += str_to_bytes_padded(self.tracker_name, 20, pad=0x20)
        
        # Version (2 bytes, little-endian: 0x0104 stored as 04 01)
        data += bytes([0x04, 0x01])
        
        # --- Variable header starts here (offset 60) ---
        
        # Header size (4 bytes) - size from this offset, which is 20 + 256 = 276
        header_size = 276
        data += header_size.to_bytes(4, byteorder='little')
        
        # Song length in pattern order table (2 bytes)
        data += len(self.pattern_seq).to_bytes(2, byteorder='little')
        
        # Restart position (2 bytes)
        data += self.song_restart.to_bytes(2, byteorder='little')
        
        # Number of channels (2 bytes)
        data += self.n_channels.to_bytes(2, byteorder='little')
        
        # Number of patterns (2 bytes)
        data += len(self.patterns).to_bytes(2, byteorder='little')
        
        # Number of instruments (2 bytes)
        data += len(self.instruments).to_bytes(2, byteorder='little')
        
        # Flags (2 bytes)
        data += self.flags.to_bytes(2, byteorder='little')
        
        # Default speed (2 bytes)
        data += self.default_speed.to_bytes(2, byteorder='little')
        
        # Default tempo (2 bytes)
        data += self.default_tempo.to_bytes(2, byteorder='little')
        
        # Pattern order table (256 bytes)
        pattern_order = bytearray(256)
        for i, p in enumerate(self.pattern_seq):
            if i < 256:
                pattern_order[i] = p
        data += pattern_order
        
        # ----------------------------
        # Write pattern data
        # ----------------------------
        
        for pat in self.patterns:
            # Pack pattern data first to know the size
            packed_pattern = bytearray()
            
            for r in range(pat.n_rows):
                for c in range(pat.n_channels):
                    note = pat.data[c][r]
                    
                    note_byte = period_to_note_byte(note.period)
                    inst_byte = note.instrument_idx
                    vol_cmd = getattr(note, 'vol_cmd', '')
                    vol_val = getattr(note, 'vol_val', -1)
                    vol_byte = volume_to_byte(vol_cmd, vol_val)
                    efx_type, efx_param = effect_to_bytes(note.effect)
                    
                    # Determine which fields are present (for packing)
                    has_note = note_byte != 0
                    has_inst = inst_byte != 0
                    has_vol = vol_byte != 0
                    has_efx = efx_type != 0
                    has_param = efx_param != 0
                    
                    # Decide whether to use packed or unpacked format
                    # Unpacked (5 bytes): write note, inst, vol, efx, param directly
                    # - Only valid when note_byte < 0x80 (otherwise conflicts with packed marker)
                    # - Makes sense when all 5 fields are non-zero
                    # Packed (1 + N bytes): write pack_byte followed by present fields
                    # - Saves space when some fields are zero
                    
                    all_fields_present = has_note and has_inst and has_vol and has_efx and has_param
                    can_use_unpacked = note_byte > 0 and note_byte < 0x80
                    
                    if all_fields_present and can_use_unpacked:
                        # Use unpacked format (5 bytes)
                        packed_pattern.append(note_byte)
                        packed_pattern.append(inst_byte)
                        packed_pattern.append(vol_byte)
                        packed_pattern.append(efx_type)
                        packed_pattern.append(efx_param)
                    else:
                        # Use packed format
                        pack_byte = 0x80
                        if has_note:
                            pack_byte |= 0x01
                        if has_inst:
                            pack_byte |= 0x02
                        if has_vol:
                            pack_byte |= 0x04
                        if has_efx:
                            pack_byte |= 0x08
                        if has_param:
                            pack_byte |= 0x10
                        
                        packed_pattern.append(pack_byte)
                        if has_note:
                            packed_pattern.append(note_byte)
                        if has_inst:
                            packed_pattern.append(inst_byte)
                        if has_vol:
                            packed_pattern.append(vol_byte)
                        if has_efx:
                            packed_pattern.append(efx_type)
                        if has_param:
                            packed_pattern.append(efx_param)
            
            # Pattern header (9 bytes)
            data += (9).to_bytes(4, byteorder='little')  # Pattern header length
            data += bytes([0])  # Packing type (always 0)
            data += pat.n_rows.to_bytes(2, byteorder='little')  # Number of rows
            data += len(packed_pattern).to_bytes(2, byteorder='little')  # Packed data size
            
            # Pattern data
            data += packed_pattern
        
        # ----------------------------
        # Write instrument data
        # ----------------------------
        
        for inst in self.instruments:
            n_samples = len(inst.samples)
            
            if n_samples == 0:
                # Empty instrument: use stored header size for round-trip, default 29
                inst_header_size = inst._header_size if inst._header_size > 0 else 29
                data += inst_header_size.to_bytes(4, byteorder='little')
                data += str_to_bytes_padded(inst.name, 22)
                data += bytes([inst._type])  # Instrument type
                data += (0).to_bytes(2, byteorder='little')  # Number of samples
                # If original header was larger than 29, pad with zeros (or the sample header size)
                extra_bytes = inst_header_size - 29
                if extra_bytes > 0:
                    # Usually this is the sample header size field (4 bytes = 0x28000000 = 40)
                    data += (40).to_bytes(4, byteorder='little')
                    extra_bytes -= 4
                    if extra_bytes > 0:
                        data += bytes(extra_bytes)
            else:
                # Full instrument header (263 bytes typical)
                # Header size covers up to (but not including) sample headers
                inst_header_size = 263
                data += inst_header_size.to_bytes(4, byteorder='little')
                data += str_to_bytes_padded(inst.name, 22)
                data += bytes([inst._type])  # Instrument type
                data += n_samples.to_bytes(2, byteorder='little')
                
                # Extended header (234 bytes)
                sample_header_size = 40
                data += sample_header_size.to_bytes(4, byteorder='little')
                
                # Internal 0-based sample_map as stored in XM instrument headers (96 bytes)
                sample_map = bytearray(96)
                for i, s in enumerate(inst.sample_map[:96]):
                    sample_map[i] = s
                data += sample_map
                
                # Volume envelope (48 bytes = 12 points * 4 bytes each)
                vol_env_data = bytearray(48)
                for i, point in enumerate(inst.volume_envelope[:12]):
                    vol_env_data[i * 4:i * 4 + 2] = point.frame.to_bytes(2, byteorder='little')
                    vol_env_data[i * 4 + 2:i * 4 + 4] = point.value.to_bytes(2, byteorder='little')
                data += vol_env_data
                
                # Panning envelope (48 bytes)
                pan_env_data = bytearray(48)
                for i, point in enumerate(inst.panning_envelope[:12]):
                    pan_env_data[i * 4:i * 4 + 2] = point.frame.to_bytes(2, byteorder='little')
                    pan_env_data[i * 4 + 2:i * 4 + 4] = point.value.to_bytes(2, byteorder='little')
                data += pan_env_data
                
                # Envelope point counts
                data += bytes([len(inst.volume_envelope)])
                data += bytes([len(inst.panning_envelope)])
                
                # Envelope control points
                data += bytes([inst.volume_sustain_point])
                data += bytes([inst.volume_loop_start])
                data += bytes([inst.volume_loop_end])
                data += bytes([inst.panning_sustain_point])
                data += bytes([inst.panning_loop_start])
                data += bytes([inst.panning_loop_end])
                
                # Envelope types
                data += bytes([inst.volume_type])
                data += bytes([inst.panning_type])
                
                # Vibrato settings
                data += bytes([inst.vibrato_type])
                data += bytes([inst.vibrato_sweep])
                data += bytes([inst.vibrato_depth])
                data += bytes([inst.vibrato_rate])
                
                # Volume fadeout
                data += inst.volume_fadeout.to_bytes(2, byteorder='little')
                
                # Reserved (22 bytes)
                data += bytes(22)
                
                # --- Sample headers (40 bytes each) ---
                for sample in inst.samples:
                    # Calculate sample length in bytes
                    sample_len_bytes = len(sample.waveform) * (2 if sample.is_16bit else 1)
                    
                    # Loop values in bytes
                    if sample.is_16bit:
                        loop_start_bytes = sample.repeat_point * 2
                        loop_len_bytes = sample.repeat_len * 2
                    else:
                        loop_start_bytes = sample.repeat_point
                        loop_len_bytes = sample.repeat_len
                    
                    data += sample_len_bytes.to_bytes(4, byteorder='little')
                    data += loop_start_bytes.to_bytes(4, byteorder='little')
                    data += loop_len_bytes.to_bytes(4, byteorder='little')
                    data += bytes([sample.volume])
                    
                    # Finetune (signed byte)
                    finetune = sample.finetune
                    if finetune < 0:
                        finetune += 256
                    data += bytes([finetune])
                    
                    # Sample type byte
                    sample_type = sample.loop_type & 0x03
                    if sample.is_16bit:
                        sample_type |= 0x10
                    data += bytes([sample_type])
                    
                    data += bytes([sample.panning])
                    
                    # Relative note (signed byte)
                    rel_note = sample.relative_note
                    if rel_note < 0:
                        rel_note += 256
                    data += bytes([rel_note])
                    
                    # Reserved byte (internal, for round-trip)
                    data += bytes([sample._reserved])
                    
                    # Sample name
                    data += str_to_bytes_padded(sample.name, 22)
                
                # --- Sample data (delta-encoded) ---
                for sample in inst.samples:
                    if len(sample.waveform) == 0:
                        continue
                    
                    if sample.is_16bit:
                        # 16-bit delta encoding
                        old = 0
                        for val in sample.waveform:
                            delta = (val - old) & 0xFFFF
                            # Convert to signed 16-bit for writing
                            if delta > 32767:
                                delta -= 65536
                            data += delta.to_bytes(2, byteorder='little', signed=True)
                            old = val
                    else:
                        # 8-bit delta encoding
                        old = 0
                        for val in sample.waveform:
                            # val is signed (-128 to 127)
                            delta = (val - old) & 0xFF
                            data += bytes([delta])
                            old = val

        # Write to file
        with open(fname, 'wb') as f:
            f.write(data)
        
        if verbose:
            print('done.')

    def load(self, fname: str, verbose: bool = True):
        """
        Loads a song from a standard XM file.

        :param fname: The path to the module file.
        :param verbose: False for silent loading.
        :return: None.
        """
        if verbose:
            print(f'Loading {fname}... ', end='', flush=True)

        with (open(fname, 'rb') as xm_file):

            data = bytearray(xm_file.read())

            # ----------------------------
            # Load fixed-size header data
            # ----------------------------

            header = data[:60]
            magic_string = header[:17].decode('ascii')

            if magic_string != "Extended Module: ":  # non-standard xm file
                raise NotImplementedError(f"Not an XM module. Magic string: {magic_string}.")
            
            self.songname = header[17:37].decode('latin-1').rstrip(' \x00')

            if header[37] != 0x1A:
                raise NotImplementedError("Invalid XM file format (header mismatch).")
            
            self.tracker_name = header[38:58].decode('latin-1').rstrip(' \x00')
            
            version = int.from_bytes(header[58:60], byteorder='little', signed=False)
            if version < 0x0104:
                raise NotImplementedError(f"Unsupported XM version {version} (expected 0x0104).")
            
            # ----------------------------
            # Load variable-size header data
            # ----------------------------

            # Header size (4 bytes at offset 60) - relative to offset 60, not file start
            header_size = int.from_bytes(data[60:64], byteorder='little', signed=False)

            # song restart position (where to loop back to)
            self.song_restart = int.from_bytes(data[66:68], byteorder='little', signed=False)

            # number of channels
            self.n_channels = int.from_bytes(data[68:70], byteorder='little', signed=False)
            if self.n_channels > 32:
                raise NotImplementedError(f"Too many channels: {self.n_channels} (XM supports 1-32).")

            # number of instruments (note : some instruments may be empty)
            self.n_instruments = int.from_bytes(data[72:74], byteorder='little', signed=False)
            if self.n_instruments > 256:
                raise NotImplementedError(f"Too many instruments: {self.n_instruments} (expected 1-256).")
            self.instruments = [Instrument() for _ in range(self.n_instruments)]

            # 0 = Amiga frequency table; 1 = Linear frequency table
            self.flags = int.from_bytes(data[74:76], byteorder='little', signed=False)
            
            # speed / ticks per row
            self.default_speed = int.from_bytes(data[76:78], byteorder='little', signed=False)

            # tempo / BPM
            self.default_tempo = int.from_bytes(data[78:80], byteorder='little', signed=False)

            # ----------------------------
            # Load pattern preamble data
            # ----------------------------

            # song length in patterns
            song_length = int.from_bytes(data[64:66], byteorder='little', signed=False)

            self.pattern_seq = [0] * song_length
            for p in range(song_length):
                self.pattern_seq[p] = data[80 + p]

            n_unique_patterns = int.from_bytes(data[70:72], byteorder='little', signed=False)
            if n_unique_patterns > 256:
                raise NotImplementedError(f"Too many patterns: {n_unique_patterns} (XM supports 1-256).")
            
            # ----------------------------
            # Load pattern data
            # ----------------------------

            def get_period(note_byte, pat_num, row, chan) -> str:

                note_val = note_byte & 0x7F

                if note_val == 0:
                    period_ = ''  # no note
                elif note_val <= 96:
                    # XM notes: 1=C-1, 2=C#1, ..., 12=B-1, 13=C-2, ...
                    note_idx = (note_val - 1) % 12
                    octave = (note_val - 1) // 12 + 1
                    period_ = f"{self.PERIOD_SEQ[note_idx]}{octave}"
                elif note_val == 97:
                    period_ = 'off'  # note off, "==" in OpenMPT
                else:
                    # Invalid note values (98-127) - warn and ignore
                    if verbose:
                        warnings.warn(f"Non-standard note value {note_val} at pattern {pat_num}, row {row}, channel {chan}. Ignoring.")
                    period_ = ''

                return period_

            def get_instrument(instrument_byte, pat_num, row, chan) -> int:

                if instrument_byte > 256:
                    return 0  # Treat as "no instrument change"
                            
                return instrument_byte

            def get_volume(volume_byte) -> tuple[str, int]:

                volume_cmd = ''
                volume_val = 0
                cmd_nibble = volume_byte & 0xF0
                val_nibble = volume_byte & 0x0F

                # FastTracker 2 leaves 0x01-0x0F and 0x51-0x5F undefined in the
                # volume column, so non-standard bytes are treated as "no volume
                # data" rather than assigned guessed semantics.

                if volume_byte == 0:
                    pass  # Empty volume column, keep defaults
                elif volume_byte < 0x10:
                    pass  # Values 0x01-0x0F are undefined in XM, ignore them
                elif cmd_nibble >= 0x10 and cmd_nibble <= 0x40:  # set volume 0-64
                    volume_cmd = 'v'
                    volume_val = volume_byte - 0x10
                elif volume_byte == 0x50:
                    volume_cmd = 'v'
                    volume_val = 64
                elif volume_byte > 0x50 and volume_byte < 0x60:
                    pass  # Values 0x51-0x5F are undefined in XM, ignore them

                elif cmd_nibble == 0x60:  # volume slide down
                    volume_cmd = 'd'
                    volume_val = val_nibble

                elif cmd_nibble == 0x70:  # volume slide up
                    volume_cmd = 'c'
                    volume_val = val_nibble

                elif cmd_nibble == 0x80:  # fine volume slide down
                    volume_cmd = 'b'
                    volume_val = val_nibble

                elif cmd_nibble == 0x90:  # fine volume slide up
                    volume_cmd = 'a'
                    volume_val = val_nibble

                elif cmd_nibble == 0xA0:  # vibrato speed
                    volume_cmd = 'u'
                    volume_val = val_nibble

                elif cmd_nibble == 0xB0:  # vibrato depth
                    volume_cmd = 'h'
                    volume_val = val_nibble

                elif cmd_nibble == 0xC0:  # set panning position
                    volume_cmd = 'p'
                    volume_val = val_nibble * 4

                elif cmd_nibble == 0xD0:  # panning slide left
                    volume_cmd = 'l'
                    volume_val = val_nibble

                elif cmd_nibble == 0xE0:  # panning slide right
                    volume_cmd = 'r'
                    volume_val = val_nibble

                elif cmd_nibble == 0xF0:  # tone portamento
                    volume_cmd = 'g'
                    volume_val = val_nibble

                return volume_cmd, volume_val

            def get_efx_type(effect_byte, pat_num, row, chan) -> str:

                if effect_byte <= 0x0D or effect_byte == 0x0F:
                    efx = f"{effect_byte:01X}"  
                    # 0=Arpeggio, 1=Porta up, 2=Porta down, 3=Tone porta, 4=Vibrato, 
                    # 5=Tone porta+Vol slide, 6=Vibrato+Vol slide, 7=Tremolo, 8=Set panning, 
                    # 9=Sample offset, A=Vol slide, B=Position jump, C=Set volume, 
                    # D=Pattern break, F=Set tempo/BPM

                elif effect_byte == 0x0E:
                    efx = "E"  # Extended effects, sub-effect in param high nibble

                elif effect_byte == 0x10:
                    efx = "G"  # Set global volume

                elif effect_byte == 0x11:
                    efx = "H"  # Global volume slide

                elif effect_byte == 0x15:
                    efx = "L"  # Set envelope position

                elif effect_byte == 0x19:
                    efx = "P"  # Panning slide

                elif effect_byte == 0x1B:
                    efx = "R"  # Multi retrig note

                elif effect_byte == 0x14:
                    efx = "K"  # Key off (parameter value is ignored)

                elif effect_byte == 0x1D:
                    efx = "T"  # Tremor

                elif effect_byte == 0x21:
                    efx = "X"  # Extra fine portamento (X1x=up, X2x=down)

                else:
                    if verbose:
                        warnings.warn(f"Non-standard effect type {effect_byte:02X} at pattern {pat_num}, row {row}, channel {chan}. Ignoring.")
                    efx = ""  # Ignore non-standard effect

                return efx

            def get_efx_param(effect_type, effect_param_byte):
                # For most effects, the parameter is the full byte (00-FF), e.g. B1A
                #
                # For effects E and X, the parameter byte is split into two nibbles:
                #   - High nibble: sub-effect type
                #   - Low nibble: actual parameter (0-F)
                #
                # Effect E sub-effects (from xm.txt):
                #   E1x - Fine porta up
                #   E2x - Fine porta down
                #   E3x - Set gliss control
                #   E4x - Set vibrato control
                #   E5x - Set finetune
                #   E6x - Set loop begin/loop
                #   E7x - Set tremolo control
                #   E9x - Retrig note
                #   EAx - Fine volume slide up
                #   EBx - Fine volume slide down
                #   ECx - Note cut
                #   EDx - Note delay
                #   EEx - Pattern delay
                #
                # Effect X sub-effects:
                #   X1x - Extra fine porta up
                #   X2x - Extra fine porta down
                #
                if effect_type == "E" or effect_type == "X":
                    sub_effect = (effect_param_byte >> 4) & 0x0F
                    param = effect_param_byte & 0x0F
                    return f"{sub_effect:01X}{param:01X}"
                else:
                    return f"{effect_param_byte:02X}"

            self.patterns = []

            # Pattern data starts at offset 60 + header_size
            pattern_data = data[60 + header_size:]
            cur_pat_idx = 0

            for p in range(n_unique_patterns):

                if pattern_data[cur_pat_idx:cur_pat_idx + 4][0] != 9:
                    raise NotImplementedError(f"Unsupported pattern header length {pattern_data[cur_pat_idx:cur_pat_idx + 4][0]} (expected 9).")

                if pattern_data[cur_pat_idx+4] != 0:
                    raise NotImplementedError("Unsupported packing type (expected 0).")

                n_rows = int.from_bytes(pattern_data[cur_pat_idx+5:cur_pat_idx+7], byteorder='little', signed=False)

                # each pattern has a different size in bytes due to packing
                pattern_data_size = int.from_bytes(pattern_data[cur_pat_idx+7:cur_pat_idx+9], byteorder='little', signed=False)

                pat = Pattern(n_rows, self.n_channels)

                # Handle empty patterns (packed_size == 0 means no note data)
                if pattern_data_size == 0:
                    # Initialize all notes as empty XMNote objects
                    for c in range(self.n_channels):
                        for r in range(n_rows):
                            pat.data[c][r] = XMNote()
                    self.patterns.append(pat)
                    cur_pat_idx += 9  # Skip pattern header only
                    continue

                # Read the pattern data byte by byte.
                # Notes are stored row-wise: first all notes across all channels for row 0, then row 1, etc.

                c = 0
                r = 0

                byte_idx = cur_pat_idx + 9
                while True:

                    packed_byte = pattern_data[byte_idx]

                    instrument = 0
                    effect = ''
                    period = ''
                    volume_cmd = ''
                    volume_val = -1

                    is_packed = packed_byte & 0x80  # MSB is set

                    if is_packed:
                        if packed_byte & 0x01:
                            byte_idx += 1
                            period = get_period(pattern_data[byte_idx], p, r, c)
                    elif not is_packed:
                        period = get_period(packed_byte, p, r, c)

                    if not is_packed or (is_packed and packed_byte & 0x02):
                        byte_idx += 1
                        instrument = get_instrument(pattern_data[byte_idx], p, r, c)

                    if not is_packed or (is_packed and packed_byte & 0x04):
                        byte_idx += 1
                        volume_cmd, volume_val = get_volume(pattern_data[byte_idx])

                    has_effect_byte = not is_packed or (is_packed and packed_byte & 0x08)
                    if has_effect_byte:
                        byte_idx += 1
                        effect = get_efx_type(pattern_data[byte_idx], p, r, c)

                    if not is_packed or (is_packed and packed_byte & 0x10):
                        byte_idx += 1
                        # If we have a param but no effect type, use effect type 0 (arpeggio)
                        # But only if we didn't just ignore a non-standard effect
                        if effect == '' and not has_effect_byte:
                            # No effect byte was present, but param is - use arpeggio
                            effect = '0'
                        if effect != '':
                            effect = f"{effect}{get_efx_param(effect, pattern_data[byte_idx])}"

                    # Create XM note with all parsed data
                    note = XMNote()
                    note.instrument_idx = instrument
                    note.period = period
                    note.effect = effect
                    note.vol_cmd = volume_cmd
                    note.vol_val = volume_val
                    
                    pat.data[c][r] = note

                    # move to the next note (next channel or row)
                    byte_idx += 1

                    c += 1  # move to the next channel
                    if c == self.n_channels:
                        c = 0
                        r += 1

                    if byte_idx == cur_pat_idx + 9 + pattern_data_size:
                        break  # next pattern

                self.patterns.append(pat)

                # next pattern
                cur_pat_idx += 9 + pattern_data_size

            # ----------------------------
            # Load instrument data
            # ----------------------------

            # Each instrument has a variable-size header. If n_samples > 0,
            # an extended header with sample mapping and envelopes follows.
            
            instrument_data = pattern_data[cur_pat_idx:]
            cur_inst_idx = 0
            
            for i in range(self.n_instruments):
                # Read instrument header size (4 bytes)
                inst_header_size = int.from_bytes(
                    instrument_data[cur_inst_idx:cur_inst_idx + 4], 
                    byteorder='little', signed=False
                )
                
                # Store original header size for round-trip saving
                self.instruments[i]._header_size = inst_header_size
                
                # Instrument name (22 bytes at offset +4)
                name_bytes = instrument_data[cur_inst_idx + 4:cur_inst_idx + 26]
                self.instruments[i].name = name_bytes.decode('latin-1').rstrip('\x00')
                
                # Instrument type at offset +26 (officially "always 0", but often random)
                self.instruments[i]._type = instrument_data[cur_inst_idx + 26]
                
                # Number of samples in this instrument (2 bytes at offset +27)
                n_samples = int.from_bytes(
                    instrument_data[cur_inst_idx + 27:cur_inst_idx + 29], 
                    byteorder='little', signed=False
                )
                
                if verbose:
                    print(f"Instrument {i + 1}: '{self.instruments[i].name}' ({n_samples} samples)")
                
                # Total bytes to skip for this instrument (header + sample headers + sample data)
                total_inst_size = inst_header_size
                
                if n_samples > 0:
                    # Extended header follows (starting at offset +29)
                    ext_offset = cur_inst_idx + 29
                    
                    # Sample header size (4 bytes) - size of each sample header that follows
                    sample_header_size = int.from_bytes(
                        instrument_data[ext_offset:ext_offset + 4], 
                        byteorder='little', signed=False
                    )
                    
                    # Internal 0-based sample_map for note indices 0-95 (96 bytes at offset +33)
                    sample_map_bytes = instrument_data[ext_offset + 4:ext_offset + 100]
                    self.instruments[i].sample_map = list(sample_map_bytes)
                    
                    # Volume envelope points (48 bytes at offset +129)
                    # Each point is 2 words (4 bytes): frame (X), value (Y)
                    vol_env_bytes = instrument_data[ext_offset + 100:ext_offset + 148]
                    
                    # Panning envelope points (48 bytes at offset +177)
                    pan_env_bytes = instrument_data[ext_offset + 148:ext_offset + 196]
                    
                    # Number of volume points (1 byte at offset +225)
                    n_vol_points = instrument_data[ext_offset + 196]
                    
                    # Number of panning points (1 byte at offset +226)
                    n_pan_points = instrument_data[ext_offset + 197]
                    
                    # Parse volume envelope points
                    for p in range(n_vol_points):
                        frame = int.from_bytes(vol_env_bytes[p * 4:p * 4 + 2], byteorder='little', signed=False)
                        value = int.from_bytes(vol_env_bytes[p * 4 + 2:p * 4 + 4], byteorder='little', signed=False)
                        self.instruments[i].volume_envelope.append(EnvelopePoint(frame, value))
                    
                    # Parse panning envelope points
                    for p in range(n_pan_points):
                        frame = int.from_bytes(pan_env_bytes[p * 4:p * 4 + 2], byteorder='little', signed=False)
                        value = int.from_bytes(pan_env_bytes[p * 4 + 2:p * 4 + 4], byteorder='little', signed=False)
                        self.instruments[i].panning_envelope.append(EnvelopePoint(frame, value))
                    
                    # Envelope control points (offsets +227 to +232)
                    self.instruments[i].volume_sustain_point = instrument_data[ext_offset + 198]
                    self.instruments[i].volume_loop_start = instrument_data[ext_offset + 199]
                    self.instruments[i].volume_loop_end = instrument_data[ext_offset + 200]
                    self.instruments[i].panning_sustain_point = instrument_data[ext_offset + 201]
                    self.instruments[i].panning_loop_start = instrument_data[ext_offset + 202]
                    self.instruments[i].panning_loop_end = instrument_data[ext_offset + 203]
                    
                    # Envelope types (offset +233, +234)
                    # bit 0: On, bit 1: Sustain, bit 2: Loop
                    self.instruments[i].volume_type = instrument_data[ext_offset + 204]
                    self.instruments[i].panning_type = instrument_data[ext_offset + 205]
                    
                    # Vibrato settings (offsets +235 to +238)
                    self.instruments[i].vibrato_type = instrument_data[ext_offset + 206]
                    self.instruments[i].vibrato_sweep = instrument_data[ext_offset + 207]
                    self.instruments[i].vibrato_depth = instrument_data[ext_offset + 208]
                    self.instruments[i].vibrato_rate = instrument_data[ext_offset + 209]
                    
                    # Volume fadeout (2 bytes at offset +239)
                    self.instruments[i].volume_fadeout = int.from_bytes(
                        instrument_data[ext_offset + 210:ext_offset + 212], 
                        byteorder='little', signed=False
                    )

                    # ----------------------------
                    # Load sample data
                    # ----------------------------
                    
                    # Pre-allocate sample slots for this instrument
                    self.instruments[i].samples = [XMSample() for _ in range(n_samples)]
                    
                    # Calculate total size: instrument header + all sample headers + all sample data
                    # Sample headers follow the instrument header
                    # Sample data follows all sample headers
                    sample_headers_start = cur_inst_idx + inst_header_size
                    total_sample_data_size = 0
                    
                    # First pass: read all sample headers and calculate total data size
                    sample_lengths = []  # Store lengths for second pass
                    sample_is_16bit = []  # Store bit depth for second pass
                    
                    for s in range(n_samples):
                        sample_hdr_offset = sample_headers_start + s * sample_header_size
                        
                        # Sample length in bytes (4 bytes at offset 0)
                        sample_length = int.from_bytes(
                            instrument_data[sample_hdr_offset:sample_hdr_offset + 4],
                            byteorder='little', signed=False
                        )
                        
                        # Loop start in bytes (4 bytes at offset 4)
                        loop_start = int.from_bytes(
                            instrument_data[sample_hdr_offset + 4:sample_hdr_offset + 8],
                            byteorder='little', signed=False
                        )
                        
                        # Loop length in bytes (4 bytes at offset 8)
                        loop_length = int.from_bytes(
                            instrument_data[sample_hdr_offset + 8:sample_hdr_offset + 12],
                            byteorder='little', signed=False
                        )
                        
                        # Volume (1 byte at offset 12)
                        volume = instrument_data[sample_hdr_offset + 12]
                        
                        # Finetune (1 signed byte at offset 13, -128 to +127)
                        finetune = instrument_data[sample_hdr_offset + 13]
                        if finetune > 127:
                            finetune -= 256  # Convert to signed
                        
                        # Sample type (1 byte at offset 14)
                        # Bits 0-1: loop type (0=none, 1=forward, 2=ping-pong)
                        # Bit 4: 16-bit sample
                        sample_type = instrument_data[sample_hdr_offset + 14]
                        loop_type = sample_type & 0x03
                        is_16bit = bool(sample_type & 0x10)
                        
                        # Panning (1 byte at offset 15, 0-255)
                        panning = instrument_data[sample_hdr_offset + 15]
                        
                        # Relative note (1 signed byte at offset 16, -96 to +95)
                        relative_note = instrument_data[sample_hdr_offset + 16]
                        if relative_note > 127:
                            relative_note -= 256  # Convert to signed
                        
                        # Reserved byte at offset 17 (used by ModPlug for ADPCM indicator)
                        # Store for byte-perfect round-trip
                        reserved = instrument_data[sample_hdr_offset + 17]
                        
                        # Sample name (22 bytes at offset 18)
                        sample_name = instrument_data[sample_hdr_offset + 18:sample_hdr_offset + 40]
                        
                        # Store sample metadata
                        sample = self.instruments[i].samples[s]
                        sample.name = sample_name.decode('latin-1').rstrip('\x00')
                        sample.volume = volume
                        sample.finetune = finetune
                        sample.panning = panning
                        sample.relative_note = relative_note
                        sample._reserved = reserved
                        sample.loop_type = loop_type
                        sample.is_16bit = is_16bit
                        
                        # Convert loop values from bytes to samples
                        if is_16bit:
                            sample.repeat_point = loop_start // 2
                            sample.repeat_len = loop_length // 2
                        else:
                            sample.repeat_point = loop_start
                            sample.repeat_len = loop_length
                        
                        sample_lengths.append(sample_length)
                        sample_is_16bit.append(is_16bit)
                        total_sample_data_size += sample_length
                    
                    # Second pass: read sample data (comes after all sample headers)
                    sample_data_start = sample_headers_start + n_samples * sample_header_size
                    sample_data_offset = 0
                    
                    for s in range(n_samples):
                        sample = self.instruments[i].samples[s]
                        sample_length = sample_lengths[s]
                        is_16bit = sample_is_16bit[s]
                        
                        if sample_length == 0:
                            continue
                        
                        raw_data = instrument_data[
                            sample_data_start + sample_data_offset:
                            sample_data_start + sample_data_offset + sample_length
                        ]
                        
                        # Decode delta-encoded sample data
                        if is_16bit:
                            # 16-bit samples: signed 16-bit integers, little-endian
                            n_samples_count = sample_length // 2
                            sample.waveform = array.array('h')  # signed short
                            old = 0
                            for j in range(n_samples_count):
                                # Read little-endian signed 16-bit value
                                delta = int.from_bytes(
                                    raw_data[j * 2:j * 2 + 2],
                                    byteorder='little', signed=True
                                )
                                new_val = (old + delta) & 0xFFFF
                                # Convert to signed
                                if new_val > 32767:
                                    new_val -= 65536
                                sample.waveform.append(new_val)
                                old = new_val
                        else:
                            # 8-bit samples: signed bytes
                            sample.waveform = array.array('b')  # signed byte
                            old = 0
                            for j in range(sample_length):
                                delta = raw_data[j]
                                if delta > 127:
                                    delta -= 256  # Convert to signed
                                new_val = (old + delta) & 0xFF
                                # Convert to signed for storage
                                if new_val > 127:
                                    new_val -= 256
                                sample.waveform.append(new_val)
                                old = new_val & 0xFF  # Keep as unsigned for delta calc
                        
                        sample_data_offset += sample_length
                    
                    total_inst_size = inst_header_size + n_samples * sample_header_size + total_sample_data_size
                
                # Move to next instrument
                cur_inst_idx += total_inst_size

        if verbose:
            print('done.')

    '''
    -------------------------------------
    SONG
    -------------------------------------
    '''

    def iter_playback_rows(
        self,
        *,
        profile: str | None = None,  # noqa: ARG002
        exact: bool = True,  # noqa: ARG002
        max_steps: int = 250_000,
    ):
        """Yield visited XM rows with source coordinates and timing metadata.

        ``profile`` and ``exact`` are reserved compatibility parameters and
        are currently accepted as explicit no-ops.
        """
        if max_steps <= 0:
            raise ValueError(f"Invalid max_steps {max_steps} (expected > 0).")

        bpm = self.default_tempo
        speed = self.default_speed
        d = Song.get_tick_duration(bpm)

        jump_to_position = -1
        jump_to_pattern = -1
        stop_song = False
        self_jump_count = 0
        self_jump_limit = 5
        start_row = 0
        elapsed = 0.0
        visit_idx = 0

        loop_start_row = [0] * self.n_channels

        seq_idx = 0
        while seq_idx < len(self.pattern_seq):
            p = self.pattern_seq[seq_idx]
            if p < 0 or p >= len(self.patterns):
                seq_idx += 1
                continue
            pat = self.patterns[p]
            n_rows = pat.n_rows
            n_channels = pat.n_channels

            if n_channels != len(loop_start_row):
                loop_start_row = [0] * n_channels

            loop_count = [0] * n_channels
            r = start_row
            while r < n_rows:
                if visit_idx >= max_steps:
                    raise RuntimeError(
                        f"iter_playback_rows exceeded max_steps={max_steps}; possible runaway playback loop."
                    )

                if jump_to_position != -1:
                    jump_to_position = -1
                    start_row = 0

                row_delay = 0
                loop_jump_row = None
                pending_jump_row = None
                pending_jump_pattern = None
                saw_self_jump = False

                for c in range(n_channels):
                    efx = pat.data[c][r].effect
                    if efx == "":
                        continue

                    if efx[0] == "F":
                        if len(efx) < 2:
                            continue
                        try:
                            v = int(efx[1:], 16)
                        except ValueError:
                            continue
                        if v <= 31:
                            if v != 0:
                                speed = v
                        else:
                            bpm = v
                        d = Song.get_tick_duration(bpm)

                    elif efx[0] == "D":
                        if len(self.pattern_seq) <= 1:
                            continue
                        if len(efx) == 1:
                            pending_jump_row = 0
                            continue
                        param = efx[1:]
                        if len(param) == 1:
                            param = f"0{param}"
                        try:
                            hi = int(param[0], 16)
                            lo = int(param[1], 16)
                        except ValueError:
                            continue
                        pending_jump_row = hi * 10 + lo

                    elif efx[0] == "B":
                        param = efx[1:] if len(efx) > 1 else "00"
                        try:
                            dest = int(param, 16)
                        except ValueError:
                            continue
                        if dest < len(self.pattern_seq):
                            if dest > seq_idx:
                                pending_jump_pattern = dest
                            elif dest == seq_idx:
                                saw_self_jump = True
                            else:
                                stop_song = True

                    elif efx[0] == "E" and len(efx) >= 3:
                        cmd = efx[1].upper()
                        try:
                            val = int(efx[2], 16)
                        except ValueError:
                            continue
                        if cmd == "6":
                            if val == 0:
                                old_loop_start = loop_start_row[c]
                                loop_start_row[c] = r
                                if loop_count[c] == -1 and r > old_loop_start:
                                    loop_count[c] = 0
                            else:
                                if loop_count[c] == 0:
                                    loop_count[c] = val
                                if loop_count[c] > 0 and loop_start_row[c] < r:
                                    loop_count[c] -= 1
                                    loop_jump_row = loop_start_row[c]
                                    if loop_count[c] == 0:
                                        loop_count[c] = -1
                        elif cmd == "E":
                            row_delay = val

                row_duration = d * speed
                if row_delay > 0:
                    row_duration *= (row_delay + 1)
                start_sec = elapsed
                elapsed += row_duration
                yield PlaybackRowView(
                    visit_idx=visit_idx,
                    sequence_idx=seq_idx,
                    pattern_idx=p,
                    row=r,
                    start_sec=start_sec,
                    end_sec=elapsed,
                    speed=speed,
                    tempo=bpm,
                )
                visit_idx += 1

                if stop_song:
                    break

                if saw_self_jump and pending_jump_row is not None and self_jump_count < self_jump_limit:
                    pending_jump_pattern = seq_idx
                    self_jump_count += 1

                if loop_jump_row is not None:
                    r = loop_jump_row
                    continue

                if pending_jump_pattern is not None:
                    jump_to_pattern = pending_jump_pattern
                    if pending_jump_row is not None:
                        start_row = pending_jump_row
                    else:
                        start_row = 0
                    break

                if pending_jump_row is not None:
                    jump_to_position = pending_jump_row
                    start_row = jump_to_position
                    break

                r += 1

            if stop_song:
                break

            if jump_to_pattern != -1:
                seq_idx = jump_to_pattern
                jump_to_pattern = -1
                start_row = 0
                continue

            seq_idx += 1

    def timestamp(self) -> list[list[tuple[float, int, int]]]:
        """
        Compute XM row-end timestamps, speeds, and BPM values.

        Each tuple stores the cumulative end time after the row has finished
        playing. This matches MOD and differs from S3M, whose timestamp tuples
        currently record row-start times.

        :return: A list of visited sequence entries, each containing
                 ``(end_time_seconds, speed, bpm)`` tuples.
        """

        bpm = self.default_tempo
        speed = self.default_speed  # ticks per row

        d = Song.get_tick_duration(bpm)

        timestamps = []
        speeds = []
        bpms = []

        jump_to_position = -1  # modified by Dxx effect
        jump_to_pattern = -1  # modified by Bxx effect (song order index)
        stop_song = False
        self_jump_count = 0
        self_jump_limit = 5

        start_row = 0

        # XM quirk: loop start positions are not reset across patterns
        loop_start_row = [0] * self.n_channels

        seq_idx = 0
        while seq_idx < len(self.pattern_seq):

            p = self.pattern_seq[seq_idx]
            if p < 0 or p >= len(self.patterns):
                seq_idx += 1
                continue
            pat = self.patterns[p]
            n_rows = pat.n_rows
            n_channels = pat.n_channels

            if n_channels != len(loop_start_row):
                loop_start_row = [0] * n_channels

            pattern_timestamps = []
            pattern_speeds = []
            pattern_bpms = []

            loop_count = [0] * n_channels

            r = start_row
            while r < n_rows:

                if jump_to_position != -1:
                    jump_to_position = -1
                    start_row = 0

                row_delay = 0
                loop_jump_row = None
                pending_jump_row = None
                pending_jump_pattern = None
                saw_self_jump = False

                for c in range(n_channels):

                    efx = pat.data[c][r].effect
                    if efx != "":

                        if efx[0] == "F":  # set speed/tempo (XM)

                            if len(efx) < 2:
                                continue
                            try:
                                v = int(efx[1:], 16)
                            except ValueError:
                                continue
                            if v <= 31:
                                if v != 0:
                                    speed = v
                            else:
                                bpm = v

                            d = Song.get_tick_duration(bpm)

                        elif efx[0] == "D":  # pattern break (BCD)
                            if len(self.pattern_seq) <= 1:
                                continue
                            if len(efx) == 1:
                                pending_jump_row = 0
                                continue
                            param = efx[1:]
                            if len(param) == 1:
                                param = f"0{param}"
                            try:
                                hi = int(param[0], 16)
                                lo = int(param[1], 16)
                            except ValueError:
                                continue
                            pending_jump_row = hi * 10 + lo

                        elif efx[0] == "B":  # pattern jump
                            param = efx[1:] if len(efx) > 1 else "00"
                            try:
                                dest = int(param, 16)
                            except ValueError:
                                continue
                            if dest < len(self.pattern_seq):
                                if dest > seq_idx:
                                    pending_jump_pattern = dest
                                elif dest == seq_idx:
                                    saw_self_jump = True
                                else:
                                    stop_song = True

                        elif efx[0] == "E" and len(efx) >= 3:
                            cmd = efx[1].upper()
                            try:
                                val = int(efx[2], 16)
                            except ValueError:
                                continue
                            if cmd == "6":  # pattern loop (per-channel)
                                if val == 0:
                                    old_loop_start = loop_start_row[c]
                                    loop_start_row[c] = r
                                    if loop_count[c] == -1 and r > old_loop_start:
                                        loop_count[c] = 0
                                else:
                                    if loop_count[c] == 0:
                                        loop_count[c] = val
                                    if loop_count[c] > 0 and loop_start_row[c] < r:
                                        loop_count[c] -= 1
                                        loop_jump_row = loop_start_row[c]
                                        if loop_count[c] == 0:
                                            loop_count[c] = -1
                            elif cmd == "E":  # pattern delay
                                row_delay = val

                row_duration = d * speed
                if row_delay > 0:
                    row_duration *= (row_delay + 1)
                pattern_timestamps.append(row_duration)
                pattern_speeds.append(speed)
                pattern_bpms.append(bpm)

                if stop_song:
                    break

                if saw_self_jump and pending_jump_row is not None and self_jump_count < self_jump_limit:
                    pending_jump_pattern = seq_idx
                    self_jump_count += 1

                if loop_jump_row is not None:
                    r = loop_jump_row
                    continue

                if pending_jump_pattern is not None:
                    jump_to_pattern = pending_jump_pattern
                    if pending_jump_row is not None:
                        start_row = pending_jump_row
                    else:
                        start_row = 0
                    break

                if pending_jump_row is not None:
                    jump_to_position = pending_jump_row
                    start_row = jump_to_position
                    break

                r += 1

            timestamps.append(pattern_timestamps)
            speeds.append(pattern_speeds)
            bpms.append(pattern_bpms)

            if stop_song:
                break

            if jump_to_pattern != -1:
                seq_idx = jump_to_pattern
                jump_to_pattern = -1
                start_row = 0
                continue

            seq_idx += 1

        cum = 0
        for p in range(len(timestamps)):
            for r in range(len(timestamps[p])):
                cum += timestamps[p][r]
                timestamps[p][r] = (cum, speeds[p][r], bpms[p][r])

        return timestamps

    '''
    -------------------------------------
    NOTES
    -------------------------------------
    '''

    def set_note(
        self,
        sequence_idx: int,
        channel: int,
        row: int,
        instrument_idx: int,
        period: str,
        effect: str = "",
        vol_cmd: str | None = None,
        vol_val: int | None = None,
    ):
        """
        Writes an XM note in the given sequence pattern, channel, and row.
        If no effect is given and the current note already has a speed effect, leaves it unchanged.
        If ``vol_cmd`` and ``vol_val`` are both omitted, the existing volume column is preserved.
        XM volume-column commands are format-specific shorthands such as ``'v'``
        (set volume), ``'d'``/``'c'`` (slides), ``'p'`` (set panning), and ``'g'``
        (tone portamento).

        :param sequence_idx: The 0-based sequence index to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param instrument_idx: The 1-based instrument index to write.
        :param period: The note period (pitch) to write, e.g. "C-4".
        :param effect: The note effect, e.g. "ED1".
        :param vol_cmd: Volume-column command, or ``None`` to keep the existing command.
        :param vol_val: Volume-column parameter, or ``None`` to keep the existing value.
        :return: None.
        """

        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        pat = self.patterns[self.pattern_seq[sequence_idx]]

        if row < 0 or row >= pat.n_rows:
            raise IndexError(f"Invalid row index {row} (expected 0-{pat.n_rows-1}).")

        if channel < 0 or channel >= pat.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{self.n_channels-1}).")

        cur_note = pat.data[channel][row]

        cur_efx = cur_note.effect
        if effect == '':
            effect = self._preserved_effect(cur_efx)

        new_note = XMNote()
        new_note.instrument_idx = instrument_idx
        new_note.period = period
        new_note.effect = effect

        if vol_cmd is None and vol_val is None:
            new_note.vol_cmd = getattr(cur_note, 'vol_cmd', '')
            new_note.vol_val = getattr(cur_note, 'vol_val', -1)
        else:
            new_note.vol_cmd = vol_cmd if vol_cmd is not None else ''
            new_note.vol_val = vol_val if vol_val is not None else -1

        pat.data[channel][row] = new_note

    def set_note_rc(
        self,
        sequence_idx: int,
        row: int,
        channel: int,
        instrument_idx: int,
        period: str,
        effect: str = "",
        vol_cmd: str | None = None,
        vol_val: int | None = None,
    ):
        """Write an XM note using canonical coordinate order (sequence, row, channel)."""
        self.set_note(sequence_idx, channel, row, instrument_idx, period, effect, vol_cmd, vol_val)

    def add_channel(self, count: int = 1) -> None:
        """Append one or more channels to every XM pattern.

        New channels are filled with empty ``XMNote`` rows in every existing pattern.

        :param count: Number of channels to add.
        """
        if count <= 0:
            raise ValueError(f"Invalid channel count {count} (expected >=1).")
        if self.n_channels + count > 32:
            raise ValueError(f"Too many channels: {self.n_channels + count} (XM supports 1-32).")
        for pat in self.patterns:
            for _ in range(count):
                pat.data.append([XMNote() for _ in range(pat.n_rows)])
            pat.n_channels += count
        self._n_channels += count

    def remove_channel(self, channel: int) -> None:
        """Remove one channel from every XM pattern.

        :param channel: 0-based channel index to remove.
        """
        if self.n_channels <= 1:
            raise ValueError("Cannot remove last channel (XM requires at least 1).")
        if channel < 0 or channel >= self.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{self.n_channels-1}).")
        for pat in self.patterns:
            pat.data.pop(channel)
            pat.n_channels -= 1
        self._n_channels -= 1

    def mute_channel(self, channel: int) -> None:
        """
        Mutes a specified channel in the entire song while preserving global effects.
        This clears notes, instruments, and channel-specific effects but keeps global effects
        like speed/BPM changes (Fxx), pattern breaks (Bxx), position jumps (Dxx), and extended effects (E**).

        :param channel: The 0-based channel index to mute.
        """
        if channel < 0 or channel >= self.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{self.n_channels-1}).")

        for pat in self.patterns:
            for r in range(pat.n_rows):
                note = pat.data[channel][r]
                global_effect = self._preserved_effect(note.effect)
                new_note = XMNote()
                if global_effect:
                    new_note.effect = global_effect
                pat.data[channel][r] = new_note

    def clear_channel(self, channel: int) -> None:
        """Clear one channel across all patterns.

        Unlike ``mute_channel()``, this also removes structural effects on that channel.

        :param channel: The 0-based channel index to clear.
        """
        if channel < 0 or channel >= self.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{self.n_channels-1}).")
        for pat in self.patterns:
            for r in range(pat.n_rows):
                pat.data[channel][r] = XMNote()

    def get_note(self, sequence_idx: int, row: int, channel: int) -> XMNote:
        """
        Returns the XMNote object at the given sequence pattern, row, and channel.

        :param sequence_idx: The 0-based sequence index to read from.
        :param row: The row index to read from, 0-based.
        :param channel: The channel index to read from, 0-based.
        :return: The ``XMNote`` object stored at that location.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        pat = self.patterns[self.pattern_seq[sequence_idx]]

        if row < 0 or row >= pat.n_rows:
            raise IndexError(f"Invalid row index {row} (expected 0-{pat.n_rows-1}).")

        if channel < 0 or channel >= pat.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{self.n_channels-1}).")

        return pat.data[channel][row]

    def iter_samples(self, *, include_empty: bool = True):
        """Yield immutable snapshots for XM samples flattened across instruments."""
        sample_idx = 1
        for inst in self.instruments:
            for sample in inst.samples:
                length = len(sample.waveform)
                if not include_empty and length == 0:
                    sample_idx += 1
                    continue
                yield SampleView(
                    sample_idx=sample_idx,
                    name=sample.name,
                    length=length,
                    finetune=sample.finetune,
                    volume=sample.volume,
                    loop_start=sample.repeat_point,
                    loop_length=sample.repeat_len,
                )
                sample_idx += 1

    '''
    -------------------------------------
    EFFECTS
    -------------------------------------
    '''

    def set_bpm(self, pattern: int, channel: int, row: int, bpm: int):
        """
        Sets the bpm (tempo) at the given pattern, row and channel, overwriting whatever other effect is there.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param bpm: The bpm value to set, from 32 to 255.
        :return: None.
        """
        if bpm < 32 or bpm > 255:
            raise ValueError(f"Invalid tempo {bpm} (expected 32-255).")

        self.set_effect(pattern, channel, row, f"F{bpm:02X}")

    def set_ticks_per_row(self, pattern: int, channel: int, row: int, ticks: int):
        """
        Sets the ticks per row (speed) at the given pattern, row and channel, overwriting whatever other effect is there.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param ticks: The speed value to set, from 1 to 31.
        :return: None.
        """
        if ticks < 1 or ticks > 31:
            raise ValueError(f"Invalid ticks per row {ticks} (expected 1-31).")

        self.set_effect(pattern, channel, row, f"F{ticks:02X}")

    def set_global_volume(self, pattern: int, channel: int, row: int, volume: int):
        if volume < 0 or volume > 64:
            raise ValueError(f"Invalid global volume {volume} (expected 0-64).")
        self.set_effect(pattern, channel, row, f"G{volume:02X}")

    def set_global_volume_slide(self, pattern: int, channel: int, row: int, slide: int):
        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid global volume slide {slide} (expected -15 to 15).")
        effect_value = 0
        if slide > 0:
            effect_value = slide << 4
        elif slide < 0:
            effect_value = -slide
        self.set_effect(pattern, channel, row, f"H{effect_value:02X}")

    def set_key_off(self, pattern: int, channel: int, row: int, ticks: int):
        if ticks < 0 or ticks > 255:
            raise ValueError(f"Invalid key-off tick {ticks} (expected 0-255).")
        self.set_effect(pattern, channel, row, f"K{ticks:02X}")

    def set_envelope_position(self, pattern: int, channel: int, row: int, pos: int):
        if pos < 0 or pos > 255:
            raise ValueError(f"Invalid envelope position {pos} (expected 0-255).")
        self.set_effect(pattern, channel, row, f"L{pos:02X}")

    def set_panning_slide(self, pattern: int, channel: int, row: int, slide: int):
        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid panning slide {slide} (expected -15 to 15).")
        if slide > 0:
            effect = f"P{slide:X}0"
        elif slide < 0:
            effect = f"P0{-slide:X}"
        else:
            effect = "P00"
        self.set_effect(pattern, channel, row, effect)

    def set_retrigger_volume_slide(self, pattern: int, channel: int, row: int, volume_slide: int, interval: int):
        if volume_slide < 0 or volume_slide > 15:
            raise ValueError(f"Invalid retrigger volume slide {volume_slide} (expected 0-15).")
        if interval < 0 or interval > 15:
            raise ValueError(f"Invalid retrigger interval {interval} (expected 0-15).")
        self.set_effect(pattern, channel, row, f"R{volume_slide:X}{interval:X}")

    def set_tremor(self, pattern: int, channel: int, row: int, on_ticks: int, off_ticks: int):
        if on_ticks < 0 or on_ticks > 15 or off_ticks < 0 or off_ticks > 15:
            raise ValueError("Tremor values must be in the range 0-15.")
        self.set_effect(pattern, channel, row, f"T{on_ticks:X}{off_ticks:X}")

    def set_extra_fine_portamento(self, pattern: int, channel: int, row: int, slide: int):
        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid extra-fine portamento {slide} (expected -15 to 15).")
        if slide > 0:
            self.set_effect(pattern, channel, row, f"X1{slide:X}")
        elif slide < 0:
            self.set_effect(pattern, channel, row, f"X2{-slide:X}")
        else:
            self.set_effect(pattern, channel, row, "X10")

    '''
    -------------------------------------
    INSTRUMENTS AND SAMPLES
    -------------------------------------
    '''

    def new_instrument(self, name: str = "") -> int:
        """
        Creates a new empty instrument and returns its 1-based instrument index.
        """
        inst = Instrument()
        inst.name = name
        self.instruments.append(inst)
        self.n_instruments = len(self.instruments)
        return self.n_instruments

    def list_instruments(self) -> list[Instrument]:
        """Return the list of instruments in order."""
        return self.instruments

    def get_instrument(self, inst_idx: int) -> Instrument:
        """Return the instrument at the given 1-based index."""
        if inst_idx <= 0 or inst_idx > len(self.instruments):
            raise IndexError(f"Invalid instrument index {inst_idx} (expected 1-{len(self.instruments)}).")
        return self.instruments[inst_idx - 1]
    def set_volume_envelope(
        self,
        inst_idx: int,
        points: list[EnvelopePoint | tuple[int, int]],
        sustain: int | None = None,
        loop: tuple[int, int] | None = None,
        enabled: bool = True,
        sustain_enabled: bool | None = None,
        loop_enabled: bool | None = None,
        raw_type: int | None = None,
    ) -> None:
        inst = self.get_instrument(inst_idx)
        inst.set_volume_envelope(
            points,
            sustain=sustain,
            loop=loop,
            enabled=enabled,
            sustain_enabled=sustain_enabled,
            loop_enabled=loop_enabled,
            raw_type=raw_type,
        )

    def set_panning_envelope(
        self,
        inst_idx: int,
        points: list[EnvelopePoint | tuple[int, int]],
        sustain: int | None = None,
        loop: tuple[int, int] | None = None,
        enabled: bool = True,
        sustain_enabled: bool | None = None,
        loop_enabled: bool | None = None,
        raw_type: int | None = None,
    ) -> None:
        inst = self.get_instrument(inst_idx)
        inst.set_panning_envelope(
            points,
            sustain=sustain,
            loop=loop,
            enabled=enabled,
            sustain_enabled=sustain_enabled,
            loop_enabled=loop_enabled,
            raw_type=raw_type,
        )

    def clear_volume_envelope(self, inst_idx: int) -> None:
        inst = self.get_instrument(inst_idx)
        inst.set_volume_envelope([], enabled=False, sustain_enabled=False, loop_enabled=False)

    def clear_panning_envelope(self, inst_idx: int) -> None:
        inst = self.get_instrument(inst_idx)
        inst.set_panning_envelope([], enabled=False, sustain_enabled=False, loop_enabled=False)

    def set_sample_for_note(self, inst_idx: int, note: str | int, sample_idx: int) -> None:
        inst = self.get_instrument(inst_idx)
        inst.set_sample_for_note(note, sample_idx)

    def duplicate_instrument(self, inst_idx: int) -> int:
        return self.copy_instrument_from(self, inst_idx)

    def duplicate_sample(self, inst_idx: int, sample_idx: int) -> int:
        return self.copy_sample_from(self, inst_idx, sample_idx, inst_idx)

    def set_sample_name(self, inst_idx: int, sample_idx: int, name: str) -> None:
        smp = self.get_sample(inst_idx, sample_idx)
        smp.name = name

    def set_sample_volume(self, inst_idx: int, sample_idx: int, volume: int) -> None:
        if volume < 0 or volume > 64:
            raise ValueError(f"Invalid volume {volume} (expected 0-64).")
        smp = self.get_sample(inst_idx, sample_idx)
        smp.volume = volume

    def set_sample_finetune(self, inst_idx: int, sample_idx: int, finetune: int) -> None:
        """
        Sets the XM sample finetune value as a signed byte in the range -128 to 127.
        """
        if finetune < -128 or finetune > 127:
            raise ValueError(f"Invalid finetune {finetune} (expected -128 to 127).")
        smp = self.get_sample(inst_idx, sample_idx)
        smp.finetune = finetune

    def set_sample_panning(self, inst_idx: int, sample_idx: int, panning: int) -> None:
        if panning < 0 or panning > 255:
            raise ValueError(f"Invalid panning {panning} (expected 0-255).")
        smp = self.get_sample(inst_idx, sample_idx)
        smp.panning = panning

    def set_sample_relative_note(self, inst_idx: int, sample_idx: int, rel: int) -> None:
        if rel < -96 or rel > 95:
            raise ValueError(f"Invalid relative note {rel} (expected -96 to 95).")
        smp = self.get_sample(inst_idx, sample_idx)
        smp.relative_note = rel

    def set_instrument_name(self, inst_idx: int, name: str) -> None:
        inst = self.get_instrument(inst_idx)
        inst.name = name

    def set_instrument_fadeout(self, inst_idx: int, fadeout: int) -> None:
        if fadeout < 0 or fadeout > 65535:
            raise ValueError(f"Invalid fadeout {fadeout} (expected 0-65535).")
        inst = self.get_instrument(inst_idx)
        inst.volume_fadeout = fadeout

    def set_instrument_vibrato(self, inst_idx: int, vib_type: int, sweep: int, depth: int, rate: int) -> None:
        if vib_type < 0 or vib_type > 3:
            raise ValueError(f"Invalid vibrato type {vib_type} (expected 0-3).")
        if sweep < 0 or sweep > 255 or depth < 0 or depth > 255 or rate < 0 or rate > 255:
            raise ValueError("Invalid vibrato parameters (expected 0-255).")
        inst = self.get_instrument(inst_idx)
        inst.vibrato_type = vib_type
        inst.vibrato_sweep = sweep
        inst.vibrato_depth = depth
        inst.vibrato_rate = rate


    def set_sample_loop(
        self,
        inst_idx: int,
        sample_idx: int,
        start: int,
        length: int,
        loop_type: int,
    ) -> None:
        if loop_type not in (0, 1, 2):
            raise ValueError(f"Invalid loop_type {loop_type}. Expected 0, 1, or 2.")
        smp = self.get_sample(inst_idx, sample_idx)
        smp.repeat_point = max(0, start)
        smp.repeat_len = max(0, length)
        smp.loop_type = loop_type

    def validate_sample_loop(self, inst_idx: int, sample_idx: int) -> None:
        smp = self.get_sample(inst_idx, sample_idx)
        n = len(smp.waveform)
        if smp.loop_type == Sample.LOOP_NONE:
            return
        if smp.repeat_len <= 0:
            raise ValueError("Loop length must be >0 when loop_type is enabled.")
        if smp.repeat_point < 0:
            raise ValueError("Loop start cannot be negative.")
        if smp.repeat_point + smp.repeat_len > n:
            raise ValueError(f"Loop end {smp.repeat_point + smp.repeat_len} exceeds sample length {n}.")

    def validate_instrument(self, inst_idx: int) -> None:
        inst = self.get_instrument(inst_idx)
        inst.validate_envelopes()
        for sidx in range(1, len(inst.samples) + 1):
            self.validate_sample_loop(inst_idx, sidx)


    def remove_instrument(self, inst_idx: int) -> None:
        """
        Clears the instrument at the given 1-based index without changing indices.
        """
        if inst_idx <= 0 or inst_idx > len(self.instruments):
            raise IndexError(f"Invalid instrument index {inst_idx} (expected 1-{len(self.instruments)}).")
        self.instruments[inst_idx - 1] = Instrument()

    def add_sample(self, inst_idx: int, sample: XMSample) -> int:
        """
        Adds a sample to an instrument and returns its 1-based sample index within that instrument.
        """
        inst = self.get_instrument(inst_idx)
        inst.samples.append(sample)
        if not inst.sample_map:
            inst.sample_map = [0] * 96
        return len(inst.samples)

    def get_sample(self, inst_idx: int, sample_idx: int) -> XMSample:
        """Return the sample at the given 1-based index within the instrument."""
        inst = self.get_instrument(inst_idx)
        if sample_idx <= 0 or sample_idx > len(inst.samples):
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-{len(inst.samples)}).")
        return inst.samples[sample_idx - 1]

    def remove_sample(self, inst_idx: int, sample_idx: int) -> None:
        """Remove a sample and update the sample map accordingly."""
        inst = self.get_instrument(inst_idx)
        if sample_idx <= 0 or sample_idx > len(inst.samples):
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-{len(inst.samples)}).")
        removed_idx = sample_idx - 1
        inst.samples.pop(removed_idx)
        if inst.sample_map:
            new_map: list[int] = []
            for v in inst.sample_map:
                if v == removed_idx:
                    new_map.append(0)
                elif v > removed_idx:
                    new_map.append(v - 1)
                else:
                    new_map.append(v)
            inst.sample_map = new_map

    def set_instrument_sample(self, inst_idx: int, sample: XMSample) -> None:
        """
        Replaces the instrument's samples with a single sample.
        """
        inst = self.get_instrument(inst_idx)
        inst.samples = [sample]
        inst.sample_map = [0] * 96

    def get_instrument_sample(self, inst_idx: int) -> XMSample | None:
        """
        Returns the single sample if the instrument has exactly one sample, otherwise None.
        """
        inst = self.get_instrument(inst_idx)
        if len(inst.samples) != 1:
            return None
        return inst.samples[0]

    def load_sample(self, inst_idx: int, fname: str) -> int:
        """
        Loads a WAV sample and stores it in the given instrument.
        Returns the 1-based sample index.
        """
        inst = self.get_instrument(inst_idx)
        audio = pydub.AudioSegment.from_wav(fname).set_channels(1)
        sample = XMSample()
        sample.waveform = audio.get_array_of_samples()
        sample.is_16bit = audio.sample_width == 2
        inst.samples.append(sample)
        if not inst.sample_map:
            inst.sample_map = [0] * 96
        return len(inst.samples)

    def load_sample_from_raw(
        self,
        inst_idx: int,
        raw_bytes: bytes | bytearray | list[float] | tuple[float, ...],
        sample_width: int,
    ) -> int:
        """
        Loads a raw PCM sample and stores it in the given instrument.

        :param inst_idx: The instrument index (1-based).
        :param raw_bytes: Raw mono PCM bytes, or normalized float samples in the range [-1.0, 1.0].
        :param sample_width: Sample width in bytes (1 for 8-bit, 2 for 16-bit).
        :return: The 1-based sample index.
        """
        if sample_width not in (1, 2):
            raise ValueError(f"Invalid sample_width {sample_width}. XM supports 1 (8-bit) or 2 (16-bit).")
        inst = self.get_instrument(inst_idx)
        sample = XMSample()

        if isinstance(raw_bytes, (list, tuple)):
            if sample_width == 1:
                sample.waveform = array.array('b')
                for s in raw_bytes:
                    if s > 1.0:
                        s = 1.0
                    elif s < -1.0:
                        s = -1.0
                    sample.waveform.append(int(s * 127))
                sample.is_16bit = False
            else:
                sample.waveform = array.array('h')
                for s in raw_bytes:
                    if s > 1.0:
                        s = 1.0
                    elif s < -1.0:
                        s = -1.0
                    sample.waveform.append(int(s * 32767))
                sample.is_16bit = True
        else:
            if sample_width == 1:
                sample.waveform = array.array('b')
                sample.waveform.frombytes(raw_bytes)
                sample.is_16bit = False
            else:
                sample.waveform = array.array('h')
                sample.waveform.frombytes(raw_bytes)
                sample.is_16bit = True

        inst.samples.append(sample)
        if not inst.sample_map:
            inst.sample_map = [0] * 96
        return len(inst.samples)

    def set_sample_map_range(
        self,
        inst_idx: int,
        sample_idx: int,
        note_low: str | int,
        note_high: str | int,
    ) -> None:
        """
        Maps a public 1-based sample index to an inclusive note range.
        Notes can be given as strings (e.g. 'C-4') or as 0-based note indices (0-95).
        """
        low_idx = self._note_str_to_idx(note_low)
        high_idx = self._note_str_to_idx(note_high)
        if low_idx < 0 or high_idx > 95 or low_idx > high_idx:
            raise ValueError(f"Invalid note range {note_low}-{note_high} (expected 0-95 and low<=high).")
        inst = self.get_instrument(inst_idx)
        if sample_idx < 1 or sample_idx > len(inst.samples):
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-{len(inst.samples)}).")
        if not inst.sample_map or len(inst.sample_map) != 96:
            inst.sample_map = [0] * 96
        for note in range(low_idx, high_idx + 1):
            inst.sample_map[note] = sample_idx - 1

    def set_sample_map_all(self, inst_idx: int, sample_idx: int) -> None:
        """
        Maps a public 1-based sample index to all note indices 0-95.
        """
        self.set_sample_map_range(inst_idx, sample_idx, 0, 95)

    def copy_sample_from(
        self,
        src: 'XMSong',
        src_inst_idx: int,
        src_sample_idx: int,
        dst_inst_idx: int,
    ) -> int:
        """
        Copies a single sample from another song into a destination instrument.
        Returns the new 1-based sample index in the destination instrument.
        """
        src_inst = src.get_instrument(src_inst_idx)
        if src_sample_idx < 1 or src_sample_idx > len(src_inst.samples):
            raise IndexError(f"Invalid sample index {src_sample_idx} (expected 1-{len(src_inst.samples)}).")
        smp = src_inst.samples[src_sample_idx - 1]
        new_smp = XMSample()
        new_smp.name = smp.name
        new_smp.volume = smp.volume
        new_smp.finetune = smp.finetune
        new_smp.panning = smp.panning
        new_smp.relative_note = smp.relative_note
        new_smp._reserved = smp._reserved
        new_smp.loop_type = smp.loop_type
        new_smp.is_16bit = smp.is_16bit
        new_smp.repeat_point = smp.repeat_point
        new_smp.repeat_len = smp.repeat_len
        new_smp.waveform = smp.waveform.__class__(smp.waveform.typecode, smp.waveform)
        return self.add_sample(dst_inst_idx, new_smp)

    def copy_samples_from(
        self,
        src: 'XMSong',
        src_inst_idx: int,
        src_sample_indices: list[int],
        dst_inst_idx: int,
    ) -> list[int]:
        """
        Copies multiple samples from another song into a destination instrument.
        Returns the list of new 1-based sample indices.
        """
        new_indices: list[int] = []
        for s_idx in src_sample_indices:
            new_indices.append(self.copy_sample_from(src, src_inst_idx, s_idx, dst_inst_idx))
        return new_indices

    def copy_instrument_from(self, src: 'XMSong', inst_idx: int) -> int:
        """
        Copies a single instrument from another XMSong and returns the new index.
        """
        inst = src.get_instrument(inst_idx)
        new_idx = self.new_instrument(inst.name)
        new_inst = self.get_instrument(new_idx)

        # Preserve internal header metadata
        new_inst._type = inst._type
        new_inst._header_size = inst._header_size

        # Envelopes (preserve raw flags for fidelity)
        new_inst.set_volume_envelope(
            [(p.frame, p.value) for p in inst.volume_envelope],
            sustain=inst.volume_sustain_point,
            loop=(inst.volume_loop_start, inst.volume_loop_end),
            enabled=(inst.volume_type & 0x01) != 0,
            sustain_enabled=(inst.volume_type & 0x02) != 0,
            loop_enabled=(inst.volume_type & 0x04) != 0,
            raw_type=inst.volume_type,
        )
        new_inst.set_panning_envelope(
            [(p.frame, p.value) for p in inst.panning_envelope],
            sustain=inst.panning_sustain_point,
            loop=(inst.panning_loop_start, inst.panning_loop_end),
            enabled=(inst.panning_type & 0x01) != 0,
            sustain_enabled=(inst.panning_type & 0x02) != 0,
            loop_enabled=(inst.panning_type & 0x04) != 0,
            raw_type=inst.panning_type,
        )

        new_inst.vibrato_type = inst.vibrato_type
        new_inst.vibrato_sweep = inst.vibrato_sweep
        new_inst.vibrato_depth = inst.vibrato_depth
        new_inst.vibrato_rate = inst.vibrato_rate
        new_inst.volume_fadeout = inst.volume_fadeout

        # Samples
        for smp in inst.samples:
            new_smp = XMSample()
            new_smp.name = smp.name
            new_smp.volume = smp.volume
            new_smp.finetune = smp.finetune
            new_smp.panning = smp.panning
            new_smp.relative_note = smp.relative_note
            new_smp._reserved = smp._reserved
            new_smp.loop_type = smp.loop_type
            new_smp.is_16bit = smp.is_16bit
            new_smp.repeat_point = smp.repeat_point
            new_smp.repeat_len = smp.repeat_len
            new_smp.waveform = smp.waveform.__class__(smp.waveform.typecode, smp.waveform)
            self.add_sample(new_idx, new_smp)

        # Public 1-based sample map converted from the internal 0-based sample_map.
        if inst.samples:
            map_1based = [v + 1 for v in inst.sample_map[:96]]
            new_inst.set_sample_map(map_1based)

        return new_idx

    def copy_instruments_from(self, src: 'XMSong', inst_indices: list[int]) -> dict[int, int]:
        """
        Copies multiple instruments from another XMSong.
        Returns a mapping from source index to new destination index.
        """
        mapping: dict[int, int] = {}
        for idx in inst_indices:
            mapping[idx] = self.copy_instrument_from(src, idx)
        return mapping

    def save_sample(
        self,
        inst_idx: int,
        sample_idx: int,
        fname: str,
        sample_rate: int | None = None,
        force_sample_rate: int | None = None,
    ):
        """
        Saves the sample at the given index as a WAV file.
        """
        smp = self.get_sample(inst_idx, sample_idx)
        if sample_rate is None:
            sample_rate = 8363

        if len(smp.waveform) == 0:
            raise ValueError(f"Sample {sample_idx} has no waveform data")

        sample_width = 2 if smp.is_16bit else 1
        audio = pydub.AudioSegment(
            data=smp.waveform.tobytes(),
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=1,
        )
        if force_sample_rate is not None and force_sample_rate != sample_rate:
            audio = audio.set_frame_rate(force_sample_rate)
        audio.export(fname, format="wav")

    '''
    -------------------------------------
    PATTERNS
    -------------------------------------
    '''

    def add_to_sequence(self, pattern_idx: int, sequence_position: int | None = None) -> None:
        if len(self.pattern_seq) + 1 > 256:
            raise ValueError(f"Pattern sequence too long ({len(self.pattern_seq) + 1}). XM supports up to 256.")
        super().add_to_sequence(pattern_idx, sequence_position)


    def set_sequence(self, seq: list[int]) -> None:
        if len(seq) > 256:
            raise ValueError(f"Pattern sequence too long ({len(seq)}). XM supports up to 256.")
        super().set_sequence(seq)


    def resize_pattern(self, sequence_idx: int, n_rows: int) -> None:
        """
        Resizes a sequence pattern to the given number of rows (1-256).
        Truncates or extends with empty notes as needed.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")
        if n_rows < 1 or n_rows > 256:
            raise ValueError(f"Invalid row count {n_rows} (expected 1-256).")
        p = self.pattern_seq[sequence_idx]
        pat = self.patterns[p]
        if n_rows == pat.n_rows:
            return
        if n_rows < pat.n_rows:
            for c in range(pat.n_channels):
                pat.data[c] = pat.data[c][:n_rows]
        else:
            for c in range(pat.n_channels):
                pat.data[c].extend([XMNote() for _ in range(n_rows - pat.n_rows)])
        pat.n_rows = n_rows


    def clear_pattern(self, sequence_idx: int):
        """
        Clears completely a specified sequence pattern.
        The pattern is not removed from the song sequence, but all the notes are set to empty.

        :param sequence_idx: The 0-based sequence index to clear.
        :return: None.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        p = self.pattern_seq[sequence_idx]
        pat = self.patterns[p]
        for r in range(pat.n_rows):
            for c in range(pat.n_channels):
                pat.data[c][r] = XMNote()

    def add_pattern(self, n_rows: int = 64) -> int:
        """
        Creates a brand new pattern, appends it to the pattern pool, and adds that pool index to the song sequence.

        :param n_rows: Number of rows in the new pattern (default 64, XM supports 1-256).
        :return: The new pattern pool index.
        """
        if n_rows < 1 or n_rows > 256:
            raise ValueError(f"Invalid row count {n_rows}. XM supports 1-256 rows.")
        
        # Create pattern with XMNote objects
        pat = Pattern(n_rows, self.n_channels)
        for c in range(self.n_channels):
            for r in range(n_rows):
                pat.data[c][r] = XMNote()
        
        self.patterns.append(pat)
        n = len(self.patterns) - 1
        self.pattern_seq.append(n)

        return n

    def get_used_instruments(
        self,
        *,
        scope: str = "sequence",
        order: str = "sorted",
    ) -> list[int]:
        """Return instrument indices referenced by notes under sequence or reachable scope."""
        self._validate_used_resource_args(scope, order)
        seen: set[int] = set()
        first_use: list[int] = []
        for note in self._iter_notes_by_scope(scope):
            inst_idx = getattr(note, 'instrument_idx', 0)
            if inst_idx > 0 and inst_idx not in seen:
                seen.add(inst_idx)
                first_use.append(inst_idx)
        return self._finalize_used_values(first_use, order)

    def get_used_samples(
        self,
        *,
        scope: str = "sequence",
        order: str = "sorted",
    ) -> list[int]:
        """Return flattened XM sample indices referenced by notes under sequence/reachable scope."""
        self._validate_used_resource_args(scope, order)

        # Flatten sample numbering to match iter_samples() stable ordering.
        flat_by_instrument: dict[int, list[int]] = {}
        flat_idx = 1
        for inst_idx, inst in enumerate(self.instruments, start=1):
            inst_flat = list(range(flat_idx, flat_idx + len(inst.samples)))
            flat_by_instrument[inst_idx] = inst_flat
            flat_idx += len(inst.samples)

        seen: set[int] = set()
        first_use: list[int] = []

        def _record(flat_sample_idx: int) -> None:
            if flat_sample_idx not in seen:
                seen.add(flat_sample_idx)
                first_use.append(flat_sample_idx)

        for note in self._iter_notes_by_scope(scope):
            inst_idx = getattr(note, 'instrument_idx', 0)
            if inst_idx <= 0 or inst_idx > len(self.instruments):
                continue
            inst = self.instruments[inst_idx - 1]
            inst_flat = flat_by_instrument.get(inst_idx, [])
            if not inst_flat:
                continue
            if len(inst_flat) == 1:
                _record(inst_flat[0])
                continue

            mapped_flat = None
            period = getattr(note, 'period', '')
            if period not in {'', 'off'} and len(inst.sample_map) == 96:
                try:
                    note_idx = Song.note_to_index(period)
                except (TypeError, ValueError):
                    note_idx = None
                if note_idx is not None:
                    mapped_local = inst.sample_map[note_idx]
                    if 0 <= mapped_local < len(inst_flat):
                        mapped_flat = inst_flat[mapped_local]

            if mapped_flat is not None:
                _record(mapped_flat)
                continue

            # Ambiguous sample selection (no note or no mapping): include all samples of the instrument.
            for candidate in inst_flat:
                _record(candidate)

        return self._finalize_used_values(first_use, order)

    def get_effective_row_count(self, sequence_idx: int, include_loops: bool = True) -> int:
        """
        Returns the effective number of rows that get played in a sequence pattern.
        Accounts for position jumps, loops, and breaks.

        TODO: do a separate version for the entire song

        :param sequence_idx: The 0-based sequence index to inspect.
        :param include_loops: True to also count the rows that get played in loops.
        :return: The effective number of rows that gets played in the pattern.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        loop_start_row = 0  # used by E6x effect

        data = copy.deepcopy(self.patterns[self.pattern_seq[sequence_idx]].data)
        n_channels = len(data)
        n_rows = len(data[0]) if data else 0

        unrolled_data = [[] for _ in range(n_channels)]

        for r in range(n_rows):

            interrupt = False  # if true, the pattern is cut short by Bxx or Dxx effects

            for c in range(n_channels):

                unrolled_data[c].append(data[c][r])

                efx = data[c][r].effect
                if efx != "":

                    if efx[0] == "B" or efx[0] == "D":
                        interrupt = True

                    if include_loops and efx[:2] == "E6":

                        if int(efx[2], 16) == 0:  # E60 means loop start
                            loop_start_row = r

                        loop_end_row = r
                        loop_count = int(efx[2], 16)

                        for _ in range(loop_count):
                            unrolled_data[c] += unrolled_data[c][loop_start_row:loop_end_row + 1]

            if interrupt:
                break

        return max([len(unrolled_data[c]) for c in range(n_channels)]) if n_channels > 0 else 0

    def get_pattern_duration(self, pattern: int) -> float:
        """
        Returns the duration of a pattern in seconds.

        :param pattern: The pattern index (within the song sequence).
        :return: The pattern duration in seconds.
        """
        raise NotImplementedError("XMSong.get_pattern_duration() is not implemented.")
