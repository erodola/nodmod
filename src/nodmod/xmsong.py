import array
import copy
import shutil
import warnings

from nodmod import Song
from nodmod import XMSample
from nodmod import Instrument
from nodmod import EnvelopePoint
from nodmod import Pattern
from nodmod import XMNote


class XMSong(Song):
    """
    XM (FastTracker 2 Extended Module) song format.
    
    Unlike MOD files where notes reference samples directly, XM files use instruments.
    Each instrument can contain 0, 1, or many samples. Notes in XM files reference
    instruments by index, and the instrument determines which sample(s) to play.
    """
    
    @property
    def file_extension(self) -> str:
        return 'xm'
    
    @property
    def uses_linear_frequency(self) -> bool:
        """True if using linear frequency table, False for Amiga frequency table."""
        return (self.flags & 0x01) == 1
    
    def __init__(self):
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

    def save_as_ascii(self, fname: str, verbose: bool = True):
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
            for seq_idx, pat_idx in enumerate(self.pattern_seq):
                pat = self.patterns[pat_idx]
                n_rows = len(pat.data[0]) if pat.data else 0
                n_channels = len(pat.data)
                
                # Write pattern header
                file.write(f"# Pattern {seq_idx} (unique pattern {pat_idx}): "
                           f"{n_rows} rows, {n_channels} channels\n")
                
                for r in range(n_rows):
                    for c in range(n_channels):
                        file.write(f"| {pat.data[c][r]} ")
                    file.write('|\n')
                file.write('\n')

        if verbose:
            print('done.')

    def save_to_file(self, fname: str, verbose: bool = True):
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
        data += self.n_instruments.to_bytes(2, byteorder='little')
        
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
                    vol_byte = volume_to_byte(note.vol_cmd, note.vol_val)
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
                
                # Sample map (96 bytes)
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

    def load_from_file(self, fname: str, verbose: bool = True):
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
                raise NotImplementedError(f"Not an XM module! Magic string: {magic_string}.")
            
            self.songname = header[17:37].decode('latin-1').rstrip(' \x00')

            if header[37] != 0x1A:
                raise NotImplementedError("Invalid XM file format.")
            
            self.tracker_name = header[38:58].decode('latin-1').rstrip(' \x00')
            
            version = int.from_bytes(header[58:60], byteorder='big', signed=False)
            if version < 0x0104:
                raise NotImplementedError(f"Unsupported XM version {version}.")
            
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
                raise NotImplementedError(f"Too many channels: {self.n_channels}.")

            # number of instruments (note : some instruments may be empty)
            self.n_instruments = int.from_bytes(data[72:74], byteorder='little', signed=False)
            if self.n_instruments > 128:
                raise NotImplementedError(f"Too many instruments: {self.n_instruments}.")
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
                raise NotImplementedError(f"Too many patterns: {n_unique_patterns}.")
            
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

                if instrument_byte > 128:
                    if verbose:
                        warnings.warn(f"Non-standard instrument value {instrument_byte} at pattern {pat_num}, row {row}, channel {chan}. Ignoring.")
                    return 0  # Treat as "no instrument change"
                            
                return instrument_byte

            def get_volume(volume_byte) -> tuple[str, int]:

                volume_cmd = ''
                volume_val = 0
                cmd_nibble = volume_byte & 0xF0
                val_nibble = volume_byte & 0x0F

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
                    raise NotImplementedError(f"Unsupported pattern header length: {pattern_data[cur_pat_idx:cur_pat_idx + 4][0]}.")

                if pattern_data[cur_pat_idx+4] != 0:
                    raise NotImplementedError(f"Unsupported packing type.")

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
                    
                    # Sample number for all notes (96 bytes at offset +33)
                    # Maps note numbers 0-95 to sample indices within this instrument
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

    def timestamp(self) -> list[list[float, int, int]]:
        """
        Annotates the time of each row in the song, taking into account the speed and bpm changes.

        :return: A list where each element is a list corresponding to pattern in the sequence.
                 Within each list, each row is a triple (timestamp [s], speed, bpm).
        """
        pass  # TODO

    '''
    -------------------------------------
    PATTERNS
    -------------------------------------
    '''

    def clear_pattern(self, pattern: int):
        """
        Clears completely a specified pattern.
        The pattern is not removed from the song sequence, but all the notes are set to empty.

        :param pattern: The pattern index (within the song sequence) to be cleared.
        :return: None.
        """
        if pattern < 0 or pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        p = self.pattern_seq[pattern]
        pat = self.patterns[p]
        for r in range(pat.n_rows):
            for c in range(pat.n_channels):
                pat.data[c][r] = XMNote()

    def add_pattern(self, n_rows: int = 64) -> int:
        """
        Creates a brand new pattern and adds it to the song sequence.

        :param n_rows: Number of rows in the new pattern (default 64, XM supports 1-256).
        :return: The index of the new pattern in the sequence.
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

    def get_effective_row_count(self, pattern: int) -> int:
        """
        Returns the effective number of rows that get played in a pattern.
        Accounts for position jumps, loops, and breaks.

        TODO: do a separate version for the entire song

        :param pattern: The pattern index (within the song sequence).
        :return: The effective number of rows that gets played in the pattern.
        """
        pass  # TODO

    def get_pattern_duration(self, pattern: int) -> float:
        """
        Returns the duration of a pattern in seconds.

        :param pattern: The pattern index (within the song sequence).
        :return: The pattern duration in seconds.
        """
        pass  # TODO