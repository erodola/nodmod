import array
import shutil

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
    
    def __init__(self):
        super().__init__()
        
        # XM-specific: instruments list (notes reference instruments, not samples directly)
        # Each Instrument can contain multiple Sample objects
        self.instruments: list[Instrument] = []
        self.n_instruments = 0  # Number from header (includes empty instruments)
        
        # Source file path (used for save_to_file until XM writing is implemented)
        self._source_file: str | None = None

    '''
    -------------------------------------
    IMPORT AND EXPORT
    -------------------------------------
    '''

    def save_to_file(self, fname: str, verbose: bool = True):
        """
        Saves the song as a standard XM file.
        
        Note: Full XM writing is not yet implemented. This method copies the
        original source file if available.
        
        :param fname: Complete file path.
        :param verbose: False for silent saving.
        :return: None.
        """
        if self._source_file is None:
            raise NotImplementedError(
                "XM file writing is not yet implemented. "
                "Can only save XM files that were loaded from disk."
            )
        
        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)
        
        shutil.copy2(self._source_file, fname)
        
        if verbose:
            print('done.')

    def load_from_file(self, fname: str, verbose: bool = True):
        """
        Loads a song from a standard XM file.

        :param fname: The path to the module file.
        :param verbose: False for silent loading.
        :return: None.
        """
        
        # Store source file path for save_to_file
        self._source_file = fname

        if verbose:
            print(f'Loading {fname}... ', end='', flush=True)

        with (open(fname, 'rb') as xm_file):

            data = bytearray(xm_file.read())

            # ----------------------------
            # Load fixed-size header data
            # ----------------------------

            header = data[:60]
            magic_string = header[:17].decode('utf-8')

            if magic_string != "Extended Module: ":  # non-standard xm file
                raise NotImplementedError(f"Unsupported module format {magic_string}.")
            
            self.songname = header[17:37].decode('utf-8').rstrip(' ')

            if header[37] != 0x1A:
                raise NotImplementedError("Invalid XM file format.")
            
            self.tracker_name = header[38:58].decode('utf-8').rstrip(' ')
            
            version = int.from_bytes(header[58:60], byteorder='big', signed=False)
            if version < 0x0104:
                raise NotImplementedError(f"Unsupported XM version {version}.")
            
            # ----------------------------
            # Load variable-size header data
            # ----------------------------

            # song restart position
            # TODO: use this
            song_restart = int.from_bytes(data[66:68], byteorder='little', signed=False)

            # number of channels
            n_channels = int.from_bytes(data[68:70], byteorder='little', signed=False)
            if n_channels > 32:
                raise NotImplementedError(f"Too many channels: {n_channels}.")

            # number of instruments (note : some instruments may be empty)
            self.n_instruments = int.from_bytes(data[72:74], byteorder='little', signed=False)
            if self.n_instruments > 128:
                raise NotImplementedError(f"Too many instruments: {self.n_instruments}.")
            self.instruments = [Instrument() for _ in range(self.n_instruments)]

            # 0 = Amiga frequency table; 1 = Linear frequency table
            # TODO: use this
            flags = int.from_bytes(data[74:76], byteorder='little', signed=False)
            
            # speed / ticks per row
            # TODO: use this
            speed = int.from_bytes(data[76:78], byteorder='little', signed=False)

            # tempo
            # TODO: use this
            tempo = int.from_bytes(data[78:80], byteorder='little', signed=False)

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

            def get_period(note_byte) -> str:

                if note_byte & 0x80:
                    raise NotImplementedError("The note has the MSB set.")
            
                note_val = note_byte & 0x7F

                if note_val > 97:
                    raise NotImplementedError("Invalid note value.")

                if note_val == 0:
                    period_ = ''  # no note
                elif note_val == 97:
                    period_ = 'off'  # note off, "==" in OpenMPT
                else:
                    period = self.PERIOD_SEQ[(note_val % len(self.PERIOD_SEQ)) - 1]
                    octave = int(note_val / len(self.PERIOD_SEQ)) + 1
                    period_ = f"{period}{octave}"

                return period_

            def get_instrument(instrument_byte) -> int:

                if instrument_byte > 127:
                    raise NotImplementedError("Invalid instrument value.")
                            
                return instrument_byte

            def get_volume(volume_byte) -> tuple[str, int]:

                cmd_nibble = volume_byte & 0xF0
                val_nibble = volume_byte & 0x0F

                if cmd_nibble >= 0x00 and cmd_nibble <= 0x0F:
                    raise NotImplementedError("Volume command does nothing")
                if cmd_nibble >= 0x51 and cmd_nibble <= 0x5F:
                    raise NotImplementedError("Volume command undefined")

                if cmd_nibble >= 0x10 and cmd_nibble <= 0x1F:  # set volume
                    volume_cmd = 'v'
                    volume_val = val_nibble
                elif cmd_nibble >= 0x20 and cmd_nibble <= 0x2F:
                    volume_cmd = 'v'
                    volume_val = val_nibble + 16
                elif cmd_nibble >= 0x30 and cmd_nibble <= 0x3F:
                    volume_cmd = 'v'
                    volume_val = val_nibble + 32
                elif cmd_nibble >= 0x40 and cmd_nibble <= 0x4F:
                    volume_cmd = 'v'
                    volume_val = val_nibble + 48
                elif cmd_nibble == 0x50:
                    volume_cmd = 'v'
                    volume_val = 64

                elif cmd_nibble >= 0x60 and cmd_nibble <= 0x6F:  # volume slide down
                    volume_cmd = 'd'
                    volume_val = val_nibble

                elif cmd_nibble >= 0x70 and cmd_nibble <= 0x7F:  # volume slide up
                    volume_cmd = 'c'
                    volume_val = val_nibble

                elif cmd_nibble >= 0x80 and cmd_nibble <= 0x8F:  # fine volume slide down
                    volume_cmd = 'b'
                    volume_val = val_nibble

                elif cmd_nibble >= 0x90 and cmd_nibble <= 0x9F:  # fine volume slide up
                    volume_cmd = 'a'
                    volume_val = val_nibble

                elif cmd_nibble >= 0xA0 and cmd_nibble <= 0xAF:  # vibrato speed
                    volume_cmd = 'u'
                    volume_val = val_nibble

                elif cmd_nibble >= 0xB0 and cmd_nibble <= 0xBF:  # vibrato depth
                    volume_cmd = 'h'
                    volume_val = val_nibble

                elif cmd_nibble >= 0xC0 and cmd_nibble <= 0xCF:  # set panning position
                    volume_cmd = 'p'
                    volume_val = val_nibble * 4

                elif cmd_nibble >= 0xD0 and cmd_nibble <= 0xDF: # panning slide left
                    volume_cmd = 'l'
                    volume_val = val_nibble

                elif cmd_nibble >= 0xE0 and cmd_nibble <= 0xEF: # panning slide right
                    volume_cmd = 'r'
                    volume_val = val_nibble

                elif cmd_nibble >= 0xF0 and cmd_nibble <= 0xFF: # tone portamento
                    volume_cmd = 'g'
                    volume_val = val_nibble

                return volume_cmd, volume_val

            def get_efx_type(effect_byte) -> str:

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

                elif effect_byte == 0x1D:
                    efx = "T"  # Tremor

                elif effect_byte == 0x21:
                    efx = "X"  # Extra fine portamento (X1x=up, X2x=down)

                else:
                    raise NotImplementedError(f"Invalid effect type {effect_byte:02X}.")

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

            pattern_data = data[336:]
            cur_pat_idx = 0

            for p in range(n_unique_patterns):

                if pattern_data[cur_pat_idx:cur_pat_idx + 4][0] != 9:
                    raise NotImplementedError(f"Unsupported pattern header length: {pattern_data[cur_pat_idx:cur_pat_idx + 4][0]}.")

                if pattern_data[cur_pat_idx+4] != 0:
                    raise NotImplementedError(f"Unsupported packing type.")

                n_rows = int.from_bytes(pattern_data[cur_pat_idx+5:cur_pat_idx+7], byteorder='little', signed=False)

                # each pattern has a different size in bytes due to packing
                pattern_data_size = int.from_bytes(pattern_data[cur_pat_idx+7:cur_pat_idx+9], byteorder='little', signed=False)

                if pattern_data_size == 0:
                    raise NotImplementedError(f"Empty pattern.")

                pat = Pattern(n_rows, n_channels)

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
                            period = get_period(pattern_data[byte_idx])
                    elif not is_packed:
                        period = get_period(packed_byte)

                    if period != '' and period != 'off':
                        volume_cmd = 'v'
                        volume_val = 64  # full volume, unless overwritten in the volume column

                    if not is_packed or (is_packed and packed_byte & 0x02):
                        byte_idx += 1
                        instrument = get_instrument(pattern_data[byte_idx])

                    if not is_packed or (is_packed and packed_byte & 0x04):
                        byte_idx += 1
                        volume_cmd, volume_val = get_volume(pattern_data[byte_idx])

                    if not is_packed or (is_packed and packed_byte & 0x08):
                        byte_idx += 1
                        effect = get_efx_type(pattern_data[byte_idx])

                    if not is_packed or (is_packed and packed_byte & 0x10):
                        byte_idx += 1
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
                    if c == n_channels:
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
                
                # Instrument name (22 bytes at offset +4)
                name_bytes = instrument_data[cur_inst_idx + 4:cur_inst_idx + 26]
                self.instruments[i].name = name_bytes.decode('latin-1').rstrip('\x00')
                
                # Instrument type at offset +26 (always 0, but often random - ignore it)
                
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
                        # Sample name (22 bytes at offset 18)
                        sample_name = instrument_data[sample_hdr_offset + 18:sample_hdr_offset + 40]
                        
                        # Store sample metadata
                        sample = self.instruments[i].samples[s]
                        sample.name = sample_name.decode('latin-1').rstrip('\x00')
                        sample.volume = volume
                        sample.finetune = finetune
                        sample.panning = panning
                        sample.relative_note = relative_note
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