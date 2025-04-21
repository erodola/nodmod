from nodmod.song import Song
from nodmod.song import Sample
from nodmod.song import Pattern
from nodmod.song import Note

class XMSong(Song):
    
    def __init__(self):
        pass

    '''
    -------------------------------------
    IMPORT AND EXPORT
    -------------------------------------
    '''

    def load_from_file(self, fname: str, verbose: bool = True):
        """
        Loads a song from a standard XM file.
        TODO

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
            # TODO: use this
            n_instruments = int.from_bytes(data[72:74], byteorder='little', signed=False)
            if n_instruments > 128:
                raise NotImplementedError(f"Too many instruments: {n_instruments}.")
            
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

            def get_period(note_byte):

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

            def get_instrument(instrument_byte):

                if instrument_byte > 127:
                    raise NotImplementedError("Invalid instrument value.")
                            
                return f"{instrument_byte:02d}"

            def get_volume(volume_byte):

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

            def get_efx_type(effect_byte):  #FIXME add effect X

                if effect_byte <= 0x0D or effect_byte == 0x0F:
                    efx = f"{effect_byte:01X}"

                elif effect_byte == 0x0E:
                    efx = "E"  # TODO: see how it's treated in .mod files

                elif effect_byte == 0x10:
                    efx = "G"

                elif effect_byte == 0x11:
                    efx = "H"

                elif effect_byte == 0x15:
                    efx = "L"

                elif effect_byte == 0x19:
                    efx = "P"

                elif effect_byte == 0x1B:
                    efx = "R"

                elif effect_byte == 0x1D:
                    efx = "T"

                else:
                    raise NotImplementedError(f"Invalid effect type {effect_byte:02X}.")

                return efx

            def get_efx_param(effect_param_byte):
                #TODO how are efx E and X treated?
                #TODO see * cases in xm.pdf
                return effect_param_byte

            self.patterns = []

            pattern_data = data[336:]
            cur_pat_idx = 0

            for p in range(n_unique_patterns):

                print("")
                print(f"-------------\nPattern {p}\n-------------")
                print("")

                if pattern_data[cur_pat_idx:cur_pat_idx + 4][0] != 9:
                    raise NotImplementedError(f"Unsupported pattern header length.")

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

                    instrument = ''
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
                        effect = f"{effect}{get_efx_param(pattern_data[byte_idx]):02X}"

                    # pretty print
                    if period == '':
                        print('--- ', end='')
                    elif period == 'off':
                        print('==  ', end='')
                    else:
                        print(f"{period} ", end='')
                    print(f"{'--' if instrument=='' else instrument} {'-' if volume_cmd=='' else volume_cmd}", end='')
                    if volume_val == -1:
                        print('-- ', end='')
                    else:
                        print(f"{volume_val:02d} ", end='')
                    print(f"{'---' if effect=='' else effect} | ", end='')

                    # FIXME: redefine note class for MOD and XM
                    note = Note()

                    note.instrument_idx = instrument
                    note.period = period

                    # TODO
                    # note.effect = effect
                    # note.volume = volume_val
                    # note.volume_cmd = volume_cmd

        #                 e_type, e_param = MODSong.get_effect_from_note(note_raw)

        #                 if e_type != 0 or e_param != 0:

        #                     # dirty way for converting hex number to string... e.g. 0xF1 -> "F1"
        #                     note.effect = hex(e_type).lstrip("0x").upper() 
        #                     # note.effect += hex(e_param)[2:].upper()  # dunno why i was doing this...
        #                     note.effect += f"{e_param:02X}"
                            
        #                     if e_type == 0:  # arpeggio effect
        #                         note.effect = "0" + note.effect
                    
                    pat.data[c][r] = note

                    # move to the next note (next channel or row)
                    byte_idx += 1

                    c += 1  # move to the next channel
                    if c == n_channels:
                        c = 0
                        r += 1
                        print("")

                    if byte_idx == cur_pat_idx + 9 + pattern_data_size:
                        break  # next pattern

                self.patterns.append(pat)

                # next pattern
                cur_pat_idx += 9 + pattern_data_size


        #     # ----------------------------
        #     # Load sample data
        #     # ----------------------------

        #     sample_lengths = [0] * MODSong.SAMPLES
        #     self.n_actual_samples = 0

        #     self.samples = [Sample() for _ in range(MODSong.SAMPLES)]

        #     # all the waveforms are stored right after the pattern data
        #     waveform_idx = 1084 + n_unique_patterns * MODSong.ROWS * MODSong.CHANNELS * 4

        #     for i in range(MODSong.SAMPLES):

        #         idx = 20 + i * 30 + 22
        #         sample_lengths[i] = 2 * int.from_bytes(data[idx:idx + 2], byteorder='big', signed=False)

        #         # some docs say this is equivalent to sample length = 0. not dealing with this now.
        #         assert sample_lengths[i] != 1

        #         if sample_lengths[i] > 0:
        #             self.n_actual_samples += 1

        #         smp = Sample()
        #         smp.name = data[idx - 22:idx].rstrip(b'\x00').decode('utf-8')

        #         # Lower four bits are the finetune value, stored as a signed 4-bit number.
        #         # The upper four bits are not used.
        #         smp.finetune = data[idx + 2]
        #         smp.finetune &= 0x0F

        #         # Volume range is 0x00-0x40 (or 0-64 decimal)
        #         smp.volume = data[idx + 3]

        #         smp.repeat_point = 2 * int.from_bytes(data[idx + 4:idx + 6], byteorder='big', signed=False)
        #         smp.repeat_len = 2 * int.from_bytes(data[idx + 6:idx + 8], byteorder='big', signed=False)

        #         # The digitized samples are raw 8-bit signed data.

        #         if sample_lengths[i] == 0:
        #             smp.waveform = array.array('b')
        #         else:
        #             smp.waveform = data[waveform_idx:waveform_idx + sample_lengths[i]]
        #             smp.waveform = array.array(
        #                 'b',
        #                 smp.waveform)  # the bytearray string is reinterpreted as signed 8bit values
        #             waveform_idx += sample_lengths[i]

        #         self.samples[i] = smp

        #     # ----------------------------
        #     # Do some sanity checks
        #     # ----------------------------

        #     # Some non-standard modules have extra patterns saved in the song data.
        #     # The MOD standard does not allow storing patterns beyond the maximum pattern number that is actually used.
        #     patterns_size = n_unique_patterns * MODSong.ROWS * MODSong.CHANNELS * 4
        #     samples_size = sum(sample_lengths)
        #     predicted_size = patterns_size + samples_size
        #     actual_size = len(data[1084:])
        #     if predicted_size != actual_size:
        #         n_extra_patterns = int((actual_size - predicted_size) / (MODSong.ROWS * MODSong.CHANNELS * 4))
        #         raise NotImplementedError(f"The module has {n_extra_patterns} unexpected extra patterns.")

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