from nodmod.song import Song
from nodmod.song import Sample
from nodmod.song import Pattern
from nodmod.song import Note
import array
import os
import subprocess
import shutil
import pydub
import copy


class MODSong(Song):

    ROWS = 64
    CHANNELS = 4
    SAMPLES = 31
    PATTERN_SIZE = ROWS * CHANNELS * 4

    # OpenMPT period table for Tuning 0, Normal
    PERIOD_TABLE = {
        3424: 'C-2', 3232: 'C#2', 3048: 'D-2', 2880: 'D#2', 2712: 'E-2', 2560: 'F-2', 2416: 'F#2', 2280: 'G-2',
        2152: 'G#2', 2032: 'A-2', 1920: 'A#2', 1812: 'B-2',
        1712: 'C-3', 1616: 'C#3', 1524: 'D-3', 1440: 'D#3', 1356: 'E-3', 1280: 'F-3', 1208: 'F#3', 1140: 'G-3',
        1076: 'G#3', 1016: 'A-3', 960: 'A#3', 906: 'B-3',
        856: 'C-4', 808: 'C#4', 762: 'D-4', 720: 'D#4', 678: 'E-4', 640: 'F-4', 604: 'F#4', 570: 'G-4', 538: 'G#4',
        508: 'A-4', 480: 'A#4', 453: 'B-4',
        428: 'C-5', 404: 'C#5', 381: 'D-5', 360: 'D#5', 339: 'E-5', 320: 'F-5', 302: 'F#5', 285: 'G-5', 269: 'G#5',
        254: 'A-5', 240: 'A#5', 226: 'B-5',
        214: 'C-6', 202: 'C#6', 190: 'D-6', 180: 'D#6', 170: 'E-6', 160: 'F-6', 151: 'F#6', 143: 'G-6', 135: 'G#6',
        127: 'A-6', 120: 'A#6', 113: 'B-6',
        107: 'C-7', 101: 'C#7', 95: 'D-7', 90: 'D#7', 85: 'E-7', 80: 'F-7', 75: 'F#7', 71: 'G-7', 67: 'G#7', 63: 'A-7',
        60: 'A#7', 56: 'B-7',
        53: 'C-8', 50: 'C#8', 47: 'D-8', 45: 'D#8', 42: 'E-8', 40: 'F-8', 37: 'F#8', 35: 'G-8', 33: 'G#8', 31: 'A-8',
        30: 'A#8', 28: 'B-8'
    }

    INV_PERIOD_TABLE = {value: key for key, value in PERIOD_TABLE.items()}

    def __init__(self):
        """
        Initializes the song with one empty pattern and an empty sample bank.
        This way, the song can be immediately saved as a valid module file.
        """

        super().__init__()

        self.patterns = [Pattern(n_rows=MODSong.ROWS, n_channels=MODSong.CHANNELS)]
        self.pattern_seq = [0]
        self.samples = [Sample() for _ in range(MODSong.SAMPLES)]

    '''
    -------------------------------------
    IMPORT AND EXPORT
    -------------------------------------
    '''

    def load_from_file(self, fname: str, verbose: bool = True):
        """
        Loads a song from a standard MOD file.

        :param fname: The path to the module file.
        :param verbose: False for silent loading.
        :return: None.
        """

        if verbose:
            print(f'Loading {fname}... ', end='', flush=True)

        self.artist, self.songname = Song.artist_songname_from_filename(fname)

        with (open(fname, 'rb') as mod_file):

            data = bytearray(mod_file.read())

            # TODO: check if the .mod file is in packed format (never happened so far)

            magic_string = data[1080:1080 + 4].decode('utf-8')
            if magic_string != "M.K.":  # non-standard mod file
                raise NotImplementedError(f"Unsupported module format {magic_string}.")

            # ----------------------------
            # Load pattern preamble data
            # ----------------------------

            song_length = data[950]  # song length in patterns
            self.pattern_seq = [0] * song_length

            n_unique_patterns = 0
            for p in range(song_length):
                x = data[952 + p]
                self.pattern_seq[p] = x
                if x > n_unique_patterns:
                    n_unique_patterns = x
            n_unique_patterns += 1
            
            self.patterns = []

            # ----------------------------
            # Load sample data
            # ----------------------------

            sample_lengths = [0] * MODSong.SAMPLES
            self.n_actual_samples = 0

            self.samples = [Sample() for _ in range(MODSong.SAMPLES)]

            # all the waveforms are stored right after the pattern data
            waveform_idx = 1084 + n_unique_patterns * MODSong.ROWS * MODSong.CHANNELS * 4

            for i in range(MODSong.SAMPLES):

                idx = 20 + i * 30 + 22
                sample_lengths[i] = 2 * int.from_bytes(data[idx:idx + 2], byteorder='big', signed=False)

                # some docs say this is equivalent to sample length = 0. not dealing with this now.
                assert sample_lengths[i] != 1

                if sample_lengths[i] > 0:
                    self.n_actual_samples += 1

                smp = Sample()
                smp.name = data[idx - 22:idx].rstrip(b'\x00').decode('utf-8')

                # Lower four bits are the finetune value, stored as a signed 4-bit number.
                # The upper four bits are not used.
                smp.finetune = data[idx + 2]
                smp.finetune &= 0x0F

                # Volume range is 0x00-0x40 (or 0-64 decimal)
                smp.volume = data[idx + 3]

                smp.repeat_point = 2 * int.from_bytes(data[idx + 4:idx + 6], byteorder='big', signed=False)
                smp.repeat_len = 2 * int.from_bytes(data[idx + 6:idx + 8], byteorder='big', signed=False)

                # The digitized samples are raw 8-bit signed data.

                if sample_lengths[i] == 0:
                    smp.waveform = array.array('b')
                else:
                    smp.waveform = data[waveform_idx:waveform_idx + sample_lengths[i]]
                    smp.waveform = array.array(
                        'b',
                        smp.waveform)  # the bytearray string is reinterpreted as signed 8bit values
                    waveform_idx += sample_lengths[i]

                self.samples[i] = smp

            # ----------------------------
            # Do some sanity checks
            # ----------------------------

            # Some non-standard modules have extra patterns saved in the song data.
            # The MOD standard does not allow storing patterns beyond the maximum pattern number that is actually used.
            patterns_size = n_unique_patterns * MODSong.ROWS * MODSong.CHANNELS * 4
            samples_size = sum(sample_lengths)
            predicted_size = patterns_size + samples_size
            actual_size = len(data[1084:])
            if predicted_size != actual_size:
                n_extra_patterns = int((actual_size - predicted_size) / (MODSong.ROWS * MODSong.CHANNELS * 4))
                raise NotImplementedError(f"The module has {n_extra_patterns} unexpected extra patterns.")

            # ----------------------------
            # Load pattern data
            # ----------------------------

            for p in range(n_unique_patterns):

                pat = Pattern(MODSong.ROWS, MODSong.CHANNELS)

                for r in range(MODSong.ROWS):
                    for c in range(MODSong.CHANNELS):

                        # byte index
                        idx = p * MODSong.ROWS * MODSong.CHANNELS * 4 + r * MODSong.CHANNELS * 4 + c * 4

                        # a full 4-byte slot in the current channel, e.g. "C-5 11 F06"
                        note_raw = data[1084 + idx:1084 + idx + 4]

                        note = Note()

                        note.sample_idx = MODSong.get_sample_from_note(note_raw)
                        note.period = MODSong.get_period_from_note(note_raw)

                        e_type, e_param = MODSong.get_effect_from_note(note_raw)

                        if e_type != 0 or e_param != 0:

                            # dirty way for converting hex number to string... e.g. 0xF1 -> "F1"
                            note.effect = hex(e_type).lstrip("0x").upper() + hex(e_param)[2:].upper()
                            
                            if e_type == 0:  # arpeggio effect
                                note.effect = "0" + note.effect

                        pat.data[c][r] = note

                self.patterns.append(pat)

        if verbose:
            print('done.')

    def save_as_ascii(self, fname: str, verbose: bool = True):
        """
        Writes the song as readable text with ASCII encoding.

        :param fname: Complete file path.
        :param verbose: False for silent saving.
        :return: None.
        """

        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)

        with open(fname, 'w', encoding='ascii') as file:

            for p in [self.patterns[i] for i in self.pattern_seq]:
                for r in range(MODSong.ROWS):
                    for c in range(MODSong.CHANNELS):
                        file.write(f"| {p.data[c][r]} ")
                    file.write('|\n')
                file.write('\n')

        if verbose:
            print('done.')

    def save_as_mod(self, fname: str, verbose: bool = True):
        """
        Saves the song as a standard MOD file.

        :param fname: Complete file path.
        :param verbose: False for silent saving.
        :return: None.
        """

        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)

        if len(self.pattern_seq) == 0 or len(self.patterns) > 128:
            raise OverflowError(f"Can't save a MOD file with {len(self.pattern_seq)} patterns.")

        data = bytearray()

        def str_to_bytes_padded(s: str, max_len: int) -> bytes:
            r = bytes(s, 'utf-8')
            if len(r) > max_len:  # truncate
                r = r[:max_len]
            else:
                r += bytes(max_len - len(r))
            return r

        # ----------------------------
        # Write the song title
        # ----------------------------

        data += str_to_bytes_padded(self.songname, 20)

        # ----------------------------
        # Write the sample headers
        # ----------------------------

        for s in range(MODSong.SAMPLES):

            smp = self.samples[s]

            data += str_to_bytes_padded(smp.name, 22)

            # A brief note on the maximum sample length.
            # The two bytes encode the sample length in terms of number of words.
            # This means that 2^16 = 64K is the maximum length in words, i.e. 128K in bytes.

            waveform = smp.waveform
            if len(waveform) % 2 != 0:
                waveform.append(0x0)
            if int(len(waveform) / 2) > 131072:
                raise ValueError(f"Sample length {int(len(waveform) / 2)} exceeds the MOD maximum of 128K.")
            data += int(len(waveform) / 2).to_bytes(2, byteorder='big', signed=False)

            data += ((smp.finetune << 4) >> 4).to_bytes(1)

            if smp.volume > 64:
                print(f"Warning: Truncating max sample volume from {smp.volume} to 64.")
            data += min(smp.volume, 64).to_bytes(1)

            data += int(smp.repeat_point / 2).to_bytes(2, byteorder='big', signed=False)
            data += int(smp.repeat_len / 2).to_bytes(2, byteorder='big', signed=False)

        # ----------------------------
        # Write the song sequence
        # ----------------------------

        data += len(self.pattern_seq).to_bytes(1)
        data += int(127).to_bytes(1)
        data += bytearray(self.pattern_seq) + bytearray(128 - len(self.pattern_seq))
        data += bytes("M.K.", 'utf-8')

        # ----------------------------
        # Write the pattern data
        # ----------------------------

        for p in range(len(self.patterns)):

            if p > max(self.pattern_seq):  # the MOD standard does not allow saving patterns beyond this index
                break

            pat = self.patterns[p]

            if len(pat.data) != MODSong.CHANNELS:
                raise NotImplementedError(f"Can't save a MOD file with {len(pat.data)} channels.")

            if len(pat.data[0]) != MODSong.ROWS:
                raise NotImplementedError(f"Can't save a MOD file with {len(pat.data[0])} rows.")

            for r in range(MODSong.ROWS):
                for c in range(MODSong.CHANNELS):

                    note = pat.data[c][r]

                    efx_type = 0x0
                    efx_param = 0x0
                    if note.effect != "":
                        efx_type = int(note.effect[0], 16)  # interpret the character as a hex digit
                        efx_param = int(note.effect[1:], 16)

                    if note.period != '':
                        pd = MODSong.INV_PERIOD_TABLE[note.period]
                    else:
                        pd = 0

                    note_raw = bytearray(4)
                    note_raw[0] = (note.sample_idx & 0xF0) | ((pd & 0xF00) >> 8)
                    note_raw[1] = pd & 0xFF
                    note_raw[2] = ((note.sample_idx & 0x0F) << 4) | efx_type
                    note_raw[3] = efx_param

                    data += note_raw

        # ----------------------------
        # Write the sample data
        # ----------------------------

        for smp in self.samples:
            data += bytearray(smp.waveform)

        # ----------------------------
        # Write the raw data to file
        # ----------------------------

        with open(fname, 'bw') as mod_file:
            mod_file.write(bytes(data))

        if verbose:
            print('done.')

    def render_as_wav(self, fname: str, verbose: bool = True, cleanup: bool = False):
        """
        Renders the current song as a WAV file.
        Note: Requires openmpt123.exe to be installed in the current working directory.

        :param fname: Complete path of the output WAV file.
        :param verbose: False for silent rendering.
        :param cleanup: True to remove the temporary MOD file generated for rendering.
        :return: None.
        """

        if verbose:
            print("Rendering as wav... ", end='', flush=True)

        if os.path.isfile(fname):
            os.remove(fname)

        noext = os.path.splitext(fname)

        if os.path.isfile(f"{noext[0]}.mod"):
            os.remove(f"{noext[0]}.mod")

        self.save_as_mod(f"{noext[0]}.mod", False)

        if os.path.isfile(f"{noext[0]}.mod.wav"):
            os.remove(f"{noext[0]}.mod.wav")

        try:
            subprocess.run(f"openmpt123.exe {noext[0]}.mod -q --channels 1 --samplerate 44100 --render", check=True)
        except FileNotFoundError as _:
            try:
                subprocess.run([f"ffmpeg", "-i", f"{noext[0]}.mod", f"{noext[0]}.mod.wav"], check=True)
            except FileNotFoundError as e:
                raise FileNotFoundError(e)

        shutil.move(f"{noext[0]}.mod.wav", fname)

        if cleanup:
            os.remove(f"{noext[0]}.mod")

        if verbose:
            print("done.")
    
    '''
    -------------------------------------
    SONG
    -------------------------------------
    '''

    def get_song_duration(self) -> float:
        """
        Returns the duration of the song in seconds.

        :return: The song duration in seconds.
        """

        # TODO: Implement this method.
        
        return 0.

    def annotate_time(self) -> list[list[float]]:
        """
        Annotates the time of each row in the song, taking into account the speed and bpm changes.

        FIXME: account for pattern delays, loops, and jumps (Bxx Dxx, E6x, EEx effects).
        FIXME: do a separate version for individual patterns, and another for the entire song.

        :return: A list where each element is a list corresponding to pattern in the sequence.
                 Within each list, each row is annotated with a time in seconds.
        """

        # default timing for MOD files, if nothing is specified
        bpm = 125
        speed = 6

        d = Song.get_tick_duration(bpm)

        annotated_song = []

        for p in self.pattern_seq:

            annotated_pattern = []  # annotate each pattern separately

            for r in range(MODSong.ROWS):

                jump = False

                for c in range(MODSong.CHANNELS):    

                    efx = self.patterns[p].data[c][r].effect
                    if efx != "":
                     
                        if efx[0] == "F":  # change of speed or bpm

                            v = int(efx[1:], 16)
                            if v <= 31:
                                speed = v
                            else:
                                bpm = v

                            d = Song.get_tick_duration(bpm)
                            # print(f"CHANGE: Pattern {p}, row {r}, channel {c}, speed {speed}, bpm {bpm}, tick duration {d}")

                        elif efx[0] == "D":  # jump to a specific position in the next pattern in the sequence
                            jump = True

                annotated_pattern.append(d * speed)

                if jump:
                    break

            annotated_song.append(annotated_pattern)

        # cumsum over the entire list of lists
        cum = 0
        for p in range(len(annotated_song)):
            for r in range(len(annotated_song[p])):
                cum += annotated_song[p][r]
                annotated_song[p][r] = cum

        return annotated_song
    
    '''
    -------------------------------------
    SAMPLES AND INSTRUMENTS
    -------------------------------------
    '''

    def load_sample(self, fname: str, sample_idx: int = 0) -> tuple[int, Sample]:
        """
        Loads a sample from a WAV file, and stores it at the given sample index.

        :param fname: The complete file path to the .wav file.
        :param sample_idx: The sample index to store the sample in the song, from 1 to 31. 
                           Use 0 to automatically use the next available slot.
        :return: A tuple (int, Sample) containing:
                 - the index of the added sample, from 1 to 31
                 - the corresponding sample object
        """

        if sample_idx < 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx}.")
        
        if sample_idx == 0:
            for i in range(MODSong.SAMPLES):
                if len(self.samples[i].waveform) == 0:
                    sample_idx = i + 1
                    break
            if sample_idx == 0:
                raise ValueError(f"Couldn't find an empty slot for the new sample.")

        self.samples[sample_idx - 1] = Sample()  # reset all attributes

        audio = pydub.AudioSegment.from_wav(fname).set_channels(1)
        if audio.sample_width != 1:
            audio = audio.set_sample_width(1)

        self.samples[sample_idx - 1].waveform = audio.get_array_of_samples()

        return sample_idx, self.samples[sample_idx - 1]

    def get_sample(self, sample_idx: int) -> Sample:

        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx}")

        return self.samples[sample_idx - 1]

    def remove_sample(self, sample_idx: int):
        """
        Deletes the sample from the sample bank.
        WARNING: This does not remove the sample notes from the song. The notes will stay, but will play mute.

        :param sample_idx: The sample index to remove, 1 to 31.
        :return: None.
        """

        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx}")

        self.samples[sample_idx] = Sample()

    def keep_sample(self, sample_idx: int):
        """
        Deletes all samples in the sample bank, except for the one specified by the given index.
        WARNING: This does not remove the sample notes from the song. The notes will stay, but will play mute.

        :param sample_idx: The sample index to be kept, 1 to 31.
        :return: None.
        """

        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx}")

        for s in range(MODSong.SAMPLES):
            if s + 1 != sample_idx:
                self.samples[s] = Sample()
    
    '''
    -------------------------------------
    PATTERNS
    -------------------------------------
    '''

    def clear_pattern(self, pattern: int):
        """
        Clears completely a specified pattern.

        :param pattern: The pattern index (within the song sequence) to be cleared.
        :return: None.
        """

        if pattern < 0 or pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        p = self.pattern_seq[pattern]
        for r in range(MODSong.ROWS):
            for c in range(MODSong.CHANNELS):
                self.patterns[p].data[c][r] = Note()

    def add_pattern(self) -> int:
        """
        Creates a brand new pattern and adds it to the song sequence.

        :return: The index of the new pattern.
        """

        self.patterns.append(Pattern(MODSong.ROWS, MODSong.CHANNELS))
        n = len(self.patterns) - 1
        self.pattern_seq.append(n)

        return n

    def get_pattern_duration(self, pattern: int) -> float:
        """
        Returns the duration of a pattern in seconds.

        :param pattern: The pattern index (within the song sequence).
        :return: The pattern duration in seconds.
        """

        # TODO
        
        return 0.
        
    def get_effective_row_count(self, pattern: int) -> int:
        """
        Returns the effective number of rows that get played in a pattern.
        Accounts for position jumps, loops, and breaks.

        TODO: Implement a version for the entire song. 
              It's not so trivial, because of position jumps effects (Dxx) and such.

        :param pattern: The pattern index (within the song sequence).
        :return: The effective number of rows that gets played in the pattern.
        """

        if pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        effective_rows = 0
        loop_start_row = 0  # used by E6x effect

        data = copy.deepcopy(self.patterns[self.pattern_seq[pattern]].data)

        unrolled_data = [[] for _ in range(MODSong.CHANNELS)]

        for r in range(MODSong.ROWS):

            interrupt = False  # if true, the pattern is cut short by Bxx or Dxx effects

            for c in range(MODSong.CHANNELS):                

                unrolled_data[c].append(data[c][r])

                efx = data[c][r].effect
                if efx != "":

                    if efx[0] == "B" or efx[0] == "D":
                        interrupt = True

                    if efx[:2] == "E6":

                        if int(efx[2], 16) == 0:  # E60 means loop start
                            loop_start_row = r
                        
                        loop_end_row = r
                        loop_count = int(efx[2], 16)

                        for loop in range(loop_count):
                            unrolled_data[c] += unrolled_data[c][loop_start_row:loop_end_row + 1]

            if interrupt:
                break
    
        return max([len(unrolled_data[c]) for c in range(MODSong.CHANNELS)])

    '''
    -------------------------------------
    CHANNELS
    -------------------------------------
    '''

    def clear_channel(self, channel: int):
        """
        Clears completely a specified channel in the entire song.
        Warning: If you use this as a way to mute a channel, be careful because it also deletes global effects like bpm.

        :param channel: The channel index to mute, 1 to 4.
        :return: None.
        """

        if channel <= 0 or channel > MODSong.CHANNELS:
            raise IndexError(f"Invalid channel index {channel}")

        for p in range(len(self.patterns)):
            for r in range(MODSong.ROWS):
                self.patterns[p].data[channel - 1][r] = Note()

    '''
    -------------------------------------
    NOTES
    -------------------------------------
    '''

    @staticmethod
    def get_sample_from_note(note: bytearray) -> int:
        """
        Returns the sample number from a 4-byte note.

        :param note: A 4-byte note.
        :return: The sample number.
        """

        u4 = note[0] & 0xF0  # upper 4 bits of sample number
        l4 = note[2] & 0xF0  # lower 4 bits
        return u4 | (l4 >> 4)

    @staticmethod
    def get_effect_from_note(note: bytearray) -> tuple[int, int]:
        """
        Returns the effect type and parameter from a note.
        Effects follow a hex format, e.g. E60 means (15,96) in decimal.

        :param note: A 4-byte note.
        :return: A tuple (int, int) containing the effect type and parameter.
        """

        return note[2] & 0x0F, note[3]

    @staticmethod
    def get_period_from_note(note: bytearray) -> str:
        """
        Returns the note period (pitch) from a 4-byte note.

        :param note: A 4-byte note.
        :return: The note period (pitch), or an empty string if no pitch is specified.
        """

        period_raw = ((note[0] & 0x0F) << 8) | note[1]
        if period_raw != 0:
            return MODSong.PERIOD_TABLE[period_raw]
        else:
            return ""

    def get_note(self, pattern_in_song: int, row: int, channel: int) -> Note:

        if row < 0 or row >= MODSong.ROWS:
            raise IndexError(f"Invalid row index {row}")

        if channel < 0 or channel >= MODSong.CHANNELS:
            raise IndexError(f"Invalid channel index {channel}")

        if pattern_in_song < 0 or pattern_in_song >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern_in_song}")

        return self.patterns[self.pattern_seq[pattern_in_song]].data[channel][row]
    
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
            raise ValueError(f"Invalid tempo {bpm}")

        self.write_effect(pattern, channel, row, f"F{bpm:02X}")
        
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
            raise ValueError(f"Invalid ticks per row {ticks}")

        self.write_effect(pattern, channel, row, f"F{ticks:02X}")

    def set_portamento(self, pattern: int, channel: int, row: int, slide: int):
        """
        Slides up or down the sample frequency by 'slide' notes per tick.
        Therefore, the slide rate depends on the number of ticks per row.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param slide: The amount of notes to slide by, -255 to 255. 0 is ignored.
        :return: None.
        """

        if slide < -255 or slide > 255:
            raise ValueError(f"Invalid portamento slide {slide}")

        if slide > 0:
            self.write_effect(pattern, channel, row, f"1{slide:02X}")
        elif slide < 0:
            self.write_effect(pattern, channel, row, f"2{slide:02X}")

    def set_tone_portamento(self, pattern: int, channel: int, row: int, speed: int):
        """
        Slides the previous note (usually of the same sample) to the current note.
        The slide happens with the given speed.
        The effect continues until the specified frequency is reached.
        If speed=0, then the last speed used on the channel is used again.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param speed: The amount of notes per tick to slide by, 0 to 255.
        :return: None.
        """

        if speed < 0 or speed > 255:
            raise ValueError(f"Invalid tone portamento speed {speed}")

        self.write_effect(pattern, channel, row, f"3{speed:02X}")

    def set_tone_portamento_slide(self, pattern: int, channel: int, row: int, slide: int):
        """
        Continue a preceding tone portamento, sliding the volume up or down by a given amount.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param slide: The volume slide to set, -15 to 15. 0 is ignored.
        :return: None.
        """

        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid tone portamento slide {slide}")
        
        efx = 0
        if slide > 0:
            efx = slide << 4
        elif slide < 0:
            efx = -slide
        
        self.write_effect(pattern, channel, row, f"5{efx:02X}")
        
    def set_volume(self, pattern: int, channel: int, row: int, volume: int):
        """
        Sets the volume.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param volume: The volume to set, 0 to 64.
        :return: None.
        """

        if volume < 0 or volume > 64:
            raise ValueError(f"Invalid volume {volume}")

        self.write_effect(pattern, channel, row, f"C{volume:02X}")

    def set_volume_slide(self, pattern: int, channel: int, row: int, slide: int):
        """
        Slides the volume up or down by a given amount.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param slide: The volume slide to set, -15 to 15. 0 is ignored.
        :return: None.
        """

        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid volume slide {slide}")
        
        efx = 0
        if slide > 0:
            efx = slide << 4
        elif slide < 0:
            efx = -slide
        
        self.write_effect(pattern, channel, row, f"A{efx:02X}")

    def set_vibrato(self, pattern: int, channel: int, row: int, speed: int, depth: int):
        """
        Sets the vibrato effect with the given speed and depth.
        If either speed or depth are 0, then reuse values from the most recent vibrato.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param speed: The vibrato speed (how fast it oscillates), 0 to 15.
        :param depth: The vibrato depth (how much it oscillates), 0 to 15.
        :return: None.
        """

        if speed < 0 or speed > 15:
            raise ValueError(f"Invalid vibrato speed {speed}")
        
        if depth < 0 or depth > 15:
            raise ValueError(f"Invalid vibrato depth {depth}")
        
        efx = 16 * speed + depth
        
        self.write_effect(pattern, channel, row, f"4{efx:02X}")

    def set_vibrato_slide(self, pattern: int, channel: int, row: int, slide: int):
        """
        Continue a preceding vibrato, sliding the volume up or down by a given amount.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param slide: The volume slide to set, -15 to 15. 0 is ignored.
        :return: None.
        """

        if slide < -15 or slide > 15:
            raise ValueError(f"Invalid vibrato slide {slide}")
        
        efx = 0
        if slide > 0:
            efx = slide << 4
        elif slide < 0:
            efx = -slide
        
        self.write_effect(pattern, channel, row, f"6{efx:02X}")

    def set_tremolo(self, pattern: int, channel: int, row: int, speed: int, depth: int):
        """
        Sets the tremolo effect with the given speed and depth.
        If either speed or depth are 0, then reuse values from the most recent tremolo.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param speed: The tremolo speed (how fast it oscillates), 0 to 15.
        :param depth: The tremolo depth (how much it oscillates), 0 to 15.
        :return: None.
        """

        if speed < 0 or speed > 15:
            raise ValueError(f"Invalid tremolo speed {speed}")
        
        if depth < 0 or depth > 15:
            raise ValueError(f"Invalid tremolo depth {depth}")
        
        efx = 16 * speed + depth
        
        self.write_effect(pattern, channel, row, f"7{efx:02X}")
