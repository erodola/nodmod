from nodmod.song import Song
from nodmod.song import Sample
from nodmod.song import Pattern
from nodmod.song import Note
import copy
import array
import os
import subprocess
import shutil
import pydub
import numpy as np


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
        1076: 'G#3', 1016: 'A-3', 960: 'A#3', 907: 'B-3',
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

    def load_from_file(self, fname: str, verbose: bool = True):
        """
        Loads a song from a standard MOD file.

        :param fname: Complete file path.
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

    def save(self, fname: str, verbose: bool = True):
        """
        Saves the song to file; the format is automatically determined by the file extension.

        :param fname: Complete file path.
        :param verbose: False for silent saving.
        :return: None.
        """

        ext = fname.split('.')[-1]
        if ext.lower() == 'mod':
            self._save_as_mod(fname, verbose)
        elif ext == 'txt':
            self._save_as_ascii(fname, verbose)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _save_as_ascii(self, fname: str, verbose: bool = True):
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

    def _save_as_mod(self, fname: str, verbose: bool = True):
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

    def clear_pattern(self, pattern_in_song: int):
        """
        Clears completely a specified pattern.

        :param pattern_in_song: The pattern index (within the song sequence) to be cleared.
        :return: None.
        """

        if pattern_in_song < 0 or pattern_in_song >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern_in_song}")

        p = self.pattern_seq[pattern_in_song]
        for r in range(MODSong.ROWS):
            for c in range(MODSong.CHANNELS):
                self.patterns[p].data[c][r] = Note()

    def new_pattern(self) -> int:
        """
        Creates a new pattern and set the composer at its beginning.

        :return: The index of the new pattern.
        """

        r = self.add_pattern()
        self.use_pattern(r)
        self.use_row(0)
        return r

    def add_pattern(self) -> int:
        """
        Creates a brand new pattern and adds it to the song sequence.

        :return: The index of the new pattern.
        """

        self.patterns.append(Pattern(MODSong.ROWS, MODSong.CHANNELS))
        n = len(self.patterns) - 1
        self.pattern_seq.append(n)

        return n

    def duplicate_pattern(self, pattern: int) -> int:
        """
        Duplicates a pattern in the song sequence, and appends it at the end of the song sequence.

        :param pattern: The pattern index (within the song sequence) to be duplicated.
        :return: The index of the new pattern.
        """

        if pattern < 0 or pattern >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern}")

        self.patterns.append(copy.deepcopy(self.patterns[self.pattern_seq[pattern]]))
        n = len(self.patterns) - 1
        self.pattern_seq.append(n)

        return n

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

    def interp_effect(self, start_efx: str, end_efx: str, n_steps: int, stride: int = 1):
        """
        Linearly interpolates an effect between given start and end points.
        The interpolated effects are written directly in the song starting from the current row.
        This can also be used to repeat an effect a number of times by setting start = end.

        :param start_efx: The start effect value given as a string, e.g. "C00"
        :param end_efx: The end effect value given as a string, e.g. "C3F"
        :param n_steps: The number of interpolation steps.
        :param stride: The stride to use, e.g. stride=1 writes the effect at successive rows.
        :return: None.
        """

        if n_steps <= 1:
            raise ValueError(f"Interpolation steps must be at least 2.")

        if stride < 1:
            raise ValueError(f"Interpolation stride must be at least 1.")

        efx = start_efx[0]

        if efx != end_efx[0]:
            raise ValueError("Start and end effects for the interpolation must be the same.")

        if self.current_row + n_steps - 1 >= MODSong.ROWS:
            raise IndexError(f"Too many interpolation steps, will break out of the current pattern.")

        start = int(start_efx[1:], 16)
        end = int(end_efx[1:], 16)
        steps = [int(round(start + (end - start) * t / (n_steps - 1))) for t in range(n_steps)]

        backup_row = self.current_row

        for r in range(n_steps):
            self.write_effect(efx + f"{steps[r]:02X}")
            self.current_row += stride

        self.current_row = backup_row

    def arpeggio(self, chord: list[str], algo: str):
        """
        Generates an arpeggio out of a given chord, according to a desired strategy.
        Writes directly in the song starting from the current row.

        :param chord: The input chord to convert to an arpeggio.
        :param algo: The arpeggiator algorithm to use. Right now only "up" and "down" are implemented.
        :return: None.
        """

        if algo == 'up' or algo == 'down':

            idx = [0 for _ in range(len(chord))]
            for i, note in enumerate(chord):
                idx[i] = Song.PERIOD_SEQ.index(note[:2]) + 20*int(note[2])

            note_seq = np.array(chord)[np.argsort(idx)]

            if algo == 'down':
                note_seq = note_seq[::-1]

        else:
            raise NotImplementedError(f"The arpeggiator algorithm {algo} is not implemented.")

        old_row = self.current_row
        for note in note_seq:
            self.write_note(note)
            self.current_row += 1

        self.current_row = old_row

    def write_note_(self, period: str, effect: str = ""):
        """
        Automagic version of Song.write_note().
        Automatically chooses the best channel to write the desired note.
        Advances to the next row.
        If no channel is free, overwrites the current channel.

        :param period: The note period (pitch) to write, e.g. "C-4".
        :param effect: The note effect, e.g. "ED1".
        :return: None.
        """

        old_channel = self.current_channel

        for c in range(MODSong.CHANNELS):
            note = self.get_note(self.current_pattern, self.current_row, c)
            if note.is_empty():
                break
            if effect == "" and note.sample_idx == 0 and note.period == '':
                if note.effect[0] == 'F':  # global effect, can write in this channel
                    break
        
        self.current_channel = c
        self.write_note(period, effect)

        self.current_channel = old_channel
        self.current_row = min(self.current_row + 1, MODSong.ROWS - 1)

    def clear_tempo(self, pattern=-1):
        """
        Clears all the tempo effects in the given pattern.

        :param pattern: The pattern index to clear the tempo effects from. -1 for the entire song.
        :return: None.
        """

        patterns_to_clear = [pattern] if pattern >= 0 else range(len(self.patterns))

        for p in patterns_to_clear:
            for r in range(MODSong.ROWS):
                for c in range(MODSong.CHANNELS):
                    efx = self.patterns[p].data[c][r].effect
                    if efx != '' and efx[0] == 'F':
                        self.patterns[p].data[c][r].effect = ""

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

    def get_note(self, pattern_in_song: int, row: int, channel: int) -> Note:

        if row < 0 or row >= MODSong.ROWS:
            raise IndexError(f"Invalid row index {row}")

        if channel < 0 or channel >= MODSong.CHANNELS:
            raise IndexError(f"Invalid channel index {channel}")

        if pattern_in_song < 0 or pattern_in_song >= len(self.pattern_seq):
            raise IndexError(f"Invalid pattern index {pattern_in_song}")

        return self.patterns[self.pattern_seq[pattern_in_song]].data[channel][row]

    def tune_sample(self, sample_idx: int, verbose: bool = False, cleanup: bool = True) -> str:
        """
        Tunes a specified sample by finding a map between true G (Sol) and the corresponding sample pitch.
        The sample wave itself is not modified; the result is written in the .tune class field.

        :param sample_idx: The sample index to tune.
        :param verbose: False for silent tuning.
        :param cleanup: True to remove the temporary MOD, WAV, and MIDI files used for tuning.
        :return: The sample pitch corresponding to true G (Sol), or an empty string if the tuning failed.
        """

        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx}")

        # Lazy import within the function, because this module does very inefficient operations upon loading.
        import basic_pitch.inference

        tmp_fname = './tmp'

        # Build dictionaries to convert pitches from MIDI format to MOD format.
        # TODO: directly hardcode the dictionaries as Song statics

        pitch_to_note = {}
        note_to_pitch = {}
        with open("./midi_notes.txt", 'r') as file:
            for line in file:
                items = line.split()
                pitch_to_note[items[0]] = items[1]
                note_to_pitch[items[1]] = items[0]

        # Will temporarily replace the song with a single pattern and chromatic scales at two consecutive octaves.

        backup_patterns = copy.deepcopy(self.patterns)
        backup_seq = self.pattern_seq

        bpm = 125
        speed = 6
        rows_per_beat = 4

        # the formula holds in the absence of tempo effects, such as EDx (note delay), Fxx, etc.
        # source: https://wiki.openmpt.org/Manual:_Song_Properties
        # FIXME: for MOD files the Classic mode applies, not the Modern mode.
        tick_duration = ((60 / bpm) / rows_per_beat) / speed  # in seconds

        self.keep_pattern_from_seq(0)
        self.use_pattern(0)
        self.use_channel(2)
        self.use_sample(sample_idx)

        found = False

        for octave in range(4, 6):

            # Write two chromatic scales and consecutive octaves, and render as WAV.

            octaves = [octave, octave + 1]

            self.clear_pattern(0)

            self.use_row(0)

            self.use_channel(0)
            self.set_bpm(bpm)

            self.use_channel(1)
            self.set_ticks_per_row(speed)

            self.use_channel(2)

            r = 0
            for n in Song.PERIOD_SEQ:
                self.use_row(r)
                self.write_note(f"{n}{octaves[0]}")
                r += 3
            for n in Song.PERIOD_SEQ:
                if r >= MODSong.ROWS:
                    break
                self.use_row(r)
                self.write_note(f"{n}{octaves[1]}")
                r += 3

            self.render_as_wav(f"{tmp_fname}.wav", verbose, False)

            # Detect the true notes from the chromatic scales.

            _, midi_data, _ = basic_pitch.inference.predict(f"{tmp_fname}.wav")

            if not cleanup:
                midi_data.write(f"{tmp_fname}.mid")

            if cleanup:
                if os.path.isfile(f"{tmp_fname}.wav"):
                    os.remove(f"{tmp_fname}.wav")
                if os.path.isfile(f"{tmp_fname}.mod"):
                    os.remove(f"{tmp_fname}.mod")
                if os.path.isfile(f"{tmp_fname}.mid"):
                    os.remove(f"{tmp_fname}.mid")

            # Store the MOD notes in a list for future comparison, and print them in a readable way.

            mod_notes = []

            for r in range(MODSong.ROWS):
                note = self.get_note(0, r, 2)
                if note.period == "":
                    if verbose:
                        print('.', end='')
                else:
                    if verbose:
                        print(note.period, end='')
                    mod_notes.append(note.period)
                if verbose:
                    print(' ', end='')
            if verbose:
                print('')

            # Store the MIDI notes in a list, and print them in a readable way.

            n_notes = len(midi_data.instruments[0].notes)
            midi_notes = ['' for _ in range(n_notes)]

            for n in range(n_notes):
                note = midi_data.instruments[0].notes[n]
                if n == 0:
                    n_rows = int(round(note.start / (tick_duration * speed)))
                else:
                    n_rows = int(round(
                        (note.start - midi_data.instruments[0].notes[n - 1].start) / (tick_duration * speed))) - 1
                if verbose:
                    for _ in range(n_rows):
                        print('. ', end='')
                midi_notes[n] = pitch_to_note[str(note.pitch)]
                if verbose:
                    print(pitch_to_note[str(note.pitch)], end='')
                    print(' ', end='')
            if verbose:
                print('')

            # Compare the original MOD notes with the detected MIDI notes.

            if len(mod_notes) != len(midi_notes):
                if verbose:
                    print(f"WARNING: ", end='')
                found = False
            else:
                if verbose:
                    print(f"OK: ", end='')
                found = True
            if verbose:
                print(f"The note detection algorithm detected {len(midi_notes)} out of {len(mod_notes)} notes.")

            is_chord = False

            if len(midi_notes) > len(mod_notes):
                if len(midi_notes) % len(mod_notes) == 0:
                    is_chord = True
                    print(f"Based on the amount of detected notes, this sample might actually be a "
                          f"{int(len(midi_notes) / len(mod_notes))}-note chord.")
                elif len(midi_notes) >= 1.5*len(mod_notes):
                    print(f"Based on the amount of detected notes, this sample might actually be a chord.")

            if not found and not is_chord:
                continue

            if is_chord:
                print(f"Chords are not implemented yet.")  # TODO

            full_scale = []
            for o in range(1, 9):
                for note in Song.PERIOD_SEQ:
                    full_scale.append(f"{note}{o}")

            mod_shifts = []
            for i in range(2, len(mod_notes)):
                mod_shifts.append(full_scale.index(mod_notes[i]) - full_scale.index(mod_notes[i - 1]))
            midi_shifts = []
            for i in range(2, len(midi_notes)):
                midi_shifts.append(full_scale.index(midi_notes[i]) - full_scale.index(midi_notes[i - 1]))

            if len(mod_shifts) == len(midi_shifts):
                if mod_shifts != midi_shifts:
                    found = False
                    if verbose:
                        print("WARNING: The two sequences have different inner pitch shifts.")
                else:
                    found = True
                    if verbose:
                        print("OK: The two sequences have the same inner pitch shifts.")

            if not found:
                continue

            shifts_mod_to_midi = []
            if len(mod_shifts) == len(midi_shifts):
                for i in range(len(mod_notes)):
                    s = full_scale.index(mod_notes[i]) - full_scale.index(midi_notes[i])
                    shifts_mod_to_midi.append(s)

            if all(n == shifts_mod_to_midi[0] for n in shifts_mod_to_midi):
                found = True
                if verbose:
                    print("OK: The two sequences are coherent.")
            else:
                found = False
                if verbose:
                    print("WARNING: The two sequences are not coherent.")

            if verbose:
                print(f"MOD notes: {mod_notes}")
                print(f"MID notes: {midi_notes}")

            if not found:
                continue

            # Compute the amount of (semi)tones required to bring the MIDI in tune with the MOD samples.

            shift = shifts_mod_to_midi[0]
            n_octaves = int(shift / 12)
            n_semitones = shift % 12

            if verbose:
                if shift > 0:
                    sign = '+'
                else:
                    sign = '-'
                if n_octaves > 0:
                    print(
                        f"Shift the MIDI notes by {sign}{n_octaves} octaves and {sign}{n_semitones} semitones to match the MOD.")
                else:
                    print(f"Shift the MIDI notes by {sign}{n_semitones} semitones to match the MOD.")

            if found:  # no need to test the next octave
                break

        self.pattern_seq = backup_seq
        self.patterns = backup_patterns

        G = ''

        if found:
            for G_idx in range(len(midi_notes)):
                if midi_notes[G_idx][:2] == 'G-':
                    G = mod_notes[G_idx][:2]
                    if G[1] == '-':
                        G = G[0]
                    break

        self.samples[sample_idx].tune = G
        return G

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

        self._save_as_mod(f"{noext[0]}.mod", False)

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

    def fade_in(self, chord: list[str], start_vol: int = 0, end_vol: int = 255, n_steps: int = 0, stride: int = 1):
        """
        Writes a chord at the current row and applies a fade-in effect.

        :param chord: The chord to write.
        :param start_vol: The starting volume for the fade-in effect (typically 0).
        :param end_vol: The end volume for the fade-in effect (typically 255).
        :param n_steps: The number of steps required to go from start_vol to end_vol. Set to 0 to use +1 increments.
        :param stride: The stride to use, e.g. stride=1 fades in at successive rows.
        :return: None.
        """

        if end_vol <= start_vol:
            raise ValueError(f"The end volume for fade-in must be larger than the start volume.")

        self.fade(chord, start_vol, end_vol, n_steps, stride)

    def fade_out(self, chord: list[str], start_vol: int = 255, end_vol: int = 0, n_steps: int = 0, stride: int = 1):
        """
        Writes a chord at the current row and applies a fade-out effect.

        :param chord: The chord to write.
        :param start_vol: The starting volume for the fade-out effect (typically 255).
        :param end_vol: The end volume for the fade-out effect (typically 0).
        :param n_steps: The number of steps required to go from start_vol to end_vol. Set to 0 to use -1 increments.
        :param stride: The stride to use, e.g. stride=1 fades out at successive rows.
        :return: None.
        """

        if end_vol >= start_vol:
            raise ValueError(f"The end volume for fade-out must be smaller than the start volume.")

        self.fade(chord, start_vol, end_vol, n_steps, stride)

    def fade_in_and_out(
            self, chord: list[str],
            start_vol: int = 0, mid_vol: int = 128, end_vol: int = 0,
            steps_in: int = 0, steps_out: int = 0, stride: int = 1):
        """
        Writes a chord at the current row and applies a fade-in-and-out effect.

        :param chord: The chord to write.
        :param start_vol: The starting volume for the fade-in part.
        :param mid_vol: The mid-volume to reach.
        :param end_vol: The end volume for the fade-out part.
        :param steps_in: The number of steps to go from start_vol to mid_vol. Set to 0 to use +1 increments.
        :param steps_out: The number of steps to go from mid_vol to end_vol. Set to 0 to use -1 increments.
        :param stride: The stride to use, e.g. stride=1 fades at successive rows.
        :return: None.
        """

        old_row = self.current_row

        self.fade_in(chord, start_vol, mid_vol, steps_in, stride)
        self.current_row += (steps_in - 1) * stride
        self.fade_out(chord, mid_vol, end_vol, steps_out, stride)

        note = self.patterns[self.pattern_seq[self.current_pattern]].data[self.current_channel][self.current_row]
        note.sample_idx = 0
        note.period = ''

        self.patterns[self.pattern_seq[self.current_pattern]].data[self.current_channel][self.current_row] = note
        self.current_row = old_row

    def fade(self, chord: list[str], start_vol: int, end_vol: int, n_steps: int = 0, stride: int = 1):
        """
        Writes a chord at the current row and applies a volume fade effect.
        Note: You probably want to directly use MODSong.fade_in() and MODSong.fade_out().

        :param chord: The chord to write.
        :param start_vol: The starting volume for the fade effect.
        :param end_vol: The end volume for the fade effect.
        :param n_steps: The number of steps required to go from start_vol to end_vol. Set to 0 to use +1 increments.
        :param stride: The stride to use, e.g. stride=1 uses successive rows.
        :return: None.
        """

        old_channel = self.current_channel

        if n_steps == 0:
            n_steps = end_vol - start_vol + 1

        self.write_chord(chord)

        ch = range(self.current_channel, self.current_channel + len(chord))
        for c in ch:
            self.current_channel = c
            self.interp_effect(f"C{start_vol:02X}", f"C{end_vol:02X}", n_steps, stride)

        self.current_channel = old_channel

    def write_chord(self, chord: list[str]):
        """
        Writes a given chord at the current row, starting from the current channel.
        Overwrites whatever is already written in the affected channels.
        Raises an error if the chord goes beyond the maximum number of channels.

        :param chord: The chord to write in the song.
        :return: None.
        """

        if self.current_channel + len(chord) > MODSong.CHANNELS:
            raise OverflowError(f"The given chord overflows the number of channels.")

        old_channel = self.current_channel

        for i, n in enumerate(chord):
            self.current_channel += i
            self.write_note(n)

        self.current_channel = old_channel

    def set_bpm_(self, bpm: int):
        """
        Automagic version of MODSong.set_bpm().
        Automatically chooses the best channel to set the effect.
        Does not advance the current row.
        If no channel is free, overwrites the current channel.

        :param bpm: The desired bpm, from 32 to 255.
        :return: None.
        """

        old_channel = self.current_channel

        done = False

        for c in range(MODSong.CHANNELS):
            efx = self.get_note(self.current_pattern, self.current_row, c).effect
            if efx != '':
                efx_param = int(efx[1:], 16)
            if (efx == '') or (efx[0] == 'F' and efx_param in range(32, 256)):
                self.current_channel = c
                self.set_bpm(bpm)
                done = True
                break

        self.current_channel = old_channel

        if not done:
            self.set_bpm(bpm)

    def set_bpm(self, bpm: int):
        """
        Sets the bpm (tempo) at the current pattern, row and channel, overwriting whatever other effect is there.

        :param bpm: The bpm value to set in the given pattern, from 32 to 255.
        :return: None.
        """

        if bpm < 32 or bpm > 255:
            raise ValueError(f"Invalid tempo {bpm}")

        self.write_effect(f"F{bpm:02X}")

    def set_ticks_per_row_(self, ticks: int):
        """
        Automagic version of MODSong.set_ticks_per_row().
        Automatically chooses the best channel to set the effect.
        Does not advance the current row.

        :param ticks: The desired speed, from 1 to 31.
        :return: None.
        """

        old_channel = self.current_channel

        done = False

        for c in range(MODSong.CHANNELS):
            efx = self.get_note(self.current_pattern, self.current_row, c).effect
            if efx != '':
                efx_param = int(efx[1:], 16)
            if (efx == '') or (efx[0] == 'F' and efx_param in range(1, 32)):
                self.current_channel = c
                self.set_ticks_per_row(ticks)
                done = True
                break

        self.current_channel = old_channel

        if not done:
            self.set_ticks_per_row(ticks)

    def set_ticks_per_row(self, ticks: int):
        """
        Sets the ticks per row (speed) at the current pattern, row and channel, overwriting whatever other effect is there.

        :param ticks: The speed value to set in the given pattern, from 1 to 31.
        :return: None.
        """

        if ticks < 1 or ticks > 31:
            raise ValueError(f"Invalid ticks per row {ticks}")

        self.write_effect(f"F{ticks:02X}")

    def use_sample(self, sample_idx: int):
        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx}")
        self.current_sample = sample_idx

    def use_channel(self, channel_idx: int):
        if channel_idx < 0 or channel_idx >= MODSong.CHANNELS:
            raise IndexError(f"Invalid channel index {channel_idx}")
        self.current_channel = channel_idx

    def use_row(self, row_idx: int):
        if row_idx < 0 or row_idx >= MODSong.ROWS:
            raise IndexError(f"Invalid row index {row_idx}")
        self.current_row = row_idx

    @staticmethod
    def get_sample_from_note(note: bytearray) -> int:
        u4 = note[0] & 0xF0  # upper 4 bits of sample number
        l4 = note[2] & 0xF0  # lower 4 bits
        return u4 | (l4 >> 4)

    # Returns a tuple (type, parameter).
    # NOTE: Effects follow a hex format, e.g. E60 means (15,96) in decimal.
    @staticmethod
    def get_effect_from_note(note: bytearray):
        return note[2] & 0x0F, note[3]

    @staticmethod
    def get_period_from_note(note: bytearray) -> str:
        period_raw = ((note[0] & 0x0F) << 8) | note[1]
        if period_raw != 0:
            return MODSong.PERIOD_TABLE[period_raw]
        else:
            return ""
