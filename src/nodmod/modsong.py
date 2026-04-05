"""Support for loading, editing, and saving classic 4-channel MOD modules."""

from __future__ import annotations
from nodmod import Song
from nodmod import Sample
from nodmod import Pattern
from nodmod import Note
import array
import os
import pydub  # needed for loading WAV samples
import copy
import struct
from .views import CellView, PlaybackRowView, RowView


class MODSong(Song):


    ROWS = 64
    CHANNELS = 4
    SAMPLES = 31
    PATTERN_SIZE = ROWS * CHANNELS * 4
    PAL_CLOCK = 7093789.2  # Hz
    NTSC_SR = 8287  # Hz (reference pitch C-5)

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

    @property
    def file_extension(self) -> str:
        """File extension used when saving MOD songs."""
        return 'mod'

    def __init__(self):
        """
        Initializes the song with one empty pattern and an empty sample bank.
        This way, the song can be immediately saved as a valid module file.
        """
        super().__init__()

        self.patterns = [Pattern(n_rows=MODSong.ROWS, n_channels=MODSong.CHANNELS)]
        self.pattern_seq = [0]
        
        # MOD files store samples directly (notes reference samples by index).
        # We always store the maximum allowed slots, possibly with empty slots.
        self.samples = [Sample() for _ in range(MODSong.SAMPLES)]
        self.n_actual_samples = 0  # The number of non-empty samples present in the song.
        self._restart_position_raw = 127

        # Mutation-versioned caches for MOD sample-memory resolution.
        self._resolution_version = 0
        self._resolved_sequence_cache_version = -1
        self._resolved_reachable_cache_version = -1
        self._resolved_sequence_cells: dict[tuple[int, int, int], int] = {}
        self._resolved_sequence_first_use: list[int] = []
        self._resolved_reachable_cells: dict[tuple[int, int], int] = {}
        self._resolved_reachable_first_use: list[int] = []
        self._resolved_reachable_rows: tuple[PlaybackRowView, ...] = ()

    def _update_n_actual_samples(self) -> None:
        """Refresh the cached count of non-empty sample slots."""
        self.n_actual_samples = sum(1 for sample in self.samples if len(sample.waveform) > 0)

    def _on_mutation(self) -> None:
        """Invalidate MOD resolved-sample caches after note/sequence mutations."""
        self._resolution_version += 1

    @staticmethod
    def _resolve_effective_sample(raw_sample: int, period: str, latched_sample: int) -> tuple[int, int]:
        """Resolve one MOD cell's effective sample using channel-local sample memory."""
        if raw_sample > 0:
            latched_sample = raw_sample
        if period != '':
            if raw_sample > 0:
                return raw_sample, latched_sample
            return latched_sample, latched_sample
        return raw_sample, latched_sample

    def _ensure_sequence_resolution_cache(self) -> None:
        """Build or reuse the sequence-scope effective-sample cache."""
        if self._resolved_sequence_cache_version == self._resolution_version:
            return
        latched = [0] * MODSong.CHANNELS
        cells: dict[tuple[int, int, int], int] = {}
        seen: set[int] = set()
        first_use: list[int] = []

        for sequence_idx, pattern_idx in enumerate(self.pattern_seq):
            if pattern_idx < 0 or pattern_idx >= len(self.patterns):
                continue
            pat = self.patterns[pattern_idx]
            n_channels = min(pat.n_channels, MODSong.CHANNELS)
            for row in range(pat.n_rows):
                for channel in range(n_channels):
                    note = pat.data[channel][row]
                    raw_sample = int(getattr(note, 'instrument_idx', 0))
                    period = getattr(note, 'period', '')
                    effective_sample, latched[channel] = self._resolve_effective_sample(
                        raw_sample,
                        period,
                        latched[channel],
                    )
                    cells[(sequence_idx, row, channel)] = effective_sample
                    used_sample = effective_sample if period != '' else raw_sample
                    if used_sample > 0 and used_sample not in seen:
                        seen.add(used_sample)
                        first_use.append(used_sample)

        self._resolved_sequence_cells = cells
        self._resolved_sequence_first_use = first_use
        self._resolved_sequence_cache_version = self._resolution_version

    def _ensure_reachable_resolution_cache(self) -> None:
        """Build or reuse the reachable-scope effective-sample cache."""
        if self._resolved_reachable_cache_version == self._resolution_version:
            return
        played_rows = tuple(self.iter_playback_rows())
        latched = [0] * MODSong.CHANNELS
        cells: dict[tuple[int, int], int] = {}
        seen: set[int] = set()
        first_use: list[int] = []

        for played_row in played_rows:
            pattern_idx = played_row.pattern_idx
            if pattern_idx < 0 or pattern_idx >= len(self.patterns):
                continue
            pat = self.patterns[pattern_idx]
            row = played_row.row
            if row < 0 or row >= pat.n_rows:
                continue
            n_channels = min(pat.n_channels, MODSong.CHANNELS)
            for channel in range(n_channels):
                note = pat.data[channel][row]
                raw_sample = int(getattr(note, 'instrument_idx', 0))
                period = getattr(note, 'period', '')
                effective_sample, latched[channel] = self._resolve_effective_sample(
                    raw_sample,
                    period,
                    latched[channel],
                )
                cells[(played_row.visit_idx, channel)] = effective_sample
                used_sample = effective_sample if period != '' else raw_sample
                if used_sample > 0 and used_sample not in seen:
                    seen.add(used_sample)
                    first_use.append(used_sample)

        self._resolved_reachable_cells = cells
        self._resolved_reachable_first_use = first_use
        self._resolved_reachable_rows = played_rows
        self._resolved_reachable_cache_version = self._resolution_version

    @staticmethod
    def period_to_note(period_value: int) -> str:
        """Convert a raw MOD period value into note text."""
        if period_value not in MODSong.PERIOD_TABLE:
            raise ValueError(f"Unknown period value {period_value}.")
        return MODSong.PERIOD_TABLE[period_value]

    @staticmethod
    def note_to_period(note_str: str) -> int:
        """Convert note text into a raw MOD period value."""
        if note_str not in MODSong.INV_PERIOD_TABLE:
            raise ValueError(f"Unknown MOD note {note_str!r}.")
        return MODSong.INV_PERIOD_TABLE[note_str]

    def copy(self) -> MODSong:
        """
        Creates a deep copy of this song.
        
        :return: A new MODSong instance with all data copied.
        """
        new_song = MODSong()
        new_song.artist = self.artist
        new_song.songname = self.songname
        new_song.patterns = copy.deepcopy(self.patterns)
        new_song.pattern_seq = copy.deepcopy(self.pattern_seq)
        new_song.samples = copy.deepcopy(self.samples)
        new_song.n_actual_samples = self.n_actual_samples
        new_song._restart_position_raw = self._restart_position_raw

        return new_song

    @property
    def restart_position(self) -> int | None:
        """Return normalized MOD restart position (127 maps to None)."""
        return self.get_restart_position(raw=False)

    def get_restart_position(self, raw: bool = False) -> int | None:
        """Return MOD restart position as raw header byte or normalized value."""
        value = self._restart_position_raw
        if raw:
            return value
        if value == 127:
            return None
        return value

    def set_restart_position(self, position: int | None, *, raw: bool = False) -> None:
        """Set MOD restart position as normalized value or raw header byte."""
        if position is None:
            self._restart_position_raw = 127
            return
        if not isinstance(position, int):
            raise TypeError(f"Invalid restart position type {type(position).__name__} (expected int or None).")
        if position < 0 or position > 255:
            raise ValueError(f"Invalid restart position {position} (expected 0-255).")
        if raw:
            self._restart_position_raw = position
            return
        self._restart_position_raw = position

    '''
    -------------------------------------
    IMPORT AND EXPORT
    -------------------------------------
    '''

    def load(self, fname: str, verbose: bool = True, *, metadata_from_filename: bool = False):
        """
        Loads a song from a standard MOD file.

        :param fname: The path to the module file.
        :param verbose: False for silent loading.
        :param metadata_from_filename: When True, override metadata after load
            using ``artist - title`` parsing from the file path. By default,
            metadata is read from MOD file bytes.
        :return: None.
        """

        if verbose:
            print(f'Loading {fname}... ', end='', flush=True)

        with (open(fname, 'rb') as mod_file):

            data = bytearray(mod_file.read())

            # TODO: check if the MOD file is in packed format (never happened so far)

            def _decode_header_bytes(raw: bytes) -> str:
                try:
                    return raw.decode('utf-8')
                except UnicodeDecodeError:
                    return raw.decode('latin-1')

            self.artist = "Unknown Artist"
            self.songname = _decode_header_bytes(bytes(data[:20])).rstrip('\x00')

            magic_string = _decode_header_bytes(data[1080:1080 + 4])
            accepted_magic = {"M.K.", "M!K!", "FLT4"}  # 4-channel variants
            if magic_string not in accepted_magic:  # non-standard mod file
                raise NotImplementedError(f"Unsupported MOD format {magic_string}. Supported: M.K., M!K!, FLT4.")

            # ----------------------------
            # Load pattern preamble data
            # ----------------------------

            song_length = data[950]  # song length in patterns
            self._restart_position_raw = data[951]
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
                smp.name = _decode_header_bytes(data[idx - 22:idx].rstrip(b'\x00'))

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
                raise NotImplementedError(f"Unexpected extra patterns: {n_extra_patterns} (non-standard MOD).")

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

                        note.instrument_idx = MODSong.get_sample_from_note(note_raw)
                        note.period = MODSong.get_period_from_note(note_raw)

                        e_type, e_param = MODSong.get_effect_from_note(note_raw)

                        if e_type != 0 or e_param != 0:

                            # dirty way for converting hex number to string... e.g. 0xF1 -> "F1"
                            note.effect = hex(e_type).lstrip("0x").upper() 
                            # note.effect += hex(e_param)[2:].upper()  # dunno why i was doing this...
                            note.effect += f"{e_param:02X}"
                            
                            if e_type == 0:  # arpeggio effect
                                note.effect = "0" + note.effect

                        pat.data[c][r] = note

                self.patterns.append(pat)

        if metadata_from_filename:
            self.artist, self.songname = Song.artist_songname_from_filename(fname)

        self._on_mutation()
        if verbose:
            print('done.')

    def save_ascii(self, fname: str, verbose: bool = True):
        """
        Writes the song as readable text with ASCII encoding.

        :param fname: Complete file path.
        :param verbose: False for silent saving.
        :return: None.
        """
        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)
        with open(fname, 'w', encoding='ascii') as file:
            file.write(self.to_ascii())
        if verbose:
            print('done.')

    def save(self, fname: str, verbose: bool = True, *, validate_samples: bool = False):
        """
        Saves the song as a standard MOD file.

        :param fname: Complete file path.
        :param verbose: False for silent saving.
        :param validate_samples: When True, fail early if any sample loop
            metadata is out of bounds.
        :return: None.
        """

        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)

        if validate_samples:
            self.validate_samples()

        if len(self.pattern_seq) == 0 or len(self.pattern_seq) > 128 or len(self.patterns) > 128:
            raise OverflowError(f"Too many patterns (sequence: {len(self.pattern_seq)}, unique: {len(self.patterns)}). MOD supports up to 128.")

        data = bytearray()

        def str_to_bytes_padded(s: str, max_len: int) -> bytes:
            r = bytes(s, 'utf-8')
            # If UTF-8 would overflow, prefer Latin-1 when it can preserve byte-width.
            if len(r) > max_len:
                try:
                    r_latin1 = bytes(s, 'latin-1')
                    if len(r_latin1) <= max_len:
                        r = r_latin1
                except UnicodeEncodeError:
                    pass
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
                raise ValueError(f"Sample length {int(len(waveform) / 2)} words exceeds MOD max 65536 words (128 KB).")
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
        data += int(self._restart_position_raw).to_bytes(1)
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
                raise NotImplementedError(f"Cannot save MOD with {len(pat.data)} channels (expected 4).")

            if len(pat.data[0]) != MODSong.ROWS:
                raise NotImplementedError(f"Cannot save MOD with {len(pat.data[0])} rows (expected 64).")

            for r in range(MODSong.ROWS):
                for c in range(MODSong.CHANNELS):

                    note = pat.data[c][r]

                    efx_type = 0x0
                    efx_param = 0x0
                    if note.effect != "":
                        efx_type = int(note.effect[0], 16)  # interpret the character as a hex digit
                        efx_param = int(note.effect[1:], 16)

                    if note.period != '':
                        if note.period not in MODSong.INV_PERIOD_TABLE:
                            raise ValueError(f"Unknown note period '{note.period}' in pattern {p}, row {r}, channel {c}.")
                        pd = MODSong.INV_PERIOD_TABLE[note.period]
                    else:
                        pd = 0

                    note_raw = bytearray(4)
                    note_raw[0] = (note.instrument_idx & 0xF0) | ((pd & 0xF00) >> 8)
                    note_raw[1] = pd & 0xFF
                    note_raw[2] = ((note.instrument_idx & 0x0F) << 4) | efx_type
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

    # Alias for backwards compatibility
    
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
        """Yield visited MOD rows with source coordinates and timing metadata."""
        if max_steps <= 0:
            raise ValueError(f"Invalid max_steps {max_steps} (expected > 0).")

        bpm = 125
        speed = 6
        d = Song.get_tick_duration(bpm)

        jump_to_position = -1
        jump_to_pattern = -1
        stop_song = False
        self_jump_count = 0
        self_jump_limit = 5
        start_row = 0
        elapsed = 0.0
        visit_idx = 0

        seq_idx = 0
        while seq_idx < len(self.pattern_seq):
            p = self.pattern_seq[seq_idx]
            loop_start_row = [0] * MODSong.CHANNELS
            loop_count = [0] * MODSong.CHANNELS

            r = start_row
            while r < MODSong.ROWS:
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

                for c in range(MODSong.CHANNELS):
                    efx = self.patterns[p].data[c][r].effect
                    if efx == "":
                        continue

                    if efx[0] == "F":
                        v = int(efx[1:], 16)
                        if v <= 31:
                            if v != 0:
                                speed = v
                        else:
                            bpm = v
                        d = Song.get_tick_duration(bpm)

                    elif efx[0] == "D":
                        if len(efx) >= 3:
                            hi = int(efx[1], 16)
                            lo = int(efx[2], 16)
                            if len(self.pattern_seq) > 1:
                                pending_jump_row = hi * 10 + lo

                    elif efx[0] == "B":
                        dest = int(efx[1:], 16)
                        if dest < len(self.pattern_seq):
                            if dest > seq_idx:
                                pending_jump_pattern = dest
                            elif dest == seq_idx:
                                saw_self_jump = True
                            else:
                                stop_song = True

                    elif efx[0] == "E" and len(efx) >= 3:
                        cmd = efx[1].upper()
                        val = int(efx[2], 16)
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
        Computes the timestamp of each row in the song.
        Takes into account speed / bpm changes, pattern breaks, and position jumps.


        :return: A list where each element is a list corresponding to pattern in the sequence.
                 Within each list, each row is a triple (timestamp [s], speed, bpm).
        """

        # default timing for MOD files, if nothing is specified
        bpm = 125
        speed = 6  # ticks per row

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

        seq_idx = 0
        while seq_idx < len(self.pattern_seq):

            p = self.pattern_seq[seq_idx]

            # we annotate each pattern separately
            pattern_timestamps = []
            pattern_speeds = []
            pattern_bpms = []

            loop_start_row = [0] * MODSong.CHANNELS
            loop_count = [0] * MODSong.CHANNELS

            r = start_row
            while r < MODSong.ROWS:

                # reset the jump flags
                if jump_to_position != -1:
                    jump_to_position = -1
                    start_row = 0

                row_delay = 0
                loop_jump_row = None
                pending_jump_row = None
                pending_jump_pattern = None
                saw_self_jump = False

                for c in range(MODSong.CHANNELS):    

                    efx = self.patterns[p].data[c][r].effect
                    if efx != "":
                     
                        if efx[0] == "F":  # change of speed or bpm

                            v = int(efx[1:], 16)
                            if v <= 31:
                                if v != 0:
                                    speed = v
                            else:
                                bpm = v

                            d = Song.get_tick_duration(bpm)
                            # print(f"CHANGE: Pattern {p}, row {r}, channel {c}, speed {speed}, bpm {bpm}, tick duration {d}")

                        elif efx[0] == "D":  # jump to a specific row in the next pattern
                            if len(efx) >= 3:
                                hi = int(efx[1], 16)
                                lo = int(efx[2], 16)
                                if len(self.pattern_seq) > 1:
                                    pending_jump_row = hi * 10 + lo

                        elif efx[0] == "B":  # break to a specific pattern
                            dest = int(efx[1:], 16)
                            if dest < len(self.pattern_seq):
                                if dest > seq_idx:
                                    pending_jump_pattern = dest
                                elif dest == seq_idx:
                                    saw_self_jump = True
                                else:
                                    stop_song = True
                        
                        elif efx[0] == "E" and len(efx) >= 3:
                            cmd = efx[1].upper()
                            val = int(efx[2], 16)
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

                # Allow limited self-jumps only when combined with Dxx in the same row.
                if saw_self_jump and pending_jump_row is not None and self_jump_count < self_jump_limit:
                    pending_jump_pattern = seq_idx
                    self_jump_count += 1

                # If a pattern loop is triggered, it takes precedence over jumps.
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

        # cumsum over the entire list of lists
        cum = 0
        for p in range(len(timestamps)):
            for r in range(len(timestamps[p])):
                cum += timestamps[p][r]
                timestamps[p][r] = (cum, speeds[p][r], bpms[p][r])

        return timestamps
    
    '''
    -------------------------------------
    SAMPLES AND INSTRUMENTS
    -------------------------------------
    '''

    def load_sample(self, fname: str, sample_idx: int | None = None) -> tuple[int, Sample]:
        """
        Loads a sample from a WAV file, and stores it at the given sample index.

        :param fname: The complete file path to the .wav file.
        :param sample_idx: The destination 1-based sample index, or None to use the next empty slot.
        :return: A tuple (int, Sample) containing:
                 - the index of the added sample, from 1 to 31
                 - the corresponding sample object
        """
        if sample_idx is not None and (sample_idx <= 0 or sample_idx > MODSong.SAMPLES):
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-31).")
        
        if sample_idx is None:
            for i in range(MODSong.SAMPLES):
                if len(self.samples[i].waveform) == 0:
                    sample_idx = i + 1
                    break
            if sample_idx is None:
                raise ValueError("No empty sample slots available (1-31 are full).")

        self.samples[sample_idx - 1] = Sample()  # reset all attributes

        audio = pydub.AudioSegment.from_wav(fname).set_channels(1)
        if audio.sample_width != 1:
            audio = audio.set_sample_width(1)

        self.samples[sample_idx - 1].waveform = audio.get_array_of_samples()
        self._update_n_actual_samples()

        return sample_idx, self.samples[sample_idx - 1]

    def load_sample_from_raw(self, raw_samples: list[float], sample_idx: int | None = None, input_sr: int = NTSC_SR) -> tuple[int, Sample]:
        """
        Loads a sample from a raw list of samples, and stores it at the given sample index.
        The raw samples should be a mono list of normalized float values in the range [-1.0, 1.0].

        If the sample rate is different from NTSC_SR, the samples will be resampled to NTSC_SR.

        :param raw_samples: Mono normalized PCM samples in the range [-1.0, 1.0].
        :param sample_idx: The destination 1-based sample index, or None to use the next empty slot.
        :param input_sr: The sample rate of the input raw samples (default NTSC_SR).
        :return: A tuple (int, Sample) containing:
                 - the index of the added sample, from 1 to 31
                 - the corresponding sample object
        """
        if sample_idx is not None and (sample_idx <= 0 or sample_idx > MODSong.SAMPLES):
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-31).")
        
        if sample_idx is None:
            for i in range(MODSong.SAMPLES):
                if len(self.samples[i].waveform) == 0:
                    sample_idx = i + 1
                    break
            if sample_idx is None:
                raise ValueError("No empty sample slots available (1-31 are full).")

        self.samples[sample_idx - 1] = Sample()  # reset all attributes

        # convert floats to signed 8-bit PCM bytes
        pcm_bytes = bytearray()
        for s in raw_samples:
            # clip for safety
            if s > 1.0:
                s = 1.0
            elif s < -1.0:
                s = -1.0

            # scale to signed int8 (-128..127)
            pcm_bytes.append(struct.pack('b', int(s * 127))[0])

        # create AudioSegment
        audio = pydub.AudioSegment(
            data=bytes(pcm_bytes),
            sample_width=1,   # 8-bit signed PCM
            frame_rate=input_sr,
            channels=1
        )
        audio = audio.set_frame_rate(self.NTSC_SR)

        self.samples[sample_idx - 1].waveform = audio.get_array_of_samples()
        self._update_n_actual_samples()

        return sample_idx, self.samples[sample_idx - 1]
    
    def _get_effective_sample_rate(self, smp: Sample, period: str = "C-5") -> int:
        """
        Return the effective playback sample rate of a MOD sample.

        MOD stores finetune as a 4-bit value that shifts pitch in steps of one
        eighth of a semitone. This helper converts that tuning plus a reference
        note such as ``C-5`` into the effective WAV sample rate needed to hear
        the sample at that pitch.

        :param smp: The sample object.
        :param period: Note text used as the playback reference pitch.
        :return: The effective sample rate in Hz.
        """
        # calculate the base frequency for the reference period.
        # frequency = PAL_CLOCK / (period * 2)
        base_freq = MODSong.PAL_CLOCK / (MODSong.INV_PERIOD_TABLE[period] * 2)
        
        # account for finetune (-8 to +7, stored as 0-15).
        # finetune shifts the pitch by 1/8 of a semitone per unit
        finetune = smp.finetune
        if finetune > 7:  # Convert from 0-15 to -8 to +7
            finetune = finetune - 16
        
        # each finetune unit is 1/8 of a semitone
        # 2^(finetune / (8 * 12)) gives the frequency multiplier
        finetune_multiplier = 2 ** (finetune / 96.0)
        
        # calculate the effective sample rate
        effective_sample_rate = int(base_freq * finetune_multiplier)

        return effective_sample_rate
    
    def set_sample_name(self, sample_idx: int, name: str) -> None:
        """Set the name of one MOD sample slot."""
        smp = self.get_sample(sample_idx)
        smp.name = name

    def set_sample_volume(self, sample_idx: int, volume: int) -> None:
        """Set the volume of one MOD sample slot."""
        if volume < 0 or volume > 64:
            raise ValueError(f"Invalid volume {volume} (expected 0-64).")
        smp = self.get_sample(sample_idx)
        smp.volume = volume

    def set_sample_finetune(self, sample_idx: int, finetune: int) -> None:
        """
        Sets the raw MOD finetune nibble.

        MOD stores finetune as 0-15, where 8-15 correspond to musical values -8 to -1.
        """
        if finetune < 0 or finetune > 15:
            raise ValueError(f"Invalid finetune {finetune} (expected 0-15).")
        smp = self.get_sample(sample_idx)
        smp.finetune = finetune

    def set_sample_loop(self, sample_idx: int, start: int, length: int) -> None:
        """Set the loop start and length for one MOD sample slot."""
        smp = self.get_sample(sample_idx)
        smp.repeat_point = max(0, start)
        smp.repeat_len = max(0, length)

    def get_sample_pcm_i8(self, sample_idx: int) -> bytes:
        """Return raw signed 8-bit PCM bytes for one MOD sample slot."""
        smp = self.get_sample(sample_idx)
        return smp.waveform.tobytes()

    def set_sample_pcm_i8(
        self,
        sample_idx: int,
        pcm_i8: bytes | bytearray | memoryview | array.array,
        *,
        reset_meta: bool = False,
    ) -> None:
        """Set one MOD sample waveform from raw signed 8-bit PCM bytes."""
        smp = self.get_sample(sample_idx)

        raw: bytes
        if isinstance(pcm_i8, array.array):
            if pcm_i8.typecode not in {'b', 'B'}:
                raise TypeError(f"Invalid PCM array typecode {pcm_i8.typecode!r} (expected 'b' or 'B').")
            raw = pcm_i8.tobytes()
        elif isinstance(pcm_i8, (bytes, bytearray, memoryview)):
            raw = bytes(pcm_i8)
        else:
            raise TypeError(
                f"Invalid pcm_i8 type {type(pcm_i8).__name__} "
                "(expected bytes, bytearray, memoryview, or array.array)."
            )

        smp.waveform = array.array('b')
        smp.waveform.frombytes(raw)

        if reset_meta:
            smp.name = ""
            smp.finetune = 0
            smp.volume = 64
            smp.repeat_point = 0
            smp.repeat_len = 0
            smp.tune = ''

        self._update_n_actual_samples()

    def set_sample_loop_bytes(self, sample_idx: int, start_byte: int, length_byte: int) -> None:
        """Set MOD sample loop points using explicit byte units."""
        if start_byte < 0:
            raise ValueError(f"Invalid loop start {start_byte} (expected >= 0).")
        if length_byte < 0:
            raise ValueError(f"Invalid loop length {length_byte} (expected >= 0).")
        smp = self.get_sample(sample_idx)
        smp.repeat_point = start_byte
        smp.repeat_len = length_byte

    def sanitize_samples(self, *, mode: str = "coerce") -> None:
        """Sanitize loop metadata for all MOD sample slots.

        This is a convenience wrapper around ``Sample.sanitize_loop``.
        """
        for smp in self.samples:
            smp.sanitize_loop(mode=mode)

    def validate_samples(self) -> None:
        """Validate loop metadata for all MOD sample slots.

        :raises ValueError: If any slot contains invalid loop metadata.
        """
        for sample_idx, smp in enumerate(self.samples, start=1):
            try:
                smp.validate_loop()
            except ValueError as exc:
                raise ValueError(f"Sample {sample_idx}: {exc}") from exc

    def validate_sample_loop(self, sample_idx: int) -> None:
        """Validate one MOD sample loop using canonical safety rules.

        This delegates to ``Sample.validate_loop`` for the selected slot.

        :param sample_idx: The 1-based sample index to validate.
        """
        smp = self.get_sample(sample_idx)
        smp.validate_loop()


    def get_sample_duration(self, sample_idx: int, period: str = "C-5") -> float:
        """
        Return the playback duration of a MOD sample in seconds.

        Duration depends on the effective playback rate, so the same waveform can
        produce a different result for a different reference note or finetune.

        :param sample_idx: The sample index to query, 1 to 31.
        :param period: Note text used as the playback reference pitch.
        :return: The sample duration in seconds.
        """
        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-31).")

        smp = self.samples[sample_idx - 1]

        if len(smp.waveform) == 0:
            raise ValueError(f"Sample {sample_idx} has no waveform data")
        
        effective_sample_rate = self._get_effective_sample_rate(smp, period)
        duration = len(smp.waveform) / effective_sample_rate

        return duration

    def save_sample(self, sample_idx: int, fname: str, period: str = "C-5", force_sample_rate: int = None):
        """
        Save one MOD sample as a WAV file.

        By default the exported WAV uses the effective playback sample rate implied
        by the MOD finetune and the chosen reference pitch. ``force_sample_rate``
        can be used to resample the exported audio to a fixed output rate instead.

        :param sample_idx: The sample index to save, 1 to 31.
        :param fname: The complete file path to the output .wav file.
        :param period: Note text used as the playback reference pitch.
        :param force_sample_rate: Optional output WAV rate to resample to.
        :return: None.
        """
        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-31).")

        smp = self.samples[sample_idx - 1]

        if len(smp.waveform) == 0:
            raise ValueError(f"Sample {sample_idx} has no waveform data")
        
        effective_sample_rate = self._get_effective_sample_rate(smp, period)
        
        audio = pydub.AudioSegment(
            data=smp.waveform.tobytes(),
            sample_width=1,
            frame_rate=effective_sample_rate,
            channels=1
        )
        
        # export at the specified sample rate
        if force_sample_rate is not None and force_sample_rate != effective_sample_rate:
            audio = audio.set_frame_rate(force_sample_rate)
        
        audio.export(fname, format="wav")

    def get_sample(self, sample_idx: int) -> Sample:
        """
        Returns the sample object at the given index.
        
        :param sample_idx: The sample index to retrieve, 1 to 31.
        :return: The sample object.
        """
        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-31).")

        return self.samples[sample_idx - 1]

    def list_samples(self) -> list[Sample]:
        """
        Return the fixed 31-slot MOD sample bank.

        Empty sample slots are included in the returned list.

        :return: The ordered list of MOD sample slots.
        """
        return self.samples

    def set_sample(self, sample_idx: int, sample: Sample) -> None:
        """
        Replaces the sample at the given index.
        """
        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-31).")
        self.samples[sample_idx - 1] = sample
        self._update_n_actual_samples()

    def copy_sample_from(self, src: 'MODSong', src_sample_idx: int, dst_sample_idx: int | None = None) -> int:
        """
        Copies a single sample from another MODSong into this song.

        :param src: The source MOD song.
        :param src_sample_idx: The 1-based sample index to copy from the source.
        :param dst_sample_idx: The destination 1-based sample index, or None to use the next empty slot.
        :return: The destination 1-based sample index.
        """
        if src_sample_idx <= 0 or src_sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid source sample index {src_sample_idx} (expected 1-31).")
        if dst_sample_idx is not None and (dst_sample_idx <= 0 or dst_sample_idx > MODSong.SAMPLES):
            raise IndexError(f"Invalid destination sample index {dst_sample_idx} (expected 1-31).")

        src_smp = src.samples[src_sample_idx - 1]
        new_smp = Sample()
        new_smp.name = src_smp.name
        new_smp.finetune = src_smp.finetune
        new_smp.volume = src_smp.volume
        new_smp.repeat_point = src_smp.repeat_point
        new_smp.repeat_len = src_smp.repeat_len
        new_smp.waveform = src_smp.waveform.__class__(src_smp.waveform.typecode, src_smp.waveform)
        new_smp.tune = src_smp.tune

        if dst_sample_idx is None:
            for i in range(MODSong.SAMPLES):
                if len(self.samples[i].waveform) == 0:
                    dst_sample_idx = i + 1
                    break
            if dst_sample_idx is None:
                raise ValueError("Couldn't find an empty slot for the new sample.")

        self.samples[dst_sample_idx - 1] = new_smp
        self._update_n_actual_samples()
        return dst_sample_idx

    def copy_samples_from(self, src: 'MODSong', src_sample_indices: list[int]) -> list[int]:
        """
        Copy multiple samples from another MOD song.

        Samples are inserted one by one into the next available empty slots.

        :param src: The source MOD song.
        :param src_sample_indices: 1-based source sample indices to copy.
        :return: The destination 1-based sample indices in copy order.
        """
        new_indices: list[int] = []
        for idx in src_sample_indices:
            new_indices.append(self.copy_sample_from(src, idx, None))
        return new_indices

    def remove_sample(self, sample_idx: int):
        """
        Deletes the sample from the sample bank.
        WARNING: This does not remove the sample notes from the song. The notes will stay, but will play mute.

        :param sample_idx: The sample index to remove, 1 to 31.
        :return: None.
        """
        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-31).")

        self.samples[sample_idx - 1] = Sample()
        self._update_n_actual_samples()

    def keep_sample(self, sample_idx: int):
        """
        Deletes all samples in the sample bank, except for the one specified by the given index.
        WARNING: This does not remove the sample notes from the song. The notes will stay, but will play mute.

        :param sample_idx: The sample index to be kept, 1 to 31.
        :return: None.
        """
        if sample_idx <= 0 or sample_idx > MODSong.SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-31).")

        for s in range(MODSong.SAMPLES):
            if s + 1 != sample_idx:
                self.samples[s] = Sample()
        self._update_n_actual_samples()

    def get_used_samples(
        self,
        *,
        scope: str = "sequence",
        order: str = "sorted",
        resolved: bool = True,
    ) -> list[int]:
        """Return MOD sample indices referenced under sequence or reachable scope.

        Definitions:
        - raw sample: the 4+4-bit sample number stored in a MOD cell (`00..31`)
        - effective sample: channel-memory resolved sample used when a note period
          is present with raw sample `00`

        By default (`resolved=True`), MOD sample-memory semantics are applied:
        note rows with raw `00` inherit the channel's most recently latched sample,
        including across pattern boundaries in sequence/reachable traversal order.
        If no sample has been latched yet in that channel, effective sample is `0`.

        `resolved=False` returns legacy raw-cell behavior.
        """
        self._validate_used_resource_args(scope, order)
        if resolved:
            if scope == "sequence":
                self._ensure_sequence_resolution_cache()
                first_use = list(self._resolved_sequence_first_use)
            else:
                self._ensure_reachable_resolution_cache()
                first_use = list(self._resolved_reachable_first_use)
            return self._finalize_used_values(first_use, order)

        seen: set[int] = set()
        first_use: list[int] = []
        for note in self._iter_notes_by_scope(scope):
            sample_idx = getattr(note, 'instrument_idx', 0)
            if sample_idx > 0 and sample_idx not in seen:
                seen.add(sample_idx)
                first_use.append(sample_idx)
        return self._finalize_used_values(first_use, order)

    def iter_cells(self, *, sequence_only: bool = True, resolved: bool = True):
        """Yield immutable MOD cell snapshots.

        With ``resolved=True`` (default) and ``sequence_only=True``, returned
        ``CellView.instrument_idx`` uses MOD effective sample-memory semantics.
        With ``resolved=False`` or ``sequence_only=False``, raw stored sample
        nibbles are exposed.
        """
        if not resolved or not sequence_only:
            yield from super().iter_cells(sequence_only=sequence_only)
            return

        self._ensure_sequence_resolution_cache()
        for sequence_idx, pattern_idx in self._iter_pattern_entries(sequence_only=True):
            pat = self.patterns[pattern_idx]
            for row in range(pat.n_rows):
                for channel in range(pat.n_channels):
                    raw_note = pat.data[channel][row]
                    raw_sample = int(getattr(raw_note, 'instrument_idx', 0))
                    effective_sample = self._resolved_sequence_cells.get((sequence_idx, row, channel), raw_sample)
                    view_note = raw_note
                    if effective_sample != raw_sample:
                        view_note = Note(effective_sample, raw_note.period, raw_note.effect)
                    yield self._make_cell_view(
                        sequence_idx=sequence_idx,
                        pattern_idx=pattern_idx,
                        row=row,
                        channel=channel,
                        note=view_note,
                    )

    def iter_rows(
        self,
        *,
        sequence_only: bool = True,
        reachable_only: bool = False,
        resolved: bool = True,
    ):
        """Yield immutable MOD row snapshots.

        With ``resolved=True`` (default), ``CellView.instrument_idx`` follows
        MOD effective sample-memory semantics under sequence or reachable
        traversal. With ``resolved=False``, raw stored sample nibbles are used.
        """
        if not resolved:
            yield from super().iter_rows(sequence_only=sequence_only, reachable_only=reachable_only)
            return

        if reachable_only:
            self._ensure_reachable_resolution_cache()
            for played_row in self._resolved_reachable_rows:
                if played_row.pattern_idx < 0 or played_row.pattern_idx >= len(self.patterns):
                    continue
                pat = self.patterns[played_row.pattern_idx]
                if played_row.row < 0 or played_row.row >= pat.n_rows:
                    continue
                cells: list[CellView] = []
                for channel in range(pat.n_channels):
                    raw_note = pat.data[channel][played_row.row]
                    raw_sample = int(getattr(raw_note, 'instrument_idx', 0))
                    effective_sample = self._resolved_reachable_cells.get((played_row.visit_idx, channel), raw_sample)
                    view_note = raw_note
                    if effective_sample != raw_sample:
                        view_note = Note(effective_sample, raw_note.period, raw_note.effect)
                    cells.append(
                        self._make_cell_view(
                            sequence_idx=played_row.sequence_idx,
                            pattern_idx=played_row.pattern_idx,
                            row=played_row.row,
                            channel=channel,
                            note=view_note,
                        )
                    )
                yield RowView(
                    sequence_idx=played_row.sequence_idx,
                    pattern_idx=played_row.pattern_idx,
                    row=played_row.row,
                    cells=tuple(cells),
                )
            return

        if not sequence_only:
            yield from super().iter_rows(sequence_only=False, reachable_only=False)
            return

        self._ensure_sequence_resolution_cache()
        for sequence_idx, pattern_idx in self._iter_pattern_entries(sequence_only=True):
            pat = self.patterns[pattern_idx]
            for row in range(pat.n_rows):
                cells: list[CellView] = []
                for channel in range(pat.n_channels):
                    raw_note = pat.data[channel][row]
                    raw_sample = int(getattr(raw_note, 'instrument_idx', 0))
                    effective_sample = self._resolved_sequence_cells.get((sequence_idx, row, channel), raw_sample)
                    view_note = raw_note
                    if effective_sample != raw_sample:
                        view_note = Note(effective_sample, raw_note.period, raw_note.effect)
                    cells.append(
                        self._make_cell_view(
                            sequence_idx=sequence_idx,
                            pattern_idx=pattern_idx,
                            row=row,
                            channel=channel,
                            note=view_note,
                        )
                    )
                yield RowView(
                    sequence_idx=sequence_idx,
                    pattern_idx=pattern_idx,
                    row=row,
                    cells=tuple(cells),
                )
    
    '''
    -------------------------------------
    PATTERNS
    -------------------------------------
    '''

    def resize_pattern(self, pattern: int, n_rows: int) -> None:
        """
        MOD patterns have a fixed length of 64 rows; resizing is not supported.
        """
        if n_rows != MODSong.ROWS:
            raise ValueError(f"MOD patterns have fixed 64 rows (got {n_rows}).")


    def add_to_sequence(self, pattern_idx: int, sequence_position: int | None = None) -> None:
        """Insert a pattern into the MOD order list, enforcing the 128-order limit."""
        if len(self.pattern_seq) + 1 > 128:
            raise ValueError(f"Pattern sequence too long ({len(self.pattern_seq) + 1}). MOD supports up to 128.")
        super().add_to_sequence(pattern_idx, sequence_position)


    def set_sequence(self, seq: list[int]) -> None:
        """Replace the MOD order list, enforcing the 128-order limit."""
        if len(seq) > 128:
            raise ValueError(f"Pattern sequence too long ({len(seq)}). MOD supports up to 128.")
        super().set_sequence(seq)


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
        for r in range(MODSong.ROWS):
            for c in range(MODSong.CHANNELS):
                self.patterns[p].data[c][r] = Note()
        self._on_mutation()

    def add_pattern(self) -> int:
        """
        Creates a brand new pattern, appends it to the pattern pool, and adds that pool index to the song sequence.

        :return: The new pattern pool index.
        """
        self.patterns.append(Pattern(MODSong.ROWS, MODSong.CHANNELS))
        n = len(self.patterns) - 1
        self.pattern_seq.append(n)
        self._on_mutation()

        return n
        
    def get_effective_row_count(self, sequence_idx: int, include_loops: bool = True) -> int:
        """
        Returns the effective number of rows that get played in a sequence pattern.
        Accounts for position jumps, loops, and breaks.

        TODO: Implement a version for the entire song. 
              It's not so trivial, because of position jumps effects (Dxx) and such.

        :param sequence_idx: The 0-based sequence index to inspect.
        :param include_loops: True to count rows replayed by ``E6x`` pattern loops.
        :return: The effective number of rows that gets played in the pattern.
        """
        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        loop_start_row = 0  # used by E6x effect

        data = copy.deepcopy(self.patterns[self.pattern_seq[sequence_idx]].data)

        unrolled_data = [[] for _ in range(MODSong.CHANNELS)]

        for r in range(MODSong.ROWS):

            interrupt = False  # if true, the pattern is cut short by Bxx or Dxx effects

            for c in range(MODSong.CHANNELS):                

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

    def add_channel(self, count: int = 1) -> None:
        """Reject channel growth because standard MOD has a fixed 4-channel layout."""
        if count <= 0:
            raise ValueError(f"Invalid channel count {count} (expected >=1).")
        raise NotImplementedError("MOD format has fixed 4 channels; cannot add channels.")

    def remove_channel(self, channel: int) -> None:
        """Reject channel removal because standard MOD has a fixed 4-channel layout."""
        raise NotImplementedError("MOD format has fixed 4 channels; cannot remove channels.")

    def clear_channel(self, channel: int):
        """
        Clears completely a specified channel in the entire song.
        WARNING: Don't use this as a way to mute channels, as it also removes global effects.

        :param channel: The channel index to mute, 0 to 3.
        :return: None.
        """
        if channel < 0 or channel >= MODSong.CHANNELS:
            raise IndexError(f"Invalid channel index {channel} (expected 0-3).")

        for p in range(len(self.patterns)):
            for r in range(MODSong.ROWS):
                self.patterns[p].data[channel][r] = Note()
        self._on_mutation()

    def mute_channel(self, channel: int):
        """
        Mutes a specified channel in the entire song while preserving global effects.
        This clears notes, instruments, and channel-specific effects but keeps global effects
        like speed/BPM changes (Fxx), pattern breaks (Bxx), position jumps (Dxx), volume set (Cxx), and extended effects (E**).

        :param channel: The channel index to mute, 0 to 3.
        :return: None.
        """
        if channel < 0 or channel >= MODSong.CHANNELS:
            raise IndexError(f"Invalid channel index {channel} (expected 0-3).")

        for p in range(len(self.patterns)):
            for r in range(MODSong.ROWS):
                note = self.patterns[p].data[channel][r]
                
                # Check if this note has a global effect that should be preserved
                global_effect = ""
                if note.effect != "":
                    effect_type = note.effect[0]
                    # Preserve global effects: F (speed/BPM), B (pattern break), D (position jump)
                    if effect_type in ['F', 'B', 'D', 'C', 'E']:
                        global_effect = note.effect
                
                # Create a new empty note
                new_note = Note()
                # Restore the global effect if there was one
                if global_effect:
                    new_note.effect = global_effect
                
                self.patterns[p].data[channel][r] = new_note
        self._on_mutation()

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
            if period_raw not in MODSong.PERIOD_TABLE:
                raise ValueError(f"Unknown period value {period_raw} in MOD note data.")
            return MODSong.PERIOD_TABLE[period_raw]
        else:
            return ""

    def get_note_raw(self, sequence_idx: int, row: int, channel: int) -> Note:
        """Return the stored raw MOD note cell without sample-memory resolution.

        Raw sample values are the exact MOD cell sample nibble (`00..31`).
        Use this accessor when byte-level fidelity is required.
        """
        if row < 0 or row >= MODSong.ROWS:
            raise IndexError(f"Invalid row index {row} (expected 0-63).")

        if channel < 0 or channel >= MODSong.CHANNELS:
            raise IndexError(f"Invalid channel index {channel} (expected 0-3).")

        if sequence_idx < 0 or sequence_idx >= len(self.pattern_seq):
            raise IndexError(f"Invalid sequence index {sequence_idx} (expected 0-{len(self.pattern_seq)-1}).")

        return self.patterns[self.pattern_seq[sequence_idx]].data[channel][row]

    def get_note(self, sequence_idx: int, row: int, channel: int, *, resolved: bool = True) -> Note:
        """Return a MOD note cell with raw or effective sample semantics.

        Definitions:
        - raw sample: sample nibble stored in the MOD cell (`00..31`)
        - effective sample: channel-memory resolved sample used when a note
          period is present and raw sample is `00`

        Default behavior (`resolved=True`):
        - if period is present and raw sample is `00`, the most recently latched
          sample for that channel is returned
        - if no sample was ever latched on that channel, effective sample is `0`
        - sample-only rows (raw sample > 0 with empty period) update latch state
          for later note rows in that same channel
        - sample memory carries across rows and pattern boundaries in sequence order

        Performance:
        - resolution is backed by a mutation-versioned lazy cache and is not
          recomputed from scratch on every query

        Return shape:
        - when ``resolved=False``, returns the stored mutable note object
        - when ``resolved=True``, returns the stored note object if raw/effective
          sample match; otherwise returns a detached note snapshot with the
          effective sample index
        """
        raw_note = self.get_note_raw(sequence_idx, row, channel)
        if not resolved:
            return raw_note

        self._ensure_sequence_resolution_cache()
        raw_sample = int(getattr(raw_note, 'instrument_idx', 0))
        effective_sample = self._resolved_sequence_cells.get((sequence_idx, row, channel), raw_sample)
        if effective_sample == raw_sample:
            return raw_note
        return Note(effective_sample, raw_note.period, raw_note.effect)
    
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

    def set_portamento(self, pattern: int, channel: int, row: int, slide: int):
        """
        Slides the current period up or down by the given amount on each tick.
        The slide rate depends on the current ticks-per-row setting.

        :param pattern: The pattern index (in the sequence) to write to.
        :param channel: The channel index to write to, 0-based.
        :param row: The row index to write to, 0-based.
        :param slide: The period delta per tick, -255 to 255. 0 is ignored.
        :return: None.
        """
        if slide < -255 or slide > 255:
            raise ValueError(f"Invalid portamento slide {slide} (expected 0-255).")

        if slide > 0:
            self.set_effect(pattern, channel, row, f"1{slide:02X}")
        elif slide < 0:
            self.set_effect(pattern, channel, row, f"2{(-slide):02X}")

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
            raise ValueError(f"Invalid tone portamento speed {speed} (expected 0-255).")

        self.set_effect(pattern, channel, row, f"3{speed:02X}")

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
            raise ValueError(f"Invalid tone portamento slide {slide} (expected 0-255).")
        
        efx = 0
        if slide > 0:
            efx = slide << 4
        elif slide < 0:
            efx = -slide
        
        self.set_effect(pattern, channel, row, f"5{efx:02X}")
        
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
            raise ValueError(f"Invalid volume {volume} (expected 0-64).")

        self.set_effect(pattern, channel, row, f"C{volume:02X}")

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
            raise ValueError(f"Invalid volume slide {slide} (expected 0-255).")
        
        efx = 0
        if slide > 0:
            efx = slide << 4
        elif slide < 0:
            efx = -slide
        
        self.set_effect(pattern, channel, row, f"A{efx:02X}")

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
            raise ValueError(f"Invalid vibrato speed {speed} (expected 0-15).")
        
        if depth < 0 or depth > 15:
            raise ValueError(f"Invalid vibrato depth {depth} (expected 0-15).")
        
        efx = 16 * speed + depth
        
        self.set_effect(pattern, channel, row, f"4{efx:02X}")

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
            raise ValueError(f"Invalid vibrato slide {slide} (expected 0-255).")
        
        efx = 0
        if slide > 0:
            efx = slide << 4
        elif slide < 0:
            efx = -slide
        
        self.set_effect(pattern, channel, row, f"6{efx:02X}")

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
            raise ValueError(f"Invalid tremolo speed {speed} (expected 0-15).")
        
        if depth < 0 or depth > 15:
            raise ValueError(f"Invalid tremolo depth {depth} (expected 0-15).")
        
        efx = 16 * speed + depth
        
        self.set_effect(pattern, channel, row, f"7{efx:02X}")
