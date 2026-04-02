from __future__ import annotations

import copy
import array
import os
import struct
import sys

from nodmod import Pattern
from nodmod import S3MNote
from nodmod import S3MSample
from nodmod import Song


class S3MSong(Song):
    ROWS = 64
    MAX_CHANNELS = 32
    MAX_SAMPLES = 99
    PRESERVED_EFFECT_PREFIXES = frozenset({'A', 'B', 'C', 'T'})

    @property
    def file_extension(self) -> str:
        return 's3m'

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @n_channels.setter
    def n_channels(self, n: int) -> None:
        if n < 1 or n > self.MAX_CHANNELS:
            raise ValueError(f"Invalid channel count {n} (expected 1-{self.MAX_CHANNELS}).")
        self._n_channels = n
        self.channel_settings = self._default_channel_settings(n)
        self._rebuild_channel_mappings()
        for pat in self.patterns:
            self._resize_pattern_channels(pat, n)

    def __init__(self):
        super().__init__()

        self.sig1 = 0x1A
        self.song_type = 0x10
        self.tracker_version = 0x1320
        self.flags = 0
        self.global_volume = 64
        self.initial_speed = 6
        self.initial_tempo = 125
        self.master_volume = 0xB0
        self.ultra_click_removal = 0
        self.sample_type = 2
        self.default_pan_flag = 0
        self.special = 0
        self.reserved1 = 0
        self.reserved2 = b'\x00' * 8
        self.default_panning: list[int] = []
        self.order_count = 0
        self.instrument_count = 0
        self.pattern_count = 0
        self.order_list_raw: list[int] = []
        self.instrument_parapointers: list[int] = []
        self.pattern_parapointers: list[int] = []
        self.instrument_offsets: list[int] = []
        self.pattern_offsets: list[int] = []

        self.samples: list[S3MSample] = [S3MSample() for _ in range(self.MAX_SAMPLES)]
        self.n_actual_samples = 0

        self._n_channels = 0
        self.channel_settings: list[int] = []
        self.raw_channel_slots: list[int] = []
        self.compact_to_raw_channel: list[int] = []
        self.raw_to_compact_channel: dict[int, int] = {}

        self.n_channels = 16
        self.add_pattern()

    def _default_channel_settings(self, n_channels: int) -> list[int]:
        default_active = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
        settings = default_active[:n_channels]
        while len(settings) < n_channels:
            settings.append(len(settings) & 0x0F)
        return settings + [255] * (self.MAX_CHANNELS - n_channels)

    def _rebuild_channel_mappings(self) -> None:
        self.raw_channel_slots = [idx for idx, value in enumerate(self.channel_settings) if value != 255]
        self.compact_to_raw_channel = list(self.raw_channel_slots)
        self.raw_to_compact_channel = {raw: compact for compact, raw in enumerate(self.compact_to_raw_channel)}
        self._n_channels = len(self.compact_to_raw_channel)

    def _new_pattern(self) -> Pattern:
        pat = Pattern(self.ROWS, self.n_channels)
        pat.data = [[S3MNote() for _ in range(pat.n_rows)] for _ in range(pat.n_channels)]
        return pat

    def _resize_pattern_channels(self, pat: Pattern, n_channels: int) -> None:
        if pat.n_channels < n_channels:
            for _ in range(n_channels - pat.n_channels):
                pat.data.append([S3MNote() for _ in range(pat.n_rows)])
        elif pat.n_channels > n_channels:
            pat.data = pat.data[:n_channels]
        pat.n_channels = n_channels

    def _update_n_actual_samples(self) -> None:
        self.n_actual_samples = sum(1 for sample in self.samples if len(sample.waveform) > 0)

    def copy(self) -> 'S3MSong':
        new_song = S3MSong()
        new_song.artist = self.artist
        new_song.songname = self.songname
        new_song.patterns = copy.deepcopy(self.patterns)
        new_song.pattern_seq = copy.deepcopy(self.pattern_seq)
        new_song.samples = copy.deepcopy(self.samples)
        new_song.n_actual_samples = self.n_actual_samples
        new_song.sig1 = self.sig1
        new_song.song_type = self.song_type
        new_song.tracker_version = self.tracker_version
        new_song.flags = self.flags
        new_song.global_volume = self.global_volume
        new_song.initial_speed = self.initial_speed
        new_song.initial_tempo = self.initial_tempo
        new_song.master_volume = self.master_volume
        new_song.ultra_click_removal = self.ultra_click_removal
        new_song.sample_type = self.sample_type
        new_song.default_pan_flag = self.default_pan_flag
        new_song.special = self.special
        new_song.reserved1 = self.reserved1
        new_song.reserved2 = self.reserved2
        new_song.default_panning = copy.deepcopy(self.default_panning)
        new_song.order_count = self.order_count
        new_song.instrument_count = self.instrument_count
        new_song.pattern_count = self.pattern_count
        new_song.order_list_raw = list(self.order_list_raw)
        new_song.instrument_parapointers = list(self.instrument_parapointers)
        new_song.pattern_parapointers = list(self.pattern_parapointers)
        new_song.instrument_offsets = list(self.instrument_offsets)
        new_song.pattern_offsets = list(self.pattern_offsets)
        new_song.channel_settings = list(self.channel_settings)
        new_song._rebuild_channel_mappings()
        return new_song

    def save(self, fname: str, verbose: bool = True):
        raise NotImplementedError("S3M save support is not implemented yet.")

    def load(self, fname: str, verbose: bool = True):
        if verbose:
            print(f'Loading {fname}... ', end='', flush=True)

        with open(fname, 'rb') as s3m_file:
            data = s3m_file.read()

        if len(data) < 96:
            raise NotImplementedError("Invalid S3M file format (header too short).")

        self.artist, _ = Song.artist_songname_from_filename(fname)
        self.songname = self._decode_text(data[:28])

        self.sig1 = data[28]
        self.song_type = data[29]
        if self.sig1 != 0x1A or self.song_type != 0x10:
            raise NotImplementedError(
                f"Not an S3M module. Signature/type mismatch: sig1={self.sig1:#04x}, type={self.song_type:#04x}."
            )

        self.reserved1 = struct.unpack_from('<H', data, 30)[0]
        (
            self.order_count,
            self.instrument_count,
            self.pattern_count,
            self.flags,
            self.tracker_version,
            self.sample_type,
        ) = struct.unpack_from('<6H', data, 32)

        if data[44:48] != b'SCRM':
            raise NotImplementedError(f"Not an S3M module. Missing SCRM signature: {data[44:48]!r}.")
        if self.sample_type not in (1, 2):
            raise NotImplementedError(f"Unsupported S3M sample type {self.sample_type}.")
        if self.instrument_count > self.MAX_SAMPLES:
            raise NotImplementedError(
                f"Too many instruments: {self.instrument_count} (S3M support currently expects <= {self.MAX_SAMPLES})."
            )

        self.global_volume = data[48]
        self.initial_speed = data[49] or self.initial_speed
        tempo = data[50]
        if tempo >= 33:
            self.initial_tempo = tempo
        self.master_volume = data[51]
        self.ultra_click_removal = data[52]
        self.default_pan_flag = data[53]
        self.reserved2 = bytes(data[54:62])
        self.special = struct.unpack_from('<H', data, 62)[0]
        self.channel_settings = list(data[64:96])
        self._rebuild_channel_mappings()

        offset = 96
        tables_size = self.order_count + 2 * self.instrument_count + 2 * self.pattern_count
        if len(data) < offset + tables_size:
            raise NotImplementedError("Invalid S3M file format (truncated order/parapointer tables).")

        self.order_list_raw = list(data[offset:offset + self.order_count])
        offset += self.order_count

        if self.instrument_count:
            fmt = '<' + 'H' * self.instrument_count
            self.instrument_parapointers = list(struct.unpack_from(fmt, data, offset))
        else:
            self.instrument_parapointers = []
        offset += 2 * self.instrument_count

        if self.pattern_count:
            fmt = '<' + 'H' * self.pattern_count
            self.pattern_parapointers = list(struct.unpack_from(fmt, data, offset))
        else:
            self.pattern_parapointers = []
        offset += 2 * self.pattern_count

        self.instrument_offsets = [ptr << 4 for ptr in self.instrument_parapointers]
        self.pattern_offsets = [ptr << 4 for ptr in self.pattern_parapointers]

        self.default_panning = []
        if self.default_pan_flag == 252:
            if len(data) < offset + 32:
                raise NotImplementedError("Invalid S3M file format (missing default panning table).")
            self.default_panning = list(data[offset:offset + 32])

        self.patterns = [self._new_pattern() for _ in range(self.pattern_count)]
        self.pattern_seq = []
        for order in self.order_list_raw:
            if order == 0xFF:
                break
            if order == 0xFE:
                continue
            if order >= self.pattern_count:
                raise NotImplementedError(
                    f"Invalid S3M order entry {order} (pattern count {self.pattern_count})."
                )
            self.pattern_seq.append(order)

        self.samples = [S3MSample() for _ in range(self.MAX_SAMPLES)]
        self._load_pcm_instruments(data, fname)
        self._load_patterns(data, fname)

        if verbose:
            print('done.')

    def timestamp(self) -> list[list[tuple[float, int, int]]]:
        tick_duration = self.get_tick_duration(self.initial_tempo)
        row_duration = self.initial_speed * tick_duration
        timestamps: list[list[tuple[float, int, int]]] = []
        elapsed = 0.0
        for pat_idx in self.pattern_seq:
            pat = self.patterns[pat_idx]
            pat_rows: list[tuple[float, int, int]] = []
            for _row in range(pat.n_rows):
                pat_rows.append((elapsed, self.initial_speed, self.initial_tempo))
                elapsed += row_duration
            timestamps.append(pat_rows)
        return timestamps

    def get_effective_row_count(self, sequence_idx: int) -> int:
        pat = self._get_sequence_pattern(sequence_idx)
        return pat.n_rows

    def add_pattern(self, n_rows: int = ROWS) -> int:
        if n_rows != self.ROWS:
            raise ValueError(f"S3M patterns have fixed 64 rows (got {n_rows}).")
        if len(self.pattern_seq) + 1 > 256:
            raise ValueError(f"Pattern sequence too long ({len(self.pattern_seq) + 1}). S3M supports up to 256 orders.")
        self.patterns.append(self._new_pattern())
        pat_idx = len(self.patterns) - 1
        self.pattern_seq.append(pat_idx)
        return pat_idx

    def add_to_sequence(self, pattern_idx: int, sequence_position: int | None = None) -> None:
        if len(self.pattern_seq) + 1 > 256:
            raise ValueError(f"Pattern sequence too long ({len(self.pattern_seq) + 1}). S3M supports up to 256 orders.")
        super().add_to_sequence(pattern_idx, sequence_position)

    def set_sequence(self, seq: list[int]) -> None:
        if len(seq) > 256:
            raise ValueError(f"Pattern sequence too long ({len(seq)}). S3M supports up to 256 orders.")
        super().set_sequence(seq)

    def clear_pattern(self, sequence_idx: int) -> None:
        pat = self._get_sequence_pattern(sequence_idx)
        for channel in range(pat.n_channels):
            for row in range(pat.n_rows):
                pat.data[channel][row] = S3MNote()

    def get_note(self, sequence_idx: int, row: int, channel: int) -> S3MNote:
        pat = self._get_sequence_pattern(sequence_idx)
        if row < 0 or row >= pat.n_rows:
            raise IndexError(f"Invalid row index {row} (expected 0-{pat.n_rows-1}).")
        if channel < 0 or channel >= pat.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{pat.n_channels-1}).")
        return pat.data[channel][row]

    def set_note(
        self,
        sequence_idx: int,
        channel: int,
        row: int,
        instrument_idx: int,
        period: str,
        effect: str = "",
        volume: int | None = None,
    ) -> None:
        pat = self._get_sequence_pattern(sequence_idx)
        if row < 0 or row >= pat.n_rows:
            raise IndexError(f"Invalid row index {row} (expected 0-{pat.n_rows-1}).")
        if channel < 0 or channel >= pat.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{pat.n_channels-1}).")
        if instrument_idx < 0 or instrument_idx > self.MAX_SAMPLES:
            raise IndexError(f"Invalid sample index {instrument_idx} (expected 0-{self.MAX_SAMPLES}).")
        cur_note = pat.data[channel][row]
        if effect == '':
            effect = self._preserved_effect(cur_note.effect)
        new_note = S3MNote(
            instrument_idx=instrument_idx,
            period=period,
            effect=effect,
            volume=cur_note.volume if volume is None else volume,
        )
        pat.data[channel][row] = new_note

    def set_bpm(self, pattern: int, channel: int, row: int, bpm: int):
        if bpm < 32 or bpm > 255:
            raise ValueError(f"Invalid tempo {bpm} (expected 32-255).")
        self.set_effect(pattern, channel, row, f"T{bpm:02X}")

    def set_ticks_per_row(self, pattern: int, channel: int, row: int, ticks: int):
        if ticks < 1 or ticks > 31:
            raise ValueError(f"Invalid ticks per row {ticks} (expected 1-31).")
        self.set_effect(pattern, channel, row, f"A{ticks:02X}")

    def set_n_channels(self, n: int) -> None:
        self.n_channels = n

    def get_sample(self, sample_idx: int) -> S3MSample:
        if sample_idx <= 0 or sample_idx > self.MAX_SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-{self.MAX_SAMPLES}).")
        return self.samples[sample_idx - 1]

    def list_samples(self) -> list[S3MSample]:
        return self.samples

    def set_sample(self, sample_idx: int, sample: S3MSample) -> None:
        if sample_idx <= 0 or sample_idx > self.MAX_SAMPLES:
            raise IndexError(f"Invalid sample index {sample_idx} (expected 1-{self.MAX_SAMPLES}).")
        self.samples[sample_idx - 1] = sample
        self._update_n_actual_samples()

    def _load_pcm_instruments(self, data: bytes, fname: str) -> None:
        self.samples = [S3MSample() for _ in range(self.MAX_SAMPLES)]

        for inst_idx, inst_offset in enumerate(self.instrument_offsets, start=1):
            if inst_offset + 80 > len(data):
                raise NotImplementedError(
                    f"Invalid S3M instrument pointer for instrument {inst_idx} in {os.path.basename(fname)}."
                )

            inst = data[inst_offset:inst_offset + 80]
            inst_type = inst[0]
            sample = S3MSample()
            sample.filename = self._decode_text(inst[1:13])
            sample.name = self._decode_text(inst[48:76])
            sample._signature = inst[76:80].decode('latin-1', errors='replace')

            if inst_type == 0:
                self.samples[inst_idx - 1] = sample
                continue

            if 2 <= inst_type <= 7:
                raise NotImplementedError(
                    f"Adlib S3M instruments are not supported yet (instrument {inst_idx} in {os.path.basename(fname)})."
                )

            if inst_type != 1:
                raise NotImplementedError(
                    f"Unsupported S3M instrument type {inst_type} at instrument {inst_idx}."
                )

            para = (inst[13] << 16) | struct.unpack_from('<H', inst, 14)[0]
            sample.sample_offset = para << 4
            byte_length = struct.unpack_from('<I', inst, 16)[0]
            loop_start = struct.unpack_from('<I', inst, 20)[0]
            loop_end = struct.unpack_from('<I', inst, 24)[0]
            sample.volume = inst[28]
            sample._reserved_byte = inst[29]
            sample.pack = inst[30]
            sample.flags = inst[31]
            sample.is_stereo = bool(sample.flags & 0x02)
            sample.is_16bit = bool(sample.flags & 0x04)
            sample.c2spd = struct.unpack_from('<I', inst, 32)[0]
            sample._internal = bytes(inst[36:48])

            if sample.pack != 0:
                raise NotImplementedError(
                    f"Packed S3M samples are not supported yet (instrument {inst_idx} in {os.path.basename(fname)})."
                )
            if sample.is_stereo:
                raise NotImplementedError(
                    f"Stereo S3M samples are not supported yet (instrument {inst_idx} in {os.path.basename(fname)})."
                )

            sample.waveform = self._decode_sample_waveform(data, sample.sample_offset, byte_length, sample.is_16bit)

            unit_size = 2 if sample.is_16bit else 1
            sample.repeat_point = loop_start // unit_size
            if sample.flags & 0x01 and loop_end > loop_start:
                sample.repeat_len = (loop_end - loop_start) // unit_size
            else:
                sample.repeat_len = 0

            self.samples[inst_idx - 1] = sample

        self._update_n_actual_samples()

    def _decode_sample_waveform(self, data: bytes, offset: int, byte_length: int, is_16bit: bool):
        if offset == 0 or byte_length == 0:
            return array.array('h' if is_16bit else 'b')
        if offset + byte_length > len(data):
            raise NotImplementedError("Invalid S3M sample data pointer (truncated sample data).")
        raw = data[offset:offset + byte_length]
        if is_16bit:
            if byte_length % 2 != 0:
                raise NotImplementedError("Invalid S3M 16-bit sample length.")
            if self.sample_type == 1:
                waveform = array.array('h')
                waveform.frombytes(raw)
                if sys.byteorder != 'little':
                    waveform.byteswap()
                return waveform
            unsigned_waveform = array.array('H')
            unsigned_waveform.frombytes(raw)
            if sys.byteorder != 'little':
                unsigned_waveform.byteswap()
            return array.array('h', (value - 32768 for value in unsigned_waveform))

        if self.sample_type == 1:
            waveform = array.array('b')
            waveform.frombytes(raw)
            return waveform

        unsigned_waveform = array.array('B')
        unsigned_waveform.frombytes(raw)
        return array.array('b', (value - 128 for value in unsigned_waveform))

    def _load_patterns(self, data: bytes, fname: str) -> None:
        self.patterns = [self._new_pattern() for _ in range(self.pattern_count)]
        for pat_idx, pat_offset in enumerate(self.pattern_offsets):
            if pat_offset == 0:
                continue
            if pat_offset + 2 > len(data):
                raise NotImplementedError(
                    f"Invalid S3M pattern pointer for pattern {pat_idx} in {os.path.basename(fname)}."
                )
            packed_len = struct.unpack_from('<H', data, pat_offset)[0]
            if packed_len < 2 or pat_offset + packed_len > len(data):
                raise NotImplementedError(
                    f"Invalid S3M packed pattern length for pattern {pat_idx} in {os.path.basename(fname)}."
                )
            packed_data = data[pat_offset + 2:pat_offset + packed_len]
            self._decode_pattern_data(self.patterns[pat_idx], packed_data)

    def _decode_pattern_data(self, pat: Pattern, packed_data: bytes) -> None:
        row = 0
        pos = 0
        while row < self.ROWS and pos < len(packed_data):
            what = packed_data[pos]
            pos += 1
            if what == 0:
                row += 1
                continue

            raw_channel = what & 0x1F
            note_value = None
            instrument_idx = 0
            volume = -1
            effect = ''

            if what & 0x20:
                if pos + 2 > len(packed_data):
                    raise NotImplementedError("Invalid S3M packed pattern data (truncated note/instrument pair).")
                note_value = packed_data[pos]
                instrument_idx = packed_data[pos + 1]
                pos += 2

            if what & 0x40:
                if pos >= len(packed_data):
                    raise NotImplementedError("Invalid S3M packed pattern data (truncated volume byte).")
                raw_volume = packed_data[pos]
                volume = raw_volume if raw_volume <= 64 else -1
                pos += 1

            if what & 0x80:
                if pos + 2 > len(packed_data):
                    raise NotImplementedError("Invalid S3M packed pattern data (truncated effect pair).")
                command = packed_data[pos]
                info = packed_data[pos + 1]
                pos += 2
                if command != 0:
                    effect = f"{chr(ord('A') + command - 1)}{info:02X}"

            compact_channel = self.raw_to_compact_channel.get(raw_channel)
            if compact_channel is None:
                continue

            note = S3MNote()
            note.instrument_idx = instrument_idx
            note.volume = volume
            note.effect = effect
            if note_value is not None:
                note.period = self._decode_note_value(note_value)
            pat.data[compact_channel][row] = note

    @staticmethod
    def _decode_note_value(note_value: int) -> str:
        if note_value == 255:
            return ''
        if note_value == 254:
            return 'off'
        semitone = note_value & 0x0F
        octave = (note_value >> 4) & 0x0F
        if semitone > 11:
            raise NotImplementedError(f"Unsupported S3M note value {note_value}.")
        return f"{Song.PERIOD_SEQ[semitone]}{octave}"

    @staticmethod
    def _decode_text(raw: bytes) -> str:
        return raw.split(b'\x00', 1)[0].decode('latin-1', errors='replace').rstrip(' ')