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

    def _default_channel_setting_for_index(self, compact_index: int) -> int:
        return self._default_channel_settings(compact_index + 1)[compact_index]

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
        if verbose:
            print(f'Saving to {fname}... ', end='', flush=True)

        order_list = self._build_order_list_for_save()
        instrument_count = self._instrument_count_for_save()
        pattern_count = len(self.patterns)
        has_panning = len(self.default_panning) == 32 and self.default_pan_flag == 252
        default_pan_flag = 252 if has_panning else 0

        header_size = 96 + len(order_list) + 2 * instrument_count + 2 * pattern_count + (32 if has_panning else 0)
        payload = bytearray(header_size)

        order_offset = 96
        instrument_ptr_offset = order_offset + len(order_list)
        pattern_ptr_offset = instrument_ptr_offset + 2 * instrument_count
        panning_offset = pattern_ptr_offset + 2 * pattern_count

        payload[order_offset:order_offset + len(order_list)] = bytes(order_list)
        if has_panning:
            payload[panning_offset:panning_offset + 32] = bytes(self.default_panning)

        self._align16(payload)

        instrument_parapointers: list[int] = []
        pattern_parapointers: list[int] = []
        instrument_header_positions: list[int] = []

        for sample_idx in range(1, instrument_count + 1):
            instrument_parapointers.append(len(payload) >> 4)
            instrument_header_positions.append(len(payload))
            payload.extend(b'\x00' * 80)

        for pat in self.patterns:
            self._align16(payload)
            pattern_parapointers.append(len(payload) >> 4)
            payload.extend(self._encode_pattern_block(pat))

        sample_paragraphs: dict[int, int] = {}
        for sample_idx in range(1, instrument_count + 1):
            sample = self.samples[sample_idx - 1]
            self._validate_sample_for_save(sample, sample_idx)
            sample_bytes = self._encode_sample_data(sample)
            if sample_bytes:
                self._align16(payload)
                sample_paragraphs[sample_idx] = len(payload) >> 4
                payload.extend(sample_bytes)
            else:
                sample_paragraphs[sample_idx] = 0

        for sample_idx in range(1, instrument_count + 1):
            sample = self.samples[sample_idx - 1]
            header = self._build_instrument_header(sample, sample_paragraphs[sample_idx])
            pos = instrument_header_positions[sample_idx - 1]
            payload[pos:pos + 80] = header

        payload[0:28] = self._encode_text(self.songname, 28)
        payload[28] = self.sig1
        payload[29] = self.song_type
        struct.pack_into('<H', payload, 30, self.reserved1)
        struct.pack_into(
            '<6H',
            payload,
            32,
            len(order_list),
            instrument_count,
            pattern_count,
            self.flags,
            self.tracker_version,
            self.sample_type,
        )
        payload[44:48] = b'SCRM'
        payload[48] = self.global_volume
        payload[49] = self.initial_speed
        payload[50] = self.initial_tempo
        payload[51] = self.master_volume
        payload[52] = self.ultra_click_removal
        payload[53] = default_pan_flag
        payload[54:62] = self.reserved2[:8].ljust(8, b'\x00')
        struct.pack_into('<H', payload, 62, self.special)
        payload[64:96] = bytes(self.channel_settings[:32])

        if instrument_count:
            struct.pack_into(
                '<' + 'H' * instrument_count,
                payload,
                instrument_ptr_offset,
                *instrument_parapointers,
            )
        if pattern_count:
            struct.pack_into(
                '<' + 'H' * pattern_count,
                payload,
                pattern_ptr_offset,
                *pattern_parapointers,
            )

        with open(fname, 'wb') as s3m_file:
            s3m_file.write(payload)

        self.order_count = len(order_list)
        self.instrument_count = instrument_count
        self.pattern_count = pattern_count
        self.order_list_raw = list(order_list)
        self.instrument_parapointers = instrument_parapointers
        self.pattern_parapointers = pattern_parapointers
        self.instrument_offsets = [ptr << 4 for ptr in instrument_parapointers]
        self.pattern_offsets = [ptr << 4 for ptr in pattern_parapointers]
        self.default_pan_flag = default_pan_flag

        if verbose:
            print('done.')

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
        timestamps: list[list[tuple[float, int, int]]] = []
        speed = self.initial_speed
        bpm = self.initial_tempo
        elapsed = 0.0
        seq_idx = 0
        start_row = 0
        self_jump_count = 0
        self_jump_limit = 5

        while seq_idx < len(self.pattern_seq):
            pat_idx = self.pattern_seq[seq_idx]
            if pat_idx < 0 or pat_idx >= len(self.patterns):
                seq_idx += 1
                start_row = 0
                continue

            pat = self.patterns[pat_idx]
            pat_rows: list[tuple[float, int, int]] = []
            r = start_row
            start_row = 0

            while r < pat.n_rows:
                pending_jump_seq = None
                pending_break_row = None
                row_delay = 0

                for channel in range(pat.n_channels):
                    effect = self._effect_text(pat.data[channel][r].effect)
                    if effect == '':
                        continue
                    if effect.startswith('A') and len(effect) == 3:
                        value = int(effect[1:], 16)
                        if value != 0:
                            speed = value
                    elif effect.startswith('T') and len(effect) == 3:
                        value = int(effect[1:], 16)
                        if value >= 33:
                            bpm = value
                    elif effect.startswith('B') and len(effect) == 3:
                        value = int(effect[1:], 16)
                        if value < len(self.pattern_seq):
                            pending_jump_seq = value
                    elif effect.startswith('C') and len(effect) == 3:
                        hi = int(effect[1], 16)
                        lo = int(effect[2], 16)
                        row_target = hi * 10 + lo
                        if row_target < pat.n_rows:
                            pending_break_row = row_target
                    elif effect.startswith('SE') and len(effect) == 3:
                        row_delay = max(row_delay, int(effect[2], 16))

                pat_rows.append((elapsed, self.initial_speed, self.initial_tempo))
                pat_rows[-1] = (elapsed, speed, bpm)
                elapsed += (1 + row_delay) * self.get_tick_duration(bpm) * speed

                if pending_jump_seq is not None or pending_break_row is not None:
                    timestamps.append(pat_rows)
                    if pending_jump_seq is not None:
                        if pending_jump_seq == seq_idx:
                            self_jump_count += 1
                            if self_jump_count > self_jump_limit:
                                return timestamps
                        else:
                            self_jump_count = 0
                    if pending_break_row is not None:
                        start_row = pending_break_row
                        seq_idx = pending_jump_seq if pending_jump_seq is not None else seq_idx + 1
                    else:
                        start_row = 0
                        seq_idx = pending_jump_seq
                    break

                r += 1
            else:
                timestamps.append(pat_rows)
                seq_idx += 1
        return timestamps

    def get_effective_row_count(self, sequence_idx: int) -> int:
        pat = self._get_sequence_pattern(sequence_idx)
        played_rows = 0
        for row in range(pat.n_rows):
            row_delay = 0
            interrupt = False
            for channel in range(pat.n_channels):
                effect = self._effect_text(pat.data[channel][row].effect)
                if effect == '':
                    continue
                if effect.startswith('SE') and len(effect) == 3:
                    row_delay = max(row_delay, int(effect[2], 16))
                if effect.startswith('B') and len(effect) == 3:
                    interrupt = True
                if effect.startswith('C') and len(effect) == 3:
                    hi = int(effect[1], 16)
                    lo = int(effect[2], 16)
                    if hi * 10 + lo < pat.n_rows:
                        interrupt = True
            played_rows += 1 + row_delay
            if interrupt:
                break
        return played_rows

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

    def resize_pattern(self, sequence_idx: int, n_rows: int) -> None:
        if n_rows != self.ROWS:
            raise ValueError(f"S3M patterns have fixed 64 rows (got {n_rows}).")
        self._get_sequence_pattern(sequence_idx)

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

    def add_channel(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError(f"Invalid channel count {count} (expected >=1).")
        if self.n_channels + count > self.MAX_CHANNELS:
            raise ValueError(f"Too many channels: {self.n_channels + count} (S3M supports 1-{self.MAX_CHANNELS}).")
        for _ in range(count):
            try:
                raw_slot = self.channel_settings.index(255)
            except ValueError as exc:
                raise ValueError("No unused S3M channel slots are available.") from exc
            self.channel_settings[raw_slot] = self._default_channel_setting_for_index(self.n_channels)
            for pat in self.patterns:
                pat.data.append([S3MNote() for _ in range(pat.n_rows)])
                pat.n_channels += 1
            self._rebuild_channel_mappings()

    def remove_channel(self, channel: int) -> None:
        if self.n_channels <= 1:
            raise ValueError("Cannot remove last channel (S3M requires at least 1).")
        if channel < 0 or channel >= self.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{self.n_channels-1}).")
        raw_slot = self.compact_to_raw_channel[channel]
        for pat in self.patterns:
            pat.data.pop(channel)
            pat.n_channels -= 1
        self.channel_settings[raw_slot] = 255
        self._rebuild_channel_mappings()

    def clear_channel(self, channel: int) -> None:
        if channel < 0 or channel >= self.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{self.n_channels-1}).")
        for pat in self.patterns:
            for row in range(pat.n_rows):
                pat.data[channel][row] = S3MNote()

    def mute_channel(self, channel: int) -> None:
        if channel < 0 or channel >= self.n_channels:
            raise IndexError(f"Invalid channel index {channel} (expected 0-{self.n_channels-1}).")
        for pat in self.patterns:
            for row in range(pat.n_rows):
                note = pat.data[channel][row]
                preserved = self._preserved_effect(note.effect)
                new_note = S3MNote()
                if preserved:
                    new_note.effect = preserved
                pat.data[channel][row] = new_note

    def copy_sample_from(self, src: 'S3MSong', src_sample_idx: int, dst_sample_idx: int | None = None) -> int:
        if src_sample_idx <= 0 or src_sample_idx > src.MAX_SAMPLES:
            raise IndexError(f"Invalid source sample index {src_sample_idx} (expected 1-{src.MAX_SAMPLES}).")
        if dst_sample_idx is not None and (dst_sample_idx <= 0 or dst_sample_idx > self.MAX_SAMPLES):
            raise IndexError(f"Invalid destination sample index {dst_sample_idx} (expected 1-{self.MAX_SAMPLES}).")
        if dst_sample_idx is None:
            for idx, sample in enumerate(self.samples, start=1):
                if len(sample.waveform) == 0:
                    dst_sample_idx = idx
                    break
            if dst_sample_idx is None:
                raise ValueError("Couldn't find an empty slot for the new sample.")
        self.samples[dst_sample_idx - 1] = copy.deepcopy(src.get_sample(src_sample_idx))
        self._update_n_actual_samples()
        return dst_sample_idx

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
            sample.instrument_type = inst_type
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
            length_units = struct.unpack_from('<I', inst, 16)[0]
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

            byte_length = length_units * (2 if sample.is_16bit else 1)
            sample.waveform = self._decode_sample_waveform(data, sample.sample_offset, byte_length, sample.is_16bit)

            sample.repeat_point = loop_start
            if sample.flags & 0x01 and loop_end > loop_start:
                sample.repeat_len = loop_end - loop_start
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
    def _align16(payload: bytearray) -> None:
        remainder = len(payload) % 16
        if remainder:
            payload.extend(b'\x00' * (16 - remainder))

    def _build_order_list_for_save(self) -> list[int]:
        raw = list(self.order_list_raw)
        if raw and self._normalized_order_list(raw) == self.pattern_seq:
            return raw
        order_list = list(self.pattern_seq)
        if not order_list or order_list[-1] != 0xFF:
            order_list.append(0xFF)
        if len(order_list) > 256:
            raise ValueError(f"Pattern sequence too long ({len(order_list)}). S3M supports up to 256 orders.")
        return order_list

    def _instrument_count_for_save(self) -> int:
        highest_non_empty = 0
        for idx, sample in enumerate(self.samples, start=1):
            if self._sample_slot_used(sample):
                highest_non_empty = idx
        return max(self.instrument_count, highest_non_empty)

    @staticmethod
    def _normalized_order_list(order_list: list[int]) -> list[int]:
        normalized: list[int] = []
        for order in order_list:
            if order == 0xFF:
                break
            if order == 0xFE:
                continue
            normalized.append(order)
        return normalized

    @staticmethod
    def _encode_text(text: str, length: int) -> bytes:
        raw = text.encode('latin-1', errors='replace')[:length]
        return raw.ljust(length, b'\x00')

    def _sample_slot_used(self, sample: S3MSample) -> bool:
        return (
            sample.instrument_type != 0
            or len(sample.waveform) > 0
            or sample.name != ''
            or sample.filename != ''
        )

    def _validate_sample_for_save(self, sample: S3MSample, sample_idx: int) -> None:
        if getattr(sample, 'instrument_type', 0) in {2, 3, 4, 5, 6, 7}:
            raise NotImplementedError(f"Adlib S3M instruments are not supported yet (sample {sample_idx}).")
        if sample.pack != 0:
            raise NotImplementedError(f"Packed S3M samples are not supported yet (sample {sample_idx}).")
        if sample.is_stereo:
            raise NotImplementedError(f"Stereo S3M samples are not supported yet (sample {sample_idx}).")

    def _build_instrument_header(self, sample: S3MSample, sample_paragraph: int) -> bytes:
        header = bytearray(80)
        instrument_type = 1 if self._sample_slot_used(sample) and (sample.instrument_type == 1 or len(sample.waveform) > 0) else 0
        if sample.instrument_type in {2, 3, 4, 5, 6, 7}:
            instrument_type = sample.instrument_type
        header[0] = instrument_type
        header[1:13] = self._encode_text(sample.filename, 12)
        if instrument_type == 1:
            header[13] = (sample_paragraph >> 16) & 0xFF
            struct.pack_into('<H', header, 14, sample_paragraph & 0xFFFF)
            struct.pack_into('<I', header, 16, len(sample.waveform))
            loop_start = sample.repeat_point
            loop_end = sample.repeat_point + sample.repeat_len
            struct.pack_into('<I', header, 20, loop_start)
            struct.pack_into('<I', header, 24, loop_end)
            header[28] = max(0, min(64, sample.volume))
            header[29] = getattr(sample, '_reserved_byte', 0)
            header[30] = sample.pack
            flags = sample.flags & ~0x07
            if sample.repeat_len > 0:
                flags |= 0x01
            if sample.is_stereo:
                flags |= 0x02
            if sample.is_16bit:
                flags |= 0x04
            header[31] = flags
            struct.pack_into('<I', header, 32, sample.c2spd or 8363)
            header[36:48] = getattr(sample, '_internal', b'\x00' * 12)[:12].ljust(12, b'\x00')
            header[76:80] = getattr(sample, '_signature', 'SCRS').encode('latin-1', errors='replace')[:4].ljust(4, b'\x00')
        else:
            header[76:80] = getattr(sample, '_signature', 'SCRS').encode('latin-1', errors='replace')[:4].ljust(4, b'\x00')
        header[48:76] = self._encode_text(sample.name, 28)
        return bytes(header)

    def _sample_byte_length(self, sample: S3MSample) -> int:
        unit_size = 2 if sample.is_16bit else 1
        return len(sample.waveform) * unit_size

    def _encode_sample_data(self, sample: S3MSample) -> bytes:
        if len(sample.waveform) == 0:
            return b''
        if sample.is_16bit:
            if self.sample_type == 1:
                waveform = array.array('h', (int(value) for value in sample.waveform))
                if sys.byteorder != 'little':
                    waveform.byteswap()
                return waveform.tobytes()
            unsigned_waveform = array.array('H', (max(0, min(65535, int(value) + 32768)) for value in sample.waveform))
            if sys.byteorder != 'little':
                unsigned_waveform.byteswap()
            return unsigned_waveform.tobytes()

        if self.sample_type == 1:
            waveform = array.array('b', (max(-128, min(127, int(value))) for value in sample.waveform))
            return waveform.tobytes()
        unsigned_waveform = array.array('B', (max(0, min(255, int(value) + 128)) for value in sample.waveform))
        return unsigned_waveform.tobytes()

    def _encode_pattern_block(self, pat: Pattern) -> bytes:
        packed_data = bytearray()
        for row in range(pat.n_rows):
            for channel in range(pat.n_channels):
                note = pat.data[channel][row]
                raw_channel = self.compact_to_raw_channel[channel]
                have_note_inst = note.period != '' or note.instrument_idx != 0
                have_volume = 0 <= getattr(note, 'volume', -1) <= 64
                have_effect = self._effect_text(note.effect) != ''
                if not have_note_inst and not have_volume and not have_effect:
                    continue
                what = raw_channel
                if have_note_inst:
                    what |= 0x20
                if have_volume:
                    what |= 0x40
                if have_effect:
                    what |= 0x80
                packed_data.append(what)
                if have_note_inst:
                    packed_data.append(self._encode_note_value(note.period))
                    packed_data.append(note.instrument_idx)
                if have_volume:
                    packed_data.append(note.volume)
                if have_effect:
                    command, info = self._encode_effect(note.effect)
                    packed_data.append(command)
                    packed_data.append(info)
            packed_data.append(0)
        return struct.pack('<H', len(packed_data) + 2) + packed_data

    @staticmethod
    def _encode_note_value(period: str) -> int:
        period = period.strip().upper()
        if period == '':
            return 255
        if period == 'OFF':
            return 254
        if len(period) != 3 or period[1] not in {'-', '#'}:
            raise ValueError(f"Invalid S3M note format {period!r}.")
        pitch = period[:2]
        octave = int(period[2])
        if octave < 0 or octave > 15:
            raise ValueError(f"Invalid S3M note octave {period!r}.")
        semitone = Song.PERIOD_SEQ.index(pitch)
        return (octave << 4) | semitone

    @staticmethod
    def _encode_effect(effect: str) -> tuple[int, int]:
        effect = effect.strip().upper()
        if effect == '':
            return 0, 0
        if len(effect) != 3:
            raise ValueError(f"Invalid S3M effect format {effect!r}.")
        command = ord(effect[0]) - ord('A') + 1
        if command < 1 or command > 26:
            raise ValueError(f"Invalid S3M effect command {effect!r}.")
        return command, int(effect[1:], 16)

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

    def _preserved_effect(self, effect: str) -> str:
        effect = self._effect_text(effect)
        if effect == '':
            return ''
        if effect[0] in {'A', 'B', 'C', 'T'}:
            return effect
        if effect.startswith('SB') or effect.startswith('SE'):
            return effect
        return ''