from __future__ import annotations

import copy

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

        self.tracker_version = 0
        self.format_version = 1
        self.flags = 0
        self.global_volume = 64
        self.initial_speed = 6
        self.initial_tempo = 125
        self.master_volume = 64
        self.ultra_click_removal = 0
        self.sample_type = 2
        self.default_pan_flag = 0
        self.special = 0
        self.default_panning: list[int] = []

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
        new_song.tracker_version = self.tracker_version
        new_song.format_version = self.format_version
        new_song.flags = self.flags
        new_song.global_volume = self.global_volume
        new_song.initial_speed = self.initial_speed
        new_song.initial_tempo = self.initial_tempo
        new_song.master_volume = self.master_volume
        new_song.ultra_click_removal = self.ultra_click_removal
        new_song.sample_type = self.sample_type
        new_song.default_pan_flag = self.default_pan_flag
        new_song.special = self.special
        new_song.default_panning = copy.deepcopy(self.default_panning)
        new_song.channel_settings = list(self.channel_settings)
        new_song._rebuild_channel_mappings()
        return new_song

    def save(self, fname: str, verbose: bool = True):
        raise NotImplementedError("S3M save support is not implemented yet.")

    def load(self, fname: str, verbose: bool = True):
        raise NotImplementedError("S3M load support is not implemented yet.")

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