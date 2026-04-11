# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.0.5] - 2026-04-11

### API Stability

All public APIs in this release are considered stable.

### Added

- Added Python compatibility guardrail tests ensuring `import nodmod` does not require `pydub` at module import time and that WAV APIs fail with clear scoped messages when optional dependencies are missing.
- Added fixture-hash WAV stability tests (`tests/test_wav_fixture_hashes.py`) asserting deterministic sample load/save behavior for MOD/XM on a canonical WAV fixture.

### Changed

- Expanded declared and tested Python support to `3.10` through `3.14`.
- Added conditional runtime dependency `audioop-lts` for Python `>=3.13`.
- Moved `pydub` imports from module import-time to WAV-only code paths in `modsong` and `xmsong` to avoid import-time failures on Python versions where `audioop` is absent.
- Updated CI test matrix to run on Python `3.10`, `3.11`, `3.12`, `3.13`, and `3.14`.

### Documentation

- Updated release metadata references in README and project version fields for `v1.0.5`.

## [1.0.4] - 2026-04-08

### Changed

- Updated package/runtime metadata to `v1.0.4`.
- Clarified current tested Python support as 3.11 and 3.12 in README.
- Restricted package install metadata to Python `>=3.11,<3.13` to match current validated support.
- Updated CI to test both Python 3.11 and 3.12.
- Set the publish workflow to use a dedicated `pypi` GitHub Actions environment for Trusted Publishing.

## [1.0.3] - 2026-04-06

### API Stability

All public APIs in this release are considered stable.

### Added

- Added MOD regression coverage for fixed four-channel reporting, unreferenced-pattern save behavior, and decoded note helper values.
- Added XM regressions for explicit `get_pattern_duration(...)` failure mode and header instrument-count serialization from live instruments.
- Added filename parsing coverage for multi-dash names (`artist - song - extra.ext`) to preserve full song titles.

### Changed

- MOD channel reporting is now invariant at four channels (`MODSong.n_channels`) even when no patterns are present.
- XM set-volume byte encoding now uses a contiguous, clamped `0x10..0x50` mapping for values `0..64`.
- XM save now serializes instrument count from the live `instruments` list.
- MOD now relies on the shared base-class ASCII save path (`Song.save_ascii(...)`) instead of a duplicated override.
- Repository ignore rules now include local `plans/` workspace artifacts.

### Fixed

- `artist_songname_from_filename(...)` now splits on the first separator only, preserving titles containing additional ` - ` substrings.
- S3M timestamp tuples now preserve effective row speed/BPM values by removing a dead overwrite.
- `XMSong.get_pattern_duration(...)` now raises `NotImplementedError` explicitly instead of returning `None`.

### Testing

- Sample waveform comparisons in MOD/XM/S3M round-trip helpers now assert full waveform equality rather than head/tail signatures.

### Documentation

- Clarified timing API semantics across Song/MOD/XM/S3M (`timestamp` and `iter_playback_rows`), including current format-specific behavior and reserved playback parameters.
- Added inline MOD timing/effect notes for PAL/NTSC constants and Dxx decimal-BCD row-break decoding.

## [1.0.2] - 2026-04-05

### API Stability

All public APIs in this release are considered stable.

### Added

- Added explicit raw MOD note access via `get_note_raw(...)`, plus `resolved` flags on MOD traversal/resource helpers where applicable.
- Added loop-safety APIs for MOD sample workflows: `Sample.validate_loop`, `Sample.sanitize_loop`, `MODSong.validate_samples`, `MODSong.sanitize_samples`, plus strict save gate `MODSong.save(..., validate_samples=True)`.
- Added Ruff lint configuration and CI lint workflow.

### Changed

- MOD read APIs now expose effective sample-memory semantics by default (`get_note`, `iter_cells`, `iter_rows`, `get_used_samples`): note rows with raw sample `00` inherit the last latched sample on that channel.
- `MODSong.load(...)` now reads `songname` from the MOD header by default; filename-derived metadata is now opt-in via `metadata_from_filename=True`.

### Performance

- MOD effective-sample resolution now uses mutation-versioned lazy caches: rebuild on first read after mutation, then O(1) lookups until the next mutation.

### Testing

- Added explicit MOD tests for sample-only latch semantics (sample-without-note updates channel memory; latch survives empty rows and pattern boundaries).
- Added dedicated tests for loop metadata validation/sanitization and strict save-time loop validation.

### Documentation

- README updated for `v1.0.2`, including release highlights and API stability link.

## [1.0.1] - 2026-04-04

### API Stability

All public APIs in this release are considered stable.

### Added

- Loader convenience API: `load_song(...)` with automatic MOD/XM/S3M dispatch.
- Motif-analysis traversal APIs: `RowView`/`iter_rows(...)`, `EffectView`/`iter_effects(...)`, and playback timeline `iter_playback_rows(...)`.
- Reachability-aware resource scans for used samples and XM instruments (`scope=sequence|reachable`, `order=first_use|sorted`).
- Extended `CellView` parity fields for XM/S3M volume-column and S3M volume semantics.

### Fixed

- Base traversal APIs now skip invalid sequence pattern indices instead of failing on malformed `pattern_seq` entries.
- XM sample I/O test now uses temporary paths and no longer assumes a pre-existing `dev/` directory in CI.

### Testing

- Added fixture-based equivalence tests that compare convenience APIs against manual workflows on the canonical MOD/XM/S3M songs in `music/`.

### Documentation

- README expanded with release highlights and usage examples for motif-analysis traversal and reachability helpers.

## [1.0.0] - 2026-04-03

### API Stability

All public APIs in this release are considered stable.

### Added

- MOD restart-position metadata support with normalized/raw accessors and round-trip persistence.
- In-memory ASCII dumps via `to_ascii(...)` and `save_ascii(...)` delegation.
- Direct MOD PCM i8 sample helpers for byte-exact import/export and explicit loop-byte setters.
- MOD effect codec helpers in `nodmod.effects` (`decode_mod_effect`, `encode_mod_effect`, nibble helpers).
- Canonical row/channel note and effect wrappers (`*_rc`) while preserving legacy method signatures.
- Immutable traversal/snapshot APIs in `nodmod.views` (`SongView`, `CellView`, `SampleView`, iterators).
- Lightweight probing APIs in `nodmod.probe` (`detect_format`, `probe_file`, structured `ProbeResult`).

### Documentation

- README updated with examples for all new API families introduced for 1.0.0.
