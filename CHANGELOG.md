# Changelog

All notable changes to this project will be documented in this file.

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
