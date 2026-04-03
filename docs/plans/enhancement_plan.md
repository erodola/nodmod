# NodMOD API Enhancements Plan

## Context

This plan defines reusable `nodmod` API improvements that will make ingestion and transformation tooling easier.

The focus is intentionally on **library-level primitives** that are broadly useful for apps that:

- read/inspect tracker data
- generate fixtures programmatically
- export samples and metadata
- perform static analysis without mutating songs

This is not a plan for app-specific compiler logic.

## Goals

1. Preserve MOD header data currently not surfaced (`restart_position` byte).
2. Provide stable read-only traversal/snapshot APIs so clients do not depend on internal mutable structures.
3. Remove indexing ambiguity in note/cell APIs (row/channel argument order).
4. Expose robust effect codec helpers for MOD-style effect fields.
5. Add direct PCM byte helpers for MOD samples (no resampling side effects).
6. Add lightweight probing API (capability/metadata without full song object usage).
7. Add in-memory ASCII dump API (`to_ascii`) to complement file-based dump methods.

## Non-goals

- No interpreter/timeline semantic engine inside `nodmod`.
- No Strudel-specific logic.
- No audio equivalence tooling.
- No breaking changes to existing method names/signatures in this phase.

## Current Pain Points (Observed)

1. `MODSong.save()` always writes restart byte `127`; load path does not expose it as public metadata.
2. External tools must read mutable internals (`patterns`, `pattern_seq`, `Pattern.data[channel][row]`) directly.
3. Existing APIs mix coordinate order:
   - `set_note(sequence, channel, row, ...)`
   - `get_note(sequence, row, channel)`
4. `Song.parse_effect()` is useful but still too low-level for many effect workflows.
5. Sample byte workflows currently go through `array('b')` internals or float/WAV APIs.
6. There is no first-class `probe_file()` for format detection + quick metadata/capabilities.
7. ASCII export is file-oriented (`save_ascii`), not string-oriented.

---

## API Proposal 1: MOD Restart Position Support

### Problem

The MOD restart/order-jump byte in the header is not represented as user-facing metadata, and save currently emits constant `127`. This blocks faithful header round-trips and metadata inspection tools.

### Public API

In `MODSong`:

```python
@property
def restart_position(self) -> int | None: ...

def get_restart_position(self, raw: bool = False) -> int | None: ...
def set_restart_position(self, position: int | None, *, raw: bool = False) -> None: ...
```

### Behavior Contract

- Store exact raw header byte internally (`0..255`).
- Normalized accessor behavior:
  - `raw=False`:
    - `127` -> `None` (traditional "no restart" sentinel)
    - `0..126` -> same value
    - out-of-range/odd values still returned as int for lossless visibility
  - `raw=True`: always return exact raw byte.
- `set_restart_position(None)` stores raw `127`.
- Save writes stored raw byte, not a hardcoded constant.
- Load always captures the raw byte.

### Backward Compatibility

- Existing code unaffected unless it depends on hardcoded `127` output.
- Defaults stay equivalent for newly created songs (`None`/raw `127`).

### Implementation Notes

- `modsong.py`:
  - add internal field initialization in `__init__`
  - read byte `data[951]` in `load()`
  - write restart byte in `save()`
- `song.py`:
  - include `restart_position` in `get_song_info()` if attribute exists

### Tests

- `dev/test_mod_restart_position.py`
  - new song defaults to raw `127`, normalized `None`
  - setting explicit value persists through save/load
  - loading existing MOD with non-default value preserves raw and normalized views
  - round-trip with unknown/raw edge values remains lossless

### Acceptance Criteria

- Save/load preserves restart byte exactly.
- Metadata APIs expose both normalized and raw views.

---

## API Proposal 2: Stable Read-Only Traversal and Snapshot

### Problem

Tools currently must bind to internal mutable layout (`Pattern.data[channel][row]`), which is fragile and easy to misuse.

### Public API

Add immutable view dataclasses in a new module (example: `src/nodmod/views.py`):

```python
@dataclass(frozen=True)
class CellView:
    sequence_idx: int
    pattern_idx: int
    row: int
    channel: int
    instrument_idx: int
    period: str
    effect: str

@dataclass(frozen=True)
class SampleView:
    sample_idx: int
    name: str
    length: int
    finetune: int
    volume: int
    loop_start: int
    loop_length: int

@dataclass(frozen=True)
class SongView:
    format: str
    songname: str
    artist: str
    sequence: tuple[int, ...]
    n_patterns: int
    n_channels: int
```

New methods:

```python
def iter_cells(self, *, sequence_only: bool = True):
    """Yield CellView in deterministic sequence,row,channel order."""

def iter_samples(self, *, include_empty: bool = True):
    """Yield SampleView for sample slots."""

def view(self) -> SongView:
    """Return immutable song-level summary view."""
```

### Behavior Contract

- Deterministic ordering.
- No mutable references returned.
- Sequence coordinates always include both `sequence_idx` and `pattern_idx`.
- Works uniformly across MOD/XM/S3M (with format-specific fields possibly omitted or set to defaults in generic views).

### Backward Compatibility

- Pure additive API.

### Implementation Notes

- Implement minimal generic base in `Song`, format-specific overrides where needed.
- Keep views intentionally lightweight; no raw waveform bytes in this layer.

### Tests

- `dev/test_views_api.py`
  - deterministic ordering
  - values match direct internal data
  - mutation of underlying song after snapshot does not mutate returned dataclasses

### Acceptance Criteria

- A client can build full effect histograms and usage scans without touching `Pattern.data`.

---

## API Proposal 3: Coordinate API Normalization (Row/Channel)

### Problem

Argument order currently differs across methods and encourages mistakes.

### Public API

Add canonical row-channel method family (do not remove legacy methods):

```python
def get_note_rc(self, sequence_idx: int, row: int, channel: int): ...
def set_note_rc(self, sequence_idx: int, row: int, channel: int, sample_idx: int, period: str, effect: str = ""): ...
def set_effect_rc(self, sequence_idx: int, row: int, channel: int, effect: str = ""): ...
def clear_note_rc(self, sequence_idx: int, row: int, channel: int): ...
```

For XM/S3M, mirror existing extra args (`vol_cmd`, `vol_val`, `volume`) in `set_note_rc`.

### Behavior Contract

- Canonical coordinate order is always `(sequence_idx, row, channel, ...)`.
- Legacy methods remain and internally delegate.
- New docs and examples prefer `_rc` APIs.

### Backward Compatibility

- No breaking changes.
- Optional deprecation warnings can be deferred to a later major release.

### Implementation Notes

- Implement wrappers first; then progressively route internals to canonical methods.
- Keep index validation and error messages identical.

### Tests

- `dev/test_coordinate_api_consistency.py`
  - legacy and canonical methods produce identical results
  - invalid indices raise same exception types/messages

### Acceptance Criteria

- Library has one obvious coordinate convention for new callers.

---

## API Proposal 4: Effect Codec Helpers

### Problem

`parse_effect()` exposes only command/value tuple. Many tools need richer effect decomposition and reliable re-encoding utilities.

### Public API

In a new module `src/nodmod/effects.py`:

```python
@dataclass(frozen=True)
class EffectInfo:
    raw: str
    command: str          # e.g. "F", "E", "0"
    arg: int              # 0..255
    x: int                # high nibble
    y: int                # low nibble
    is_extended: bool
    extended_cmd: str | None  # "E0".."EF" when command == "E"

def decode_mod_effect(effect: str) -> EffectInfo: ...
def encode_mod_effect(command: str, arg: int) -> str: ...
def split_xy(arg: int) -> tuple[int, int]: ...
def merge_xy(x: int, y: int) -> int: ...
```

Optional helpers:

```python
def is_speed_effect(effect: str) -> bool: ...
def is_tempo_effect(effect: str) -> bool: ...
```

### Behavior Contract

- Strict validation with clear exceptions for malformed strings.
- Output is uppercase canonical format.
- `decode_mod_effect(encode_mod_effect(cmd, arg))` round-trips exactly.

### Backward Compatibility

- Keep `Song.parse_effect()` unchanged.
- `Song.parse_effect()` may internally call new codec (optional).

### Implementation Notes

- Constrain this API to MOD-like 3-char effects first.
- Future extension for XM/S3M-specific command alphabets can be additive.

### Tests

- `dev/test_effect_codec.py`
  - parse/encode round-trip for representative commands (`0`, `B`, `C`, `D`, `E*`, `F`)
  - malformed inputs fail with precise errors

### Acceptance Criteria

- External analyzers can classify effects without ad-hoc string slicing.

---

## API Proposal 5: Direct PCM i8 Sample Helpers

### Problem

Programmatic tooling needs exact byte-level sample I/O. Existing APIs are WAV-based or float/resample-based and not ideal for raw extraction/import workflows.

### Public API

In `MODSong`:

```python
def get_sample_pcm_i8(self, sample_idx: int) -> bytes: ...

def set_sample_pcm_i8(
    self,
    sample_idx: int,
    pcm_i8: bytes | bytearray | memoryview | array.array,
    *,
    reset_meta: bool = False,
) -> None: ...

def set_sample_loop_bytes(self, sample_idx: int, start_byte: int, length_byte: int) -> None: ...
```

### Behavior Contract

- `get_sample_pcm_i8` returns exact signed 8-bit PCM bytes from sample waveform.
- `set_sample_pcm_i8` accepts raw signed byte stream; no normalization/resampling.
- `set_sample_loop_bytes` is explicit about units and aliases MOD loop semantics.
- Preserve existing metadata unless `reset_meta=True`.

### Backward Compatibility

- Additive only.

### Implementation Notes

- Handle `array('b')` and bytes-like inputs robustly.
- Keep sample index validation identical to existing sample APIs.

### Tests

- `dev/test_mod_pcm_helpers.py`
  - byte-for-byte round-trip
  - accepts all supported input types
  - loop byte helpers map correctly to internal loop fields

### Acceptance Criteria

- External tools can export/import sample bytes without touching internals.

---

## API Proposal 6: Lightweight Probe API

### Problem

There is no fast, standardized format/capability probe for a file path without client-specific try/except loading logic.

### Public API

New module `src/nodmod/probe.py`:

```python
@dataclass(frozen=True)
class ProbeResult:
    path: str
    detected_format: str | None        # "mod" | "xm" | "s3m" | None
    supported: bool
    loader: str | None                 # "MODSong" | "XMSong" | "S3MSong"
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    metadata: dict[str, object]        # lightweight, format-specific

def probe_file(path: str) -> ProbeResult: ...
def detect_format(path: str) -> str | None: ...
```

### Behavior Contract

- Does not instantiate full song unless needed.
- Reports clear unsupported reasons (e.g., unsupported MOD magic, unsupported S3M Adlib).
- MOD metadata should include at least:
  - `magic`
  - `song_length`
  - `restart_raw`
  - `max_pattern_index`
  - inferred channel count (if determinable)

### Backward Compatibility

- Pure additive API.

### Implementation Notes

- Keep probe resilient and never throw for expected malformed/unsupported cases; encode in `supported/errors`.
- Reserve exceptions for I/O failures (missing file, permissions) only.

### Tests

- `dev/test_probe_api.py`
  - valid MOD/XM/S3M detection
  - malformed file returns structured unsupported result
  - unsupported variants expose meaningful errors

### Acceptance Criteria

- Callers can build capability reports without manually loading all formats.

---

## API Proposal 7: In-memory ASCII Dumps (`to_ascii`)

### Problem

`save_ascii()` is file-path oriented. Many tools need the text representation in memory for logs, snapshots, APIs, and diffing.

### Public API

In `Song` (and/or per format overrides):

```python
def to_ascii(self, *, sequence_only: bool = True, include_headers: bool = False) -> str: ...
```

`save_ascii()` implementations should become thin wrappers around `to_ascii()`.

### Behavior Contract

- Deterministic output for stable snapshot testing.
- Optional headers for song/pattern metadata.
- Default output matches current visual style as closely as possible.

### Backward Compatibility

- Existing `save_ascii()` behavior preserved.

### Implementation Notes

- Implement once per format where needed; share common formatter helpers.
- Keep encoding concerns in `save_ascii()` (`ascii`/`utf-8`) while `to_ascii()` returns Python `str`.

### Tests

- `dev/test_ascii_api.py`
  - `save_ascii(path)` equals `to_ascii()` file contents
  - deterministic output across runs
  - includes expected row/channel formatting

### Acceptance Criteria

- Client code can generate debug dumps without temporary files.

---

## Cross-cutting Engineering Rules

1. Additive-first rollout: no removals or hard deprecations in this iteration.
2. Keep strict index/type validation style consistent with existing API.
3. Prefer immutable dataclasses for new read-only outputs.
4. Update docs (`README.md`) with concise examples for each new API family.
5. Add targeted tests before broad refactors.

## Suggested Implementation Order

1. Proposal 1 (`restart_position`) - smallest, highest fidelity impact.
2. Proposal 7 (`to_ascii`) - straightforward and low risk.
3. Proposal 5 (PCM helpers) - practical utility for many apps.
4. Proposal 4 (effect codec) - foundational for analyzers.
5. Proposal 3 (coordinate normalization wrappers) - API ergonomics.
6. Proposal 2 (views/snapshot iterators) - larger but high value.
7. Proposal 6 (probe API) - new module and structured reporting.

## File-level Change Map (Expected)

- `modsong.py`
  - restart metadata
  - PCM helpers
  - `to_ascii` and `save_ascii` delegation
- `song.py`
  - optional canonical wrappers / shared helper scaffolding
  - `get_song_info` extension
- `__init__.py`
  - export new APIs/modules
- `effects.py` (new)
- `views.py` (new)
- `probe.py` (new)
- `README.md`
  - examples and migration guidance
- `dev/` tests (new files listed above)

## QA / Acceptance Checklist

- Unit tests for each proposal pass.
- Existing `dev` test suite passes unchanged.
- Save/load round-trip parity preserved for unaffected fields.
- New APIs documented and imported from package root where appropriate.
- No behavior regressions in existing public methods.

## Migration Notes for Existing Callers

- Existing code can remain unchanged.
- New tooling should prefer:
  - `*_rc` APIs for coordinate clarity
  - `iter_cells()`/`view()` for inspection
  - `decode_mod_effect()` for effect analysis
  - `get_sample_pcm_i8()` for byte-level sample export
  - `to_ascii()` for in-memory diagnostics

