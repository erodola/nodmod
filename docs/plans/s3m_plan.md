# S3M Implementation Plan

This file is the implementation plan for adding S3M support to NodMOD.

The goal is to add S3M in a way that is:

- clean
- modular
- correct
- aligned with the existing MOD/XM API shape
- validated against both synthetic tests and a real external corpus

## Constraints

- The external corpus at `G:\My Drive\Moduli` is read-only and must never be modified.
- Every meaningful implementation step should be committed separately.
- Strong tests must live under `dev/`.
- Simplicity and correctness take priority over speculative abstraction.

## Research Summary

Sources consulted:

- https://moddingwiki.shikadi.net/wiki/S3M_Format
- https://wiki.multimedia.cx/index.php?title=Scream_Tracker_3_Module

Important format facts confirmed from those sources:

- S3M uses a header with order list, instrument parapointers, and pattern parapointers.
- Parapointers are paragraph offsets and must be multiplied by 16.
- Patterns are always 64 rows and use packed row data.
- Packed pattern cells can independently carry note+instrument, volume, and effect bytes.
- S3M timing uses `Axx` for speed and `Txx` for tempo, not `Fxx`.
- `Bxx` is order jump and `Cxy` is decimal row break.
- PCM instruments are sample-based and carry `c2spd`, loop info, flags, and titles.
- S3M can also contain Adlib / OPL instruments.
- Header `sampleType` controls signed vs unsigned sample storage; the real corpus is consistently unsigned.
- Channel layout is defined by a 32-byte header table and can include unused and disabled channels.

Empirical scan of the external corpus:

- 87 files with `.s3m` / `.S3M` extension were found.
- 86 are valid S3M files with the `SCRM` signature.
- 86 valid files are PCM-only.
- 0 valid files in the scanned corpus contain Adlib / OPL instruments.
- The valid corpus uses header `sampleType = 2` (unsigned sample storage).
- Default panning is present in most files (`defaultPan = 252` is common).
- Real files use a range of active channel counts up to 32.

That corpus data matters. It means we should implement PCM S3M end-to-end immediately, and handle Adlib explicitly and honestly rather than faking support.

## Design Decisions

### 1. Add a dedicated `S3MSong`

Add a new source module:

- `src/nodmod/s3msong.py`

and export it from:

- `src/nodmod/__init__.py`

This should be a peer of `MODSong` and `XMSong`, not a variant hidden inside either.

### 2. Keep S3M sample-based in the public API

S3M notes reference instrument numbers, but for PCM S3M those instrument numbers behave like sample slots from the user point of view.

Public API alignment should therefore follow the MOD side more than the XM side:

- `get_sample()` / `set_sample()` / `load_sample()` / `save_sample()` should exist on `S3MSong`
- `set_note()` should take a 1-based sample slot index in the note cell, as MOD does
- sample-slot APIs should be 1-based in the public API

### 3. Do not force S3M through the existing base tempo/effect helpers

The current `Song` helpers are MOD/XM-shaped around `Fxx` tempo/speed semantics.

S3M is different enough that forcing it into those helpers would create the wrong model.

Decision:

- keep `Song` stable unless a tiny refactor is clearly beneficial
- implement S3M timing/effect helpers directly in `S3MSong`
- override `set_bpm()`, `set_ticks_per_row()`, and `timestamp()` in `S3MSong`
- use S3M-specific preserved-effect rules instead of broadening `Song` prematurely

### 4. Preserve raw S3M metadata needed for stable round-trips

To make load->save->load and load->save byte comparisons realistic, `S3MSong` should preserve format-specific metadata rather than normalizing it away.

At minimum:

- header flags
- tracker version
- sample type
- global volume
- initial speed
- initial tempo
- master volume
- ultra-click removal
- default-panning flag
- reserved header bytes
- special parapointer presence and panning table
- raw channel settings
- sample filenames, titles, `c2spd`, flags, pack type, and loop fields

### 5. Represent channels in the public API as compact indices

S3M stores 32 possible raw channel slots, but only some are active.

Public API should stay consistent with MOD/XM and expose channel indices as compact `0..n_channels-1`.

Internally we should preserve:

- the raw 32-entry channel settings table
- a compact-to-raw slot mapping
- a raw-to-compact reverse mapping for pattern unpacking/packing

This avoids leaking S3M header quirks into the user-facing API.

### 6. Scope Adlib support explicitly

Because the corpus is PCM-only, the first clean target is:

- full PCM S3M load/save/edit support
- explicit `NotImplementedError` when a file contains Adlib instruments

This is not a hand-wave. It is a scoped implementation boundary backed by corpus evidence.

If Adlib support becomes necessary later, it should be added as a distinct follow-up with dedicated types and tests rather than polluting the PCM implementation.

## Proposed Data Model

### `S3MNote`

Add a dedicated note class for S3M.

Reason:

- S3M has a volume column like XM, but not XM's volume-command model
- reusing `XMNote` would be semantically wrong and would drag unnecessary fields through the code

Suggested shape:

- inherit from `Note`
- add `volume: int = -1` where `-1` means "no explicit volume column"
- represent note-off as `period = 'off'`

### `S3MSample`

Add a dedicated sample class for S3M.

Reason:

- S3M PCM samples need metadata not present on `Sample`
- we want round-trip fidelity without polluting MOD/XM sample classes

Suggested fields in addition to the base `Sample` fields:

- `filename`
- `c2spd`
- `pack`
- `is_16bit`
- `is_stereo`
- `dsk`
- raw reserved/internal bytes used for round-trip fidelity

If we later support Adlib, that should be a separate type, not crammed into `S3MSample`.

## Implementation Phases

### Phase 1. Scaffolding and type definitions

Files:

- `src/nodmod/types.py`
- `src/nodmod/s3msong.py`
- `src/nodmod/__init__.py`

Work:

- add `S3MNote`
- add `S3MSample`
- create `S3MSong` skeleton with header fields, channel mapping fields, and a default empty song
- export the new classes from the package root

Validation:

- import smoke test from a new `dev/test_s3m_basic.py`
- constructor invariants: default pattern, default sequence, default sample bank, sane channel mapping

Commit boundary:

- one commit for types + song skeleton + import/export wiring

### Phase 2. Header parsing and song-level metadata load

Files:

- `src/nodmod/s3msong.py`
- `dev/test_s3m_basic.py`

Work:

- parse the S3M header
- validate signatures and basic fields
- load order list, including handling `0xFE` marker and `0xFF` end-of-song correctly
- load raw channel settings and derive compact channel mappings
- load optional default panning table if present
- store tracker/version flags needed for round-trip saving

Validation:

- synthetic header parsing tests
- a corpus smoke test that opens a handful of real files and verifies header invariants

Commit boundary:

- one commit for header-level load support and smoke tests

### Phase 3. PCM instrument/sample loading

Files:

- `src/nodmod/s3msong.py`
- `dev/test_s3m_sample_api.py`
- `dev/test_s3m_load_all.py`

Work:

- parse PCM instrument headers
- load waveform data according to `sampleType`, 8-bit / 16-bit flags, loop info, and `c2spd`
- convert unsigned on-disk sample data into the internal signed array representation used by the library
- implement sample-slot public API aligned with MOD where appropriate:
  - `get_sample()`
  - `set_sample()`
  - `load_sample()`
  - `load_sample_from_raw()`
  - `save_sample()`
  - `copy_sample_from()`
- detect Adlib instruments and raise clear `NotImplementedError`

Validation:

- synthetic sample API tests
- real-corpus load-all test that counts successes and explicit Adlib skips/failures
- sample load/save smoke tests to WAV through existing helper patterns

Commit boundary:

- one commit for PCM instrument/sample loading and sample API support

### Phase 4. Pattern unpacking and note decoding

Files:

- `src/nodmod/s3msong.py`
- `dev/test_s3m_basic.py`

Work:

- unpack the 64-row packed pattern format
- map raw S3M channel slots to compact public channel indices
- decode:
  - note byte
  - instrument/sample slot
  - volume column
  - effect command/info
- convert S3M note values into library note strings
- represent empty note as `''` and note-off as `'off'`

Validation:

- synthetic packed-pattern tests with hand-constructed bytes
- corpus smoke tests checking row counts, channel counts, and basic note extraction

Commit boundary:

- one commit for pattern decoding and note extraction

### Phase 5. Editing API alignment

Files:

- `src/nodmod/s3msong.py`
- `dev/test_s3m_basic.py`
- `dev/test_song_base.py` if shared expectations need extension

Work:

- implement or align:
  - `add_pattern()`
  - `resize_pattern()`
  - `clear_pattern()`
  - `get_note()`
  - `set_note()`
  - `set_effect()`
  - channel add/remove/clear/mute behavior
- keep naming and index validation aligned with the recent indexing cleanup work
- preserve explicit volume column semantics in S3M note writes

Validation:

- strong unit-style dev tests mirroring the MOD/XM basic suites
- negative-index and out-of-range checks aligned with current library conventions

Commit boundary:

- one commit for public editing API support and validation

### Phase 6. Saving and packed pattern encoding

Files:

- `src/nodmod/s3msong.py`
- `dev/test_s3m_roundtrip_write.py`

Work:

- write header, order table, instrument parapointers, pattern parapointers
- write PCM instrument/sample headers and sample data
- pack patterns back into S3M row encoding
- write optional panning table when needed
- preserve enough raw metadata for stable output on previously loaded files

Validation:

- generated-song save/load round-trip tests
- semantic round-trip comparison on corpus files
- byte-for-byte round-trip test where preservation is achievable for untouched PCM-only files

Commit boundary:

- one commit for save support and generated round-trip tests

### Phase 7. Timing and effect behavior

Files:

- `src/nodmod/s3msong.py`
- `dev/test_s3m_timing_functions.py`

Work:

- implement `timestamp()` using S3M timing rules
- implement S3M-specific speed/tempo handling via `Axx` and `Txx`
- support order jump / row break handling for timing
- implement `set_bpm()` and `set_ticks_per_row()` using S3M effects
- preserve S3M global effects when muting or rewriting notes where appropriate

Initial correctness target:

- timing correctness for header speed/tempo, `Axx`, `Txx`, `Bxx`, and `Cxy`

Later-effect fidelity such as full ST3 memory behavior should be added only if needed by tests.

Validation:

- synthetic timing tests similar in spirit to current MOD/XM timing suites

Commit boundary:

- one commit for timing/effect helpers and timing tests

### Phase 8. Real-corpus validation

Files:

- `dev/test_s3m_load_all.py`
- `dev/test_s3m_roundtrip_write.py`

Work:

- add broad load-all coverage over the read-only corpus
- add semantic round-trip coverage over real files
- record and skip unsupported files explicitly rather than silently passing
- make failures easy to diagnose by reporting first mismatch locations or field mismatches

Validation:

- run the full S3M corpus pass
- confirm PCM-only files load cleanly
- confirm untouched PCM files save and reload equivalently

Commit boundary:

- one commit for full corpus validation tooling and tests

### Phase 9. Documentation and package surface

Files:

- `README.md`
- optionally `dev/test_full_suite.py`

Work:

- update README to list S3M as supported
- add S3M tests to the aggregated dev runner if appropriate
- keep documentation concise and honest about current Adlib scope

Validation:

- smoke run of the updated test aggregator if modified

Commit boundary:

- one commit for docs and suite wiring

## Test Plan

Planned dev files:

- `dev/test_s3m_basic.py`
- `dev/test_s3m_sample_api.py`
- `dev/test_s3m_load_all.py`
- `dev/test_s3m_roundtrip_write.py`
- `dev/test_s3m_timing_functions.py`

Comparison helpers to add to `dev/test_helpers.py`:

- `compare_s3m_songs()`

That helper should compare at least:

- song header fields relevant to playback and save fidelity
- sequence
- compact channel count and channel settings
- pattern note/effect/volume content
- sample metadata
- waveform signatures

## Validation Commands

During development, validate after each phase with targeted scripts rather than waiting for one large end run.

Expected command style:

- `PYTHONPATH=src c:/Users/erodo/Workspace/nodmod/.venv/Scripts/python.exe dev/test_s3m_basic.py`
- `PYTHONPATH=src c:/Users/erodo/Workspace/nodmod/.venv/Scripts/python.exe dev/test_s3m_sample_api.py`
- `PYTHONPATH=src c:/Users/erodo/Workspace/nodmod/.venv/Scripts/python.exe dev/test_s3m_timing_functions.py`
- `PYTHONPATH=src c:/Users/erodo/Workspace/nodmod/.venv/Scripts/python.exe dev/test_s3m_load_all.py`
- `PYTHONPATH=src c:/Users/erodo/Workspace/nodmod/.venv/Scripts/python.exe dev/test_s3m_roundtrip_write.py`

## Refactoring Assessment

Refactors that are acceptable early:

- adding dedicated S3M note/sample classes
- adding a comparison helper for S3M tests
- adding tiny shared binary helpers if they clearly remove duplication

Refactors to avoid unless proven necessary:

- redesigning `Song` around a generic effect grammar
- merging MOD and S3M sample code prematurely
- changing XM/MOD behavior while introducing S3M

The guiding rule is to isolate S3M-specific complexity inside `S3MSong` unless a shared abstraction is already obviously present.

## Recommended First Execution Order

1. add `S3MNote`, `S3MSample`, and `S3MSong` scaffolding
2. parse S3M header/order/channel metadata
3. load PCM instruments and samples
4. decode packed patterns and expose `get_note()`
5. implement editing APIs aligned with MOD/XM
6. implement save path
7. add generated round-trip tests
8. add real-corpus load and round-trip tests
9. implement and validate timing helpers
10. update docs and suite wiring

This order keeps the risky binary I/O and validation work in small, reviewable slices.