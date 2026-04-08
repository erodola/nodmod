# NodMOD

[![CI](https://github.com/erodola/nodmod/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/erodola/nodmod/actions/workflows/ci.yml?query=branch%3Amain)
[![Ruff](https://github.com/erodola/nodmod/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/erodola/nodmod/actions/workflows/lint.yml?query=branch%3Amain)
[![Release](https://img.shields.io/github/v/release/erodola/nodmod?sort=semver)](https://github.com/erodola/nodmod/releases/tag/v1.0.4)
[![PyPI version](https://img.shields.io/pypi/v/nodmod)](https://pypi.org/project/nodmod/)
[![Python versions](https://img.shields.io/pypi/pyversions/nodmod)](https://pypi.org/project/nodmod/)
[![PyPI downloads](https://img.shields.io/pypi/dm/nodmod)](https://pypi.org/project/nodmod/)
[![Wheel](https://img.shields.io/pypi/wheel/nodmod)](https://pypi.org/project/nodmod/)
[![License](https://img.shields.io/pypi/l/nodmod)](https://pypi.org/project/nodmod/)
[![Last Commit](https://img.shields.io/github/last-commit/erodola/nodmod)](https://github.com/erodola/nodmod/commits/main)
[![Formats](https://img.shields.io/badge/formats-MOD%20%7C%20XM%20%7C%20S3M-4c8bf5)](https://github.com/erodola/nodmod)

NodMOD is a Python library for reading, editing, and writing tracker modules.

Current stable release: `v1.0.4`.
Current tested Python support: `3.11`, `3.12`.

It currently focuses on three classic formats:

- MOD
- XM
- S3M

The project is built around direct programmatic editing. You load or create a song, manipulate patterns, notes, samples, instruments, and effects in Python, then save the result back to disk.

## What It Does

- Load MOD, XM, and S3M files
- Save edited or newly created MOD, XM, and S3M files
- Create, duplicate, resize, clear, and reorder patterns
- Edit notes, effects, rows, channels, samples, and XM instruments
- Import WAV audio into MOD samples or XM instrument samples
- Export human-readable ASCII dumps for inspection
- Render modules to WAV through external tools when available
- Validate/sanitize MOD sample-loop metadata before strict saves

## Installation

```bash
git clone https://github.com/erodola/nodmod.git
cd nodmod
uv venv
uv pip install -e .
```

Verify with:

```bash
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

python -c "import nodmod; print('nodmod installed successfully!')"
```

## Quick Start

```python
from nodmod import MODSong

song = MODSong()
song.load("music/spice_it_up.mod")

# MOD channels are 0-based (0-3)
song.mute_channel(2)
song.mute_channel(3)

song.save("music/ch1_2.mod")
```

```python
from nodmod import XMSong
from nodmod.types import XMSample

song = XMSong()
song.set_n_channels(4)

inst = song.new_instrument("Lead")
smp = XMSample()
import array

smp.waveform = array.array('b', [0, 10, -10, 0])
song.add_sample(inst, smp)
song.set_sample_map_all(inst, 1)

song.set_note(0, 0, 0, inst, "C-4", "")
song.set_sample_panning(inst, 1, 192)
song.set_global_volume(0, 0, 0, 64)

song.save("music/lead.xm")
```

```python
from nodmod import S3MSong

song = S3MSong()
song.set_n_channels(8)

song.set_note(0, 0, 0, 1, "C-4", effect="A03", volume=32)
song.set_note(0, 1, 4, 2, "G-4", effect="T96", volume=40)

song.save("music/sketch.s3m")
```

## Core Model

At a high level, the library exposes a small set of central objects:

- `MODSong` for MOD modules
- `XMSong` for XM modules
- `S3MSong` for S3M modules
- `Pattern` for pattern data
- `Note`, `XMNote`, and `S3MNote` for note cells
- `Sample`, `XMSample`, and `S3MSample` for waveform data
- `Instrument` for XM instruments and sample maps

The API is intentionally close to tracker structure rather than trying to hide it behind a higher-level composition DSL.

## Format Notes

- MOD notes reference samples directly.
- MOD read APIs (`get_note`, `iter_cells`, `iter_rows`, `get_used_samples`) resolve sample-memory semantics by default: note rows with sample `00` inherit the last latched sample in the same channel.
- In MOD, sample-only rows (sample set with empty note) are valid and update that channel's latched sample for later note rows.
- MOD sample slots expose loop safety helpers (`Sample.validate_loop`, `Sample.sanitize_loop`, `MODSong.validate_samples`, `MODSong.sanitize_samples`) and `save(..., validate_samples=True)` for strict pre-save validation.
- Use `get_note_raw(...)` (and `resolved=False` where available) when you need exact raw MOD cell sample nibbles.
- XM notes reference instruments, and instruments contain samples.
- S3M notes reference sample / instrument slots directly for PCM modules.
- MOD sample slots are 1-based in the public API.
- XM instrument indices and XM sample indices are 1-based in the public API.
- S3M sample slots are 1-based in the public API.
- Pattern order and pattern storage are separate concepts, as they are in tracker files.

Current S3M scope:

- PCM S3M modules are supported for load, edit, save, and round-trip tests.
- Adlib / OPL S3M instruments are detected and rejected explicitly; they are not supported yet.

For most day-to-day use, the practical rule is simple: sequence positions, rows, and channels behave like normal Python indices, while tracker sample and instrument slots follow tracker conventions.

## Extras

ASCII dumps are available for both formats:

```python
song.save_ascii("music/debug.txt")
```

Rendering can target mono or multi-channel output when `openmpt123` or a compatible `ffmpeg` build is available:

```python
song.render("music/render.wav", channels=2)
```

## New API Additions

Recent enhancements add inspection-focused, additive APIs without breaking existing method signatures.
The `v1.0.3` release hardens MOD/XM/S3M edge-case behavior, including fixed MOD channel invariants, XM serialization correctness, and timing/documentation consistency improvements.

```python
from nodmod import MODSong

song = MODSong()
song.set_restart_position(3)            # normalized view
raw = song.get_restart_position(raw=True)  # exact MOD header byte
```

```python
# Canonical coordinate order: (sequence_idx, row, channel, ...)
song.set_note_rc(0, 8, 1, 4, "C-4", "F06")
song.set_effect_rc(0, 8, 1, "B01")
cell = song.get_note_rc(0, 8, 1)
```

```python
from nodmod import decode_mod_effect, encode_mod_effect

info = decode_mod_effect("E6F")
assert info.extended_cmd == "E6"
assert encode_mod_effect("F", 125) == "F7D"
```

```python
# Immutable snapshots for read-only analysis
summary = song.view()
cells = list(song.iter_cells(sequence_only=True))
rows = list(song.iter_rows(sequence_only=True))
effects = list(song.iter_effects(sequence_only=True, include_empty=False))
samples = list(song.iter_samples(include_empty=False))
```

```python
# MOD raw signed 8-bit PCM helpers
pcm = song.get_sample_pcm_i8(1)
song.set_sample_pcm_i8(1, pcm, reset_meta=False)
song.set_sample_loop_bytes(1, start_byte=128, length_byte=256)
```

```python
from nodmod import probe_file

probe = probe_file("music/demo.mod")
print(probe.detected_format, probe.supported, probe.metadata)
```

```python
# In-memory ASCII dump (no temp files needed)
text = song.to_ascii(sequence_only=True, include_headers=False)
```

```python
# Playback-aware row timeline with source coordinates
playback = list(song.iter_playback_rows(max_steps=250_000))
first = playback[0]
print(first.sequence_idx, first.pattern_idx, first.row, first.start_sec, first.end_sec)
```

```python
# Reachability-aware used-resource scans
mod_used = song.get_used_samples(scope="reachable", order="first_use")
xm_insts = xm_song.get_used_instruments(scope="sequence", order="sorted")
xm_samples = xm_song.get_used_samples(scope="reachable", order="sorted")
s3m_used = s3m_song.get_used_samples(scope="reachable", order="sorted")
```

```python
# One-call loading with format dispatch
from nodmod import load_song

song = load_song("music/demo.mod")
```

Scope semantics:

- `scope="sequence"` inspects every row in sequence-referenced patterns.
- `scope="reachable"` inspects rows actually visited during playback flow.

## Requirements

- Python 3.11 or 3.12 (current tested support)
- Python 3.13+ compatibility is tracked in [#30](https://github.com/erodola/nodmod/issues/30)
- pydub 0.25.1+
- Optional: `openmpt123` or `ffmpeg` for WAV rendering

## Project Status

This is an editing-focused library, not yet a full tracker toolkit. The codebase is strongest in direct manipulation of existing songs and in generating small-to-medium scripted edits. The public API is still evolving.

## Contributing

Useful contributions include:

- bug fixes and behavioral cleanup
- stronger round-trip coverage for MOD, XM, and S3M files
- better public examples
- support for more tracker operations and formats

## Reference Material

Large collections of legal-to-study modules can be found at [The Mod Archive](https://modarchive.org/) and [Amiga Music Preservation](https://amp.dascene.net/).

Good players and editors for checking output include [XMPlay](https://www.un4seen.com/), [Qmmp](https://qmmp.ylsoftware.com/), and [MilkyTracker](https://milkytracker.org/downloads/).

## License

MIT License. See [LICENSE](LICENSE).
