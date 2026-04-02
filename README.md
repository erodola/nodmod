# NodMOD

NodMOD is a Python library for reading, editing, and writing tracker modules.

It currently focuses on two classic formats:

- MOD
- XM

The project is built around direct programmatic editing. You load or create a song, manipulate patterns, notes, samples, instruments, and effects in Python, then save the result back to disk.

## What It Does

- Load MOD and XM files
- Save edited or newly created MOD and XM files
- Create, duplicate, resize, clear, and reorder patterns
- Edit notes, effects, rows, channels, samples, and XM instruments
- Import WAV audio into MOD samples or XM instrument samples
- Export human-readable ASCII dumps for inspection
- Render modules to WAV through external tools when available

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

## Core Model

At a high level, the library exposes a small set of central objects:

- `MODSong` for MOD modules
- `XMSong` for XM modules
- `Pattern` for pattern data
- `Note` and `XMNote` for note cells
- `Sample` and `XMSample` for waveform data
- `Instrument` for XM instruments and sample maps

The API is intentionally close to tracker structure rather than trying to hide it behind a higher-level composition DSL.

## Format Notes

- MOD notes reference samples directly.
- XM notes reference instruments, and instruments contain samples.
- MOD sample slots are 1-based in the public API.
- XM instrument indices and XM sample indices are 1-based in the public API.
- Pattern order and pattern storage are separate concepts, as they are in tracker files.

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

## Requirements

- Python 3.11
- pydub 0.25.1+
- Optional: `openmpt123` or `ffmpeg` for WAV rendering

## Project Status

This is an editing-focused library, not yet a full tracker toolkit. The codebase is strongest in direct manipulation of existing songs and in generating small-to-medium scripted edits. The public API is still evolving.

## Contributing

Useful contributions include:

- bug fixes and behavioral cleanup
- stronger round-trip coverage for MOD and XM files
- better public examples
- support for more tracker operations and formats

## Reference Material

Large collections of legal-to-study modules can be found at [The Mod Archive](https://modarchive.org/) and [Amiga Music Preservation](https://amp.dascene.net/).

Good players and editors for checking output include [XMPlay](https://www.un4seen.com/), [Qmmp](https://qmmp.ylsoftware.com/), and [MilkyTracker](https://milkytracker.org/downloads/).

## License

MIT License. See [LICENSE](LICENSE).
