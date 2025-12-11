# 🎹 NodMOD

NodMOD is a Python library for manipulating MOD / XM files produced with the music trackers of yore.

The library simplifies core operations, abstracting away from the intricacies of the MOD format.

 - Load and save MOD / XM files
 - Create, edit, duplicate, and delete patterns
 - Load WAV samples with automatic bitrate conversion
 - Full editing, down to the individual notes and effects
 - Render the song as WAV

You can find thousands of open-source songs at [The Mod Archive](https://modarchive.org/) or [Amiga Music Preservation](https://amp.dascene.net/), among others.

Suggested players are [XMPlay](https://www.un4seen.com/) (Windows), [Qmmp](https://qmmp.ylsoftware.com/) (Linux), [MilkyTracker](https://milkytracker.org/downloads/) (MacOS).

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
song.load_from_file("music/spice_it_up.mod")

song.mute_channel(3)
song.mute_channel(4)

song.save_as_mod("music/ch1_2.mod")
```

## Requirements

- Python 3.11+
- Pydub 0.25.1+
- (*optional*) [ffmpeg](https://ffmpeg.org/download.html) with the `--enable-libopenmpt` configuration is required to render MOD files as WAV.

## How you can help

We seek to expand the library in several ways:

 - Add support for other formats: IT, S3M, MED, etc.
 - More advanced functions for composing music
 - Build a script language
 - Improve usability

Contributions are welcome through dm or pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.
