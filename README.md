# 🎹 NodMOD

NodMOD is a library for manipulating Amiga MOD files produced with the music trackers of yore.

The library simplifies core operations, abstracting away from the intricacies of the MOD format.

 * Load and save MOD files
 * Create, edit, duplicate, and delete patterns
 * Load WAV samples with automatic bitrate conversion
 * Full editing, down to the individual notes and effects
 * Render the song as WAV

Related repositories:
 * AmiGen: https://github.com/erodola/amigen

## Install

Tested with Python 3.11.

It's highly recommended to set up a virtual Python environment with `conda` or `virtualenv`:
```
conda create --name nodmod python=3.11
conda activate nodmod
```

Install Python dependencies:
```
python -m pip install -r requirements.txt
```

In order to use the render to WAV functions, [ffmpeg](https://ffmpeg.org/download.html) with the `--enable-libopenmpt` configuration is required.

## Music archive

There are many sources of free MOD music tracks to play with.

You can find thousands of open-source songs at [The Mod Archive](https://modarchive.org/) or [Amiga Music Preservation](https://amp.dascene.net/), among others.

## How you can help

We seek to expand the library in several ways:

 * Add support for other formats: XM, S3M, IT, MED, etc.
 * More advanced functions for composing music
 * Build a script language
 * Train a LLM to compose tracker modules
 * Improve usability

Contributions are welcome through pull requests.
