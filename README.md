# audio_guru
Audio device test utilities

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Scripts

### analyze.py
Analyzes audio files matching a specified pattern. It provides detailed statistics and can display signal components and noise floor information.

**Usage:**
```bash
python analyze.py [options] <pattern>
```

### generate.py
Generates a set of example audio files (sine waves, noise, sweeps, distortion tests, etc.) in the `examples/` directory.

**Usage:**
```bash
python generate.py
```

### join.py
Joins multiple audio files into a single continuous file. You can specify a duration of silence to insert between each file.

**Usage:**
```bash
python join.py -w <output_file> [-s <silence_seconds>] <input_files>
```

### split.py
Splits audio files into separate segments (typically based on silence) and saves them as individual files.

**Usage:**
```bash
python split.py [options] <pattern>
```

## Examples

The `examples/` directory contains various audio test files used for automated testing and device verification. These files can be generated using `generate.py`.

Examples include:
- **Tones**: Pure sine waves at various frequencies and THD levels for distortion testing.
- **Noise**: White noise for noise floor analysis.
- **Two-Tone Signals**: Combinations like 60 Hz/7 kHz (SMPTE IMD) and 19 kHz/20 kHz (CCIF IMD).
- **Sweep**: Logarithmic frequency sweep (20 Hz - 20 kHz) for frequency response analysis.

See [examples/README.md](examples/README.md) for more details.
