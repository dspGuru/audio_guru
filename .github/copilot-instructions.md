## Purpose

This repository is an extensible audio analysis toolkit with CLI support. It identifies tones, distortion products, noise floor, and computes audio metrics (THD, SNR, SFDR, etc). These instructions guide AI agents to be productive quickly through architecture patterns, workflows, conventions, and known implementation details.

## Big picture (what to know first)

**Layered Architecture:**
- **Audio Layer** (`audio.py`): Core data class — loads/generates audio, segments buffers, manages sample rate, detects silence, computes noise floor using `scipy.signal` and `soundfile`/`sounddevice`.
- **Frequency Analysis** (`freq_bins.py`): FFT pipeline — applies Blackman–Harris window → `scipy.fft.rfft` → stores window-compensated single-sided power in `self.bins`; provides bin↔freq conversion helpers.
- **Component Extraction** (`component.py`, `analyzer.py`): Abstract `Analyzer` base class + concrete implementations (`ToneAnalyzer`, `NoiseAnalyzer`, `SweepAnalyzer`, `TimeAnalyzer`). Each `analyze()` method identifies components, classifies them (tone, distortion, spur, noise), and computes statistics.
- **Statistics Output** (`tone_stats.py`, `time_stats.py`, `audio_stats.py`, `unit_stats.py`): Results containers with `to_dict()` and `summary()` formatting for tables, CSV export.
- **CLI Entry Points** (`analyze.py`, `generate.py`, `split.py`, `join.py`, `play.py`, `device_selector.py`) and supporting orchestration/I/O classes (`audio_analyzer.py`, `audio_player.py`, `audio_select.py`): argparse-based scripts and helpers.

**Data Flow:** Load audio (file or device) → segment/detect silence → FFT (freq_bins) → extract components (analyzer subclass) → classify (tone/distortion/spur) → compute metrics → format output.

## How to run (Windows PowerShell examples)

- Install dependencies:
```powershell
pip install -r requirements.txt
# or via pyproject.toml / uv: uv sync
```

- Generate test tones (`signals/` directory):
```powershell
python .\generate.py
```

- Analyze audio files matching a pattern (shows stats, components, noise floor):
```powershell
python .\analyze.py "samples\*.wav" -a -c 10 -s -v
# -a: audio stats (noise floor), -c N: top N frequency components, -s: unit specs, -v: verbose
```

- Play audio:
```powershell
python .\play.py "samples\*.wav" -l 15.0
```

- Run tests:
```powershell
uv run pytest tests/ -v --cov=.
```

## Project-specific conventions & patterns

**Analyzer Pattern:**
- All `Analyzer` subclasses inherit from `analyzer.Analyzer` (abstract base class).
- Each concrete analyzer (`ToneAnalyzer`, `NoiseAnalyzer`, `SweepAnalyzer`, `TimeAnalyzer`) implements:
  - `analyze()` → calls `super().analyze()` (which computes `TimeStats` and calls `get_components()`) and returns its specific stats object (`ToneStats`, `FreqResp`, or `TimeStats`).
  - `get_components()` → returns `Components` (list of `Component` objects with freq, power, category).
  - `reset()` → reinitialize working state.
  - `print()` → format and print results.
- Component categories (from `util.Category`): `Tone`, `Harmonic`, `Distortion`, `Spurious`, `Noise`, `Band`, `Sweep`, `Silence`, `Unknown`.

**Audio / Metadata:**
- `Audio` class in `audio.py` manages sample loading, segmentation, silence detection, and noise floor computation.
- `Metadata` in `metadata.py` stores audio name, segment info, sample rate (`fs`), etc.

**FFT / Bins conventions:**
- `FreqBins` in `freq_bins.py` applies a 7-term Blackman–Harris window, runs `scipy.fft.rfft`, and stores window-compensated single-sided power in `self.bins`.
- `bin_to_freq(bin)` and `freq_to_bin(freq)` are canonical conversions — use them for unit consistency.
- `tones` (`@cached_property`) extracts up to `MAX_TONES` peaks in descending power order and zeroes out their bins in a local copy to compute `self.noise`.

**Statistics Output:**
- `ToneAnalyzer` returns a `ToneStats` object with fields like `freq`, `thd_pct`, `snr_db`, `sfdr_db`; `TimeAnalyzer` returns a `TimeStats` object.
- `get_stats()`, `get_dataframe()`, and `summary()` methods provide formatting for tables and CSV.

**CLI conventions:**
- Add flags in `main()` function with argparse; document in script docstring.
- Use consistent argument names across scripts (e.g., `-c`/`--components`, `-s`/`--specs`).

## Integration points & dependencies

- **External libs:** `numpy`, `scipy` (signal, fft), `soundfile` (I/O), `sounddevice` (device audio), `pandas` (data frames), `pytest` / `pytest-cov` (testing), `ruff` (linting).
- **I/O:** `soundfile` + `Audio.read()` load WAV/FLAC preserving original sample rate; `sounddevice.InputStream` captures from device.
- **Audio generation:** `generate.py` creates test signals (tones, sweeps, noise, IMD products) and saves to `signals/` and `examples/` directories.

## Important implementation notes & gotchas

- **Two-tone classification:** `ToneAnalyzer.get_components()` compares second-largest power against adaptive `_two_tone_threshold` and distance to second harmonic to distinguish two-tone from single-tone with distortion.
- **Silence detection:** `Audio._detect_segments()` / `Audio.get_segments()` finds active and quiet regions; thresholds are configurable via `DEFAULT_SILENCE_THRESH_RATIO`, `SILENCE_MIN_THRESHOLD`, and `SILENCE_WINDOW_SECS` in `constants.py`.
- **DC removal:** `FreqBins.__init__()` subtracts the mean (`self.dc`) from the time-domain signal before windowing and FFT.
- **Noise floor:** Tracked via `Audio.avg_noise_floor` across noise/silence segments.
- **dB scaling:** `decibels.py` provides `db()` and `db_to_pwr_ratio()` bounded by `MIN_PWR` (`1e-12`, `-120 dB`) so zero/tiny powers yield `-120.0 dB` rather than `-inf`.
- **Sample rate preservation:** `Audio.read()` respects original file sample rate; any resampling must be explicit.

## Editing guidance (how to change code safely)

- **Numeric/algorithmic changes:** Add deterministic unit test in `tests/` using synthetic signals (`generate.py`) or files from `signals/` / `examples/`. Assert THD%, frequency, SNR within expected tolerances.
- **New metrics:** Update the relevant `Analysis` subclass (`TimeStats`, `ToneStats`, `FreqResp`, `UnitStats`) fields and formatting methods to stay consistent.
- **New analyzer:** Inherit from `Analyzer`, implement `analyze()`, `get_components()`, `reset()`, `print()`. Add corresponding test file `test_<analyzer_name>.py`.
- **CLI flags:** Add to argparse in script's `main()` function; ensure help text is clear and defaults match intended behavior.
- **Dependencies:** Update both `requirements.txt` and `pyproject.toml`; note that `pyproject.toml` is primary (`requires-python = ">=3.13"`).

## Key files to inspect for future changes

- `analyzer.py` — abstract base class for all analyzer implementations.
- `audio.py` — core Audio data class, I/O, silence detection, segment management.
- `tone_analyzer.py` — primary single/two-tone analyzer; exemplifies component extraction and distortion classification.
- `analyze.py` — main CLI entry point for batch file analysis.
- `constants.py` — all configurable thresholds and defaults (FFT window, silence detection, min/max frequencies, etc).
- `freq_bins.py` — FFT pipeline and bin↔frequency conversions.
- `component.py` — Component and Components data classes.
- `tests/test_tone_analyzer.py` — example of pattern-based test discovery and tolerance assertion.
