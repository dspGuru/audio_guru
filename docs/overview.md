# Audio Analysis Methods in audio_guru

## Executive Summary

**audio_guru** is a Python-based audio test and measurement toolkit for analyzing audio signals and characterizing audio equipment performance. It implements digital signal processing (DSP) techniques to automatically segment recordings, classify signals (silence, tone, sweep, noise), and compute audio quality metrics including THD (Total Harmonic Distortion), THD+N, SNR (Signal-to-Noise Ratio), SFDR (Spurious-Free Dynamic Range), passband ripple, `-3 dB` band edges, and quadrature phase/magnitude balance.

For architectural and class-hierarchy details, see [design.md](design.md).

---

## Repository Architecture

### Core Data & I/O Modules

| Module | Purpose |
|--------|---------|
| `audio.py` | Audio sample container, file/device I/O, silence segmentation, category & quadrature detection |
| `segment.py` | Contiguous audio region descriptors (`Segment`, `Segments`) |
| `metadata.py` | Source audio metadata (`name`, `segment`, `fs`, `unit_id`) |
| `audio_list.py` | Collection of `Audio` objects for batch file reading/writing |
| `audio_player.py` | Audio file playback via `sounddevice` |
| `audio_select.py` | Audio device enumeration, categorization, and sorting |
| `constants.py` | Library-wide physical and empirical constants and thresholds |
| `util.py` | Core enums (`Category`, `Channel`) and frequency aliasing helper (`alias_freq`) |

### Core Analysis Modules

| Module | Purpose |
|--------|---------|
| `audio_analyzer.py` | Multi-segment orchestration, background worker thread, and result aggregation |
| `analyzer.py` | Abstract `Analyzer` base class |
| `tone_analyzer.py` | Single-tone and two-tone analysis (harmonics, intermodulation distortion, spurs, noise) |
| `noise_analyzer.py` | ISO 266 one-third octave band analysis and noise frequency response |
| `sweep_analyzer.py` | STFT-based frequency sweep tracking and frequency response analysis |
| `time_analyzer.py` | Time-domain signal analysis (silence and unclassified segments) |

### Statistics & Metrics Modules

| Module | Purpose |
|--------|---------|
| `analysis.py` | Abstract `Analysis` base class for all result containers |
| `tone_stats.py` | Tone metrics (`thd`, `thdn`, `snr`, `sfdr` in linear ratio, dB, and %) |
| `time_stats.py` | Time-domain metrics (`rms`, `dc`, `min`, `max`, `crest`, quadrature `mag_mean`, `mag_std`, `phase_err`) |
| `freq_resp.py` | Frequency response metrics (passband ripple and `-3 dB` lower/upper band edges) |
| `audio_stats.py` | Aggregate statistics across analyzed segments |
| `unit_stats.py` | Device-under-test (DUT) unit specification summary across measurements |

### Signal Processing Utilities

| Module | Purpose |
|--------|---------|
| `freq_bins.py` | 7-term Blackman-Harris windowing (`blackmanharris7`), normalized `rfft`, peak & tone extraction |
| `component.py` | Spectral component representation (`Component`, `Components`), band aggregation, and ripple |
| `decibels.py` | Power and voltage decibel conversions (`db`, `dbv`, `db_to_pwr_ratio`, `dbv_to_amp_ratio`) |

---

## Signal Analysis Methods

### 1. Fast Fourier Transform (FFT) & Spectral Power (`freq_bins.py`)

Spectral analysis is encapsulated in `FreqBins`:

- **DC Removal**: The time-domain mean (`self.dc = float(np.mean(x))`) is computed and subtracted prior to windowing.
- **Windowing (`blackmanharris7`)**: Applies a periodic (`sym=False`) 7-term minimum-sidelobe Blackman-Harris window evaluated via a degree-6 Chebyshev polynomial in `c = cos(θ)` (from O. M. Solomon, Jr., *IEEE Trans. Instrum. Meas.*, 1994).
- **Window Power Compensation**: Forward-normalized `scipy.fft.rfft(xw, norm="forward")` squared magnitudes are divided by the window's mean-square coherent/incoherent power factor $S_2 = \frac{1}{M}\sum w[n]^2 = a_0^2 + \frac{1}{2}\sum_{k=1}^6 a_k^2 \approx 0.193363$ and doubled across non-DC/non-Nyquist bins so that integrated tone bin power equals true time-domain $\text{RMS}^2$ (`0.5 * A^2` for a sine wave of amplitude `A`).
- **Peak & Tone Extraction (`FreqBins.tones`)**:
  - Identifies local maxima up to `MAX_TONES` (40) above `MIN_PWR` (`1e-12`).
  - Expands each tone's skirt monotonically out to `MAX_BINS_PER_TONE` (25) bins to capture the full Blackman-Harris main lobe.
  - Estimates sub-bin frequency by blending log-magnitude parabolic peak interpolation with the 7-bin main-lobe power-weighted centroid.
  - Zeroes extracted tone bins in a working copy of the spectrum to compute residual noise power (`self.noise`).

### 2. Tone Analysis (`tone_analyzer.py` & `tone_stats.py`)

`ToneAnalyzer` supports both single-tone and two-tone signals:

- **Adaptive Two-Tone Detection**: Computes an adaptive power ratio threshold between `TWO_TONE_THRESHOLD_MIN_DB` (`-20 dB`) and `TWO_TONE_THRESHOLD_MAX_DB` (`-10 dB`) based on the noise floor, and verifies that the candidate second tone is not simply the second harmonic of the primary tone.
- **Component Classification**:
  - **Harmonics (`Category.Harmonic`)**: Integer multiples $2f_0, \dots, \text{MAX\_HARM\_ORDER} \cdot f_0$ (up to 10th harmonic), accounting for sampling aliases via `alias_freq()`.
  - **Intermodulation Distortion (`Category.Distortion`)**: For two-tone signals, identifies sum and difference products $|m f_1 \pm n f_2|$ for orders $m + n \le \text{MAX\_DIST\_ORDER}$ (5).
  - **Spurious (`Category.Spurious`)**: Non-harmonic peaks exceeding `MIN_SPUR_RATIO` (`1e-8`, `-80 dB`) relative to total tone power.
- **Metrics (`ToneStats`)**:
  - **THD**: $\text{THD}_{\text{pwr}} = \frac{P_{\text{harmonics}} + P_{\text{IMD}}}{P_{\text{fundamental}}}$, reported in dB and percentage ($\sqrt{\text{THD}_{\text{pwr}}} \times 100\%$).
  - **THD+N**: $\text{THDN}_{\text{pwr}} = \frac{P_{\text{harmonics}} + P_{\text{IMD}} + P_{\text{noise}}}{P_{\text{fundamental}}}$.
  - **SNR**: $\text{SNR}_{\text{pwr}} = \frac{P_{\text{fundamental}}}{P_{\text{noise}}}$, reported in dB.
  - **SFDR**: $\text{SFDR}_{\text{pwr}} = \frac{P_{\text{fundamental}}}{\max(P_{\text{harmonics/IMD}}, P_{\text{max\_spur}})}$, reported in dB.

### 3. Noise Analysis (`noise_analyzer.py`)

`NoiseAnalyzer` evaluates broadband noise segments:
- Aggregates FFT bin power into ISO 266 standard one-third octave bands (`STD_BANDS`, 20 Hz to 20 kHz) via `FreqBins.get_bands()`.
- Computes a `FreqResp` summary reporting passband ripple (dB) between 100 Hz and 10 kHz and the `-3 dB` lower and upper band edges.

### 4. Sweep Analysis (`sweep_analyzer.py`)

`SweepAnalyzer` measures frequency response from frequency sweeps:
- Computes a Short-Time Fourier Transform (`scipy.signal.stft`) using a Hann window (`nperseg = min(4096, max(64, n_samples // 8))`, 50% overlap).
- Tracks the dominant frequency in each STFT time frame using 3-point log-magnitude parabolic interpolation and sums total frame power across bins.
- Combines consecutive frames with duplicate frequencies via `Components.combine()` and computes `FreqResp` (ripple and `-3 dB` band edges).

### 5. Time-Domain & Quadrature Analysis (`time_analyzer.py`, `time_stats.py`, `audio.py`)

- **Standard Time-Domain Metrics**: RMS level, DC offset, minimum/maximum sample values, peak dBFS (`max(|min|, |max|)`), and crest factor ($\frac{\max(|\min|, |\max|)}{\text{RMS}}$).
- **Quadrature (I/Q) Detection & Metrics**:
  - `Audio.is_quadrature` verifies 2-channel consistency of instantaneous envelope $|I + jQ|$ within `QUADRATURE_MAGNITUDE_TOLERANCE` (20%) across at least `QUADRATURE_SAMPLE_RATIO` (50%) of samples and checks channel peak frequency agreement (`QUADRATURE_MIN_FREQ_RATIO = 0.95`).
  - `Audio.phase_error()` computes RMS phase deviation (in radians) from a linear phase fit $\bar{\omega} n + \phi_0$ on quadrature sine tones.

---

## Audio Performance Metrics & CLI Options

### Primary Metrics (`analyze.py`)

| Metric | Source Class | Units |
|--------|--------------|-------|
| RMS, DC, Min, Max, Crest | `TimeStats` | Linear / dBFS / dB |
| Quadrature Mag Mean/Std, Phase Error | `TimeStats` | Linear / radians |
| Fundamental Frequency (`freq`, `freq2`) | `ToneStats` | Hz |
| THD, THD+N | `ToneStats` | % and dB |
| SNR, SFDR | `ToneStats` | dB |
| Passband Ripple, Lower/Upper `-3 dB` Edges | `FreqResp` | dB / Hz |

### CLI Options (`analyze.py`)

- `-a`, `--audio-stats`: Show audio statistics (default: True)
- `-c`, `--components [N]`: Show top `N` frequency components (default when flag is passed: 10)
- `-csv`, `--write-csv <file>`: Write statistics table to CSV file
- `-d`, `--device`: Capture input from default sound device
- `-l`, `--max-length <secs>`: Maximum analysis length in seconds (default: 60.0)
- `-p`, `--play <pattern>`: Play audio files matching pattern
- `-q`, `--quiet`: Quiet output
- `-s`, `--specs`: Show unit specifications (`UnitStats`)
- `-v`, `--verbose`: Verbose output
- Sorting flags (mutually exclusive): `-t` (THD, default), `-tf` (frequency), `-tf2` (second frequency), `-tn` (name), `-ts` (SNR), `-tt` (start time)

---

## Test Signal Generation (`generate.py`)

`generate.py` generates both standardized test signals (`signals/`) and demonstration examples (`examples/`):

1. **Pure Tones (`sine`)**: Stereo quadrature sinusoids at 20 Hz, 100 Hz, 250 Hz, 400 Hz, 1 kHz, 10 kHz, and 20 kHz.
2. **Two-Tone Signals (`two_tone`)**: 60 Hz + 7 kHz, 7 kHz + 60 Hz, 19 kHz + 20 kHz, and 20 kHz + 19 kHz (second tone at `-12 dB` relative to first tone, rescaled to 0.8 peak).
3. **Logarithmic Sweeps (`sweep`)**: 20 Hz to 20 kHz quadrature sweeps (plus a 300 Hz–3000 Hz bandpass-filtered sweep in `examples/`).
4. **Noise (`noise`)**: Quadrature uniform white noise (plus `-80 dBFS` noise floor and 300 Hz–3000 Hz bandpass-filtered noise in `examples/`).
5. **Controlled Distortion (`distortion`)**: 1 kHz tones with 0%, 0.001%, 0.01%, and 1.0% THD in `examples/`.
6. **Silence (`silence`)**: Reference zero-amplitude segments.

---

## Decibel Calculations (`decibels.py`)

| Function | Formula | Description |
|----------|---------|-------------|
| `db(pwr, ref_pwr=1.0)` | $10 \log_{10}(\max(\text{pwr}, \text{MIN\_PWR}) / \text{ref\_pwr})$ | Power ratio to dB (floored at `-120 dB` via `MIN_PWR = 1e-12`) |
| `dbv(v, ref_v=1.0)` | $20 \log_{10}(\max(|v|, \sqrt{\text{MIN\_PWR}}) / \text{ref\_v})$ | Amplitude/voltage ratio to dB |
| `db_to_pwr_ratio(db)` | $10^{\text{db} / 10}$ | dB to linear power ratio |
| `dbv_to_amp_ratio(db)` | $10^{\text{db} / 20}$ | dB to linear amplitude ratio |
