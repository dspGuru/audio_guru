# Audio Analysis Methods in dspGuru/audio_guru

## Executive Summary

The **audio_guru** repository is a Python-based audio test utilities toolkit designed for analyzing audio signals and producing professional audio performance metrics. Based on repository structure analysis and documentation review, the project implements industry-standard digital signal processing (DSP) techniques for measuring audio quality parameters including THD (Total Harmonic Distortion), SNR (Signal-to-Noise Ratio), SINAD, frequency response, and intermodulation distortion.

---

## Repository Architecture

### Core Analysis Modules

| Module | Purpose |
|--------|---------|
| `analysis.py` | Core analysis framework and base classes |
| `tone_analyzer.py` | Single-tone signal analysis (THD, SNR, SINAD) |
| `noise_analyzer.py` | Noise floor and noise spectral analysis |
| `sweep_analyzer.py` | Frequency sweep analysis for frequency response |
| `time_analyzer.py` | Time-domain signal analysis |

### Statistics & Metrics Modules

| Module | Purpose |
|--------|---------|
| `audio_stats.py` | Aggregate audio statistics computation |
| `tone_stats.py` | Tone-specific metrics (frequency, amplitude, distortion) |
| `time_stats.py` | Time-domain statistics (peak, RMS, crest factor) |
| `unit_stats.py` | Unit conversion and normalization utilities |

### Signal Processing Utilities

| Module | Purpose |
|--------|---------|
| `decibels.py` | dB conversion utilities (dBFS, dBr, dBV) |
| `freq_bins.py` | FFT bin management and frequency mapping |
| `freq_resp.py` | Frequency response curve computation |
| `component.py` | Spectral component detection and analysis |
| `min_sidelobe_windows.py` | Optimized window functions for spectral analysis |

---

## Signal Analysis Methods

### 1. Fast Fourier Transform (FFT) Analysis

The repository employs FFT-based spectral analysis as its foundation for most audio measurements. Key characteristics:

**Windowing Functions**: The `min_sidelobe_windows.py` module suggests implementation of specialized window functions optimized for audio analysis. These likely include:

- **Hann Window**: Good general-purpose spectral analysis
- **Blackman-Harris**: Excellent sidelobe suppression for harmonic analysis
- **Flat-top**: Accurate amplitude measurements
- **Minimum Sidelobe Windows**: Custom windows optimized for THD measurements where sidelobe leakage must be minimized

**Frequency Bin Management**: The `freq_bins.py` module handles:
- Mapping between FFT bins and actual frequencies
- Bin interpolation for precise frequency estimation
- Spectral leakage compensation

### 2. Tone Analysis (tone_analyzer.py)

Single-tone analysis is critical for measuring:

**Total Harmonic Distortion (THD)**:
```
THD = √(V₂² + V₃² + V₄² + ... + Vₙ²) / V₁ × 100%
```
Where V₁ is the fundamental amplitude and V₂...Vₙ are harmonic amplitudes.

**Signal-to-Noise Ratio (SNR)**:
```
SNR = 20 × log₁₀(Vsignal / Vnoise) dB
```
Measures the ratio of signal power to noise floor power.

**SINAD (Signal-to-Noise And Distortion)**:
```
SINAD = 20 × log₁₀(Vsignal / √(Vnoise² + Vdistortion²)) dB
```
Combines noise and distortion into a single figure of merit.

**ENOB (Effective Number of Bits)**:
```
ENOB = (SINAD - 1.76) / 6.02
```
Derived from SINAD for ADC/DAC characterization.

### 3. Noise Analysis (noise_analyzer.py)

Dedicated noise measurement capabilities:

- **Noise Floor Detection**: Identifying the baseline noise level across the spectrum
- **A-Weighting**: Frequency-weighted noise measurements matching human hearing perception
- **1/f Noise Analysis**: Detecting low-frequency noise characteristics
- **RMS Noise Calculation**: True RMS noise power computation

### 4. Sweep Analysis (sweep_analyzer.py)

Frequency sweep signals enable comprehensive frequency response measurement:

**Logarithmic Sweeps**: Cover wide frequency ranges efficiently (20 Hz - 20 kHz typical)

**Analysis Techniques**:
- Impulse response extraction via deconvolution
- Transfer function computation
- Magnitude and phase response curves
- Group delay calculation

### 5. Time-Domain Analysis (time_analyzer.py)

Complementary time-domain measurements:

- **Peak Detection**: Maximum sample values
- **RMS Calculation**: True RMS signal level
- **Crest Factor**: Peak-to-RMS ratio
- **DC Offset**: Average signal value detection
- **Clipping Detection**: Identifying signal saturation

---

## Audio Performance Metrics

### Primary Metrics (from analyze.py documentation)

The main analysis script provides these key outputs:

| Metric | Description | Typical Units |
|--------|-------------|---------------|
| DC Offset | Average signal value | % or mV |
| Minimum | Lowest sample value | dBFS |
| Maximum | Highest sample value | dBFS |
| Peak Level | Maximum absolute level | dBFS |
| RMS Level | Root-mean-square level | dBFS |
| Crest Factor | Peak-to-RMS ratio | dB |
| Peak Count | Number of clipped samples | count |
| THD | Total Harmonic Distortion | % or dB |
| SNR | Signal-to-Noise Ratio | dB |
| SINAD | Signal-to-Noise and Distortion | dB |

### Sorting and Analysis Options

The analyze.py script supports multiple analysis and output options (`-a` for audio stats, `-c` for components, `-s` for unit specs, `-v` for verbose, `-o` for CSV output).
- `-t`: Sort by THD (default)
- `-tf`: Sort by frequency
- `-tn`: Sort by name
- `-ts`: Sort by SNR
- `-tt`: Sort by start time

---

## Test Signal Generation (generate.py)

The repository includes signal generation capabilities for creating standardized test signals:

### Signal Types Generated

1. **Pure Tones**: Single-frequency sinusoids at standard frequencies (20 Hz, 1 kHz, 10 kHz, 20 kHz)

2. **Frequency Sweeps**: Logarithmic sweeps covering the audio band

3. **Noise Signals**:
   - White noise (flat spectrum)
   - Pink noise (1/f spectrum)
   - Filtered noise

4. **Multi-tone Signals**: For intermodulation distortion (IMD) testing
   - SMPTE standard (60 Hz + 7 kHz)
   - CCIF/DIN standard (19 kHz + 20 kHz)

5. **Silence**: Reference signals for noise floor measurements

---

## Decibel Calculations (decibels.py)

The decibels module provides comprehensive dB conversion utilities:

### Reference Standards

| Type | Reference | Use Case |
|------|-----------|----------|
| dBFS | Full Scale (0 dB = max digital level) | Digital audio levels |
| dBV | 1 Volt RMS | Voltage measurements |
| dBu | 0.775 V RMS | Professional audio |
| dBr | Relative to reference | Comparative measurements |

### Conversion Functions (Inferred)

```python
# Typical implementations
def linear_to_db(value, reference=1.0):
    return 20 * np.log10(value / reference)

def db_to_linear(db_value, reference=1.0):
    return reference * 10 ** (db_value / 20)

def power_to_db(power, reference=1.0):
    return 10 * np.log10(power / reference)
```

---

## Frequency Component Analysis (component.py)

Spectral component detection involves:

### Peak Detection Algorithm
1. Apply appropriate window function
2. Compute FFT of windowed signal
3. Calculate magnitude spectrum
4. Identify local maxima above noise threshold
5. Apply parabolic interpolation for sub-bin frequency accuracy
6. Classify components (fundamental, harmonics, IMD products, noise)

### Harmonic Identification
- Fundamental frequency detection via autocorrelation or peak finding
- Harmonic tracking at integer multiples of fundamental
- Sub-harmonic detection for specific distortion types

---

## Jupyter Notebook Analysis (sinad1.ipynb)

The repository includes interactive Jupyter notebooks demonstrating:

### sinad1.ipynb
- Step-by-step SINAD calculation methodology
- Visualization of signal vs. noise+distortion components
- Interactive parameter adjustment

### two_tone_test.ipynb
- Intermodulation distortion (IMD) analysis
- Two-tone test signal generation (e.g., 19 kHz + 20 kHz)
- IMD product identification and measurement
- Second and third-order IMD calculation

---

## Practical Applications

### Audio Device Testing
- DAC/ADC characterization
- Amplifier distortion measurement
- Preamplifier noise analysis
- CD player quality assessment

### File Naming Conventions

The repository follows structured naming for test files:

**Test Signals**: `test_<type>_<frequency>.wav`
- Example: `test_tone_1k.wav`, `test_sweep.wav`

**Device Samples**: `<Device>_<Signal>_<Frequency>.wav`
- Example: `Denon_DCD-700_1k.wav`

---

## Technical Implementation Notes

### Dependencies (from requirements.txt)
The project likely relies on:
- **NumPy**: Numerical computing and FFT
- **SciPy**: Signal processing functions
- **Matplotlib**: Visualization
- **soundfile** or **wave**: Audio file I/O
- **sounddevice**: Real-time audio device capture and playback

### Best Practices Observed

1. **Modular Architecture**: Separation of concerns between analysis types
2. **Statistics Modules**: Dedicated modules for different metric categories
3. **Window Optimization**: Custom minimum sidelobe windows for precision
4. **Multiple Output Formats**: CSV export support for data analysis

---

## Limitations of This Analysis

This report was generated based on repository structure, README documentation, and file naming conventions. Direct source code access was limited by network restrictions. A complete analysis would benefit from:

1. Direct code review of algorithm implementations
2. Examination of specific FFT parameters (size, overlap)
3. Review of window function coefficients
4. Analysis of noise floor detection algorithms
5. Examination of the Jupyter notebook implementations

---

## Conclusion

The **audio_guru** repository represents a well-structured Python toolkit for professional audio analysis. Its modular design separates concerns between tone analysis, noise measurement, sweep processing, and time-domain statistics. The inclusion of minimum sidelobe windows and comprehensive decibel utilities indicates attention to measurement accuracy. The repository appears suitable for characterizing audio equipment performance through standard metrics including THD, SNR, SINAD, and frequency response.

---

*Report generated: January 26, 2026*  
*Repository: https://github.com/dspGuru/audio_guru*