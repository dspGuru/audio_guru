"""Constants for audio analysis.

This module defines constants used in the audio analysis library. These values
are based on common audio processing practices or are empirical, derived from
testing with real-world audio samples and the example audio files that are
produced by generate.py.
"""

import math

# -----------------------------------------------------------------------------
# Decibel and Power Limits
# -----------------------------------------------------------------------------

MIN_PWR: float = 1e-12  # -120 dB
"""Minimum power value for audio measurements."""

MIN_DB: float = 10.0 * math.log10(MIN_PWR)
"""Minimum decibel value for audio measurements (corresponds to MIN_PWR)."""


# -----------------------------------------------------------------------------
# General Audio Defaults
# -----------------------------------------------------------------------------

DEFAULT_AMP: float = 0.8
"""Default amplitude for generated audio."""

DEFAULT_BLOCKSIZE_SECS: float = 5.0
"""Default audio sample block size in seconds."""

DEFAULT_FREQ: float = 1000.0
"""Default frequency for generated audio in Hz."""

DEFAULT_FS: float = 44100.0
"""Default sampling frequency in Hz."""

DEFAULT_GEN_SECS: float = 5.0
"""Default duration for generated audio in seconds."""

DEFAULT_GEN_SILENCE_SECS: float = 0.0
"""Default duration of silence in seconds."""


# -----------------------------------------------------------------------------
# Frequency Analysis
# -----------------------------------------------------------------------------

MIN_FREQ: float = 20.0
"""Minimum frequency for audio analysis in Hz."""

MAX_FREQ: float = 20000.0
"""Maximum frequency for audio analysis in Hz."""

REF_FREQ: float = 1000.0
"""Reference frequency for audio analysis in Hz."""

SWEEP_SEGMENT_SECS: float = 0.1
"""Duration of frequency sweep segment in seconds."""

AUDIO_BAND: tuple[float, float] = (MIN_FREQ, MAX_FREQ)
"""Full audio frequency band range in Hz."""

EQ_BANDS = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
"""Standard one-octave band center frequencies per ISO 266, which are commonly
used for audio equalizers."""

STD_BANDS = (
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
)
"""Standard one-third octave band center frequencies per ISO 266."""


# -----------------------------------------------------------------------------
# Segmentation and Silence Detection
# -----------------------------------------------------------------------------

SEGMENT_MIN_SECS: float = 1.0
"""Minimum audio segment length to analyze in seconds."""

SEGMENT_MAX_SECS: float = 60.0
"""Maximum audio segment length to analyze in seconds."""

DEFAULT_MAX_ANALYSIS_SECS: float = 30.0
"""Default maximum duration of audio to analyze in seconds."""

DEFAULT_SILENCE_THRESH_RATIO: float = 0.1
"""Ratio of peak amplitude to threshold for silence detection."""

DEFAULT_SILENCE_MIN_SECS: float = 0.5
"""Minimum duration of silence in seconds."""

SILENCE_WINDOW_SECS: float = DEFAULT_SILENCE_MIN_SECS / 10.0
"""Duration of silence window in seconds."""

SILENCE_MIN_PEAK: float = 1e-8
"""Minimum peak amplitude for silence detection."""

SILENCE_MIN_THRESHOLD: float = 1e-4
"""Minimum threshold for silence detection."""

# -----------------------------------------------------------------------------
# Analysis Thresholds and Limits
# -----------------------------------------------------------------------------

DEFAULT_MAX_NF_DBFS: float = -60.0
"""Default maximum allowable noise floor in dBFS."""

MAX_BINS_PER_TONE: int = 25
"""Number of frequency bins which constitute a single component after
applying seven-term Blackman-Harris window (empirical)"""

MAX_TONES: int = 40
"""Maximum number of tones to identify"""


# -----------------------------------------------------------------------------
# Tone Analysis
# -----------------------------------------------------------------------------

MAX_DIST_ORDER: int = 5
"""Maximum distortion order to consider"""

MAX_HARM_ORDER: int = 10
"""Maximum harmonic order to consider"""

MIN_SPUR_RATIO: float = 1e-8
"""Threshold for distinguishing spurs from noise (-80 dB)"""

MIN_TWO_TONE_RATIO: float = 1e-2
"""Minimum ratio for two-tone detection (-20 dB). Second tone must be at
least this large relative to the first tone."""


# -----------------------------------------------------------------------------
# Quadrature Detection
# -----------------------------------------------------------------------------

QUADRATURE_SAMPLE_RATIO: float = 0.5
"""Ratio of samples that must have consistent magnitude for quadrature detection."""

QUADRATURE_MAGNITUDE_TOLERANCE: float = 0.2
"""Relative tolerance for comparing complex magnitudes to peak amplitude."""

QUADRATURE_MIN_FREQ_RATIO: float = 0.95
"""Ratio of frequencies that must be close for quadrature detection."""
