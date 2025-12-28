"""Constants for audio analysis."""

import math

__all__ = [
    "DEFAULT_AMP",
    "DEFAULT_FREQ",
    "DEFAULT_FS",
    "EQ_BANDS",
    "MAX_BINS_PER_TONE",
    "MAX_TONES",
    "MIN_DB",
    "MIN_PWR",
    "MIN_SEGMENT_SECS",
    "STD_BANDS",
]


# -----------------------------------------------------------------------------
# Audio Generation and Defaults
# -----------------------------------------------------------------------------

DEFAULT_AMP: float = 0.5
"""Default amplitude for generated audio."""

DEFAULT_BLOCKSIZE_SECS: float = 5.0
"""Default audio sample block size in seconds."""

DEFAULT_FREQ: float = 1000.0
"""Default frequency for generated audio in Hz."""

DEFAULT_FS = 44100.0
"""Default sampling frequency in Hz."""

DEFAULT_GEN_SECS = 5.0
"""Default duration for generated audio in seconds."""

DEFAULT_GEN_SILENCE_SECS = 0.0
"""Default duration of silence in seconds."""


# -----------------------------------------------------------------------------
# Frequency Analysis
# -----------------------------------------------------------------------------

MIN_FREQ = 20.0
"""Minimum frequency for audio analysis in Hz."""

MAX_FREQ = 20000.0
"""Maximum frequency for audio analysis in Hz."""

REF_FREQ = 1000.0
"""Reference frequency for audio analysis in Hz."""

AUDIO_BAND = (MIN_FREQ, MAX_FREQ)
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

MIN_SEGMENT_SECS = 1.0
"""Minimum audio segment length to analyze in seconds."""

MAX_SEGMENT_SECS = 60.0
"""Maximum audio segment length to analyze in seconds."""

DEFAULT_MAX_ANALYSIS_SECS = 30.0
"""Default maximum duration of audio to analyze in seconds."""

SILENCE_THRESH_RATIO = 0.1
"""Ratio of peak amplitude to threshold for silence detection."""

SILENCE_MIN_SECS = 0.5
"""Minimum duration of silence in seconds."""

SILENCE_WINDOW_SECS = SILENCE_MIN_SECS / 10.0
"""Duration of silence window in seconds."""

MIN_SILENCE_PEAK = 1e-6
"""Minimum peak amplitude for silence detection."""


# -----------------------------------------------------------------------------
# Analysis Thresholds and Limits
# -----------------------------------------------------------------------------

MIN_PWR = 1e-12  # -120 dB
"""Minimum power value for audio measurements."""

MIN_DB = 10.0 * math.log10(MIN_PWR)
"""Minimum decibel value for audio measurements (corresponds to MIN_PWR)."""

DEFAULT_MAX_NF_DBFS = -60.0
"""Default maximum allowable noise floor in dBFS."""

MAX_BINS_PER_TONE = 25
"""Number of frequency bins which constitute a single component after
applying seven-term Blackman-Harris window (empirical)"""

MAX_TONES = 20
"""Maximum number of tones to identify"""


# -----------------------------------------------------------------------------
# Tone Analysis
# -----------------------------------------------------------------------------

MAX_DIST_ORDER = 3
"""Maximum distortion order to consider"""

MAX_HARM_ORDER = 5
"""Maximum harmonic order to consider"""

MIN_SPUR_RATIO = 1e-8
"""Threshold for distinguishing spurs from noise (-80 dB)"""

MIN_TWO_TONE_RATIO = 1e-2
"""Minimum ratio for two-tone detection (-20 dB). Second tone must be at
least this large relative to the first tone."""
