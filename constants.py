"""Constants for audio analysis."""

import math

__all__ = [
    "DEFAULT_AMP",
    "DEFAULT_FREQ",
    "DEFAULT_FS",
    "MIN_DB",
    "MIN_PWR",
    "MIN_AUDIO_LEN",
    "BINS_PER_TONE",
    "MAX_TONES",
    "EQ_BANDS",
    "STD_BANDS",
]

DEFAULT_AMP: float = 0.6
"""Default amplitude for generated audio."""

DEFAULT_BLOCKSIZE_SECS: float = 10.0
"""Default audio sample block size in seconds."""

DEFAULT_FREQ: float = 1000.0
"""Default frequency for generated audio in Hz."""

DEFAULT_FS = 44100.0
"""Default sampling frequency in Hz."""

DEFAULT_MAX_NF_DBFS = -60.0
"""Default maximum allowable noise floor in dBFS."""

MIN_PWR = 1e-12  # -120 dB
"""Minimum power value for audio measurements."""

MIN_DB = 10.0 * math.log10(MIN_PWR)
"""Minimum decibel value for audio measurements."""

MIN_AUDIO_LEN = 256
"""Minimum audio length to analyze in samples."""

BINS_PER_TONE = 5
"""Number of frequency bins which constitute a single component after
applying Blackman-Harris window (empirical)"""

MAX_TONES = 20
"""Maximum number of tones to identify"""

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
