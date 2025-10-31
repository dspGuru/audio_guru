"""Implements decibel conversion for audio
"""

import math

min_db = -120.0         # Minimum dB value which makes sense for audio

def db(pwr: float, ndigits:int =1):
    """Return dB of input power ratio."""
    try:
        return round(10.0 * math.log10(pwr), ndigits)
    except ValueError:
        return min_db
