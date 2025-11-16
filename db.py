"""Decibel utilities
"""

import math

MIN_DB = -120.0
"""The minimum dB value which makes sense for audio"""

MIN_PWR = 1e-12
"""Power ratio corresponding to MIN_DB"""

UNKNOWN_DB = MIN_DB - 1.0
"""Tag value for unknown dB"""

def db(pwr: float, ndigits:int =1):
    """
    Return dB of input power ratio, rounded to the specified number of digits.
    dB = 10*log10(pwr)
    """
    pwr = max(pwr, MIN_PWR)
    return round(10.0*math.log10(pwr), ndigits)


def dbv(v: float, ndigits:int =1):
    """
    Return dB of amplitude, eg, Volts, rounded to the specified number of digits.
    dBv = 20*log10(v)
    """
    return db(v*v, ndigits)


def db_str(pwr: float, suffix: str='', ndigits:int =1):
    """
    Return dB of input power ratio.
    
    If power is below MIN_PWR, return '-----'.

    Parameters
    ----------
    pwr : float
        Power ratio.
    suffix : str
        Suffix to append to the dB value.
    ndigits : int
        Number of digits to round the dB value to.
    """
    if pwr >= MIN_PWR:
        d = db(pwr, ndigits)
        return f'{d:6.1f}{suffix}'
    else:
        return '  -----'


def db_to_pwr(db_val: float) -> float:
    """Return power ratio of input dB.

    Values below MIN_DB are clamped to prevent underflow / -inf behavior.
    """
    return 10.0**(max(db_val, MIN_DB)/10.0)
