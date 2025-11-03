"""Audio utilities
"""

import math, os

min_db = -120.0      # Minimum dB value which makes sense for audio

def db(pwr: float, ndigits:int =1):
    """Return dB of input power ratio."""
    try:
        return round(10.0 * math.log10(pwr), ndigits)
    except ValueError:
        return min_db

def split_fname(pathname: str):
    """Split the given file pathname into three components, delimited by
     underscore, and return the result as a tuple."""
    base = os.path.splitext(os.path.basename(pathname))[0]
    parts = base.split('_')
    if len(parts) >= 3:
        return (parts[0], parts[1], '_'.join(parts[2:]))
    elif len(parts) == 2:
        return (parts[0], parts[1], '')
    elif len(parts) == 1:
        return (parts[0], '', '')
    else:
        return ('', '', '')

def alias_freq(freq: float, fs: float) -> float:
    """Alias the specified frequency into the Nyquist band [0, fs/2].

    This maps any real-valued frequency into the principal Nyquist interval
    by folding frequencies around multiples of the Nyquist frequency.
    """
    # Convert negative frequency to positive
    freq = abs(freq)

    # Wrap positive frequency above fs
    while freq > fs:
        freq -= fs

    # If frequency is above Nyquist, calculate the aliased frequency
    if freq > 0.5*fs:
        freq = fs - freq
    
    # Return aliased frequency
    return freq
