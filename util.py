"""Audio utilities"""

import os

def split_fname(pathname: str) -> tuple[str]:
    """Split the given file pathname into three components which have been
    delimited by underscores and return the result as a tuple."""
    base = os.path.splitext(os.path.basename(pathname))[0]
    parts = base.split('_')

    if len(parts) >= 3:
        (mfr, model, desc) = (parts[0], parts[1], ' '.join(parts[2:]))
    elif len(parts) == 2:
        (mfr, model, desc) = (parts[0], parts[1], '')
    elif len(parts) == 1:
        mfr = ''
        model = ''
        desc = parts[0]
    
    return (mfr, model, desc)

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

def is_harmonic(freq: float, fund: float, fs: float, order: int, tol_hz:float =5.0) -> bool:
    """Return True if freq is a harmonic of fundamental up to the specified
     order, considering aliasing into the Nyquist band.

    The function aliases both the candidate harmonic (k*freq) and freq2 into
    [0, fs/2] using alias_freq() and tests for proximity within a tolerance.
    """
    # Return fale if any input is invalid
    if fund <= 0.0 or freq < 0.0 or order < 1 or fs <= 0.0:
        return False

    # Alias the observed frequency once
    target = alias_freq(freq, fs)

    # Return True if the frequency is a harmonic
    for k in range(1, order + 1):
        harmonic = alias_freq(k * fund, fs)
        if abs(harmonic - target) <= tol_hz:
            return True

    return False