"""Audio utilities"""

from enum import Enum


__all__ = ["Category", "Channel", "alias_freq", "is_harmonic"]


class Category(Enum):
    """Enumerates categories assigned to audio signals.

    Members
    -------
    Unknown
        The category is not determined or does not fit any other class.
    Tone
        Narrowband, near-sinusoidal signal tone content.
    Harmonic
        Harmonic of signal tone content.
    Distortion
        Harmonically related artifacts produced by nonlinear processing or
        clipping.
    Spurious
        Tone which is not harmonically related to any signal tones.
    Noise
        Noise-like content.
    Sweep
        Frequency-swept tone content.
    Band
        Band-limited noise or signal content.
    Silence
        Near-silence or very low-level content.
    DC
        Direct current (zero frequency) content.
    """

    Unknown = 0
    Tone = 1
    Harmonic = 2
    Distortion = 3
    Spurious = 4
    Noise = 5
    Sweep = 6
    Band = 7
    Silence = 8
    DC = 9


class Channel(Enum):
    """Enumerates audio channel assignments.

    Members
    -------
    Unknown
        The channel is not determined or does not fit any other class.
    Left
        Left channel.
    Right
        Right channel.
    Mean
        Mean of left and right channels.
    Difference
        Difference between left and right channels.
    """

    Unknown = 0
    Left = 1
    Right = 2
    Stereo = 3
    Difference = 4


def alias_freq(freq: float, fs: float) -> float:
    """
    Alias the specified frequency into the Nyquist band [0, fs/2].

    This maps any real-valued frequency into the principal Nyquist interval
    by folding frequencies around multiples of the Nyquist frequency.

    Parameters
    ----------
    freq : float
        Frequency to alias.
    fs : float
        Sampling frequency, in the same units as freq.
    """
    # Convert negative frequency to positive
    freq = abs(freq)

    # Wrap positive frequency above fs
    while freq > fs:
        freq -= fs

    # If frequency is above Nyquist, calculate the aliased frequency
    if freq > 0.5 * fs:
        freq = fs - freq

    # Return aliased frequency
    return freq


def is_harmonic(
    freq: float, fund: float, fs: float, order: int, tol_hz: float = 5.0
) -> bool:
    """
    Return True if freq is a harmonic of fundamental up to the specified
    order, considering aliasing into the Nyquist band.

    The function aliases both the candidate harmonic (n*freq) and freq2 into
    [0, fs/2] using alias_freq() and tests for proximity within a tolerance.

    Parameters
    ----------
    freq : float
        Frequency to test.
    fund : float
        Fundamental frequency.
    fs : float
        Sampling frequency.
    order : int
        Maximum harmonic order to test.
    tol_hz : float
        Tolerance in Hz for harmonic matching.
    """
    # Return False if any input is invalid
    if fund <= 0.0 or freq < 0.0 or order < 1 or fs <= 0.0:
        return False

    # Alias the test frequency
    target = alias_freq(freq, fs)

    # Return True if the frequency is a harmonic of the fundamental
    for k in range(2, order + 1):
        harmonic = alias_freq(k * fund, fs)
        if abs(harmonic - target) <= tol_hz:
            return True

    # No harmonic found: return False
    return False
