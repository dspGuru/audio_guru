"""Generates common audio signals such as silence, sine waves, and sweeps."""

import math

import numpy as np
from scipy import signal as scipy_signal

from audio import Audio
from constants import (
    DEFAULT_AMP,
    DEFAULT_FREQ,
    DEFAULT_FS,
    DEFAULT_GEN_SECS,
    DEFAULT_GEN_SILENCE_SECS,
)
from decibels import db_to_pwr_ratio, MIN_PWR


__all__ = ["distortion", "noise", "silence", "sine", "sweep", "write_examples"]


def distortion(
    thd: float,
    amp: float = DEFAULT_AMP,
    freq: float = DEFAULT_FREQ,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    fs: float = DEFAULT_FS,
) -> Audio:
    """
    Generate samples with the specified THD values.

    Parameters
    ----------
    thd : float
        Total harmonic distortion (THD) as a fraction (e.g., 0.01 for 1%).
    amp : float
        Amplitude of the tone (0.0 to 1.0).
    freq : float
        Frequency of the tone in Hz.
    secs : float
        Duration of the generated tone in seconds.
    silence_secs : float
        Duration of silence to append after each tone, in seconds.
    fs : float
        Sampling frequency in Hz.
    """
    audio = sine(freq=freq, amp=amp, secs=secs, thd=thd, fs=fs)
    audio.name = "Distortion"

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs))

    return audio


def noise(
    amp: float = DEFAULT_AMP,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    fs: float = DEFAULT_FS,
) -> Audio:
    """
    Generate noise audio.

    Parameters
    ----------
    amp : float
        Amplitude of the noise (0.0 to 1.0).
    secs : float
        Duration of the generated noise in seconds.
    silence_secs : float
        Duration of silence to append after the noise, in seconds.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Audio
        The generated noise audio.
    """
    if secs <= 0:
        raise ValueError("Duration must be positive")
    if not (0.0 <= amp <= 1.0):
        raise ValueError("Amplitude must be between 0.0 and 1.0")

    audio = Audio(fs)
    audio.name = "Noise"

    # Generate white noise
    n_samples = round(secs * fs)
    audio.samples = np.random.uniform(low=-amp, high=amp, size=n_samples).astype(
        audio.DTYPE
    )

    # Select the entire audio
    audio.select()

    # Set metadata
    desc = f"White Noise {secs:0.1f}s"
    audio.md.set(mfr=audio.name, model="", desc=desc)

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs))

    return audio


def silence(secs: float = DEFAULT_GEN_SECS, fs: float = DEFAULT_FS) -> Audio:
    """
    Generate silence audio.

    Parameters
    ----------
    secs : float
        Duration of the generated silence in seconds.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Audio
        The generated silence audio.
    """
    if secs <= 0.0:
        raise ValueError("Duration must be positive")

    audio = Audio(fs)
    audio.name = "Silence"

    # Generate silence
    audio.samples = np.zeros(round(secs * fs), dtype=audio.DTYPE)

    # Select the entire audio
    audio.select()

    # Set metadata
    desc = f"Silence {secs:0.1f}s"
    audio.md.set(mfr=audio.name, model="", desc=desc)

    return audio


def sine(
    amp: float = DEFAULT_AMP,
    freq: float = 1000.0,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    thd: float = MIN_PWR,
    fs: float = DEFAULT_FS,
) -> Audio:
    """
    Generate sine wave audio with optional first-harmonic distortion.

    Parameters
    ----------
    amp : float
        Amplitude of the sine wave (0.0 to 1.0).
    freq : float
        Frequency of the sine wave in Hz.
    secs : float
        Duration of the generated sine wave in seconds.
    silence_secs : float
        Duration of silence to append after the sine wave, in seconds.
    thd : float
        Total harmonic distortion (THD) as a fraction (e.g., 0.01 for 1%).
        If greater than MIN_PWR, adds a first harmonic at the specified THD.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Audio
        The generated sine wave audio.
    """
    if freq <= 0.0:
        raise ValueError("Frequency must be positive")
    if secs <= 0.0:
        raise ValueError("Duration must be positive")
    if not (0.0 <= amp <= 1.0):
        raise ValueError("Amplitude must be between 0.0 and 1.0")

    audio = Audio(fs)
    audio.name = "Sine"

    # Generate sine wave signal
    t = np.arange(secs * fs) / fs
    audio.samples = amp * np.sin(2.0 * np.pi * freq * t).astype(audio.DTYPE)
    audio.select()

    # If specified, generate distortion tone at twice the nominal frequency
    if thd > MIN_PWR:
        # Recursive call to generate the harmonic
        harmonic = sine(freq=freq * 2.0, amp=amp * thd, secs=secs, thd=MIN_PWR, fs=fs)
        audio += harmonic
        thd_str = f"{(thd * 100.0):0.3f}%"
    else:
        thd_str = ""

    # Set metadata for sine or tone (sine with distortion)
    if thd <= MIN_PWR:
        audio.name = "Sine"
        desc = f"Sine {freq:0.1f}Hz"
        audio.md.set(mfr=audio.name, model="", desc=desc)
    else:
        audio.name = "Tone"
        desc = f"{freq:0.1f}" if freq < 1000.0 else f"{math.trunc(freq / 1000.0)}k"
        audio.md.set(mfr=audio.name, model=thd_str, desc=desc)

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs))

    return audio


def sweep(
    amp: float = DEFAULT_AMP,
    f_start: float = 20.0,
    f_stop: float = 20000.0,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    fs: float = DEFAULT_FS,
    method: str = "logarithmic",
) -> Audio:
    """
    Generate sweep audio.

    Parameters
    ----------
    amp : float
        Amplitude of the sweep (0.0 to 1.0).
    f_start : float
        Starting frequency of the sweep in Hz.
    f_stop : float
        Stopping frequency of the sweep in Hz.
    secs : float
        Duration of the generated sweep in seconds.
    silence_secs : float
        Duration of silence to append after the sweep, in seconds.
    fs : float
        Sampling frequency in Hz.
    method : str
        Method of frequency sweep ('linear', 'logarithmic', 'quadratic', 'hyperbolic').

    Returns
    -------
    Audio
        The generated sweep audio.
    """
    if f_start <= 0.0 or f_stop <= 0.0:
        raise ValueError("Frequencies must be positive")
    if secs <= 0.0:
        raise ValueError("Duration must be positive")
    if not (0.0 <= amp <= 1.0):
        raise ValueError("Amplitude must be between 0.0 and 1.0")

    audio = Audio(fs)
    audio.name = "Sweep"

    # Generate sweep signal
    t = np.arange(secs * fs) / fs
    audio.samples = amp * scipy_signal.chirp(
        t, f0=f_start, t1=secs, f1=f_stop, method=method
    ).astype(audio.DTYPE)

    # Select the entire audio
    audio.select()

    # Set metadata
    desc = f"Sweep {f_start:0.1f}Hz to {f_stop:0.1f}Hz"
    audio.md.set(mfr=audio.name, model="", desc=desc)

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs))

    return audio


def two_tone(
    amp: float = DEFAULT_AMP,
    f1: float = 60.0,
    f2: float = 7000.0,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    fs: float = DEFAULT_FS,
) -> Audio:
    """
    Generate two-tone audio.

    Parameters
    ----------
    amp : float
        Amplitude of the first tone (0.0 to 1.0). The second tone is 12 dB
        lower.
    f1 : float
        Frequency of the first tone in Hz.
    f2 : float
        Frequency of the second tone in Hz.
    secs : float
        Duration of the generated two-tone signal in seconds.
    silence_secs : float
        Duration of silence to append after the two-tone signal, in seconds.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Audio
        The generated two-tone audio.
    """
    # Generate first tone
    audio = sine(freq=f1, secs=secs, amp=amp, fs=fs)

    # Generate second tone
    amp2 = amp * math.sqrt(db_to_pwr_ratio(-12.0))
    audio2 = sine(freq=f2, secs=secs, amp=amp2, fs=fs)

    # Combine tones
    audio += audio2

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs))

    return audio


def write_examples(path: str = "./examples") -> None:
    """
    Write example audio files.

    Parameters
    ----------
    path : str
        Directory to write the example files to (default "./examples").
    """

    DEFAULT_GEN_SILENCE_SECS = 0.5
    """Default duration of silence in seconds."""

    import pathlib

    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Generate tones with distortion
    print("Generating tones with various distortion values")
    thds = (0.0, 0.01, 0.0001, 1e-5)
    for thd in thds:
        audio = distortion(thd)
        pct = f"{thd*100:.3f}"
        pct = pct.replace(".", "-")
        fname = f"test_tone_1k-{pct}-pct.wav"
        audio.write(path / fname)
        print(f"Wrote tone with {thd*100:.3f}% THD to '{fname}'")
    print()

    # Generate noise audio
    print("Generating noise audio")
    audio = noise()
    audio.write(path / "test_noise.wav")
    print("Wrote noise audio to 'test_noise.wav'")
    print()

    # Generate sine wave audio
    freqs: list[int] = [20, 100, 250, 400, 1000, 10000, 20000]
    for freq in freqs:
        if freq < 1000:
            freq_name: str = f"{freq}"
        else:
            freq_name: str = f"{freq // 1000}k"
        print(f"Generating {freq_name} sine wave audio")
        audio = sine(freq=freq)
        fname = f"test_tone_{freq_name}.wav"
        audio.write(path / fname)
        print(f"Wrote sine wave audio to '{fname}'")
    print()

    # Generate two-tone signal with 60 Hz and 7 kHz
    print("Generating two-tone signal with 60 Hz and 7 kHz")
    audio = two_tone(f1=60, f2=7000)
    fname = "test_tones_60_7k.wav"
    audio.write(path / fname)
    print(f"Wrote two-tone signal to '{fname}'")
    print()

    # Generate two-tone signal with 7 kHz and 60 Hz
    print("Generating two-tone signal with 7 kHz and 60 Hz")
    audio = two_tone(f1=7000, f2=60)
    fname = "test_tones_7k_60.wav"
    audio.write(path / fname)
    print(f"Wrote two-tone signal to '{fname}'")
    print()

    # Generate two-tone signal with 19 kHz and 20 kHz
    print("Generating two-tone signal with 19 kHz and 20 kHz")
    audio = two_tone(f1=19000, f2=20000)
    fname = "test_tones_19k_20k.wav"
    audio.write(path / fname)
    print(f"Wrote two-tone signal to '{fname}'")
    print()

    # Generate two-tone signal with 20 kHz and 19 kHz
    print("Generating two-tone signal with 20 kHz and 19 kHz")
    audio = two_tone(f1=20000, f2=19000)
    fname = "test_tones_20k_19k.wav"
    audio.write(path / fname)
    print(f"Wrote two-tone signal to '{fname}'")
    print()

    # Generate sweep audio
    print("Generating sweep audio")
    audio = sweep()
    fname = "test_sweep.wav"
    audio.write(path / fname)
    print(f"Wrote sweep audio to '{fname}'")
    print()

    # Design a bandpass filter from 300 Hz to 3 kHz
    sos = scipy_signal.butter(
        4, [300, 3000], btype="bandpass", fs=audio.fs, output="sos"
    )

    # Generate filtered sweep audio
    print("Generating filtered sweep audio")
    audio = sweep()
    audio.samples = scipy_signal.sosfilt(sos, audio.samples).astype(audio.DTYPE)

    fname = "test_filt-sweep.wav"
    audio.write(path / fname)
    print(f"Wrote filtered sweep audio to '{fname}'")
    print()

    # Generate filtered noise audio
    print("Generating filtered noise audio")
    audio = noise(secs=5.0, silence_secs=1.0)
    audio.samples = scipy_signal.sosfilt(sos, audio.samples).astype(audio.DTYPE)

    fname = "test_filt-noise.wav"
    audio.write(path / fname)
    print(f"Wrote filtered noise audio to '{fname}'")
    print()


if __name__ == "__main__":
    write_examples()
