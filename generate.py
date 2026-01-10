#!/usr/bin/env python3

"""Generates common audio signals such as silence, sine waves, and sweeps."""

import math

import numpy as np
from scipy import signal

from audio import Audio
from audio_list import AudioList
from constants import (
    DEFAULT_AMP,
    DEFAULT_FREQ,
    DEFAULT_FS,
    DEFAULT_GEN_SECS,
    DEFAULT_GEN_SILENCE_SECS,
)
from decibels import db_to_pwr_ratio, MIN_PWR
from util import Channel, Category

__all__ = [
    "distortion",
    "noise",
    "silence",
    "sine",
    "sweep",
    "write_signals",
    "write_examples",
]


def distortion(
    thd: float,
    amp: float = DEFAULT_AMP,
    freq: float = DEFAULT_FREQ,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    fs: float = DEFAULT_FS,
    channel: Channel = Channel.Stereo,
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
    channel : Channel
        Channel to generate (Left, Right, or Stereo).

    Returns
    -------
    Audio
        The generated distortion audio.

    Raises
    ------
    ValueError
        If channel is not Left, Right, or Stereo.
    """
    if channel not in (Channel.Left, Channel.Right, Channel.Stereo):
        raise ValueError("Channel must be Left, Right, or Stereo")
    audio = sine(freq=freq, amp=amp, secs=secs, thd=thd, fs=fs, channel=channel)
    audio.name = "Distortion"

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs, channel=channel))

    # Rescale to the specified amplitude
    audio.rescale(amp)

    return audio


def noise(
    amp: float = DEFAULT_AMP,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    fs: float = DEFAULT_FS,
    channel: Channel = Channel.Stereo,
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
    channel : Channel
        Channel to generate (Left, Right, or Stereo).

    Returns
    -------
    Audio
        The generated noise audio.

    Raises
    ------
    ValueError
        If `secs` is not positive, `amp` is not between 0.0 and 1.0, or
        `channel` is not Left, Right, or Stereo.
    """
    if secs <= 0:
        raise ValueError("Duration must be positive")
    if not (0.0 <= amp <= 1.0):
        raise ValueError("Amplitude must be between 0.0 and 1.0")
    if channel not in (Channel.Left, Channel.Right, Channel.Stereo):
        raise ValueError("Channel must be Left, Right, or Stereo")

    audio = Audio(fs)
    audio.name = "Noise"

    # Generate white noise for the specified channel
    n_samples = round(secs * fs)
    noise = np.random.uniform(low=-amp, high=amp, size=n_samples).astype(audio.DTYPE)

    # Generate noise channels for the specified channel type
    if channel == Channel.Stereo:
        # Generate quadrature noise: each sample pair has magnitude = amp
        # Left channel is always random in [-amp, amp]
        left = noise

        # Right channel has magnitude such that |left + j*right| = amp
        # |right| = sqrt(amp^2 - left^2), with random sign
        right_mag = np.sqrt(amp**2 - left**2)
        right_sign = np.random.choice([-1.0, 1.0], size=n_samples).astype(audio.DTYPE)
        right = (right_mag * right_sign).astype(audio.DTYPE)
    elif channel == Channel.Left:
        # Left channel is always random in [-amp, amp]
        left = noise
        right = np.zeros(n_samples, dtype=audio.DTYPE)
    else:  # Channel.Right
        left = np.zeros(n_samples, dtype=audio.DTYPE)
        right = noise

    # Set samples and channel
    audio.samples = np.column_stack((left, right))
    audio.channel = channel
    audio.cat = Category.Noise

    # Select the entire audio
    audio.select()

    # Set metadata
    desc = f"White Noise {secs:0.1f}s"
    audio.md.set(mfr=audio.name, model="", desc=desc)

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs, channel=channel))

    return audio


def silence(
    secs: float = DEFAULT_GEN_SECS,
    fs: float = DEFAULT_FS,
    channel: Channel = Channel.Stereo,
) -> Audio:
    """
    Generate silence audio.

    Parameters
    ----------
    secs : float
        Duration of the generated silence in seconds.
    fs : float
        Sampling frequency in Hz.
    channel : Channel
        Channel to generate (Left, Right, or Stereo).

    Returns
    -------
    Audio
        The generated silence audio.

    Raises
    ------
    ValueError
        If secs is not positive or channel is not Left, Right, or Stereo.
    """
    # Validate parameters
    if secs <= 0.0:
        raise ValueError("Duration must be positive")
    if channel not in (Channel.Left, Channel.Right, Channel.Stereo):
        raise ValueError("Channel must be Left, Right, or Stereo")

    # Create audio object
    audio = Audio(fs)

    # Generate silence for specified channel
    n_samples = round(secs * fs)
    if channel == Channel.Stereo:
        audio.samples = np.zeros((n_samples, 2), dtype=audio.DTYPE)
    else:
        # Left or Right: create one silent channel
        audio.samples = np.zeros(n_samples, dtype=audio.DTYPE)
    audio.channel = channel

    # Select the entire audio
    audio.select()

    # Set metadata
    audio.name = "Silence"
    audio.cat = Category.Silence
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
    channel: Channel = Channel.Stereo,
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
    channel : Channel
        Channel to generate (Left, Right, or Stereo).

    Returns
    -------
    Audio
        The generated sine wave audio.

    Raises
    ------
    ValueError
        If `freq` or `secs` are not positive, `amp` is not between 0.0 and
        1.0, or `channel` is not Left, Right, or Stereo.
    """
    # Validate parameters
    if freq <= 0.0:
        raise ValueError("Frequency must be positive")
    if secs <= 0.0:
        raise ValueError("Duration must be positive")
    if not (0.0 <= amp <= 1.0):
        raise ValueError("Amplitude must be between 0.0 and 1.0")
    if channel not in (Channel.Left, Channel.Right, Channel.Stereo):
        raise ValueError("Channel must be Left, Right, or Stereo")

    # Create audio object
    audio = Audio(fs)

    # Generate sine and cosine waves and assign to left and right channels
    t = np.arange(secs * fs) / fs
    cos_samples = amp * np.cos(2.0 * np.pi * freq * t).astype(audio.DTYPE)
    sin_samples = amp * np.sin(2.0 * np.pi * freq * t).astype(audio.DTYPE)
    if channel == Channel.Stereo:
        audio.samples = np.column_stack((cos_samples, sin_samples))
    elif channel == Channel.Left:
        audio.samples = cos_samples
    else:  # Channel.Right
        audio.samples = sin_samples
    audio.channel = channel
    audio.select()

    # If specified, generate distortion tone at twice the nominal frequency
    if thd > MIN_PWR:
        # Recursive call to generate the harmonic
        harmonic = sine(
            freq=freq * 2.0,
            amp=amp * thd,
            secs=secs,
            thd=MIN_PWR,
            fs=fs,
            channel=channel,
        )
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
    audio.cat = Category.Tone

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs, channel=channel))

    # Rescale to the specified amplitude
    audio.rescale(amp)

    return audio


def sweep(
    amp: float = DEFAULT_AMP,
    f_start: float = 20.0,
    f_stop: float = 20000.0,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    fs: float = DEFAULT_FS,
    method: str = "logarithmic",
    channel: Channel = Channel.Stereo,
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
    channel : Channel
        Channel to generate (Left, Right, or Stereo).

    Returns
    -------
    Audio
        The generated sweep audio.

    Raises
    ------
    ValueError
        If `f_start`, `f_stop`, or `secs` are not positive, `amp` is not
        between 0.0 and 1.0, or `channel` is not Left, Right, or Stereo.
    """
    # Validate parameters
    if f_start <= 0.0 or f_stop <= 0.0:
        raise ValueError("Frequencies must be positive")
    if secs <= 0.0:
        raise ValueError("Duration must be positive")
    if not (0.0 <= amp <= 1.0):
        raise ValueError("Amplitude must be between 0.0 and 1.0")
    if channel not in (Channel.Left, Channel.Right, Channel.Stereo):
        raise ValueError("Channel must be Left, Right, or Stereo")

    # Create audio object
    audio = Audio(fs)

    # Generate sweep signal for specified channel
    t = np.arange(secs * fs) / fs
    cplx = channel == Channel.Stereo
    samples = amp * signal.chirp(
        t,
        f0=f_start,
        t1=secs,
        f1=f_stop,
        method=method,
        complex=cplx,
    )
    if cplx:
        # Extract real and imag parts BEFORE converting to float32
        # because astype(float32) on complex array discards imaginary part
        re = samples.real.astype(audio.DTYPE)
        im = samples.imag.astype(audio.DTYPE)
        audio.samples = np.column_stack((re, im))
    else:
        audio.samples = samples.astype(audio.DTYPE)
    audio.channel = channel

    # Select the entire audio
    audio.select()

    # Set metadata
    audio.name = "Sweep"
    audio.cat = Category.Sweep
    desc = f"Sweep {f_start:0.1f}Hz to {f_stop:0.1f}Hz"
    audio.md.set(mfr=audio.name, model="", desc=desc)

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs, channel=channel))

    # Rescale to the specified amplitude
    audio.rescale(amp)

    return audio


def two_tone(
    amp: float = DEFAULT_AMP,
    f1: float = 60.0,
    f2: float = 7000.0,
    secs: float = DEFAULT_GEN_SECS,
    silence_secs: float = DEFAULT_GEN_SILENCE_SECS,
    fs: float = DEFAULT_FS,
    channel: Channel = Channel.Stereo,
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
    channel : Channel
        Channel to generate (Left, Right, or Stereo).

    Returns
    -------
    Audio
        The generated two-tone audio.

    Raises
    ------
    ValueError
        If `channel` is not Left, Right, or Stereo.
    """
    # Validate parameters
    if channel not in (Channel.Left, Channel.Right, Channel.Stereo):
        raise ValueError("Channel must be Left, Right, or Stereo")

    # Generate first tone
    audio = sine(freq=f1, secs=secs, amp=amp, fs=fs, channel=channel)
    audio.cat = Category.Tone

    # Generate second tone
    amp2 = amp * math.sqrt(db_to_pwr_ratio(-12.0))
    audio2 = sine(freq=f2, secs=secs, amp=amp2, fs=fs, channel=channel)

    # Combine tones
    audio += audio2

    # Add silence if specified
    if silence_secs > 0.0:
        audio.append(silence(silence_secs, fs, channel=channel))

    # Rescale to the specified amplitude
    audio.rescale(amp)

    return audio


def write_signals(path: str = "./signals") -> None:
    """
    Write test signal audio files.

    These are basic test signals used for testing and verification.
    File names start with 'test_'.

    Parameters
    ----------
    path : str
        Directory to write the signal files to (default "./signals").
    """
    # Create audio signals list
    signals = AudioList()

    # Define trailing silence duration
    silence_secs = 1.0

    # Generate sine wave audio
    freqs: list[int] = [20, 100, 250, 400, 1000, 10000, 20000]
    for freq in freqs:
        if freq < 1000:
            freq_name: str = f"{freq}"
        else:
            freq_name: str = f"{freq // 1000}k"
        print(f"Generating {freq_name} sine wave audio")
        audio = sine(freq=freq, silence_secs=silence_secs)
        fname = f"test_tone_{freq_name}.wav"
        audio.name = fname
        signals.append(audio)

    # Generate two-tone signal with 60 Hz and 7 kHz
    print("Generating two-tone signal with 60 Hz and 7 kHz")
    audio = two_tone(f1=60, f2=7000, silence_secs=silence_secs)
    fname = "test_tones_60_7k.wav"
    audio.name = fname
    signals.append(audio)

    # Generate two-tone signal with 7 kHz and 60 Hz
    print("Generating two-tone signal with 7 kHz and 60 Hz")
    audio = two_tone(f1=7000, f2=60, silence_secs=silence_secs)
    fname = "test_tones_7k_60.wav"
    audio.name = fname
    signals.append(audio)

    # Generate two-tone signal with 19 kHz and 20 kHz
    print("Generating two-tone signal with 19 kHz and 20 kHz")
    audio = two_tone(f1=19000, f2=20000, silence_secs=silence_secs)
    fname = "test_tones_19k_20k.wav"
    audio.name = fname
    signals.append(audio)

    # Generate two-tone signal with 20 kHz and 19 kHz
    print("Generating two-tone signal with 20 kHz and 19 kHz")
    audio = two_tone(f1=20000, f2=19000, silence_secs=silence_secs)
    fname = "test_tones_20k_19k.wav"
    audio.name = fname
    signals.append(audio)

    # Generate sweep audio
    print("Generating sweep audio")
    audio = sweep(silence_secs=silence_secs)
    fname = "test_sweep.wav"
    audio.name = fname
    signals.append(audio)

    # Generate noise audio
    print("Generating noise audio")
    audio = noise(silence_secs=silence_secs)
    audio.name = "test_noise.wav"
    signals.append(audio)

    # Generate silence audio
    print("Generating silence audio")
    audio = silence()
    audio.append(silence(silence_secs))
    audio.name = "test_silence.wav"
    signals.append(audio)

    # Write audio signal files with track numbers prepended to file names
    signals.write(path, prepend_tracknumber=True)


def write_examples(path: str = "./examples") -> None:
    """
    Write example audio files.

    These are example files demonstrating various audio processing scenarios.
    File names start with 'example_'.

    Parameters
    ----------
    path : str
        Directory to write the example files to (default "./examples").
    """
    # Create audio examples list
    examples = AudioList()

    # Define trailing silence duration
    silence_secs = 1.0

    # Generate tones with distortion
    print("Generating tones with various distortion values")
    thds = (0.0, 0.01, 0.0001, 1e-5)
    for thd in thds:
        audio = distortion(thd, silence_secs=silence_secs)
        pct = f"{thd*100:.3f}"
        pct = pct.replace(".", "-")
        audio.name = f"example_tone_1k-{pct}-pct.wav"
        examples.append(audio)

    # Generate noise-floor audio
    print("Generating noise-floor audio")
    audio = noise(amp=1e-4, silence_secs=silence_secs)  # -80 dBFS
    audio.name = "example_noise-floor.wav"
    examples.append(audio)

    # Generate noisy tone audio
    print("Generating noisy tone audio")
    amp = 0.5
    audio = sine(amp=amp, secs=5.0, silence_secs=silence_secs)
    audio += noise(amp=amp * 1e-1, secs=len(audio) / audio.fs)
    audio.name = "example_tone_noisy.wav"
    examples.append(audio)

    # Design a bandpass filter from 300 Hz to 3 kHz
    sos = signal.butter(4, [300, 3000], btype="bandpass", fs=DEFAULT_FS, output="sos")

    # Generate filtered sweep audio
    print("Generating filtered sweep audio")
    audio = sweep(silence_secs=silence_secs)
    audio.samples = signal.sosfilt(sos, audio.samples).astype(audio.DTYPE)
    audio.name = "example_filt-sweep.wav"
    examples.append(audio)

    # Generate filtered noise audio
    print("Generating filtered noise audio")
    audio = noise(secs=5.0, silence_secs=silence_secs)
    audio.samples = signal.sosfilt(sos, audio.samples).astype(audio.DTYPE)
    audio.name = "example_filt-noise.wav"
    examples.append(audio)

    # Write examples without track numbers prepended to file names
    examples.write(path, prepend_tracknumber=False)


if __name__ == "__main__":
    write_signals()
    write_examples()
