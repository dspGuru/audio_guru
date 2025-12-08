"""Tests for generate.py"""

import pytest

import numpy as np

from audio import Audio
from constants import DEFAULT_FS
from generate import silence, noise, sine, sweep, distortion, write_examples
from util import Category


def test_silence():
    fs = DEFAULT_FS
    secs = 1.0
    a = silence(secs=secs, fs=fs)

    assert isinstance(a, Audio)
    assert a.fs == fs
    assert len(a.samples) == round(secs * fs)
    assert np.all(a.samples == 0)
    assert a.get_category() == Category.Silence


def test_noise():
    fs = DEFAULT_FS
    secs = 10.0
    amp = 0.5
    a = noise(amp=amp, secs=secs, fs=fs)

    assert isinstance(a, Audio)
    assert len(a.samples) == round(secs * fs)
    assert a.rms > 0
    assert np.max(np.abs(a.samples)) <= amp + 1e-6  # allow float noise
    assert a.get_category() == Category.Noise


def test_sine():
    fs = DEFAULT_FS
    freq = 1000.0
    secs = 10.0
    a = sine(freq=freq, secs=secs, fs=fs)

    assert isinstance(a, Audio)
    assert a.freq == pytest.approx(freq, abs=5.0)
    assert a.get_category() == Category.Tone


def test_sine_with_silence_append():
    fs = DEFAULT_FS
    secs = 10.0
    silence_secs = 1.0
    a = sine(secs=secs, silence_secs=silence_secs, fs=fs)

    expected_len = round((secs + silence_secs) * fs)
    # Allow off-by-one due to separate rounding of parts
    assert abs(len(a) - expected_len) <= 1

    # Last part should be silent
    tail = a.samples[-100:]
    assert np.all(tail == 0)


def test_sweep():
    fs = DEFAULT_FS
    a = sweep(f_start=100, f_stop=1000, secs=1.0, fs=fs)

    assert isinstance(a, Audio)
    assert a.get_category() == Category.Sweep
    assert a.is_sweep(secs=0.1)


def test_distortion():
    fs = DEFAULT_FS
    # 1% THD
    a = distortion(thd=0.01, freq=1000.0, secs=0.5, fs=fs)

    assert isinstance(a, Audio)
    # Fundamental should be present
    assert a.freq == pytest.approx(1000.0, abs=5.0)
    # Hard to test exact THD without analysis, but we can check it returns audio
    assert len(a) > 0


def test_write_examples(tmp_path, monkeypatch):
    # Set examples path
    p = tmp_path / "examples"

    # Run it
    write_examples(str(p))

    assert p.exists()
    assert (p / "test_noise.wav").exists()
    assert (p / "test_tone_100.wav").exists()
    assert (p / "test_tone_400.wav").exists()
    assert (p / "test_tone_1k.wav").exists()
    assert (p / "test_tone_10k.wav").exists()
    assert (p / "test_60_7k.wav").exists()
    assert (p / "test_19k_20k.wav").exists()
    assert (p / "test_sweep.wav").exists()

    # Check for at least one THD tone file
    assert list(p.glob("test_tone_1k-*pct.wav"))
