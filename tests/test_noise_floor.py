import pytest
import numpy as np
from audio import Audio


def test_ongoing_noise_floor():
    """Test ongoing noise floor calculation."""
    fs = 1000
    a = Audio(fs=fs, name="test")

    # Two tones separated by noise; noise amp must be < 0.05 (SILENCE_THRESH_RATIO * peak)
    noise_amp = 0.01

    # 1. Tone 1
    t = np.linspace(0, 0.5, 500, endpoint=False)
    tone1 = np.sin(2 * np.pi * 100 * t).astype(np.float32)

    # 2. Noise 1 (constant amp below thresh)
    noise1 = np.ones(500, dtype=np.float32) * noise_amp

    # 3. Tone 2
    tone2 = np.sin(2 * np.pi * 200 * t).astype(np.float32)

    # 4. Noise 2
    noise2 = np.ones(500, dtype=np.float32) * noise_amp

    # Assembly: Construct a composite signal with alternating tones and noise
    # segments.
    a.samples = np.concatenate((tone1, noise1, tone2, noise2))

    # Process segments (triggers get_segments -> accumulates noise stats)
    list(a)

    # Expected: 1000 samples at 0.01 -> avg = 0.01
    assert a._noise_count > 0
    assert a.avg_noise_floor == pytest.approx(noise_amp)


def test_no_noise():
    """Test noise floor when no valid silence segments exist."""
    fs = 1000
    a = Audio(fs=fs)
    # Just a tone, no silence
    t = np.linspace(0, 1.0, 1000)
    a.samples = np.sin(2 * np.pi * 10 * t).astype(np.float32)

    list(a)

    assert a._noise_count == 0
    assert a.avg_noise_floor == 0.0
