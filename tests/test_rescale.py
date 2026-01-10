import pytest
import numpy as np
from audio import Audio
from segment import Segment


def test_rescale_sine():
    """Test rescaling a sine wave."""
    audio = Audio(fs=1000, name="test")
    # Sine wave with amplitude 0.5
    t = np.linspace(0, 1, 1000)
    audio.samples = 0.5 * np.sin(2 * np.pi * 10 * t).astype(np.float32)
    audio.segment = Segment(1000, 0, 1000)

    # Rescale to 1.0
    audio.rescale(1.0)

    assert np.isclose(np.max(np.abs(audio.samples)), 1.0, atol=1e-5)


def test_rescale_silence():
    """Test rescaling silence (should do nothing)."""
    audio = Audio(fs=1000, name="test")
    audio.samples = np.zeros(1000, dtype=np.float32)
    audio.segment = Segment(1000, 0, 1000)

    audio.rescale(1.0)

    assert np.max(np.abs(audio.samples)) == 0.0


def test_rescale_selection():
    """Test rescaling only the selected segment."""
    audio = Audio(fs=1000, name="test")
    # Amplitude 0.5 everywhere
    audio.samples = 0.5 * np.ones(1000, dtype=np.float32)

    # Select first half
    seg = Segment(1000, 0, 500)
    audio.select(seg)

    # Rescale selection to 1.0
    audio.rescale(1.0)

    # First half should be 1.0, second half 0.5
    assert np.isclose(np.max(np.abs(audio.samples[:500])), 1.0, atol=1e-5)
    assert np.isclose(np.max(np.abs(audio.samples[500:])), 0.5, atol=1e-5)
