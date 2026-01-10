"""Tests for generate.py"""

import pytest

import numpy as np

from audio import Audio
from constants import DEFAULT_AMP, DEFAULT_FS, DEFAULT_FREQ
from generate import silence, noise, sine, sweep, distortion, two_tone, write_examples
from util import Category, Channel


def test_silence():
    fs = DEFAULT_FS
    secs = 1.0
    a = silence(secs=secs, fs=fs)

    assert isinstance(a, Audio)
    assert a.fs == fs
    assert len(a.samples) == round(secs * fs)
    assert np.all(a.samples == 0)
    assert a.cat == Category.Silence


def test_noise():
    fs = DEFAULT_FS
    secs = 10.0
    amp = 0.5

    # Seed noise for repeatable tests
    np.random.seed(0)
    a = noise(amp=amp, secs=secs, fs=fs)

    assert isinstance(a, Audio)
    assert len(a.samples) == round(secs * fs)
    assert a.rms > 0
    assert np.max(np.abs(a.samples)) <= amp + 1e-6  # allow float noise
    assert a.cat == Category.Noise


def test_sine():
    fs = DEFAULT_FS
    freq = DEFAULT_FREQ
    secs = 10.0
    a = sine(freq=freq, secs=secs, fs=fs)

    assert isinstance(a, Audio)
    assert a.freq == pytest.approx(freq, abs=5.0)
    assert a.cat == Category.Tone


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
    assert a.cat == Category.Sweep
    assert a.is_sweep()


def test_distortion():
    fs = DEFAULT_FS
    # 1% THD
    a = distortion(thd=0.01, freq=DEFAULT_FREQ, secs=0.5, fs=fs)

    assert isinstance(a, Audio)
    # Fundamental should be present
    assert a.freq == pytest.approx(DEFAULT_FREQ, abs=5.0)
    # Hard to test exact THD without analysis
    assert len(a) > 0


def test_write_signals(tmp_path):
    """Test write_signals generates test_ prefixed files with trailing silence."""
    from generate import write_signals

    p = tmp_path / "signals"
    write_signals(str(p))

    assert p.exists()
    assert (p / "01-test_tone_20.wav").exists()
    assert (p / "05-test_tone_1k.wav").exists()
    assert (p / "12-test_sweep.wav").exists()
    assert (p / "13-test_noise.wav").exists()
    assert (p / "14-test_silence.wav").exists()

    # Verify duration of one representative file (e.g., 1k tone)
    # Expected: 5.0s (default) + 1.0s (trailing silence) = 6.0s
    a = Audio()
    a.read(str(p / "05-test_tone_1k.wav"))
    assert len(a) == round(6.0 * a.fs)

    # Verify silence file also has 1s trailing (total 6s)
    a_silence = Audio()
    a_silence.read(str(p / "14-test_silence.wav"))
    assert len(a_silence) == round(6.0 * a_silence.fs)


def test_write_examples(tmp_path):
    """Test write_examples generates example_ prefixed files."""
    p = tmp_path / "examples"
    write_examples(str(p))

    assert p.exists()

    # Check for THD tone files
    assert list(p.glob("example_tone_1k-*pct.wav"))

    # Check other example files
    assert (p / "example_noise-floor.wav").exists()
    assert (p / "example_filt-sweep.wav").exists()
    assert (p / "example_filt-noise.wav").exists()

    # Verify example_tone_noisy.wav
    fname = p / "example_tone_noisy.wav"
    assert fname.exists()
    a = Audio()
    a.read(str(fname))

    # Expected length 6.0s
    assert len(a) == round(6.0 * a.fs)

    # Last 0.5s is pure noise (quadrature noise has constant magnitude)
    last_samples = a.samples[int(-0.5 * a.fs) :]

    peak = np.max(np.abs(last_samples))
    avg = np.mean(np.abs(last_samples))

    assert peak == pytest.approx(0.05, rel=0.2)
    # Quadrature noise avg abs ~0.032
    assert avg == pytest.approx(0.032, rel=0.2)


# -----------------------------------------------------------------------------
# Channel Parameter Tests
# -----------------------------------------------------------------------------


class TestSilenceChannel:
    """Tests for silence() channel parameter."""

    def test_silence_stereo_shape(self):
        """Stereo silence has 2D shape (n_samples, 2)."""
        a = silence(secs=0.1, channel=Channel.Stereo)
        assert a.samples.ndim == 2
        assert a.samples.shape[1] == 2
        assert a.channel == Channel.Stereo
        assert np.all(a.samples == 0)

    def test_silence_left_shape(self):
        """Left channel silence has 1D shape."""
        a = silence(secs=0.1, channel=Channel.Left)
        assert a.samples.ndim == 1
        assert a.channel == Channel.Left
        assert np.all(a.samples == 0)

    def test_silence_right_shape(self):
        """Right channel silence has 1D shape."""
        a = silence(secs=0.1, channel=Channel.Right)
        assert a.samples.ndim == 1
        assert a.channel == Channel.Right
        assert np.all(a.samples == 0)

    def test_silence_invalid_channel(self):
        """Invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="Channel must be Left, Right, or Stereo"):
            silence(secs=0.1, channel=Channel.Unknown)


class TestNoiseChannel:
    """Tests for noise() channel parameter."""

    def test_noise_stereo_shape(self):
        """Stereo noise is quadrature with constant magnitude."""
        np.random.seed(42)
        a = noise(secs=0.1, channel=Channel.Stereo)
        assert a.samples.ndim == 2
        assert a.samples.shape[1] == 2
        assert a.channel == Channel.Stereo
        # Channels should be different (not identical)
        assert not np.allclose(a.samples[:, 0], a.samples[:, 1])
        # Quadrature property: Magnitude (sqrt(I^2 + Q^2)) should be constant
        # for complex/quadrature signals even if the I and Q components vary.
        magnitudes = np.sqrt(a.samples[:, 0] ** 2 + a.samples[:, 1] ** 2)
        assert np.allclose(magnitudes, DEFAULT_AMP, atol=1e-6)

    def test_noise_left_shape(self):
        """Left channel noise has 2D shape with signal on left, silence on right."""
        np.random.seed(42)
        a = noise(secs=0.1, channel=Channel.Left)
        assert a.samples.ndim == 2
        assert a.samples.shape[1] == 2
        assert a.channel == Channel.Left
        # Left channel should have noise, right should be zeros
        assert np.any(a.samples[:, 0] != 0)
        assert np.all(a.samples[:, 1] == 0)

    def test_noise_right_shape(self):
        """Right channel noise has 2D shape with silence on left, signal on right."""
        np.random.seed(42)
        a = noise(secs=0.1, channel=Channel.Right)
        assert a.samples.ndim == 2
        assert a.samples.shape[1] == 2
        assert a.channel == Channel.Right
        # Left channel should be zeros, right should have noise
        assert np.all(a.samples[:, 0] == 0)
        assert np.any(a.samples[:, 1] != 0)

    def test_noise_invalid_channel(self):
        """Invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="Channel must be Left, Right, or Stereo"):
            noise(secs=0.1, channel=Channel.Difference)


class TestSineChannel:
    """Tests for sine() channel parameter."""

    def test_sine_stereo_shape(self):
        """Stereo sine has 2D shape with cos on left, sin on right."""
        a = sine(freq=1000, secs=0.1, channel=Channel.Stereo)
        assert a.samples.ndim == 2
        assert a.samples.shape[1] == 2
        assert a.channel == Channel.Stereo
        # Cosine (Left) and Sine (Right) are 90 degrees out of phase, forming a
        # perfect quadrature pair
        assert not np.allclose(a.samples[:, 0], a.samples[:, 1])

    def test_sine_left_shape(self):
        """Left channel sine has 1D shape (cosine)."""
        a = sine(freq=1000, secs=0.1, channel=Channel.Left)
        assert a.samples.ndim == 1
        assert a.channel == Channel.Left
        # Should be non-zero (cosine wave)
        assert np.any(a.samples != 0)

    def test_sine_right_shape(self):
        """Right channel sine has 1D shape (sine)."""
        a = sine(freq=1000, secs=0.1, channel=Channel.Right)
        assert a.samples.ndim == 1
        assert a.channel == Channel.Right
        # Should be non-zero (sine wave)
        assert np.any(a.samples != 0)

    def test_sine_left_right_phase_difference(self):
        """Left and Right channels have 90-degree phase difference."""
        left = sine(freq=1000, secs=0.01, channel=Channel.Left)
        right = sine(freq=1000, secs=0.01, channel=Channel.Right)
        # Cosine starts at 1, sine starts at 0
        assert left.samples[0] > right.samples[0]

    def test_sine_invalid_channel(self):
        """Invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="Channel must be Left, Right, or Stereo"):
            sine(freq=1000, secs=0.1, channel=Channel.Unknown)


class TestSweepChannel:
    """Tests for sweep() channel parameter."""

    def test_sweep_stereo_shape(self):
        """Stereo sweep uses complex chirp (2D real/imag components)."""
        a = sweep(f_start=100, f_stop=1000, secs=0.1, channel=Channel.Stereo)
        # Complex chirp results in complex samples which get stored
        assert a.channel == Channel.Stereo

    def test_sweep_left_shape(self):
        """Left channel sweep has real samples."""
        a = sweep(f_start=100, f_stop=1000, secs=0.1, channel=Channel.Left)
        assert a.channel == Channel.Left
        # Should have signal content
        assert np.any(a.samples != 0)

    def test_sweep_right_shape(self):
        """Right channel sweep has real samples."""
        a = sweep(f_start=100, f_stop=1000, secs=0.1, channel=Channel.Right)
        assert a.channel == Channel.Right
        # Should have signal content
        assert np.any(a.samples != 0)

    def test_sweep_invalid_channel(self):
        """Invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="Channel must be Left, Right, or Stereo"):
            sweep(f_start=100, f_stop=1000, secs=0.1, channel=Channel.Difference)


class TestTwoToneChannel:
    """Tests for two_tone() channel parameter."""

    def test_two_tone_stereo(self):
        """Two-tone with stereo channel."""
        a = two_tone(f1=60, f2=7000, secs=0.1, channel=Channel.Stereo)
        assert a.channel == Channel.Stereo

    def test_two_tone_left(self):
        """Two-tone with left channel."""
        a = two_tone(f1=60, f2=7000, secs=0.1, channel=Channel.Left)
        assert a.channel == Channel.Left

    def test_two_tone_right(self):
        """Two-tone with right channel."""
        a = two_tone(f1=60, f2=7000, secs=0.1, channel=Channel.Right)
        assert a.channel == Channel.Right

    def test_two_tone_invalid_channel(self):
        """Invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="Channel must be Left, Right, or Stereo"):
            two_tone(f1=60, f2=7000, secs=0.1, channel=Channel.Unknown)


class TestDistortionChannel:
    """Tests for distortion() channel parameter."""

    def test_distortion_stereo(self):
        """Distortion with stereo channel."""
        a = distortion(thd=0.01, secs=0.1, channel=Channel.Stereo)
        assert a.channel == Channel.Stereo

    def test_distortion_left(self):
        """Distortion with left channel."""
        a = distortion(thd=0.01, secs=0.1, channel=Channel.Left)
        assert a.channel == Channel.Left

    def test_distortion_right(self):
        """Distortion with right channel."""
        a = distortion(thd=0.01, secs=0.1, channel=Channel.Right)
        assert a.channel == Channel.Right

    def test_distortion_invalid_channel(self):
        """Invalid channel raises ValueError."""
        with pytest.raises(ValueError, match="Channel must be Left, Right, or Stereo"):
            distortion(thd=0.01, secs=0.1, channel=Channel.Difference)


# -----------------------------------------------------------------------------
# Quadrature Detection Tests for Generated Signals
# -----------------------------------------------------------------------------

from scipy import signal as scipy_signal


class TestGeneratedQuadratureSignals:
    """Tests verifying that generated quadrature signals are detected by is_quadrature."""

    def test_sine_stereo_is_quadrature(self):
        """Stereo sine wave (cos/sin pair) is detected as quadrature."""
        a = sine(freq=1000, secs=0.5, channel=Channel.Stereo)
        assert a.is_quadrature is True

    def test_sweep_stereo_is_quadrature(self):
        """Stereo sweep (complex chirp) is detected as quadrature."""
        a = sweep(f_start=100, f_stop=10000, secs=0.5, channel=Channel.Stereo)
        assert a.is_quadrature is True

    def test_filtered_sweep_stereo_is_quadrature_in_passband(self):
        """Filtered stereo sweep is quadrature within the passband region."""
        # Generate stereo sweep
        a = sweep(f_start=100, f_stop=10000, secs=0.5, channel=Channel.Stereo)

        # Apply bandpass filter (same as write_examples)
        sos = scipy_signal.butter(
            4, [300, 3000], btype="bandpass", fs=a.fs, output="sos"
        )
        a.samples = scipy_signal.sosfilt(sos, a.samples, axis=0).astype(a.DTYPE)

        # 0.7 threshold captures region with consistent amplitude
        peak = np.max(np.abs(a.samples))
        above_threshold = np.any(np.abs(a.samples) > 0.7 * peak, axis=1)

        # Find contiguous region above threshold
        indices = np.where(above_threshold)[0]
        start, stop = indices[0], indices[-1] + 1

        # Select only the passband segment
        from segment import Segment

        passband_segment = Segment(a.fs, start, stop)
        a.select(passband_segment)

        # Within the passband, the signal should maintain quadrature
        assert a.is_quadrature is True

    def test_sine_mono_not_quadrature(self):
        """Mono sine wave is not detected as quadrature."""
        a = sine(freq=1000, secs=0.5, channel=Channel.Left)
        assert a.is_quadrature is False

    def test_sweep_mono_not_quadrature(self):
        """Mono sweep is not detected as quadrature."""
        a = sweep(f_start=100, f_stop=10000, secs=0.5, channel=Channel.Left)
        assert a.is_quadrature is False


def test_example_noise_file_is_quadrature():
    """Verify examples/test_noise.wav quadrature status.

    Note: Regenerate examples to update if file is outdated.
    """
    from pathlib import Path

    signals_dir = Path(__file__).parent.parent / "signals"
    fname = signals_dir / "13-test_noise.wav"
    if not fname.exists():
        pytest.skip("13-test_noise.wav not found in signals")

    a = Audio()
    a.read(str(fname))
    # File needs to be regenerated with write_examples() to be quadrature
    assert a.is_quadrature is True
