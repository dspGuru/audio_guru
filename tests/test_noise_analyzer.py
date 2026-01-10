"""Test for noise_analyzer.py"""

import pytest
from scipy import signal

from audio import Audio
from component import Components
from constants import DEFAULT_FS
from decibels import db
from generate import noise
from noise_analyzer import NoiseAnalyzer
from util import Channel


def test_noise_analyzer_with_example_file(signals_dir):
    # Load example noise file
    fname = signals_dir / "13-test_noise.wav"
    if not fname.exists():
        pytest.skip("13-test_noise.wav not found in signals")

    # Read audio from file
    audio = Audio()
    assert audio.read(str(fname))

    # Initialize analyzer
    analyzer = NoiseAnalyzer(audio)
    assert analyzer.audio is audio
    analysis = analyzer.analyze()
    bands = analysis.bands

    # Test get_components()
    comps = analyzer.get_components()
    assert len(comps) > 0

    # Test analyze() (returns bands on EQ centers)
    assert len(bands) > 0

    # Standard bands len is fixed. Verify reasonable power values.
    assert any(b.pwr > 0 for b in bands)
    assert bands.name == "Noise Bands"


def test_noise_analyzer_with_filtered_noise():
    # Create noise (Left channel output is 2D stereo, extract left for filtering)
    audio = noise(channel=Channel.Left)
    left_channel = audio.samples[:, 0].copy()

    # Filter the 1D left channel
    numtaps = 200
    cutoff = [1000, 10000]
    fs = DEFAULT_FS
    coeffs = signal.firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=False)
    filtered = signal.convolve(left_channel, coeffs, mode="same")
    audio.samples = filtered

    # Initialize analyzer
    analyzer = NoiseAnalyzer(audio)

    # Test get_components
    comps = analyzer.get_components()
    assert len(comps) > 0

    # Test analyze (returns bands on EQ centers)
    analysis = analyzer.analyze()
    bands = analysis.bands
    assert len(bands) > 0
    assert bands.name == "Noise Bands"

    # Check that we have components in the passband
    band_components = [
        band for band in bands if band.freq > cutoff[0] and band.freq < cutoff[1]
    ]
    assert len(band_components) > 0

    # Create passband components
    passband = Components(bands.md, "Passband")
    passband.extend(band_components)

    # Get primary frequency component of passband
    center = passband.average()
    assert center.freq > cutoff[0] and center.freq < cutoff[1]
    lower = bands.find_freq(cutoff[0])
    upper = bands.find_freq(cutoff[1])

    # Check frequency bounds
    assert lower.upper_freq < center.freq < upper.lower_freq
    assert lower.lower_freq < center.lower_freq < upper.lower_freq
    assert lower.upper_freq < center.upper_freq < upper.upper_freq

    # Check power ratio
    center_db = db(center.pwr)
    lower_db = db(lower.pwr)
    upper_db = db(upper.pwr)
    assert (center_db - lower_db) > 2.0
    assert (center_db - upper_db) > 2.0


def test_reset():
    audio = Audio()

    # Dummy
    analyzer = NoiseAnalyzer(audio)
    analyzer.components = [1, 2, 3]  # Mock
    analyzer.reset()
    assert analyzer.components is None or len(analyzer.components) == 0


def test_filtered_noise_response():
    """Verify that a filtered noise matches the expected bandpass filter response."""
    # Generate filtered noise exactly as in generate.py
    # Filter is 4th order Butterworth bandpass from 300 Hz to 3000 Hz
    import scipy.signal as scipy_signal

    fs = 48000
    secs = 2.0
    # Create noise (Left channel output is 2D stereo)
    audio = noise(secs=secs, fs=fs, channel=Channel.Left)
    left_channel = audio.samples[:, 0].copy()

    sos = scipy_signal.butter(4, [300, 3000], btype="bandpass", fs=fs, output="sos")
    filtered = scipy_signal.sosfilt(sos, left_channel).astype(audio.DTYPE)
    audio.samples = filtered

    analyzer = NoiseAnalyzer(audio)
    analysis = analyzer.analyze()  # Returns bands on ISO/EQ centers
    bands = analysis.bands

    # Expect high power in passband (300 Hz-3 kHz), low power in stopbands
    passband_centers = [
        f for f in [315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500]
    ]
    stopband_low_centers = [f for f in [63, 125]]
    stopband_high_centers = [f for f in [4000, 8000, 16000]]

    # Helper to get average power of a list of bands
    def get_avg_pwr(centers):
        pwrs = []
        for c in centers:
            band = bands.find_freq(c)
            if band:
                pwrs.append(band.pwr)
        return sum(pwrs) / len(pwrs) if pwrs else 0.0

    pass_pwr = get_avg_pwr(passband_centers)
    stop_low_pwr = get_avg_pwr(stopband_low_centers)
    stop_high_pwr = get_avg_pwr(stopband_high_centers)

    assert pass_pwr > 0

    # Expect >12 dB difference for 4th order filter
    pass_db = db(pass_pwr)
    stop_low_db = db(stop_low_pwr)
    stop_high_db = db(stop_high_pwr)
    assert (
        pass_db - stop_low_db
    ) > 12.0, f"Expected > 12 dB attenuation low, got {pass_db - stop_low_db:.1f} dB"
    assert (
        pass_db - stop_high_db
    ) > 12.0, f"Expected > 12 dB attenuation high, got {pass_db - stop_high_db:.1f} dB"
