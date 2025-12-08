"""Test for noise_analyzer.py"""

import pytest
from scipy import signal

from audio import Audio
from component import Components
from constants import DEFAULT_FS
from decibels import db
from freq_bins import FreqBins
from generate import noise
from metadata import Metadata
from noise_analyzer import NoiseAnalyzer


def test_noise_analyzer_with_example_file(examples_dir):
    # Load example noise file
    fname = examples_dir / "test_noise.wav"
    if not fname.exists():
        pytest.skip("test_noise.wav not found in examples")

    audio = Audio()
    assert audio.read(str(fname))

    # Initialize analyzer
    analyzer = NoiseAnalyzer(audio)
    assert analyzer.audio is audio
    bands = analyzer.analyze()

    # Test get_components
    # Noise components are those NOT matching a tone?
    # Or NoiseAnalyzer just treats everything as noise?
    # NoiseAnalyzer.get_components calls bins.get_components.
    comps = analyzer.get_components()
    assert len(comps) > 0
    # Should be effectively just bands of noise?

    # Test analyze (returns bands on EQ centers)
    assert len(bands) > 0
    # Standard bands (EQ_BANDS) len is usually fixed (e.g. 31 or 10 or similar ISO bands)
    # Let's verify we got reasonable power values (not -inf everywhere)
    assert any(b.pwr > 0 for b in bands)
    assert bands.name == "Noise Bands"


def test_noise_analyzer_with_filtered_noise():
    # Create noise
    audio = noise()

    # Filter noise
    numtaps = 200
    cutoff = [1000, 10000]
    fs = DEFAULT_FS
    coeffs = signal.firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=False)
    audio.filter(coeffs)

    # Initialize analyzer
    analyzer = NoiseAnalyzer(audio)

    # Test get_components
    comps = analyzer.get_components()
    assert len(comps) > 0
    # Should be effectively just bands of noise?

    # Test analyze (returns bands on EQ centers)
    bands = analyzer.analyze()
    assert len(bands) > 0
    assert bands.name == "Noise Bands"

    # Check that we have components in the passband
    band_components = [
        band for band in bands if band.freq > cutoff[0] and band.freq < cutoff[1]
    ]
    assert len(band_components) > 0

    passband = Components(bands.md, "Passband")
    passband.extend(band_components)

    # Get frequency components
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
    # Implementation: super().reset() sets self.analysis=None?
    # Actually analyzer.Analyzer.reset needs checking, but usually clears state.
