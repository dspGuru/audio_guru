import numpy as np
import pytest
from freq_bins import FreqBins
from metadata import Metadata
from segment import Segment


def test_bin_to_freq_and_back():
    # Setup simple FreqBins
    fs = 1000
    n = 1000
    # dummy samples
    samples = np.zeros(n, dtype=np.float32)
    md = Metadata("test", Segment(fs, 0, n))
    bins = FreqBins(samples, md)

    # Check bin to freq
    # Current implementation uses len(bins) which includes DC, so it's slightly offset from N/2+1 logic if strictly N.
    # We check approximate or just round trip primarily.
    f1 = bins.bin_to_freq(1)
    assert f1 > 0.0

    # Check freq to bin
    b1 = bins.freq_to_bin(f1)
    assert b1 == 1

    # Round trip
    assert bins.freq_to_bin(bins.bin_to_freq(50)) == 50
    assert bins.bin_to_freq(bins.freq_to_bin(50.0)) == pytest.approx(50.0, rel=1e-2)


def test_mean_power_edge_cases():
    fs = 1000
    n = 100
    samples = np.zeros(n, dtype=np.float32)
    md = Metadata("test", Segment(fs, 0, n))
    bins = FreqBins(samples, md)

    # Edge case: f1 > f2
    # Implementation might return 0.0 for empty range or handle it.
    # We just ensure it doesn't crash.
    m = bins.mean(f1=100, f2=50)
    assert isinstance(m, float)

    # Edge case: f1 or f2 out of Nyquist?
    # Nyquist is 500.
    # If we ask for mean up to 600?
    # It should clip to max bin.
    m = bins.mean(f1=0, f2=600)
    assert m >= 0.0  # Should be 0 since samples are 0.


def test_ripple_edge_cases():
    fs = 1000
    n = 1000
    t = np.arange(n) / fs
    # Generate flat noise (approximately) or just a single tone
    # Single tone has infinite ripple technically if range covers it and silence?
    # Ripple is max - min power in band.

    # Flat spectrum (impulse in time = all freqs constant?)
    samples = np.zeros(n, dtype=np.float32)
    samples[0] = 1.0  # Impulse
    md = Metadata("test", Segment(fs, 0, n))
    bins = FreqBins(samples, md)

    # FFT of impulse is constant (flat)
    # So ripple should be near 0
    r = bins.ripple(f1=10, f2=400)
    assert r == pytest.approx(0.0, abs=1e-5)

    # Case with no bins in range
    # e.g. narrow range between bins?
    # bin resolution is 1 Hz.
    # range 10.1 to 10.2 ?
    # freq_to_bin might map to same bin -> 1 sample -> max-min=0
    r_narrow = bins.ripple(f1=10.1, f2=10.2)
    assert r_narrow == 0.0
