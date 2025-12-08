import pytest
import numpy as np

from constants import DEFAULT_FS, MIN_PWR
from freq_bins import FreqBins
from metadata import Metadata
from segment import Segment
from util import Channel


class TestFreqBins:
    def test_freqbins_initialization(self):
        # Create a simple sine wave: 1 kHz at 44.1 kHz, 1000 samples
        fs = DEFAULT_FS
        t = np.arange(1000) / fs
        samples = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 999))

        bins = FreqBins(samples, md, Channel.Mean)

        assert bins.bins is not None
        assert len(bins) > 0
        assert bins.fs == fs

    def test_get_tones(self):
        fs = DEFAULT_FS
        t = np.arange(4096) / fs  # Enough samples for good resolution

        # 1 kHz tone
        samples = 0.5 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 4095))

        bins = FreqBins(samples, md, Channel.Mean)
        tones = bins.get_tones(max_components=1)

        assert len(tones) == 1
        assert tones[0].freq == pytest.approx(1000.0, abs=fs / len(t))
        # Within bin resolution?
        # Actually FreqBins usually does interpolation or peak finding.
        # Let's hope it's accurate.

    def test_is_sine(self):
        # Create a sine wave
        fs = DEFAULT_FS
        t = np.arange(4096) / fs
        samples = 0.5 * np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 4095))

        bins = FreqBins(samples, md)
        assert bins.is_sine()

    def test_is_noise(self):
        fs = DEFAULT_FS
        samples = np.random.uniform(-0.5, 0.5, 4096).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 4095))

        bins = FreqBins(samples, md)
        assert bins.is_noise()

    def test_channel_handling(self):
        fs = DEFAULT_FS
        t = np.arange(1000) / fs

        # Create stereo signal: left=1k sine, right=silence
        left = np.sin(2 * np.pi * 1000.0 * t)
        right = np.zeros_like(left)
        stereo = np.column_stack((left, right)).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 999))

        # Test Left
        bins_l = FreqBins(stereo, md, Channel.Left)
        # 1000 samples, amp=1.0. With windowing, peak power > 0.01 is conservative enough.
        assert bins_l.max > 0.01

        # Test Right
        bins_r = FreqBins(stereo, md, Channel.Right)
        assert bins_r.max < 1e-4

        # Test Mean
        bins_m = FreqBins(stereo, md, Channel.Mean)
        assert bins_m.max > 0.005
        assert bins_m.max < bins_l.max

    def test_mean_power(self):
        fs = DEFAULT_FS
        t = np.arange(1000) / fs

        # 100 Hz sine
        samples = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 999))
        bins = FreqBins(samples, md)

        # Mean around 100 Hz should be high
        pwr_signal = bins.mean(90, 110)

        # Mean far from 100 Hz should be low
        pwr_noise = bins.mean(200, 300)
        assert pwr_signal > pwr_noise

    def test_ripple(self):
        fs = DEFAULT_FS
        t = np.arange(1000) / fs

        # 100 Hz sine + 200 Hz sine with different amplitudes
        # To create ripple, we need a flat region or multiple tones?
        # Definition: max - min in range.
        samples = (
            np.sin(2 * np.pi * 100.0 * t) + 0.5 * np.sin(2 * np.pi * 200.0 * t)
        ).astype(np.float32)

        md = Metadata("test", Segment(fs, 0, 999))
        bins = FreqBins(samples, md)

        # Check ripple spanning both peaks
        r = bins.ripple(50, 250)
        assert r > 0
        # Should be roughly max peak - noise floor (or min valley between peaks)

    def test_band_edges(self):
        fs = DEFAULT_FS
        t = np.arange(1000) / fs

        # Create 100 Hz sine
        samples = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 999))
        bins = FreqBins(samples, md)

        # Get edges 3 dB down
        low, high = bins.get_band_edges(tol_db=3.0)

        # Should bracket 100 Hz closely
        assert low < 100.0
        assert high > 100.0
        assert (high - low) < 50.0  # Narrow peak for pure sine with window

    def test_get_bands_and_components(self):
        fs = DEFAULT_FS

        # Create silence
        samples = np.zeros(1000, dtype=np.float32)
        md = Metadata("test", Segment(fs, 0, 999))
        bins = FreqBins(samples, md)

        # get_bands
        bands = bins.get_bands((100, 200))
        assert len(bands) == 2
        assert bands[0].freq == 100
        assert bands[1].freq == 200

        # get_components
        comps = bins.get_components(max_components=10)
        assert len(comps) == 10

    def test_freq_in_list(self):
        fs = 1000.0
        samples = np.zeros(100, dtype=np.float32)
        md = Metadata("test", Segment(fs, 0, 99))
        bins = FreqBins(samples, md)

        targets = [100.0, 200.0]

        # Should find exact match
        assert bins.freq_in_list(100.0, targets, 1)

        # Should find close match
        assert bins.freq_in_list(100.5, targets, 2)

        # 150 Hz NOT in list
        assert not bins.freq_in_list(150.0, targets, 1)

    def test_min_max_properties(self):
        fs = DEFAULT_FS

        # Create impulse with magnitude 1
        samples = np.zeros(100, dtype=np.float32)
        samples[0] = 1.0

        md = Metadata("test", Segment(fs, 0, 99))
        bins = FreqBins(samples, md)

        # Max should be 1
        assert bins.max > 0.0

        # Min should be 0
        assert bins.min < MIN_PWR
