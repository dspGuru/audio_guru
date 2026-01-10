import pytest
import unittest.mock
import numpy as np

from constants import DEFAULT_FS, DEFAULT_FREQ
from freq_bins import FreqBins
from metadata import Metadata
from segment import Segment
from util import Channel


class TestFreqBins:
    def test_freqbins_initialization(self):
        # Create a simple sine wave: 1 kHz at 44.1 kHz, 1000 samples
        fs = DEFAULT_FS
        t = np.arange(1000) / fs
        samples = np.sin(2 * np.pi * DEFAULT_FREQ * t).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 999))

        bins = FreqBins(samples, md, Channel.Stereo)

        assert bins.bins is not None
        assert len(bins) > 0
        assert bins.fs == fs

    def test_get_tones(self):
        fs = DEFAULT_FS
        t = np.arange(4096) / fs  # Enough samples for good resolution

        # 1 kHz tone
        samples = 0.5 * np.sin(2 * np.pi * DEFAULT_FREQ * t).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 4095))

        bins = FreqBins(samples, md, Channel.Stereo)
        tones = bins.tones[:1]

        assert len(tones) == 1
        assert tones[0].freq == pytest.approx(DEFAULT_FREQ, abs=fs / len(t))

    def test_is_sine(self):
        # Create a sine wave
        fs = DEFAULT_FS
        t = np.arange(4096) / fs
        samples = 0.5 * np.sin(2 * np.pi * DEFAULT_FREQ * t).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 4095))

        bins = FreqBins(samples, md)
        assert bins.is_sine()

    def test_is_noise(self):
        fs = DEFAULT_FS
        np.random.seed(0)
        samples = np.random.uniform(-0.5, 0.5, 4096).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 4097))

        bins = FreqBins(samples, md)
        assert bins.is_noise()

    def test_channel_handling(self):
        fs = DEFAULT_FS
        t = np.arange(1000) / fs

        # Create stereo signal: left=1k sine, right=silence
        left = np.sin(2 * np.pi * DEFAULT_FREQ * t)
        right = np.zeros_like(left)
        stereo = np.column_stack((left, right)).astype(np.float32)
        md = Metadata("test", Segment(fs, 0, 999))

        # Test Left
        bins_l = FreqBins(stereo, md, Channel.Left)
        # With windowing, peak power > 0.01
        assert bins_l.max > 0.01

        # Test Right
        bins_r = FreqBins(stereo, md, Channel.Right)
        assert bins_r.max < 1e-4

        # Test Mean
        bins_m = FreqBins(stereo, md, Channel.Stereo)
        assert bins_m.max > 0.003
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

    def test_freq_to_bin_negative(self):
        fs = 1000
        samples = np.zeros(100, dtype=np.float32)
        md = Metadata("test", Segment(fs, 0, 99))
        bins = FreqBins(samples, md)

        with pytest.raises(ValueError, match="Frequency must be non-negative"):
            bins.freq_to_bin(-10.0)

    def test_freq_bins_channel_logic(self):
        fs = 1000
        # Stereo signal L=1, R=0.5
        # Mean = 0.75
        # Difference = 0.5
        samples = np.ones((100, 2), dtype=np.float32)
        samples[:, 1] = 0.5
        md = Metadata("test", Segment(fs, 0, 99))

        # Test Difference
        bins_diff = FreqBins(samples, md, Channel.Difference)
        # DC removed in init; verify channel case logic runs
        assert bins_diff.channel == Channel.Difference

        # Test Unknown channel -> defaults to Mean
        bins_unk = FreqBins(samples, md, Channel.Unknown)
        assert bins_unk.channel == Channel.Unknown

        # 1D array sets channel to Unknown
        samples_1d = np.ones(100, dtype=np.float32)
        bins_1d = FreqBins(samples_1d, md, Channel.Left)
        assert bins_1d.channel == Channel.Unknown

    def test_mean_value_error(self):
        fs = 1000
        samples = np.zeros(100, dtype=np.float32)
        md = Metadata("test", Segment(fs, 0, 99))
        bins = FreqBins(samples, md)

        # freq_to_bin: freq=-10 raises; freq=10000 -> bin > len
        with pytest.raises(ValueError, match="Frequency range is out of bounds"):
            bins.mean(10000, 20000)

    def test_get_band_edges_empty(self):
        fs = 1000
        samples = np.zeros(100, dtype=np.float32)
        md = Metadata("test", Segment(fs, 0, 99))
        bins = FreqBins(samples, md)

        # Force empty above_threshold logic
        bins.bins[:] = 0.0
        edges = bins.get_band_edges(tol_db=3.0)
        assert edges == (0.0, 0.0)

    def test_is_noise_sine_no_tones(self):
        fs = 1000
        samples = np.zeros(100, dtype=np.float32)
        md = Metadata("test", Segment(fs, 0, 99))
        bins = FreqBins(samples, md)
        # Mock tones to be empty
        bins.tones = []
        assert bins.is_noise() is True
        assert bins.is_sine() is False

    def test_freq_in_list_edge_cases(self):
        fs = 1000
        samples = np.zeros(100, dtype=np.float32)
        md = Metadata("test", Segment(fs, 0, 99))
        bins = FreqBins(samples, md)

        # freq <= 0
        assert bins.freq_in_list(-5, [100], 1) is False
        assert bins.freq_in_list(0, [100], 1) is False

        # empty list
        assert bins.freq_in_list(100, [], 1) is False
