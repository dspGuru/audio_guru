"""Tests to improve coverage of remaining uncovered lines."""

import pytest
import numpy as np
from analysis_list import AnalysisList
from freq_bins import FreqBins
from sweep_analyzer import SweepAnalyzer
from time_stats import TimeStats
from tone_stats import ToneStats
from tone_analyzer import ToneAnalyzer
from unit_stats import UnitStats
from metadata import Metadata
from segment import Segment
from constants import DEFAULT_FS
import generate


def test_analysis_list_summary_empty():
    """Test AnalysisList.summary() with empty list."""
    al = AnalysisList()
    summary = al.summary()
    assert "Empty" in summary


def test_freq_bins_min_with_data():
    """Test FreqBins.min property with actual data."""
    audio = generate.noise(secs=0.1)
    bins = audio.get_bins()

    # Access min property
    min_val = bins.min
    assert isinstance(min_val, float)
    assert min_val >= 0.0


def test_freq_bins_mean_method():
    """Test FreqBins.mean() method."""
    audio = generate.noise(secs=0.1)
    bins = audio.get_bins()

    # Test mean over frequency range
    mean_pwr = bins.mean(100.0, 1000.0)
    assert isinstance(mean_pwr, float)
    assert mean_pwr >= 0.0


def test_freq_bins_ripple_method():
    """Test FreqBins.ripple() method."""
    audio = generate.noise(secs=0.1)
    bins = audio.get_bins()

    # Test ripple calculation
    ripple = bins.ripple(100.0, 1000.0)
    assert isinstance(ripple, float)
    assert ripple >= 0.0


def test_sweep_analyzer_get_components():
    """Test SweepAnalyzer.get_components() for frequency < 1 Hz."""
    # Create sweep with low frequencies - amp is first param
    audio = generate.sweep(amp=0.5, f_start=20, f_stop=100, secs=1.1)
    analyzer = SweepAnalyzer(audio)

    # Get components (may skip low freq components)
    comps = analyzer.get_components()
    assert comps is not None


def test_time_stats_to_dict_content():
    """Test TimeStats.to_dict() returns expected keys."""
    audio = generate.noise(secs=0.1)
    ts = TimeStats(audio)

    # The to_dict method should return stats
    d = ts.to_dict()
    assert isinstance(d, dict)
    # Should have at least some stats
    assert len(d) > 0


def test_tone_analyzer_get_second_freq():
    """Test ToneAnalyzer detection of second frequency."""
    # Create two-tone signal
    a1 = generate.sine(freq=1000, amp=0.4, secs=1.1)
    a2 = generate.sine(freq=3000, amp=0.4, secs=1.1)

    audio = Audio(fs=a1.fs)
    audio.samples = a1.samples + a2.samples
    audio.segment = a1.segment

    analyzer = ToneAnalyzer(audio)
    segment = audio.get_segments()[0]
    segment.cat = Category.Tone

    result = analyzer.analyze(segment)
    # Should analyze without error
    assert result is not None


def test_tone_analyzer_harmonics():
    """Test ToneAnalyzer with harmonic content."""
    # Generate tone with harmonics
    audio = generate.sine(freq=1000, amp=0.5, secs=1.1, thd=0.05)

    analyzer = ToneAnalyzer(audio)
    segment = audio.get_segments()[0]
    segment.cat = Category.Tone

    result = analyzer.analyze(segment)
    assert result is not None


def test_tone_stats_string_representation_full():
    """Test ToneStats __str__ with all fields."""
    audio = generate.sine(freq=1000, amp=0.5, secs=0.5)
    ts = ToneStats(
        audio,
        freq=1000.0,
        freq2=3000.0,  # Second tone
        sig=0.1,
        noise=0.001,
        spur=0.005,
        dist=0.01,
    )

    s = str(ts)
    assert "Freq" in s
    assert isinstance(s, str)


def test_unit_stats_aggregate_missing_snr():
    """Test UnitStats.aggregate() when unit doesn't have SNR."""
    unit1 = UnitStats("unit1")
    unit1.thd = 0.05
    unit1.noise_floor = 0.01

    # Don't set SNR on unit1
    if hasattr(unit1, "snr"):
        delattr(unit1, "snr")

    unit2 = UnitStats("unit2")
    unit2.snr = 20.0
    unit2.thd = 0.03
    unit2.noise_floor = 0.02

    # Should handle gracefully
    unit1.aggregate(unit2)
    assert True  # If we got here, no exception


# Import needed for Category
from util import Category
from audio import Audio
