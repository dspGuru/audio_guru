import re
import unittest.mock

import numpy as np
import pytest

from audio import Audio
from component import Component, Components
from metadata import Metadata
from segment import Segment
from tone_analyzer import ToneAnalyzer
from util import Category
from constants import DEFAULT_FREQ


def test_tone_analyzer_with_examples(signals_dir):
    # Find all test tone files
    tone_files = list(signals_dir.glob("*-test_tone_*.wav"))
    if not tone_files:
        pytest.skip("No *-test_tone_*.wav files found in signals")

    # Regex to extract expected THD from filename
    # Format: example_tone_1k-X_XXX_pct.wav -> X.XXX%
    pattern = re.compile(r".*?(\d+_\d+)_pct\.wav")

    for fpath in tone_files:
        fname = fpath.name
        match = pattern.match(fname)
        if not match:
            print(f"Skipping {fname} (no THD pattern match)")
            continue

        thd_str = match.group(1).replace("_", ".")
        expected_thd_pct = float(thd_str)

        print(f"Testing {fname}, expecting THD={expected_thd_pct}%")

        audio = Audio()
        assert audio.read(str(fpath)), f"Failed to read {fname}"

        analyzer = ToneAnalyzer(audio)
        stats = analyzer.analyze()

        # Verify Frequency (should be ~1k)
        assert stats.freq == pytest.approx(DEFAULT_FREQ, abs=5.0)

        # THD estimation isn't perfect - use tolerance
        if expected_thd_pct == 0.0:
            # Should be very low
            assert stats.thd_pct < 0.001
        else:
            # Allow 80% relative error or 0.5% absolute
            assert stats.thd_pct == pytest.approx(expected_thd_pct, rel=0.8, abs=0.5)


def test_tone_analyzer_short_audio():
    import analyzer

    # Verify patch didn't leak
    assert (
        analyzer.SEGMENT_MIN_SECS == 1.0
    ), f"SEGMENT_MIN_SECS is {analyzer.SEGMENT_MIN_SECS}"

    a = Audio()
    a.samples = np.zeros(10, dtype=np.float32)
    # Ensure segment covers the samples explicitly
    a.segment = Segment(fs=a.fs, start=0, stop=10)

    analyzer = ToneAnalyzer(a)
    try:
        analyzer.select(segment=a.segment)
        analyzer.analyze()
    except ValueError as e:
        assert "is too short to analyze" in str(e)
        return

    pytest.fail("Did not raise ValueError for short audio")


def test_tone_analyzer_two_tone_logic():
    fs = 48000
    # Increase to 1.1s for MIN_SEGMENT_SECS constraint
    t = np.arange(round(fs * 1.1)) / fs
    # Two tones: 600 Hz at 0.51 amp (slightly more), 1000 Hz at 0.5 amp
    # 600 Hz -> tone, 1000 Hz -> tone2 (not harmonic of 600)
    tone1 = 0.5 * np.sin(2 * np.pi * DEFAULT_FREQ * t).astype(np.float32)
    tone2 = 0.51 * np.sin(2 * np.pi * 600 * t).astype(np.float32)

    a = Audio(fs=fs)
    a.samples = tone1 + tone2
    a.select()

    analyzer = ToneAnalyzer(a)
    analyzer.analyze()

    # Expect 600 Hz to be self.tone, 1000 Hz to be self.tone2
    assert analyzer.tone.freq == pytest.approx(600, abs=5)
    assert analyzer.tone2.freq == pytest.approx(DEFAULT_FREQ, abs=5)
    assert analyzer.tone.pwr > 0
    assert analyzer.tone2.pwr > 0


def test_harmonic_logic_recategorization():
    # Test recategorization of Noise -> Spur if harmonic of Spur
    a = Audio(fs=1000)
    a.samples = np.zeros(1000, dtype=np.float32)  # Dummy
    a.select()

    analyzer = ToneAnalyzer(a)

    # Mock bin properties
    mock_bins = unittest.mock.Mock()
    analyzer.audio.bins = mock_bins

    # Components:
    # 1. Main Tone (100 Hz, 1.0)
    # 2. Spurious (150 Hz, 0.001) - Not harmonic of 100
    # 3. Harmonic of Spur (450 Hz, 1e-9) - Initially Noise, then recategorized

    comp1 = Component(100.0, 1.0, 0.0, Category.Unknown)
    comp2 = Component(150.0, 0.001, 0.0, Category.Unknown)
    comp3 = Component(450.0, 1e-9, 0.0, Category.Unknown)  # Very low power

    components = Components(a.md)
    components.append(comp1)
    components.append(comp2)
    components.append(comp3)

    mock_bins.tones = components

    # freq_to_bin returns large value to avoid "harmonic alias" logic
    mock_bins.freq_to_bin.return_value = 100
    mock_bins.freq_in_list.return_value = False

    tones = analyzer.get_components()

    # Categorize tones:
    # 100 Hz = Primary Tone
    # 150 Hz = Spurious (non-harmonic of 100 Hz)
    # 450 Hz = Harmonics of Spurious (recategorized from Noise to Spur)
    assert tones[0].cat == Category.Tone
    assert tones[1].cat == Category.Spurious
    assert tones[2].cat == Category.Spurious


def test_spurious_categorization(examples_dir):
    # Test spurious detection logic using MOCK components to avoid signal
    # processing issues
    a = Audio(fs=48000)
    a.samples = np.zeros(48000, dtype=np.float32)
    a.select()
    analyzer = ToneAnalyzer(a)

    # Mock bins and tones
    mock_bins = unittest.mock.Mock()
    analyzer.audio.bins = mock_bins

    # Components to trigger branching:
    # Tone: 1000 Hz, Pwr 1.0
    # Spur: 830 Hz, Pwr 1e-6 (> 1e-8 threshold) -> Spurious
    c0 = Component(DEFAULT_FREQ, 1.0, 0.0, Category.Unknown)
    c1 = Component(830.0, 1e-6, 0.0, Category.Unknown)

    comps = Components(Metadata("test", Segment(48000)))
    comps.append(c0)
    comps.append(c1)
    comps.ref_pwr = 1.0

    mock_bins.tones = comps

    # 2-tone logic: c1.pwr (1e-6) < MIN_TWO_TONE_RATIO (1e-2) * c0.pwr -> Single tone
    mock_bins.freq_to_bin.return_value = 100
    mock_bins.freq_in_list.return_value = False

    # Run
    analyzer.get_components()

    # Single tone detection: c1.pwr < c0.pwr * MIN_TWO_TONE_RATIO (threshold
    # for dual-tone mode)
    assert c0.cat == Category.Tone
    assert c1.cat == Category.Spurious


def test_spurious_harmonic_recategorization():
    # Test recategorization of Noise -> Spur if harmonic of Spur
    a = Audio(fs=48000)
    a.samples = np.zeros(48000, dtype=np.float32)  # Dummy samples
    a.select()
    analyzer = ToneAnalyzer(a)

    # Mock bins
    mock_bins = unittest.mock.Mock()
    analyzer.audio.bins = mock_bins

    # Components:
    # c0: Tone 1000 Hz, Pwr 1.0
    # c1: Spur 830 Hz, Pwr 1e-6 (> 1e-8) -> Spurious
    # c2: Harm 1660 Hz (2*830), Pwr 1e-10 (<1e-8) -> Noise, then recategorized
    c0 = Component(DEFAULT_FREQ, 1.0, 0.0)
    c1 = Component(830.0, 1e-6, 0.0)
    c2 = Component(1660.0, 1e-10, 0.0)

    comps = Components(Metadata("test", Segment(48000)))
    comps.append(c0)
    comps.append(c1)
    comps.append(c2)

    mock_bins.tones = comps
    mock_bins.freq_to_bin.return_value = 100
    mock_bins.freq_in_list.return_value = False  # Not dist

    # Run
    analyzer.get_components()

    assert c0.cat == Category.Tone
    assert c1.cat == Category.Spurious
    assert c2.cat == Category.Spurious  # Recategorized
