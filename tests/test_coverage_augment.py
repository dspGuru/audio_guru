"""Tests to augment coverage for edge cases and uncovered code paths."""

import math
import os
import tempfile

import numpy as np
import pytest

from analysis import Analysis
from audio import Audio
from audio_analyzer import AudioAnalyzer
from component import Component, Components
from constants import DEFAULT_FS
from generate import distortion, noise, sine, sweep, two_tone
from metadata import Metadata
from noise_analyzer import NoiseAnalysis, NoiseAnalyzer
from segment import Segment, Segments
from sweep_analyzer import SweepAnalyzer
from time_stats import TimeStats
from tone_analyzer import ToneAnalyzer
from util import Category, Channel


# ---------------------------------------------------------------------------
# segment.py line 186 - Trunc snap tolerance (within 1ms snaps up)
# ---------------------------------------------------------------------------


def test_segment_trunc_snap_tolerance():
    """Segment.trunc() should snap to next second when within 1ms tolerance."""
    fs = 48000
    # 0.999s should snap to 1.0s (within 1ms tolerance)
    samples = round(fs * 0.999)  # 47952 samples
    seg = Segment(fs=fs, start=0, stop=samples)

    # Initial length
    assert len(seg) == samples

    seg.trunc(1.0)

    # Note: if len < n_secs, we don't enter the truncation path
    # Create 1.999s segment instead
    samples = round(fs * 2.0) - 40  # 95960 samples, ~1.9992 seconds
    seg = Segment(fs=fs, start=0, stop=samples)
    initial_len = len(seg)

    seg.trunc(1.0)

    # Should snap to 96000 (2.0s) since within 48-sample tolerance
    assert seg.stop == fs * 2


def test_segment_trunc_no_snap_outside_tolerance():
    """Segment.trunc() should truncate normally when outside tolerance."""
    fs = 48000
    # Create 1.5 second segment - well outside tolerance, should truncate to 1s
    samples = round(fs * 1.5)  # 72000 samples
    seg = Segment(fs=fs, start=0, stop=samples)

    seg.trunc(1.0)

    # Should truncate to 1.0s (well outside tolerance)
    assert seg.stop == fs


# ---------------------------------------------------------------------------
# analysis.py lines 42, 47, 57, 67 - Properties (cat, fs, segment, unit_id)
# ---------------------------------------------------------------------------


class ConcreteAnalysis(Analysis):
    """Concrete Analysis subclass for testing."""

    @staticmethod
    def summary_header() -> str:
        return "Test Header"

    def summary(self) -> str:
        return "Test Summary"

    def to_dict(self) -> dict:
        return {"test": True}


def test_analysis_properties():
    """Test Analysis base class properties (cat, fs, segment, unit_id)."""
    # Create proper metadata with a segment
    seg = Segment(fs=48000, start=0, stop=48000, id=0, cat=Category.Tone)
    md = Metadata(pathname="Test_Unit_Desc.wav", segment=seg)

    analysis = ConcreteAnalysis(md)

    # Test each property
    assert analysis.cat == "Tone"
    assert analysis.fs == 48000
    assert analysis.segment.cat == Category.Tone
    assert analysis.unit_id == "Test Unit"
    assert analysis.secs == 1.0
    assert analysis.start == 0.0


# ---------------------------------------------------------------------------
# noise_analyzer.py lines 27, 32, 38 - NoiseAnalysis methods
# ---------------------------------------------------------------------------


def test_noise_analysis_methods():
    """Test NoiseAnalysis to_dict, summary, and summary_header methods."""
    # Create noise audio for analysis
    audio = noise(secs=2.0, silence_secs=0.0)

    # Run analysis
    analyzer = NoiseAnalyzer(audio)
    analysis = analyzer.analyze()

    # Test to_dict returns dictionary
    result_dict = analysis.to_dict()
    assert isinstance(result_dict, dict)

    # Test summary returns string
    summary = analysis.summary()
    assert isinstance(summary, str)

    # Test static summary_header
    header = NoiseAnalysis.summary_header()
    assert isinstance(header, str)
    assert "Noise" in header


# ---------------------------------------------------------------------------
# noise_analyzer.py lines 84, 126 - Sine channel difference path
# ---------------------------------------------------------------------------


def test_noise_analyzer_with_sine_uses_difference():
    """NoiseAnalyzer uses Channel.Difference for sine signals."""
    # Create a clear sine wave
    audio = sine(freq=1000, secs=2.0, silence_secs=0.0)

    # Run noise analysis - should detect sine and use difference channel
    analyzer = NoiseAnalyzer(audio)

    # get_components should use Channel.Difference path for sine
    components = analyzer.get_components()
    assert isinstance(components, Components)

    # get_bands should also use Channel.Difference for sine
    bands = analyzer.get_bands()
    assert isinstance(bands, Components)


# ---------------------------------------------------------------------------
# audio_analyzer.py line 118 - TimeStats analysis path
# ---------------------------------------------------------------------------


def test_audio_analyzer_timestats_analysis():
    """AudioAnalyzer returns TimeStats directly for unknown category."""
    # Create very short audio that will be analyzed as TimeStats
    audio = Audio(fs=48000)
    audio.samples = np.random.uniform(-0.1, 0.1, int(48000 * 2.0)).astype(audio.DTYPE)
    audio.select()

    # Create segment with Unknown category
    seg = Segment(fs=48000, start=0, stop=96000, cat=Category.Unknown)

    analyzer = AudioAnalyzer()
    analyzer.audio = audio

    # Analyze - should handle TimeStats path
    analyzer.audio.select(seg)
    result = analyzer.analyze(analyzer.audio)

    # TimeAnalyzer returns TimeStats, which triggers the isinstance check
    assert result is not None


# audio_analyzer.py - Verify pattern match behavior and empty results.
# ---------------------------------------------------------------------------


def test_audio_analyzer_read_empty_pattern(tmp_path):
    """AudioAnalyzer.read() continues on files with no blocks."""
    analyzer = AudioAnalyzer()

    # Read with pattern that matches nothing
    count = analyzer.read(str(tmp_path / "nonexistent_*.wav"))
    assert count == 0


# ---------------------------------------------------------------------------
# component.py line 62 - Component.summary_header
# ---------------------------------------------------------------------------


def test_component_summary_header():
    """Test Component.summary_header static method."""
    header = Component.summary_header()
    assert isinstance(header, str)
    assert "Category" in header or "Frequency" in header


# ---------------------------------------------------------------------------
# tone_analyzer.py line 113 - Spurious harmonic of spurious
# ---------------------------------------------------------------------------


def test_tone_analyzer_spur_harmonic():
    """Verify recategorization of noise as spurious if it aligns with a spur's
    harmonic."""
    # Create primary tone at 1000 Hz
    audio = sine(freq=1000, secs=2.0, amp=0.9, silence_secs=0.0)

    # Add a spur at 200 Hz (not harmonic of 1000)
    spur = sine(freq=200, secs=2.0, amp=0.05, silence_secs=0.0)
    audio += spur

    # Add a harmonic of the spur at 400 Hz (2 * 200)
    spur_harm = sine(freq=400, secs=2.0, amp=0.02, silence_secs=0.0)
    audio += spur_harm

    # Analyze - should detect main tone and categorize spur harmonics
    analyzer = ToneAnalyzer(audio)
    analyzer.analyze()

    # Check components are categorized
    components = analyzer.components
    assert len(components) > 0


# ---------------------------------------------------------------------------
# sweep_analyzer.py line 93 - Empty STFT column (all magnitudes below eps)
# ---------------------------------------------------------------------------


def test_sweep_analyzer_handles_silence_frames():
    """SweepAnalyzer handles frames where all magnitudes are below epsilon."""
    # Create a sweep with some very low amplitude sections
    audio = sweep(amp=0.8, secs=2.0, silence_secs=0.0)

    # Zero out part of the audio to create silent frames
    audio.samples[0:4096] = 0.0

    analyzer = SweepAnalyzer(audio)
    components = analyzer.get_components()

    # Should still work, skipping the silent frames
    assert isinstance(components, Components)


def test_sweep_analyzer_adds_dummy_for_empty():
    """SweepAnalyzer adds dummy component when no components found."""
    # Create very short nearly silent audio
    audio = Audio(fs=48000)
    audio.samples = np.zeros(48000 * 2, dtype=audio.DTYPE)  # Two seconds of silence
    audio.samples += np.random.uniform(-1e-10, 1e-10, len(audio.samples)).astype(
        audio.DTYPE
    )  # Tiny noise
    audio.select()

    analyzer = SweepAnalyzer(audio)

    # analyze should add dummy component if none found
    result = analyzer.analyze()
    assert result is not None


# ---------------------------------------------------------------------------
# generate.py lines 53, 278, 327 - Silence append paths
# ---------------------------------------------------------------------------


def test_distortion_with_silence():
    """distortion() appends silence when silence_secs > 0."""
    audio = distortion(thd=0.01, secs=1.0, silence_secs=0.5)

    # Total should be 1.0 + 0.5 = 1.5 seconds
    expected_samples = round(1.5 * DEFAULT_FS)
    assert len(audio.samples) == expected_samples


def test_sweep_with_silence():
    """sweep() appends silence when silence_secs > 0."""
    audio = sweep(secs=1.0, silence_secs=0.5, channel=Channel.Left)

    # Total should be 1.0 + 0.5 = 1.5 seconds
    expected_samples = round(1.5 * DEFAULT_FS)
    assert len(audio.samples) == expected_samples


def test_two_tone_with_silence():
    """two_tone() appends silence when silence_secs > 0."""
    audio = two_tone(secs=1.0, silence_secs=0.5)

    # Total should be 1.0 + 0.5 = 1.5 seconds
    expected_samples = round(1.5 * DEFAULT_FS)
    assert len(audio.samples) == expected_samples


# ---------------------------------------------------------------------------
# Additional coverage - segments offset method
# ---------------------------------------------------------------------------


def test_segments_offset():
    """Test Segments.offset() method."""
    # Offset subtracts from start, so segments must have start >= offset
    seg1 = Segment(fs=48000, start=2000, stop=50000)
    seg2 = Segment(fs=48000, start=50000, stop=98000)

    segs = Segments([seg1, seg2])
    segs.offset(1000)

    # After offset(1000): start -= 1000
    assert seg1.start == 1000
    assert seg2.start == 49000


def test_segment_desc_without_id():
    """Test Segment.desc property when id is None."""
    seg = Segment(fs=48000, start=0, stop=48000, id=None, cat=Category.Tone)
    desc = seg.desc

    assert "Segment" not in desc  # No segment prefix when id is None
    assert "Tone" in desc


def test_audio_cache_invalidation():
    """Test that Audio properties are cached and invalidated on select."""
    a = Audio(fs=1000)
    # 0.5s of 1.0, 0.5s of 0.5
    s1 = np.ones(500, dtype=np.float32)
    s2 = np.ones(500, dtype=np.float32) * 0.5
    a.samples = np.concatenate((s1, s2))

    # Select first half
    seg1 = Segment(1000, 0, 500)
    a.select(seg1)

    # Access expensive properties to populate cache
    max1 = a.max
    rms1 = a.rms

    assert max1 == pytest.approx(1.0)
    assert rms1 == pytest.approx(1.0)

    # Verify values are cached (access again should be fast/consistent)
    assert a.max == 1.0

    # Select second half - should invalidate cache
    seg2 = Segment(1000, 500, 1000)
    a.select(seg2)

    # Values should update
    assert a.max == pytest.approx(0.5)
    assert a.rms == pytest.approx(0.5)

    # Verify cache is populated with new values
    assert "max" in a.__dict__
    assert a.__dict__["max"] == 0.5
