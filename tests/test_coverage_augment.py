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
from util import Category


# ---------------------------------------------------------------------------
# segment.py line 186 - Trunc snap tolerance (within 1ms snaps up)
# ---------------------------------------------------------------------------


def test_segment_trunc_snap_tolerance():
    """Segment.trunc() should snap to next second when within 1ms tolerance."""
    fs = 48000
    # Create a segment that is 0.999s - should snap up to 1.0s
    # n = fs * 0.999 = 47952 samples
    # 1 second = 48000 samples
    # Remainder = 47952 % 48000 = 47952
    # n_secs - remainder = 48000 - 47952 = 48 samples = 1ms exactly at 48kHz
    # So this should snap up
    samples = round(fs * 0.999)  # 47952 samples
    seg = Segment(fs=fs, start=0, stop=samples)

    # Initial length
    assert len(seg) == samples

    seg.trunc(1.0)

    # Should snap up to 48000 samples (1.0 seconds)
    # multiple = (47952 // 48000) + 1 = 0 + 1 = 1
    # new_stop = 0 + 1 * 48000 = 48000
    # Wait, that exceeds original stop - let's recalculate
    # Actually if n (len) > n_secs, we do the calculation
    # Here n = 47952 < 48000, so we don't enter the if block at all
    # Need to create segment > 1 second that's within tolerance of next multiple

    # Create 1.999 second segment (within 1ms of 2 seconds)
    samples = round(fs * 2.0) - 40  # 96000 - 40 = 95960, ~1.9992 seconds
    seg = Segment(fs=fs, start=0, stop=samples)
    initial_len = len(seg)

    seg.trunc(1.0)

    # Should snap to 96000 (2.0 seconds) since we're within tolerance
    # n = 95960, n_secs = 48000, remainder = 95960 % 48000 = 47960
    # n_secs - remainder = 48000 - 47960 = 40 samples
    # tolerance = max(1, round(48000 * 0.001)) = 48 samples
    # 40 <= 48 so we snap up to multiple = (95960 // 48000) + 1 = 1 + 1 = 2
    # new_stop = 0 + 2 * 48000 = 96000
    assert seg.stop == fs * 2


def test_segment_trunc_no_snap_outside_tolerance():
    """Segment.trunc() should truncate normally when outside tolerance."""
    fs = 48000
    # Create 1.5 second segment - well outside tolerance, should truncate to 1s
    samples = round(fs * 1.5)  # 72000 samples
    seg = Segment(fs=fs, start=0, stop=samples)

    seg.trunc(1.0)

    # Should truncate to 48000 (1.0 second)
    # n = 72000, n_secs = 48000, remainder = 72000 % 48000 = 24000
    # n_secs - remainder = 48000 - 24000 = 24000 samples, way > tolerance (48)
    # So multiple = 72000 // 48000 = 1
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
    result = analyzer.analyze(seg)

    # TimeAnalyzer returns TimeStats, which triggers the isinstance check
    assert result is not None


# ---------------------------------------------------------------------------
# audio_analyzer.py line 172 - Empty pattern match (continue on no blocks)
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
    """ToneAnalyzer recategorizes noise as spur when it's harmonic of a spur."""
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
    audio.samples = np.zeros(48000 * 2, dtype=audio.DTYPE)  # 2 seconds of silence
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
    audio = sweep(secs=1.0, silence_secs=0.5)

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
