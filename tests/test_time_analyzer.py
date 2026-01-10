"""Test for time_analyzer.py"""

import pytest

from audio import Audio
from component import Components
from constants import DEFAULT_FS, DEFAULT_FREQ
import generate
from time_analyzer import TimeAnalyzer
from time_stats import TimeStats


@pytest.fixture
def tone_audio():
    """Create a simple tone audio for testing."""
    audio = generate.sine(freq=DEFAULT_FREQ, secs=1.1, fs=DEFAULT_FS)
    return audio


def test_initialization(tone_audio):
    """Test that TimeAnalyzer initializes correctly."""
    analyzer = TimeAnalyzer(tone_audio)
    assert analyzer.audio is tone_audio
    assert isinstance(analyzer.audio, Audio)


def test_reset(tone_audio):
    """Test reset method."""
    analyzer = TimeAnalyzer(tone_audio)
    analyzer.analysis = "dummy"
    analyzer.reset()


def test_get_components(tone_audio):
    """Test get_components method."""
    analyzer = TimeAnalyzer(tone_audio)
    components = analyzer.get_components()

    assert isinstance(components, Components)
    assert components.name == "Frequency"
    assert len(components) == 1

    comp = components[0]
    assert comp.freq == tone_audio.freq
    assert comp.pwr == tone_audio.rms


def test_analyze(tone_audio):
    """Test analyze method."""
    analyzer = TimeAnalyzer(tone_audio)
    stats = analyzer.analyze()

    assert isinstance(stats, TimeStats)
    assert stats.rms is not None
    assert stats.max is not None
    assert analyzer.analysis is stats


def test_analyze_with_segment(tone_audio):
    """Test analyze method with a specific segment."""
    analyzer = TimeAnalyzer(tone_audio)
    analyzer.select(segment=None)
    stats = analyzer.analyze()
    assert isinstance(stats, TimeStats)
