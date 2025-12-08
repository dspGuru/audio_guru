"""Test for time_analyzer.py"""

from pathlib import Path
import sys

import pytest
from unittest.mock import MagicMock, patch

from audio import Audio
from component import Components
from segment import Segment
from time_analyzer import TimeAnalyzer
from time_stats import TimeStats


@pytest.fixture
def tone_audio():
    """Create a simple tone audio for testing."""
    audio = Audio(fs=44100)
    audio.generate_tone(freq=1000, secs=1.0)
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
    stats = analyzer.analyze(segment=None)
    assert isinstance(stats, TimeStats)


def test_print(capsys, tone_audio):
    """Test print method output."""
    analyzer = TimeAnalyzer(tone_audio)
    analyzer.analyze()

    # Test without components
    analyzer.print()
    captured = capsys.readouterr()
    assert str(analyzer.time_stats) in captured.out

    # Test with components
    analyzer.print(components=True)
    captured = capsys.readouterr()
    assert "Frequency" in captured.out
    assert str(analyzer.time_stats) in captured.out


def test_main_generate(capsys):
    """Test main function with --generate."""
    with patch.object(sys, "argv", ["time_analyzer.py", "--generate"]):
        with patch("audio.Audio.write") as mock_write:
            import time_analyzer

            time_analyzer.main()

            mock_write.assert_called_with("test_tone.wav")
            captured = capsys.readouterr()
            assert "Generated test tone audio" in captured.out


def test_main_no_args(capsys):
    """Test main function with no arguments."""
    with patch.object(sys, "argv", ["time_analyzer.py"]):
        import time_analyzer

        time_analyzer.main()
        captured = capsys.readouterr()
        assert "No input audio specified" in captured.out


def test_main_with_file(capsys, tmp_path):
    """Test main function with input file."""
    audio_file = tmp_path / "test.wav"
    audio = Audio()
    audio.generate_tone()
    audio.write(str(audio_file))

    with patch.object(sys, "argv", ["time_analyzer.py", str(audio_file)]):
        import time_analyzer

        time_analyzer.main()
        captured = capsys.readouterr()
        assert "Time Stats" in captured.out or "RMS" in captured.out


def test_main_file_not_found(capsys):
    """Test main function with non-existent file."""
    with patch.object(sys, "argv", ["time_analyzer.py", "nonexistent.wav"]):
        import time_analyzer

        time_analyzer.main()
        captured = capsys.readouterr()
        assert "Error: Could not read audio file" in captured.out
