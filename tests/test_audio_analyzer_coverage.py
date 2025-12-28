"""Tests for AudioAnalyzer coverage."""

import pytest
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np

from audio_analyzer import AudioAnalyzer, AudioAnalysis
from segment import Segment
from audio_stats import AudioStats
from freq_resp import FreqResp
from tone_stats import ToneStats
from util import Category
from constants import DEFAULT_FS
from time_stats import TimeStats
from component import Components
from metadata import Metadata
from noise_analyzer import NoiseAnalysis


def test_init():
    """Test initialization of AudioAnalyzer."""
    analyzer = AudioAnalyzer()
    assert analyzer.audio.name == "Analyzer Audio"
    assert analyzer.analyzers == []
    assert len(analyzer.analysis_list) == 0
    assert analyzer.results == {}
    assert analyzer.unit_ids == set()


def test_audio_analysis_str():
    """Test string representation of AudioAnalysis."""
    seg = Segment(fs=DEFAULT_FS, start=0, stop=100)
    # Mock analysis objects
    mock_stats = MagicMock(spec=TimeStats)
    mock_stats.__str__.return_value = "Stats"
    mock_analysis = MagicMock()
    mock_analysis.__str__.return_value = "AnalysisResult"
    mock_comps = MagicMock(spec=Components)
    mock_comps.__str__.return_value = "Comps"

    aa = AudioAnalysis(seg, mock_stats, mock_analysis, mock_comps)
    s = str(aa)
    assert "Analysis(segment=" in s
    assert "result=AnalysisResult" in s


def test_analyze_too_short(capsys):
    """Test analyze with segment shorter than minimum."""
    analyzer = AudioAnalyzer()

    # Create very short segment
    seg = Segment(fs=DEFAULT_FS, start=0, stop=1)  # 1 sample
    # Check constants.MIN_SEGMENT_SECS logic (usually 1.0)
    # 1 sample at 44.1k is definitely < 1.0s

    result = analyzer.analyze(seg)
    assert result is None

    captured = capsys.readouterr()
    assert "too short to analyze" in captured.out


def test_analyze_unknown_no_analyzer():
    """Test analyze with a category that has no analyzer class configured."""
    analyzer = AudioAnalyzer()

    seg = Segment(fs=DEFAULT_FS, start=0, stop=int(DEFAULT_FS))
    seg.cat = Category.Band  # Band has no analyzer in analyzer.analyzer_classes

    # Ensure audio buffer is large enough
    analyzer.audio.samples = np.zeros(int(DEFAULT_FS), dtype=np.float32)

    # Bypass unknown resolution by setting cat manually
    result = analyzer.analyze(seg)
    assert result is None


def test_analyze_unknown_resolves_category():
    """Test analyze resolves Unknown category."""
    analyzer = AudioAnalyzer()

    # Mock audio.get_category
    analyzer.audio.get_category = MagicMock(return_value=Category.Tone)
    analyzer.audio.select = MagicMock()

    seg = Segment(
        fs=DEFAULT_FS, start=0, stop=int(DEFAULT_FS) + 100, cat=Category.Unknown
    )

    # Mocking analyzer_classes to avoid actual analysis overhead/errors
    with patch.object(analyzer, "analyzer_classes", {Category.Tone: MagicMock()}):
        analyzer.analyze(seg)
        analyzer.audio.get_category.assert_called_once()
        assert seg.cat == Category.Tone


def test_read_success(capsys):
    """Test successful read of multiple files."""
    analyzer = AudioAnalyzer()

    with (
        patch("glob.glob", return_value=["file1.wav", "file2.wav"]),
        patch("audio_analyzer.Audio") as MockAudio,
        patch("audio_analyzer.AudioStats"),
    ):

        # Create distinct mocks for each file
        mock_audio1 = MagicMock()
        mock_audio1.read_block.side_effect = [True, False]
        mock_audio1.__iter__.return_value = iter([])

        mock_audio2 = MagicMock()
        mock_audio2.read_block.side_effect = [True, False]
        mock_audio2.__iter__.return_value = iter([])

        # MockAudio constructor returns these in sequence
        MockAudio.side_effect = [mock_audio1, mock_audio2]

        num = analyzer.read("*.wav")

        # Debug output if failed
        captured = capsys.readouterr()
        if captured.out:
            print(captured.out)

        assert num == 2

        # Verify open calls on respective mocks
        mock_audio1.open.assert_called_with("file1.wav")
        mock_audio2.open.assert_called_with("file2.wav")


def test_read_no_files():
    """Test read when no files match pattern."""
    analyzer = AudioAnalyzer()

    with patch("glob.glob", return_value=[]):
        num = analyzer.read("*.wav")
        assert num == 0


def test_read_empty_file_skipped():
    """Test read when a file is empty (read_block returns False immediately)."""
    analyzer = AudioAnalyzer()

    with (
        patch("glob.glob", return_value=["empty.wav"]),
        patch("audio_analyzer.Audio") as MockAudio,
    ):
        mock_audio = MockAudio.return_value
        # read_block returns False immediately (empty file)
        mock_audio.read_block.return_value = False

        num = analyzer.read("*.wav")

        # Should not increment num_read
        assert num == 0

        # Verify open was called
        mock_audio.open.assert_called_with("empty.wav")


def test_analyze_success():
    """Test successful analysis of a segment."""
    analyzer = AudioAnalyzer()

    # Mock audio selection/validation
    analyzer.audio.select = MagicMock()
    analyzer.audio.get_category = MagicMock(return_value=Category.Tone)

    seg = Segment(fs=DEFAULT_FS, start=0, stop=int(DEFAULT_FS), cat=Category.Tone)

    # Mock ToneAnalyzer
    mock_analyzer_class = MagicMock()
    mock_analyzer_instance = mock_analyzer_class.return_value
    mock_analyzer_instance.components = MagicMock()
    mock_analysis_result = MagicMock()
    mock_analyzer_instance.analyze.return_value = mock_analysis_result

    with patch.object(
        analyzer, "analyzer_classes", {Category.Tone: mock_analyzer_class}
    ):
        result = analyzer.analyze(seg)

        assert result is not None
        assert isinstance(result, AudioAnalysis)
        assert result.segment == seg
        assert result.analysis == mock_analysis_result

        # Check side effects
        assert len(analyzer.analyzers) == 1
        assert len(analyzer.analysis_list) == 2  # TimeStats + Analysis
        assert f"Analyzer Audio:{seg}" in analyzer.results


def test_analyze_category_unknown_returns_timestats():
    """Test verify Category.Unknown results in TimeStats."""
    analyzer = AudioAnalyzer()

    # Mock audio
    analyzer.audio.select = MagicMock()
    # Mock Unknown category that persists
    analyzer.audio.get_category = MagicMock(return_value=Category.Unknown)

    seg = Segment(fs=DEFAULT_FS, start=0, stop=int(DEFAULT_FS), cat=Category.Unknown)

    # Mock TimeAnalyzer
    mock_analyzer_class = MagicMock()
    mock_analyzer_instance = mock_analyzer_class.return_value
    mock_analyzer_instance.components = MagicMock()

    # Return a TimeStats object from analyze
    mock_stats = MagicMock(spec=TimeStats)
    mock_analyzer_instance.analyze.return_value = mock_stats

    # Inject TimeAnalyzer mock
    with patch.object(
        analyzer, "analyzer_classes", {Category.Unknown: mock_analyzer_class}
    ):
        result = analyzer.analyze(seg)

        assert result is not None
        assert isinstance(result, AudioAnalysis)
        # Verify result.time_stats is the same as result.analysis when it is TimeStats
        assert result.time_stats is mock_stats
        assert result.analysis is mock_stats


def test_print_options(capsys):
    """Test print method execution with various options."""
    analyzer = AudioAnalyzer()

    # Mock analysis_list entirely to avoid internal logic issues with mocks
    analyzer.analysis_list = MagicMock()

    # Setup filtered return values
    mock_audio_stats_list = MagicMock()
    mock_audio_stats_list.summary.return_value = "Audio Stats Summary"

    mock_time_stats_list = MagicMock()
    mock_time_stats_list.summary.return_value = "Time Stats Summary"

    mock_freq_list = MagicMock()
    mock_freq_list.summary.return_value = "Freq Resp Summary"

    mock_tone_list = MagicMock()
    mock_tone_list.summary.return_value = "Tone Stats Summary"

    mock_noise_list = MagicMock()
    mock_noise_list.summary.return_value = "Noise Stats Summary"

    # filtered is called multiple times with different class args
    # use side_effect to return appropriate mock list
    def filtered_side_effect(**kwargs):
        if kwargs.get("__class__") == AudioStats:
            return mock_audio_stats_list
        if kwargs.get("__class__") == TimeStats:
            return mock_time_stats_list
        if kwargs.get("__class__") == FreqResp:
            return mock_freq_list
        if kwargs.get("__class__") == ToneStats:
            return mock_tone_list
        if kwargs.get("__class__") == NoiseAnalysis:
            return mock_noise_list
        return []

    analyzer.analysis_list.filtered.side_effect = filtered_side_effect

    # Mock analyzers
    mock_analyzer = MagicMock()
    analyzer.analyzers.append(mock_analyzer)

    analyzer.print(
        tone_sort_attr="thd", time_sort_attr="start", components=True, noise_floor=True
    )

    # Verify sub-analyzer call IS NOT MADE (logic removed)
    mock_analyzer.print.assert_not_called()

    captured = capsys.readouterr()
    assert "Audio Stats Summary" in captured.out
    assert "Time Stats Summary" in captured.out
    assert "Freq Resp Summary" in captured.out
    assert "Tone Stats Summary" in captured.out


def test_read_exception(capsys):
    """Test read method exception handling."""
    analyzer = AudioAnalyzer()

    with patch("glob.glob", side_effect=Exception("Glob Error")):
        num = analyzer.read("*.wav")
        assert num == 0
        captured = capsys.readouterr()
        assert "Error reading audio file: Glob Error" in captured.out


def test_read_device_keyboard_interrupt(capsys):
    """Test read_device handles KeyboardInterrupt."""
    analyzer = AudioAnalyzer()

    with patch("audio_analyzer.Audio") as MockAudio:
        mock_audio_instance = MockAudio.return_value
        mock_audio_instance.read_block.side_effect = KeyboardInterrupt

        result = analyzer.read_device()

        assert result is True  # Returns True on interrupt if handled
        assert "Stopping recording..." in capsys.readouterr().out
        mock_audio_instance.close.assert_called()


def test_read_device_keyboard_interrupt_with_blocks(capsys):
    """Test read_device handles KeyboardInterrupt after reading some blocks."""
    analyzer = AudioAnalyzer()

    with patch("audio_analyzer.Audio") as MockAudio:
        mock_audio_instance = MockAudio.return_value
        # return True once, then Raise
        mock_audio_instance.read_block.side_effect = [True, KeyboardInterrupt]
        mock_audio_instance.__iter__.return_value = iter([])
        mock_audio_instance.get_noise_floor.return_value = (0.0, 0)
        # Also need num_channels property mock since AudioStats accesses it
        type(mock_audio_instance).num_channels = unittest.mock.PropertyMock(
            return_value=1
        )

        # Mock print method to verify it's called
        with patch.object(analyzer, "print") as mock_print:
            result = analyzer.read_device()

            assert result is True
            assert "Stopping recording..." in capsys.readouterr().out
            mock_audio_instance.close.assert_called()

            # Since num_blocks > 0 (it was 1), verify steps
            # AudioStats appended
            assert len(analyzer.analysis_list) == 1
            assert isinstance(analyzer.analysis_list[0], AudioStats)

            # Print called
            mock_print.assert_called_once()


def test_read_device_exception(capsys):
    """Test read_device handles generic Exception."""
    analyzer = AudioAnalyzer()

    with patch("audio_analyzer.Audio") as MockAudio:
        mock_audio_instance = MockAudio.return_value
        mock_audio_instance.open_device.side_effect = Exception("Device Error")

        result = analyzer.read_device()

        assert result is False
        assert "Error reading device: Device Error" in capsys.readouterr().out


def test_read_device_duration_limit():
    """Test read_device respects duration limit."""
    analyzer = AudioAnalyzer()

    with patch("audio_analyzer.Audio") as MockAudio:
        mock_audio_instance = MockAudio.return_value
        mock_audio_instance.fs = 1000
        mock_audio_instance.blocksize = 100
        # read_block returns True (success)
        mock_audio_instance.read_block.return_value = True
        # Mock iterator to return empty list so we don't crash in analysis loop
        mock_audio_instance.__iter__.return_value = iter([])

        # Duration 1s -> 1000 samples -> 10 blocks (plus 1 buffer maybe)
        # Limit logic: max_blocks = int(duration * fs / blocksize) + 1
        # 1.0 * 1000 / 100 + 1 = 11 blocks.

        # We need read_block to return True enough times to hit limit
        # side_effect allows us to control how many times it returns True successfully
        # but the analyzer loop breaks when max_blocks is reached.
        # So we can just set it to always return True, and ensure it stops.
        # But to be safe against infinite loops if logic is wrong, we give it a limit > expected.
        mock_audio_instance.read_block.side_effect = [True] * 20

        analyzer.read_device(duration=1.0)

        # Should have called roughly 11 times
        assert mock_audio_instance.read_block.call_count >= 10
        assert mock_audio_instance.read_block.call_count < 20


def test_read_device_analyze_segments():
    """Test that read_device calls analyze on found segments."""
    analyzer = AudioAnalyzer()
    seg = Segment(fs=DEFAULT_FS, start=0, stop=100)

    with patch("audio_analyzer.Audio") as MockAudio:
        mock_audio_instance = MockAudio.return_value
        # read_block returns True once then False
        mock_audio_instance.read_block.side_effect = [True, False]
        # iterator yields one segment when called
        # iterator yields one segment when called
        mock_audio_instance.__iter__.return_value = iter([seg])
        mock_audio_instance.get_noise_floor.return_value = (0.0, 0)
        type(mock_audio_instance).num_channels = unittest.mock.PropertyMock(
            return_value=1
        )

        with patch.object(analyzer, "analyze") as mock_analyze:
            # Return a mock analysis object so summary() call works
            mock_analysis_res = MagicMock()
            mock_analyze.return_value = mock_analysis_res

            analyzer.read_device()

            mock_analyze.assert_called_with(seg)

            # Verify AudioStats appended at end (num_blocks > 0)
            # analysis_list should contain [AudioAnalysis, AudioStats]
            assert len(analyzer.analysis_list) == 2
            assert isinstance(analyzer.analysis_list[-1], AudioStats)
