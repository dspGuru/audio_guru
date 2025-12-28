from unittest.mock import patch, MagicMock
import analyze
from constants import DEFAULT_MAX_NF_DBFS


@patch("analyze.AudioAnalyzer")
@patch("analyze.get_args")
def test_main_success(mock_get_args, mock_audio_analyzer):
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.pattern = "*.wav"
    mock_args.table = True
    mock_args.components = False
    mock_args.noise_floor = DEFAULT_MAX_NF_DBFS
    mock_args.device = False
    mock_args.max_length = 60.0  # Add max_length to mock args
    mock_get_args.return_value = mock_args

    # Setup mock analyzer
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read.return_value = 1  # Successfully read 1 file

    # Run main
    analyze.main()

    # Verify calls
    mock_audio_analyzer.assert_called_once()
    mock_analyzer_instance.read.assert_called_once_with("*.wav")
    mock_analyzer_instance.print.assert_called_once_with(
        tone_sort_attr=True,
        components=False,
        noise_floor=DEFAULT_MAX_NF_DBFS,
    )


@patch("analyze.AudioAnalyzer")
@patch("analyze.get_args")
@patch("builtins.print")
def test_main_no_files_found(mock_print, mock_get_args, mock_audio_analyzer):
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.pattern = "*.mp3"
    mock_args.device = False
    mock_args.max_length = 60.0
    mock_get_args.return_value = mock_args

    # Setup mock analyzer to fail reading
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read.return_value = 0  # 0 files read

    # Run main
    analyze.main()

    # Verify calls
    mock_analyzer_instance.read.assert_called_once_with("*.mp3")
    mock_analyzer_instance.print.assert_not_called()
    mock_print.assert_called_once_with("No files found.")


@patch("analyze.AudioAnalyzer")
@patch("analyze.get_args")
def test_main_device(mock_get_args, mock_audio_analyzer):
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.device = True
    mock_args.max_length = 5.0
    mock_args.table = "thd"
    mock_args.components = False
    mock_args.noise_floor = True
    mock_get_args.return_value = mock_args

    # Setup mock analyzer
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read_device.return_value = True

    # Run main
    analyze.main()

    # Verify calls
    mock_audio_analyzer.assert_called_once_with(max_segment_secs=5.0)
    mock_analyzer_instance.read_device.assert_called_once_with()
    mock_analyzer_instance.print.assert_called_once_with(
        tone_sort_attr="thd",
        components=False,
        noise_floor=True,
    )
