from unittest.mock import patch, MagicMock
import analyze


@patch("analyze.AudioAnalyzer")
@patch("analyze.get_args")
@patch("analyze.glob.glob")
def test_main_success(mock_glob, mock_get_args, mock_audio_analyzer):
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.pattern = "*.wav"
    mock_args.table = True
    mock_args.components = False
    mock_args.noise_floor = -60
    mock_get_args.return_value = mock_args

    # Setup mock glob
    mock_glob.return_value = ["dummy.wav"]

    # Setup mock analyzer
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read.return_value = True

    # Run main
    analyze.main()

    # Verify calls
    mock_audio_analyzer.assert_called_once()
    mock_glob.assert_called_once_with("*.wav")
    mock_analyzer_instance.read.assert_called_once_with("dummy.wav")
    mock_analyzer_instance.print.assert_called_once_with(
        tone_sort_attr=True,
        components=False,
        noise_floor=-60,
    )


@patch("analyze.AudioAnalyzer")
@patch("analyze.get_args")
@patch("analyze.glob.glob")
@patch("builtins.print")
def test_main_no_files_found(mock_print, mock_glob, mock_get_args, mock_audio_analyzer):
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.pattern = "*.mp3"
    mock_get_args.return_value = mock_args

    # Setup mock glob to return empty list
    mock_glob.return_value = []

    # Setup mock analyzer to fail reading (shouldn't be called, but good practice)
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read.return_value = False

    # Run main
    analyze.main()

    # Verify calls
    mock_glob.assert_called_once_with("*.mp3")
    mock_analyzer_instance.read.assert_not_called()
    mock_analyzer_instance.print.assert_not_called()
    mock_print.assert_called_once_with("No files found.")
