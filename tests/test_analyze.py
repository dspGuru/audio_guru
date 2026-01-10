from unittest.mock import patch, MagicMock
import analyze
from constants import DEFAULT_MAX_NF_DBFS
from audio_player import AudioPlayer


@patch("analyze.AudioAnalyzer")
@patch("analyze.get_args")
def test_main_success(mock_get_args, mock_audio_analyzer):
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.pattern = "*.wav"
    mock_args.table = True
    mock_args.components = False
    mock_args.audio_stats = DEFAULT_MAX_NF_DBFS
    mock_args.device = False
    mock_args.max_length = 60.0  # Add max_length to mock args
    mock_args.play = None
    mock_args.verbose = False
    mock_args.quiet = False
    mock_args.write_csv = None
    mock_get_args.return_value = mock_args

    # Setup mock analyzer
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read.return_value = 1  # Successfully read 1 file

    # Run main
    analyze.main()

    # Verify calls
    mock_audio_analyzer.assert_called_once_with(max_segment_secs=60.0, verbose=False)
    mock_analyzer_instance.read.assert_called_once_with("*.wav", quiet=False)
    mock_analyzer_instance.print.assert_called_once_with(
        tone_sort_attr=True,
        components=False,
        audio_stats=DEFAULT_MAX_NF_DBFS,
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
    mock_args.play = None
    mock_args.verbose = False
    mock_args.quiet = False
    mock_args.write_csv = None
    mock_get_args.return_value = mock_args

    # Setup mock analyzer to fail reading
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read.return_value = 0  # 0 files read

    # Run main
    analyze.main()

    # Verify calls
    mock_analyzer_instance.read.assert_called_once_with("*.mp3", quiet=False)
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
    mock_args.audio_stats = True
    mock_args.play = None
    mock_args.verbose = False
    mock_args.quiet = False
    mock_args.write_csv = None
    mock_get_args.return_value = mock_args

    # Setup mock analyzer
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read_device.return_value = True

    # Run main
    analyze.main()

    # Verify calls
    mock_audio_analyzer.assert_called_once_with(max_segment_secs=5.0, verbose=False)
    mock_analyzer_instance.read_device.assert_called_once_with(quiet=False)
    mock_analyzer_instance.print.assert_called_once_with(
        tone_sort_attr="thd",
        components=False,
        audio_stats=True,
    )


def test_main_device_failure():
    """Test main function with device read failure."""
    # Mock get_args
    mock_args = MagicMock()
    mock_args.device = True
    mock_args.max_length = 5.0
    mock_args.table = "thd"
    mock_args.components = False
    mock_args.audio_stats = False
    mock_args.play = None
    mock_args.verbose = False
    mock_args.quiet = False
    mock_args.write_csv = None

    with patch("analyze.get_args", return_value=mock_args):
        # Mock AudioAnalyzer
        with patch("analyze.AudioAnalyzer") as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            # read_device returns False
            mock_instance.read_device.return_value = False

            with patch("builtins.print") as mock_print:
                analyze.main()

                MockAnalyzer.assert_called_with(max_segment_secs=5.0, verbose=False)
                mock_instance.read_device.assert_called_once()
                mock_print.assert_any_call("Failed to read from device")


@patch("analyze.AudioPlayer")
@patch("analyze.AudioAnalyzer")
@patch("analyze.get_args")
def test_main_play(mock_get_args, mock_audio_analyzer, mock_audio_player):
    # Setup mock arguments
    mock_args = MagicMock()
    mock_args.pattern = "*.wav"
    mock_args.device = False
    mock_args.max_length = 5.0
    mock_args.play = "sine.wav"
    mock_args.verbose = False
    mock_args.quiet = True
    mock_args.write_csv = None
    mock_args.table = "thd"
    mock_args.components = None
    mock_args.audio_stats = True
    mock_get_args.return_value = mock_args

    # Setup mock analyzer
    mock_analyzer_instance = mock_audio_analyzer.return_value
    mock_analyzer_instance.read.return_value = 1
    mock_analyzer_instance.get_msg_queue.return_value = "dummy_q"

    # Run main
    analyze.main()

    # Verify AudioPlayer was called
    mock_audio_player.assert_called_once_with("dummy_q")
    mock_player_instance = mock_audio_player.return_value
    mock_player_instance.play.assert_called_once_with(
        "sine.wav", duration=5.0, quiet=True
    )

    # Verify read was called
    mock_analyzer_instance.read.assert_called_once_with("*.wav", quiet=True)
