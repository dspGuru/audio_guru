"""Tests for play.py."""

import unittest
from unittest.mock import patch, MagicMock
import play


class TestPlay(unittest.TestCase):
    """Test play.py."""

    @patch("play.AudioPlayer")
    def test_main(self, MockPlayer):
        """Test main function."""
        # Mock sys.argv
        with patch("sys.argv", ["play.py", "*.wav", "-l", "10.0"]):
            play.main()

            # Verify Player instantiated
            MockPlayer.assert_called_once()

            # Verify play called with correct args
            instance = MockPlayer.return_value
            instance.play.assert_called_with("*.wav", duration=10.0, quiet=False)

    @patch("play.AudioPlayer")
    def test_main_no_pattern(self, MockPlayer):
        """Test main with no pattern."""
        with patch("sys.argv", ["play.py"]):
            with patch("builtins.print") as mock_print:
                play.main()
                mock_print.assert_called_with("Please specify a file pattern.")

                # Should not play
                instance = MockPlayer.return_value
                instance.play.assert_not_called()

    @patch("play.AudioPlayer")
    def test_main_quiet(self, MockPlayer):
        """Test main function with quiet flag."""
        with patch("sys.argv", ["play.py", "*.wav", "-q"]):
            play.main()

            # Verify play called with quiet=True
            instance = MockPlayer.return_value
            instance.play.assert_called_with("*.wav", duration=15.0, quiet=True)

    def test_get_args_with_pattern(self):
        """Test get_args with pattern argument."""
        with patch("sys.argv", ["play.py", "test.wav"]):
            args = play.get_args()
            self.assertEqual(args.pattern, "test.wav")
            self.assertEqual(args.max_length, 15.0)

    def test_get_args_with_max_length(self):
        """Test get_args with --max-length argument."""
        with patch("sys.argv", ["play.py", "test.wav", "--max-length", "5.5"]):
            args = play.get_args()
            self.assertEqual(args.pattern, "test.wav")
            self.assertEqual(args.max_length, 5.5)

    def test_get_args_with_short_option(self):
        """Test get_args with -l short option."""
        with patch("sys.argv", ["play.py", "*.wav", "-l", "10"]):
            args = play.get_args()
            self.assertEqual(args.pattern, "*.wav")
            self.assertEqual(args.max_length, 10.0)

    def test_get_args_empty_pattern_default(self):
        """Test get_args defaults to empty pattern when not provided."""
        with patch("sys.argv", ["play.py"]):
            args = play.get_args()
            self.assertEqual(args.pattern, "")
            self.assertEqual(args.max_length, 15.0)

    def test_get_args_quiet(self):
        """Test get_args with quiet flag."""
        with patch("sys.argv", ["play.py", "test.wav", "-q"]):
            args = play.get_args()
            self.assertTrue(args.quiet)
