import unittest
from unittest.mock import MagicMock, patch
import queue
import time

from audio_player import AudioPlayer
from constants import DEFAULT_BLOCKSIZE_SECS


# These tests use extensive mocking of SoundFile and OutputStream context
# managers to verify that the threaded player correctly manages file I/O
# and audio device interactions without needing actual hardware.
class TestAudioPlayerThreaded(unittest.TestCase):

    def _create_mock_data(self, content):
        mock_data = MagicMock(spec=bytes)
        mock_data.__len__.return_value = len(content)
        mock_data.dtype = "float32"
        return mock_data

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    def test_play_pushes_to_queue(self, mock_sf_ctor, mock_stream_ctor):
        """Test that play method pushes items to queue and worker processes them."""

        # Mock SoundFile context manager
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2
        mock_block1 = self._create_mock_data(b"block1")
        mock_block2 = self._create_mock_data(b"block2")
        mock_sf_instance.read.side_effect = [
            mock_block1,
            mock_block2,
            self._create_mock_data(b""),
        ]

        # Mock OutputStream context manager

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        player = AudioPlayer()

        # Patch the queue put method to verify calls
        with patch.object(player.play_q, "put", wraps=player.play_q.put) as mock_put:
            # Also patch glob to return a dummy file
            with patch("audio_player.glob.glob", return_value=["test.wav"]):
                # Use duration > 0 so playback loop runs
                player.play("test.wav", duration=2.0)
                # Verify puts
                mock_put.assert_called_with(("test.wav", 2.0))

        # Join queue to ensure worker processed it
        player.play_q.join()

        # Verify sf.SoundFile was initialized
        mock_sf_ctor.assert_called_with("test.wav")
        # Verify read called with blocksize (samplerate * DEFAULT_BLOCKSIZE_SECS)
        expected_blocksize = int(44100 * DEFAULT_BLOCKSIZE_SECS)
        mock_sf_instance.read.assert_called_with(expected_blocksize)
        # Verify bits were written to stream
        self.assertEqual(mock_stream_instance.write.call_count, 2)
        mock_stream_instance.write.assert_any_call(mock_block1)
        mock_stream_instance.write.assert_any_call(mock_block2)

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    def test_play_stops_at_eof(self, mock_sf_ctor, mock_stream_ctor):
        """Test that playback stops at EOF even if duration is longer."""

        # Mock SoundFile
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2

        # Mock read: returns data once, then empty (EOF)
        mock_sf_instance.read.side_effect = [
            self._create_mock_data(b"block"),
            self._create_mock_data(b""),
        ]

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        player = AudioPlayer()

        with patch("audio_player.glob.glob", return_value=["short.wav"]):
            # Play for 10 seconds, but file is short
            player.play("short.wav", duration=10.0)
            player.play_q.join()

        # Verify seek was NOT called (no looping)
        mock_sf_instance.seek.assert_not_called()

        # Stream writes: 1 block
        self.assertEqual(mock_stream_instance.write.call_count, 1)

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    def test_msg_queue_receives_filename(self, mock_sf_ctor, mock_stream_ctor):
        """Test that msg_q receives filename when provided."""
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2
        mock_sf_instance.blocks.return_value = [b"block"]

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        msg_q = queue.Queue()
        player = AudioPlayer(msg_q=msg_q)

        with patch("audio_player.glob.glob", return_value=["notify.wav"]):
            player.play("notify.wav", duration=1.0)
            player.play_q.join()

        # Verify msg_q received the filename
        self.assertFalse(msg_q.empty())
        self.assertEqual(msg_q.get_nowait(), "notify.wav")

    @patch("audio_player.sf.SoundFile")
    def test_worker_handles_exception(self, mock_sf_ctor):
        """Test that worker handles exceptions gracefully."""
        mock_sf_ctor.return_value.__enter__.side_effect = Exception("File error")

        player = AudioPlayer()

        with patch("audio_player.glob.glob", return_value=["bad.wav"]):
            with patch("builtins.print") as mock_print:
                player.play("bad.wav", duration=1.0)
                player.play_q.join()

                # Verify error was printed
                mock_print.assert_any_call("Error playing bad.wav: File error")

    def test_play_no_files_found(self):
        """Test that play handles no files found gracefully."""
        player = AudioPlayer()

        with patch("audio_player.glob.glob", return_value=[]):
            with patch("builtins.print") as mock_print:
                player.play("nonexistent*.wav", duration=1.0)
                mock_print.assert_called_with("No files found matching the pattern.")

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    def test_play_multiple_files(self, mock_sf_ctor, mock_stream_ctor):
        """Test playing multiple files sequentially."""
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2
        # Mock read blocks for play once
        mock_block = self._create_mock_data(b"block")
        # Ensure we have enough data for 3 files: [block, eof, block, eof, block, eof]
        mock_sf_instance.read.side_effect = [
            mock_block,
            self._create_mock_data(b""),
            mock_block,
            self._create_mock_data(b""),
            mock_block,
            self._create_mock_data(b""),
        ]

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        msg_q = queue.Queue()
        player = AudioPlayer(msg_q=msg_q)

        files = ["file1.wav", "file2.wav", "file3.wav"]
        with patch("audio_player.glob.glob", return_value=files):
            player.play("*.wav", duration=2.0)
            player.play_q.join()

        # Verify all files were played
        played_files = []
        while not msg_q.empty():
            played_files.append(msg_q.get_nowait())
        self.assertEqual(played_files, files)

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    def test_loop_exact_duration(self, mock_sf_ctor, mock_stream_ctor):
        """Test looping stops exactly when duration is reached."""
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 1

        # Simulate file where read returns exact frames requested
        mock_sf_instance.read.return_value = self._create_mock_data(
            [0] * 44100
        )  # One second of data

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        player = AudioPlayer()

        with patch("audio_player.glob.glob", return_value=["exact.wav"]):
            # Play for exactly 1 second (should read once)
            player.play("exact.wav", duration=1.0)
            player.play_q.join()

        # Should have written once
        self.assertEqual(mock_stream_instance.write.call_count, 1)


class TestAudioPlayerStopBehavior(unittest.TestCase):
    """Tests for stop() behavior and queue clearing."""

    def test_stop_clears_queue_with_items(self):
        """Test that stop() clears any remaining items from the queue."""
        player = AudioPlayer()

        # Add items to the queue directly (without triggering playback)
        player.play_q.put(("file1.wav", 1.0))
        player.play_q.put(("file2.wav", 2.0))
        player.play_q.put(("file3.wav", 3.0))

        # Verify queue has items
        self.assertFalse(player.play_q.empty())

        # Call stop
        player.stop()

        # Verify stop_event is set
        self.assertTrue(player.stop_event.is_set())

        # Verify queue is empty (items were cleared)
        self.assertTrue(player.play_q.empty())

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    def test_worker_skips_playback_when_stopped(self, mock_sf_ctor, mock_stream_ctor):
        """Test that worker skips playback when stop_event is set before processing."""
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2

        player = AudioPlayer()

        # Set stop event BEFORE adding item
        player.stop_event.set()

        # Add item to queue
        player.play_q.put(("skipped.wav", 1.0))

        # Wait for worker to process
        player.play_q.join()

        # SoundFile should NOT have been called (playback skipped)
        mock_sf_ctor.assert_not_called()


class TestAudioPlayerDtypeConversion(unittest.TestCase):
    """Tests for dtype conversion during playback."""

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    def test_converts_non_float32_data(self, mock_sf_ctor, mock_stream_ctor):
        """Test that non-float32 data is converted to float32 before writing."""
        import numpy as np

        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2

        # Create mock data that is NOT float32 (e.g., int16)
        mock_data = MagicMock()
        mock_data.__len__.return_value = 1000
        mock_data.dtype = "int16"  # Not float32
        mock_data.astype.return_value = MagicMock()  # Return converted data

        mock_sf_instance.read.side_effect = [
            mock_data,
            MagicMock(__len__=lambda self: 0),  # EOF
        ]

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        player = AudioPlayer()

        with patch("audio_player.glob.glob", return_value=["int16.wav"]):
            player.play("int16.wav", duration=1.0)
            player.play_q.join()

        # Verify astype was called to convert to float32
        mock_data.astype.assert_called_with("float32")


class TestAudioPlayerStopDuringPlayback(unittest.TestCase):
    """Tests for stop during active playback."""

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    def test_stop_during_playback_returns_early(self, mock_sf_ctor, mock_stream_ctor):
        """Test that setting stop_event during playback causes early return."""
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2

        # Create mock data with side effect that sets stop event
        player = AudioPlayer()

        def read_and_stop(block_size):
            # First read returns data, second triggers stop
            if not hasattr(read_and_stop, "called"):
                read_and_stop.called = True
                mock_data = MagicMock()
                mock_data.__len__.return_value = 1000
                mock_data.dtype = "float32"
                # Set stop during read
                player.stop_event.set()
                return mock_data
            return MagicMock(__len__=lambda self: 0)

        mock_sf_instance.read.side_effect = read_and_stop

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        with patch("audio_player.glob.glob", return_value=["interrupt.wav"]):
            player.play("interrupt.wav", duration=10.0)
            player.play_q.join()

        # Only one write should have occurred before stop
        self.assertEqual(mock_stream_instance.write.call_count, 1)


class TestAudioPlayerQuietParameter(unittest.TestCase):
    """Tests for the quiet parameter in AudioPlayer.play()."""

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    @patch("audio_player.sd.query_devices")
    def test_play_quiet_suppresses_device_output(
        self, mock_query_devices, mock_sf_ctor, mock_stream_ctor
    ):
        """Test that quiet=True suppresses 'Audio output' print."""
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2
        mock_sf_instance.read.return_value = MagicMock(__len__=lambda self: 0)

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        player = AudioPlayer()

        with patch("audio_player.glob.glob", return_value=["test.wav"]):
            with patch("builtins.print") as mock_print:
                player.play("test.wav", duration=1.0, quiet=True)
                player.play_q.join()

                # "Audio output" should NOT have been printed
                for call in mock_print.call_args_list:
                    assert "Audio output" not in str(call)

        # query_devices should NOT have been called
        mock_query_devices.assert_not_called()

    @patch("audio_player.sd.OutputStream")
    @patch("audio_player.sf.SoundFile")
    @patch("audio_player.sd.query_devices")
    @patch("audio_player.sd.default")
    def test_play_not_quiet_prints_device_output(
        self, mock_default, mock_query_devices, mock_sf_ctor, mock_stream_ctor
    ):
        """Test that quiet=False (default) prints 'Audio output'."""
        mock_sf_instance = MagicMock()
        mock_sf_ctor.return_value.__enter__.return_value = mock_sf_instance
        mock_sf_instance.samplerate = 44100
        mock_sf_instance.channels = 2
        mock_sf_instance.read.return_value = MagicMock(__len__=lambda self: 0)

        mock_stream_instance = MagicMock()
        mock_stream_ctor.return_value.__enter__.return_value = mock_stream_instance

        # Mock default device
        mock_default.device = (0, 1)  # (input, output)
        mock_query_devices.return_value = {"name": "Test Output Device"}

        player = AudioPlayer()

        with patch("audio_player.glob.glob", return_value=["test.wav"]):
            with patch("builtins.print") as mock_print:
                player.play("test.wav", duration=1.0, quiet=False)
                player.play_q.join()

                # "Audio output" should have been printed
                output_printed = any(
                    "Audio output" in str(call) for call in mock_print.call_args_list
                )
                assert output_printed, "Expected 'Audio output' to be printed"

        # query_devices should have been called
        mock_query_devices.assert_called_once()
