"""Tests for audio_player.py module."""

import queue
import threading
import time
import unittest.mock

import numpy as np
import pytest


def test_audio_player_init():
    """Test AudioPlayer initialization."""
    from audio_player import AudioPlayer

    with unittest.mock.patch("audio_player.sd"):
        with unittest.mock.patch("audio_player.sf"):
            player = AudioPlayer()
            assert player.msg_q is None
            assert isinstance(player.play_q, queue.Queue)
            assert isinstance(player.stop_event, threading.Event)
            assert player.thread.is_alive()
            player.stop()


def test_audio_player_init_with_msg_queue():
    """Test AudioPlayer initialization with message queue."""
    from audio_player import AudioPlayer

    msg_q = queue.Queue()

    with unittest.mock.patch("audio_player.sd"):
        with unittest.mock.patch("audio_player.sf"):
            player = AudioPlayer(msg_q=msg_q)
            assert player.msg_q is msg_q
            player.stop()


def test_audio_player_stop():
    """Test AudioPlayer stop method clears queue."""
    from audio_player import AudioPlayer

    with unittest.mock.patch("audio_player.sd"):
        with unittest.mock.patch("audio_player.sf"):
            player = AudioPlayer()

            # Add items to queue
            player.play_q.put(("file1.wav", 0.0))
            player.play_q.put(("file2.wav", 0.0))

            player.stop()

            assert player.stop_event.is_set()
            # Queue should be cleared
            assert player.play_q.empty()


def test_audio_player_play_no_files(tmp_path):
    """Test AudioPlayer.play with no matching files."""
    from audio_player import AudioPlayer

    with unittest.mock.patch("audio_player.sd"):
        with unittest.mock.patch("audio_player.sf"):
            player = AudioPlayer()

            # Use a pattern that won't match any files
            player.play(str(tmp_path / "nonexistent_*.wav"))

            player.stop()


def test_audio_player_play_with_files(tmp_path, examples_dir):
    """Test AudioPlayer.play with actual files."""
    from audio_player import AudioPlayer

    # Create a mock sound file context manager
    mock_soundfile = unittest.mock.MagicMock()
    mock_soundfile.__enter__ = unittest.mock.Mock(return_value=mock_soundfile)
    mock_soundfile.__exit__ = unittest.mock.Mock(return_value=False)
    mock_soundfile.samplerate = 44100
    mock_soundfile.channels = 2
    # Return data once, then empty to signal end
    mock_soundfile.read = unittest.mock.Mock(side_effect=[
        np.zeros((1024, 2), dtype=np.float32),
        np.array([], dtype=np.float32)
    ])

    # Create mock output stream
    mock_stream = unittest.mock.MagicMock()
    mock_stream.__enter__ = unittest.mock.Mock(return_value=mock_stream)
    mock_stream.__exit__ = unittest.mock.Mock(return_value=False)

    mock_sd = unittest.mock.MagicMock()
    mock_sd.OutputStream.return_value = mock_stream
    mock_sd.default.device = (0, 1)
    mock_sd.query_devices.return_value = {"name": "Test Output"}

    with unittest.mock.patch("audio_player.sd", mock_sd):
        with unittest.mock.patch("audio_player.sf.SoundFile", return_value=mock_soundfile):
            player = AudioPlayer()

            # Find an existing example file
            example_files = list(examples_dir.glob("*.wav"))
            if example_files:
                player.play(str(example_files[0]), duration=0.1, quiet=False)

            player.stop()


def test_audio_player_play_quiet_mode(examples_dir):
    """Test AudioPlayer.play in quiet mode."""
    from audio_player import AudioPlayer

    mock_soundfile = unittest.mock.MagicMock()
    mock_soundfile.__enter__ = unittest.mock.Mock(return_value=mock_soundfile)
    mock_soundfile.__exit__ = unittest.mock.Mock(return_value=False)
    mock_soundfile.samplerate = 44100
    mock_soundfile.channels = 2
    mock_soundfile.read = unittest.mock.Mock(return_value=np.array([], dtype=np.float32))

    mock_stream = unittest.mock.MagicMock()
    mock_stream.__enter__ = unittest.mock.Mock(return_value=mock_stream)
    mock_stream.__exit__ = unittest.mock.Mock(return_value=False)

    mock_sd = unittest.mock.MagicMock()
    mock_sd.OutputStream.return_value = mock_stream

    with unittest.mock.patch("audio_player.sd", mock_sd):
        with unittest.mock.patch("audio_player.sf.SoundFile", return_value=mock_soundfile):
            player = AudioPlayer()

            example_files = list(examples_dir.glob("*.wav"))
            if example_files:
                player.play(str(example_files[0]), quiet=True)

            # query_devices should not be called in quiet mode
            mock_sd.query_devices.assert_not_called()

            player.stop()


def test_audio_player_worker_stop_event():
    """Test that worker respects stop event."""
    from audio_player import AudioPlayer

    with unittest.mock.patch("audio_player.sd"):
        with unittest.mock.patch("audio_player.sf"):
            player = AudioPlayer()

            # Set stop event before adding item
            player.stop_event.set()

            # Add item to queue
            player.play_q.put(("test.wav", 0.0))

            # Wait for worker to process
            player.play_q.join()

            # Worker should have skipped playback
            player.stop()


def test_audio_player_worker_exception():
    """Test worker handles exceptions gracefully."""
    from audio_player import AudioPlayer

    mock_sf = unittest.mock.MagicMock()
    mock_sf.SoundFile.side_effect = Exception("Test error")

    with unittest.mock.patch("audio_player.sd"):
        with unittest.mock.patch("audio_player.sf", mock_sf):
            player = AudioPlayer()

            # This should trigger the exception handler
            player.play_q.put(("nonexistent.wav", 1.0))
            player.play_q.join()

            # Player should have stopped due to error
            assert player.stop_event.is_set()
            player.stop()


def test_audio_player_play_file_or_loop_with_msg_queue():
    """Test _play_file_or_loop sends message to queue."""
    from audio_player import AudioPlayer

    msg_q = queue.Queue()

    mock_soundfile = unittest.mock.MagicMock()
    mock_soundfile.__enter__ = unittest.mock.Mock(return_value=mock_soundfile)
    mock_soundfile.__exit__ = unittest.mock.Mock(return_value=False)
    mock_soundfile.samplerate = 44100
    mock_soundfile.channels = 1
    mock_soundfile.read = unittest.mock.Mock(return_value=np.array([], dtype=np.float32))

    mock_stream = unittest.mock.MagicMock()
    mock_stream.__enter__ = unittest.mock.Mock(return_value=mock_stream)
    mock_stream.__exit__ = unittest.mock.Mock(return_value=False)

    mock_sd = unittest.mock.MagicMock()
    mock_sd.OutputStream.return_value = mock_stream
    mock_sd.default.device = (0, 1)
    mock_sd.query_devices.return_value = {"name": "Test"}

    with unittest.mock.patch("audio_player.sd", mock_sd):
        with unittest.mock.patch("audio_player.sf.SoundFile", return_value=mock_soundfile):
            player = AudioPlayer(msg_q=msg_q)

            player._play_file_or_loop("test.wav", 1.0)

            # Check message was sent
            assert not msg_q.empty()
            assert msg_q.get() == "test.wav"

            player.stop()


def test_audio_player_data_type_conversion():
    """Test that non-float32 data is converted."""
    from audio_player import AudioPlayer

    mock_soundfile = unittest.mock.MagicMock()
    mock_soundfile.__enter__ = unittest.mock.Mock(return_value=mock_soundfile)
    mock_soundfile.__exit__ = unittest.mock.Mock(return_value=False)
    mock_soundfile.samplerate = 44100
    mock_soundfile.channels = 1

    # Return float64 data first, then empty
    float64_data = np.zeros(1024, dtype=np.float64)
    mock_soundfile.read = unittest.mock.Mock(side_effect=[
        float64_data,
        np.array([], dtype=np.float32)
    ])

    mock_stream = unittest.mock.MagicMock()
    mock_stream.__enter__ = unittest.mock.Mock(return_value=mock_stream)
    mock_stream.__exit__ = unittest.mock.Mock(return_value=False)

    mock_sd = unittest.mock.MagicMock()
    mock_sd.OutputStream.return_value = mock_stream

    with unittest.mock.patch("audio_player.sd", mock_sd):
        with unittest.mock.patch("audio_player.sf.SoundFile", return_value=mock_soundfile):
            player = AudioPlayer()

            player._play_file_or_loop("test.wav", 1.0)

            # Verify write was called with float32 data
            write_calls = mock_stream.write.call_args_list
            if write_calls:
                written_data = write_calls[0][0][0]
                assert written_data.dtype == np.float32

            player.stop()
