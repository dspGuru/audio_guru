"""Tests for select.py CLI module."""

import sys
import unittest.mock

import pytest


def test_select_main_list_mode():
    """Test select.py in list-only mode."""
    from device_selector import main

    # Mock AudioSelect
    mock_audio_select = unittest.mock.Mock()
    mock_audio_select.devices = [
        {
            "id": 0,
            "name": "Test Device",
            "api_name": "Test API",
            "category": "Speakers",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 44100,
        }
    ]
    mock_audio_select.__str__ = unittest.mock.Mock(return_value="Mock Device List")

    with unittest.mock.patch(
        "device_selector.AudioSelect", return_value=mock_audio_select
    ):
        with unittest.mock.patch.object(sys, "argv", ["device_selector.py", "--list"]):
            # Should exit normally after listing
            main()

    mock_audio_select.sort.assert_called_once_with(["category", "name", "api_name"])


def test_select_main_quit_input():
    """Test select.py when user types 'q' to quit."""
    from device_selector import main

    mock_audio_select = unittest.mock.Mock()
    mock_audio_select.devices = []
    mock_audio_select.__str__ = unittest.mock.Mock(return_value="Mock Device List")

    with unittest.mock.patch(
        "device_selector.AudioSelect", return_value=mock_audio_select
    ):
        with unittest.mock.patch.object(sys, "argv", ["device_selector.py"]):
            with unittest.mock.patch("builtins.input", return_value="q"):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0


def test_select_main_valid_device_selection():
    """Test select.py with valid device ID selection."""
    from device_selector import main

    mock_audio_select = unittest.mock.Mock()
    mock_audio_select.devices = [
        {
            "id": 0,
            "name": "Test Device",
            "api_name": "Test API",
            "category": "Speakers",
            "max_input_channels": 2,
            "max_output_channels": 2,
            "default_samplerate": 48000,
        }
    ]
    mock_audio_select.__str__ = unittest.mock.Mock(return_value="Mock Device List")

    with unittest.mock.patch(
        "device_selector.AudioSelect", return_value=mock_audio_select
    ):
        with unittest.mock.patch.object(sys, "argv", ["device_selector.py"]):
            with unittest.mock.patch("builtins.input", return_value="0"):
                # Should complete after selecting device
                main()


def test_select_main_invalid_device_id():
    """Test select.py with invalid device ID then quit."""
    from device_selector import main

    mock_audio_select = unittest.mock.Mock()
    mock_audio_select.devices = [
        {
            "id": 0,
            "name": "Test Device",
            "api_name": "Test API",
            "category": "Speakers",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 44100,
        }
    ]
    mock_audio_select.__str__ = unittest.mock.Mock(return_value="Mock Device List")

    # First return invalid ID (99), then quit
    input_values = iter(["99", "q"])

    with unittest.mock.patch(
        "device_selector.AudioSelect", return_value=mock_audio_select
    ):
        with unittest.mock.patch.object(sys, "argv", ["device_selector.py"]):
            with unittest.mock.patch(
                "builtins.input", side_effect=lambda _: next(input_values)
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0


def test_select_main_invalid_input():
    """Test select.py with non-numeric input then quit."""
    from device_selector import main

    mock_audio_select = unittest.mock.Mock()
    mock_audio_select.devices = []
    mock_audio_select.__str__ = unittest.mock.Mock(return_value="Mock Device List")

    # First return invalid input, then quit
    input_values = iter(["abc", "q"])

    with unittest.mock.patch(
        "device_selector.AudioSelect", return_value=mock_audio_select
    ):
        with unittest.mock.patch.object(sys, "argv", ["device_selector.py"]):
            with unittest.mock.patch(
                "builtins.input", side_effect=lambda _: next(input_values)
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0


def test_select_main_keyboard_interrupt():
    """Test select.py with keyboard interrupt."""
    from device_selector import main

    mock_audio_select = unittest.mock.Mock()
    mock_audio_select.devices = []
    mock_audio_select.__str__ = unittest.mock.Mock(return_value="Mock Device List")

    with unittest.mock.patch(
        "device_selector.AudioSelect", return_value=mock_audio_select
    ):
        with unittest.mock.patch.object(sys, "argv", ["device_selector.py"]):
            with unittest.mock.patch("builtins.input", side_effect=KeyboardInterrupt):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
