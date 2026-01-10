import unittest
from unittest.mock import patch
from audio_select import AudioSelect


class TestAudioSelect(unittest.TestCase):

    def setUp(self):
        # Mock data for sounddevice
        self.mock_devices = [
            {
                "name": "Microphone (USB)",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "Speakers (Realtek)",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {
                "name": "MONITOR (NVIDIA)",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {
                "name": "Line In",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
        ]
        self.mock_hostapis = [
            {"name": "MME"},
            {"name": "DirectSound"},
        ]

    @patch("audio_select.sd.query_devices")
    @patch("audio_select.sd.query_hostapis")
    def test_init_and_categorization(self, mock_hostapis, mock_devices):
        mock_devices.return_value = self.mock_devices
        mock_hostapis.return_value = self.mock_hostapis

        as_sel = AudioSelect()

        # Check enrichment
        self.assertEqual(len(as_sel.devices), 4)
        self.assertEqual(as_sel.devices[0]["category"], "Microphone")
        self.assertEqual(as_sel.devices[1]["category"], "Speakers")
        self.assertEqual(as_sel.devices[2]["category"], "Monitor")
        self.assertEqual(as_sel.devices[3]["category"], "Line In")

        self.assertEqual(as_sel.devices[0]["api_name"], "MME")
        self.assertEqual(as_sel.devices[2]["api_name"], "DirectSound")

    @patch("audio_select.sd.query_devices")
    @patch("audio_select.sd.query_hostapis")
    def test_sort_by_category(self, mock_hostapis, mock_devices):
        mock_devices.return_value = self.mock_devices
        mock_hostapis.return_value = self.mock_hostapis

        as_sel = AudioSelect()

        # Default sort (category, name, api_name)
        # Categories: Microphone, Speakers, Monitor, Line In
        # Alphabetical: Line In, Microphone, Monitor, Speakers
        as_sel.sort()

        self.assertEqual(as_sel.devices[0]["category"], "Line In")
        self.assertEqual(as_sel.devices[1]["category"], "Microphone")
        self.assertEqual(as_sel.devices[2]["category"], "Monitor")
        self.assertEqual(as_sel.devices[3]["category"], "Speakers")

    @patch("audio_select.sd.query_devices")
    @patch("audio_select.sd.query_hostapis")
    def test_sort_by_api(self, mock_hostapis, mock_devices):
        mock_devices.return_value = self.mock_devices
        mock_hostapis.return_value = self.mock_hostapis

        as_sel = AudioSelect()

        # Sort by API name then name
        as_sel.sort(by=["api_name", "name"])

        # DirectSound devices first (alphabetical)
        self.assertEqual(as_sel.devices[0]["api_name"], "DirectSound")
        self.assertEqual(as_sel.devices[1]["api_name"], "DirectSound")
        # MME is after DirectSound
        self.assertEqual(as_sel.devices[2]["api_name"], "MME")

    @patch("audio_select.sd.query_devices")
    @patch("audio_select.sd.query_hostapis")
    def test_repr(self, mock_hostapis, mock_devices):
        mock_devices.return_value = self.mock_devices
        mock_hostapis.return_value = self.mock_hostapis

        as_sel = AudioSelect()
        r = repr(as_sel)
        self.assertIn("Category", r)
        self.assertIn("Microphone", r)
        self.assertIn("MME", r)
