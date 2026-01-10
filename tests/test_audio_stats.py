import unittest.mock
from audio_stats import AudioStats
from audio import Audio
from metadata import Metadata
from segment import Segment


def test_init():
    # Mock Audio object
    audio = unittest.mock.Mock(spec=Audio)
    audio.md = Metadata("test_unit", Segment(48000))
    # Simulate property access for noise floor and channel count.
    type(audio).avg_noise_floor = unittest.mock.PropertyMock(return_value=-60.0)
    type(audio).num_channels = unittest.mock.PropertyMock(return_value=2)

    stats = AudioStats(audio)

    # AudioStats calls super().__init__(audio.md) with shallow copy
    assert stats.md is not audio.md
    assert stats.md.pathname == audio.md.pathname

    assert stats.noise_floor == -60.0
    assert stats.num_channels == 2


def test_to_dict():
    audio = unittest.mock.Mock(spec=Audio)
    audio.md = Metadata("test_unit", Segment(48000))
    type(audio).avg_noise_floor = unittest.mock.PropertyMock(return_value=-80.5)
    type(audio).num_channels = unittest.mock.PropertyMock(return_value=2)

    stats = AudioStats(audio)
    d = stats.to_dict()

    assert d["Noise Floor"] == -80.5
    assert d["Num Channels"] == 2


def test_summary_valid_noise_floor():
    audio = unittest.mock.Mock(spec=Audio)
    # Verify metadata extraction from underscored filenames (Manufacturer_Model_Description).
    audio.md = Metadata("Unit123_Model_Description.wav", Segment(48000))

    # Mock avg_noise_floor (input 0.001 -> -60 dB)
    type(audio).avg_noise_floor = unittest.mock.PropertyMock(return_value=0.001)
    type(audio).num_channels = unittest.mock.PropertyMock(return_value=2)

    stats = AudioStats(audio)
    s = stats.summary()

    # Expected format: "Unit123 Description 2 -60.000 dBFS"
    assert "Unit123" in s
    assert "Description" in s
    assert "2" in s
    assert "-60.000 dBFS" in s


def test_summary_no_noise_floor():
    audio = unittest.mock.Mock(spec=Audio)
    audio.md = Metadata("Unit123_Model_Description.wav", Segment(48000))

    # If noise floor is 0.0
    type(audio).avg_noise_floor = unittest.mock.PropertyMock(return_value=0.0)
    type(audio).num_channels = unittest.mock.PropertyMock(return_value=1)

    stats = AudioStats(audio)
    s = stats.summary()

    assert "Not found" in s


def test_summary_header():
    h = AudioStats.summary_header()
    assert "Audio Statistics" in h
    assert "Unit" in h
    assert "Noise Floor" in h
