import unittest.mock
from audio_stats import AudioStats
from audio import Audio
from metadata import Metadata
from segment import Segment


def test_init():
    # Mock Audio object
    audio = unittest.mock.Mock(spec=Audio)
    audio.md = Metadata("test_unit", Segment(48000))
    audio.get_noise_floor.return_value = (-60.0, 1000)  # -60 dB, sample 1000
    audio.num_channels.return_value = 2

    stats = AudioStats(audio)

    # AudioStats calls super().__init__(audio.md) which does a shallow copy
    assert stats.md is not audio.md
    assert stats.md.pathname == audio.md.pathname

    assert stats.noise_floor == -60.0
    assert stats.noise_floor_idx == 1000

    # audio.num_channels is a method, not property, creating Mock(spec=Audio) ensures it mocks the method.
    # But wait, does audio.num_channels() need to be called?
    # Source: self.num_channels = audio.num_channels()
    # So it calls it. Mock.return_value handles it.
    assert stats.num_channels == 2


def test_noise_floor_secs():
    audio = unittest.mock.Mock(spec=Audio)
    # fs = 48000. idx = 48000 -> 1.0 sec.
    audio.md = Metadata("test_unit", Segment(48000))
    audio.get_noise_floor.return_value = (-60.0, 24000)
    audio.num_channels.return_value = 1

    stats = AudioStats(audio)
    assert stats.noise_floor_secs == 0.5


def test_to_dict():
    audio = unittest.mock.Mock(spec=Audio)
    audio.md = Metadata("test_unit", Segment(48000))
    audio.get_noise_floor.return_value = (-80.5, 4800)  # 0.1s
    audio.num_channels.return_value = 2

    stats = AudioStats(audio)
    d = stats.to_dict()

    assert d["Noise Floor"] == -80.5
    assert d["Noise Floor Secs"] == 0.1
    assert d["Num Channels"] == 2


def test_summary_valid_noise_floor():
    audio = unittest.mock.Mock(spec=Audio)
    # Metadata splits string by underscores.
    # "Unit123_Model_Description" -> mfr="Unit123", model="Model", desc="Description"
    audio.md = Metadata("Unit123_Model_Description.wav", Segment(48000))

    # Audio.get_noise_floor return convention:
    # If using dbv later, checks usually imply it returns value in DB or linear?
    # "dbv(self.noise_floor)" implies noise_floor is linear.
    # 20 * log10(x).
    # If we want -60dB, input should be 0.001.
    audio.get_noise_floor.return_value = (0.001, 48000)  # 1.0s. 0.001 -> -60dBV
    audio.num_channels.return_value = 2

    stats = AudioStats(audio)
    s = stats.summary()

    # "Unit123             Description        2           -60.000 dBFS at   1.0s"
    assert "Unit123" in s
    assert "Description" in s
    assert "2" in s
    assert "-60.000 dBFS" in s
    assert "1.0s" in s


def test_summary_no_noise_floor():
    audio = unittest.mock.Mock(spec=Audio)
    audio.md = Metadata("Unit123_Model_Description.wav", Segment(48000))

    # If noise floor is 0.0
    audio.get_noise_floor.return_value = (0.0, 0)
    audio.num_channels.return_value = 1

    stats = AudioStats(audio)
    s = stats.summary()

    assert "Not found" in s


def test_summary_header():
    h = AudioStats.summary_header()
    assert "Audio Statistics" in h
    assert "Unit" in h
    assert "Noise Floor" in h
