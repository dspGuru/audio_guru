import pytest
import numpy as np
from time_stats import TimeStats, TimeStatsList
from audio import Audio
from segment import Segment


class TestTimeStats:
    def test_timestats_initialization(self):
        # Create audio with known properties
        audio = Audio(fs=1000.0, name="TestAudio")

        # Audio default is empty, let's create a segment
        # Create DC signal: all 1.0
        audio.samples = np.ones(1000, dtype=np.float32)
        # We must select the segment so Audio properties work
        audio.select(Segment(1000.0, 0, 1000))  # 1 second

        # In Audio class:
        # dc property calculates mean
        # rms calculates norm/sqrt(size)
        # max/min are from selected samples

        stats = TimeStats(audio)

        # Verify stats properties
        assert stats.dc == pytest.approx(1.0)
        assert stats.max == pytest.approx(1.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.rms == pytest.approx(1.0)
        assert stats.secs == pytest.approx(1.0)
        assert stats.start == 0.0

        # Crest factor: max / rms
        assert stats.crest == pytest.approx(1.0)

        # dbfs: 20*log10(max) -> 20*log10(1) = 0
        assert stats.dbfs == 0.0

    def test_timestats_str(self):
        # TimeStats reads properties from Audio, so we need to setup Audio properly
        audio = Audio(fs=1000.0)
        audio.samples = np.zeros(1000, dtype=np.float32)
        audio.select(Segment(1000.0, 0, 999))

        stats = TimeStats(audio)
        s = str(stats)

        assert "Time Stats" in s
        assert "Fs    =" in s
        assert "RMS   =" in s
        assert "Crest =" in s

    def test_to_dict(self):
        audio = Audio(fs=1000.0)
        audio.samples = np.zeros(1000, dtype=np.float32)
        audio.select(Segment(1000.0, 0, 999))

        stats = TimeStats(audio)
        d = stats.to_dict()

        assert "Crest" in d
        assert "Secs" in d

    def test_summary(self):
        audio = Audio(fs=1000.0)
        audio.samples = np.zeros(1000, dtype=np.float32)
        audio.select(Segment(1000.0, 0, 999))

        stats = TimeStats(audio)
        s = stats.summary()

        # Check for duration and values (zeros)
        assert "1.000s" in s  # secs
        assert "0.0000" in s  # rms/dc/crest

    def test_timestats_list_print(self, capsys):
        # Create two segments with different start times
        s1 = Segment(1000.0, 0, 999)  # start 0.0
        s2 = Segment(1000.0, 1000, 1999)  # start 1.0

        # Audio 1
        a1 = Audio(fs=1000.0)
        a1.samples = np.zeros(2000, dtype=np.float32)  # Enough samples
        a1.select(s1)
        ts1 = TimeStats(a1)

        # Audio 2
        a2 = Audio(fs=1000.0)
        a2.samples = np.zeros(2000, dtype=np.float32)  # Enough samples
        a2.select(s2)
        ts2 = TimeStats(a2)

        ts_list = TimeStatsList([ts2, ts1])  # unsorted

        ts_list.print(sort_attr="start")

        captured = capsys.readouterr()

        # Verify output contains both
        assert "Start = 0.000s" in captured.out
        assert "Start = 1.000s" in captured.out

        # Verify sorting: start=0 should appear before start=1
        idx1 = captured.out.find("Start = 0.000s")
        idx2 = captured.out.find("Start = 1.000s")
        assert idx1 != -1
        assert idx2 != -1
        assert idx1 < idx2
