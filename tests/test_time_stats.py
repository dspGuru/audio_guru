import pytest
import numpy as np
from time_stats import TimeStats
from audio import Audio
from segment import Segment, Segments
from util import Category


class TestTimeStats:
    def test_timestats_initialization(self):
        # Create audio with known properties
        audio = Audio(fs=1000.0, name="TestAudio")

        # Audio default is empty, let's create a segment
        # Create DC signal: all 1.0
        audio.samples = np.ones(1000, dtype=np.float32)

        # Select the segment so Audio properties work
        audio.select(Segment(1000.0, 0, 1000))  # 1 second

        # In Audio class:
        # dc property calculates mean
        # rms calculates norm/sqrt(size)
        # max/min are from selected samples

        stats = TimeStats(audio)

        # Verify stats properties
        assert stats.dc == 1.0
        assert stats.max == 1.0
        assert stats.min == 1.0
        assert stats.rms == pytest.approx(1.0)
        assert stats.secs == 1.0
        assert stats.start == 0.0

        # Crest factor: max / rms
        assert stats.crest == pytest.approx(1.0)

        # dbfs: 20*log10(max) -> 20*log10(1) = 0
        assert stats.dbfs == 0.0

        # noise_floor_secs property
        assert stats.noise_floor_secs == pytest.approx(0.0)

    def test_timestats_str(self):
        # TimeStats reads properties from Audio, so we need to setup Audio properly
        audio = Audio(fs=1000.0)
        audio.samples = np.zeros(1000, dtype=np.float32)
        audio.select(Segment(1000.0, 0, 999))

        stats = TimeStats(audio)
        s = str(stats)

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
        assert "All:  0.0-1.0s" in s  # secs
        assert "0.000" in s  # rms/dc/crest


def test_len_secs_and_slice_properties():
    s = Segment(fs=100.0, start=0, stop=100, id=0)
    assert len(s) == 100
    assert s.secs == pytest.approx(1.0)
    # slice is inclusive of stop index in this implementation -> stop
    assert s.slice == slice(0, 100)
    assert s.start_secs == pytest.approx(0.0)
    assert s.stop_secs == pytest.approx(1.0)


def test_title_str_and_desc_formatting():
    s = Segment(fs=100.0, start=0, stop=99, id=0)  # default Category.Unknown
    # title uses rounded start/stop seconds
    assert s.title == f"{s.cat.name}_{round(s.start_secs)}-{round(s.stop_secs)}s"
    # __str__ and desc include formatted seconds and category name
    assert "0:" in str(s)
    assert s.desc.endswith(f" {s.cat.name}")
    # exact desc formatting expectation
    assert s.desc == f"Segment 0: {s.start_secs:0.1f}-{s.stop_secs:0.1f}s {s.cat.name}"


def test_limit_applies_max_seconds():
    s = Segment(fs=100.0, start=0, stop=199)
    # limit to 0.5 seconds => trunc(0.5 * 100) = 50 => stop becomes start + 50
    s.limit(0.5)
    assert s.stop == 50
    # limit 0.0 should leave segment unchanged
    prev = Segment(fs=100.0, start=0, stop=50)
    prev.limit(0.0)
    assert prev.stop == 50


def test_trunc_behavior_and_noop_for_short_segments():
    # trunc only acts when length >= fs
    short = Segment(fs=100.0, start=0, stop=50)  # len = 50 < fs
    short.trunc(1.0)
    assert short.stop == 50  # unchanged

    # longer segment: use stop=300 so length=300, fs=100 -> n_secs=100, multiple=3 => stop = 300
    long = Segment(fs=100.0, start=0, stop=300)
    long.trunc(1.0)
    assert long.stop == 300


def test_segments_str_lists_all_segments():
    a = Segment(fs=100.0, start=0, stop=101, id=1, cat=Category.Unknown)
    b = Segment(fs=100.0, start=100, stop=201, id=2, cat=Category.Unknown)
    segs = Segments([a, b])
    s = str(segs)
    assert "1:" in s
    assert "2:" in s
    # two lines expected
    lines = [line for line in s.splitlines() if line.strip()]
    assert len(lines) == 2


def test_segments_str_empty_and_all_id_none():
    # empty segments -> empty string
    segs = Segments([])
    assert str(segs) == ""

    # id None -> "All" prefix in representation
    s = Segment(fs=100.0, start=0, stop=99, id=None, cat=Category.Unknown)
    assert "All:" in str(s)
