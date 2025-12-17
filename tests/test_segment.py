from segment import Segment, Segments
from util import Category


def test_len_secs_and_slice_properties():
    s = Segment(fs=100.0, start=0, stop=101, id=0)  # stop is exclusive
    assert s.secs == 1.0
    # slice is inclusive of stop index in this implementation -> stop + 1
    assert s.slice == slice(0, 101)
    assert s.start_secs == 0.0
    assert s.stop_secs == 1.0


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
    short = Segment(fs=100.0, start=0, stop=51)  # len = 51 < fs
    short.trunc(1.0)
    assert short.stop == 51  # unchanged

    # longer segment: start=0 stop=300 -> len=300, fs=100 -> n_secs=100, multiple=3 => stop = 300
    long = Segment(fs=100.0, start=0, stop=301)
    long.trunc(1.0)
    assert long.stop == 301


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
