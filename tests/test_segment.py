from segment import Segment, Segments
from util import Category


def test_len_secs_and_slice_properties():
    s = Segment(fs=100.0, start=0, stop=100, id=0)  # Stop is exclusive
    assert s.secs == 1.0
    # Slice is inclusive of stop index
    assert s.slice == slice(0, 100)
    assert s.start_secs == 0.0
    assert s.stop_secs == 1.0


def test_title_str_and_desc_formatting():
    s = Segment(fs=100.0, start=0, stop=99, id=0)  # Default Category.Unknown
    # Title uses rounded start/stop seconds
    assert s.title == f"{s.cat.name}_{round(s.start_secs)}-{round(s.stop_secs)}s"
    # __str__ and desc include formatted seconds and category name
    assert "0:" in str(s)
    assert s.desc.endswith(f" {s.cat.name}")
    assert s.desc == f"Segment 0: {s.start_secs:0.1f}-{s.stop_secs:0.1f}s {s.cat.name}"


def test_limit_applies_max_seconds():
    s = Segment(fs=100.0, start=0, stop=199)
    # limit to 0.5s => stop = start + 50
    s.limit(0.5)
    assert s.stop == 50
    # Limit 0.0 should leave segment unchanged
    prev = Segment(fs=100.0, start=0, stop=50)
    prev.limit(0.0)
    assert prev.stop == 50


def test_trunc_behavior_and_noop_for_short_segments():
    # trunc only acts when length >= fs
    short = Segment(fs=100.0, start=0, stop=51)  # len = 51 < fs
    short.trunc(1.0)
    assert short.stop == 51  # Unchanged

    # len=301, fs=100 -> trunc to len=300
    long = Segment(fs=100.0, start=0, stop=301)
    long.trunc(1.0)
    assert long.stop == 300


def test_segments_str_lists_all_segments():
    a = Segment(fs=100.0, start=0, stop=101, id=1, cat=Category.Unknown)
    b = Segment(fs=100.0, start=100, stop=201, id=2, cat=Category.Unknown)
    segs = Segments([a, b])
    s = str(segs)
    assert "1:" in s
    assert "2:" in s
    lines = [line for line in s.splitlines() if line.strip()]
    assert len(lines) == 2


def test_short_segment_secs():
    s = Segment(fs=100.0, start=0, stop=1)
    # len = 1-0 = 1
    assert len(s) == 1
    assert s.secs == 0.01

    # 2 samples
    s2 = Segment(fs=100.0, start=0, stop=2)
    # len = 2-0 = 2
    assert len(s2) == 2
    assert s2.secs == 0.02


def test_segments_combine():
    # Segments: [Cat1, Cat1, Cat2, Cat2, Cat1]
    # Should combine to: [Cat1, Cat2, Cat1]
    fs = 100.0
    s1 = Segment(fs=fs, start=0, stop=100, cat=Category.Tone)
    s2 = Segment(fs=fs, start=100, stop=200, cat=Category.Tone)  # Adjacent
    s3 = Segment(fs=fs, start=200, stop=300, cat=Category.Noise)
    s4 = Segment(fs=fs, start=300, stop=400, cat=Category.Noise)
    s5 = Segment(fs=fs, start=400, stop=500, cat=Category.Tone)

    segs = Segments([s1, s2, s3, s4, s5])
    segs.combine()

    assert len(segs) == 3
    # Combined segment 1
    assert segs[0].cat == Category.Tone
    assert segs[0].start == 0
    assert segs[0].stop == 200

    # Combined segment 2
    assert segs[1].cat == Category.Noise
    assert segs[1].start == 200
    assert segs[1].stop == 400

    # Combined segment 3 (unchanged)
    assert segs[2].cat == Category.Tone
    assert segs[2].start == 400
    assert segs[2].stop == 500


def test_segments_trunc():
    # Verify trunc calls trunc on elements
    fs = 100.0
    # s1: 1.5s -> trunc to 1.0s
    s1 = Segment(fs=fs, start=0, stop=150)

    # s2: 0.5s -> no change (len=50 < fs)
    s2 = Segment(fs=fs, start=151, stop=201)

    segs = Segments([s1, s2])
    segs.trunc(1.0)

    # s1 should be truncated
    assert s1.secs == 1.0
    assert s1.stop == 100

    # s2 should be unchanged
    assert s2.secs == 0.5
    assert s2.stop == 201
