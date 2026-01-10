"""Tests for Segment class coverage."""

import pytest
from segment import Segment, Segments
from util import Category, Channel
from constants import DEFAULT_FS


def test_segment_eq_type_mismatch():
    """Test equality with non-Segment object."""
    s = Segment()
    assert s != "NotASegment"
    assert s != 123


def test_segments_merge_logic():
    """Test Segments.merge logic with contiguous, same-category segments."""
    # Create two segments list
    # List 1: Ends with incomplete segment
    s1 = Segment(fs=DEFAULT_FS, start=0, stop=100, cat=Category.Tone, complete=False)
    segs1 = Segments([s1])

    # List 2: Starts with segment that continues s1
    s2 = Segment(fs=DEFAULT_FS, start=100, stop=200, cat=Category.Tone, complete=True)
    segs2 = Segments([s2])

    # Merge
    segs1.merge(segs2)

    # Check result: s1 should be extended and marked complete
    assert len(segs1) == 1
    assert segs1[0].stop == 200
    assert segs1[0].complete is True

    # s2 was popped from segs2. The code pops from input list.
    # Note: list is mutable, passed by reference.
    # Logic: segments.pop(0)
    assert len(segs2) == 0


def test_segments_merge_no_match():
    """Test Segments.merge when segments do not merge."""
    # s1 complete
    s1 = Segment(fs=DEFAULT_FS, start=0, stop=100, cat=Category.Tone, complete=True)
    segs1 = Segments([s1])

    # s2 adjacent but s1 is complete
    s2 = Segment(fs=DEFAULT_FS, start=100, stop=200, cat=Category.Tone)
    segs2 = Segments([s2])

    segs1.merge(segs2)

    assert len(segs1) == 2
    assert segs1[0] == s1
    assert segs1[1] == s2
