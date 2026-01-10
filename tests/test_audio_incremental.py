import numpy as np
import pytest
from audio import Audio, Segment, Category
from util import Channel
import generate


def test_incremental_segmentation():
    """Verify that get_segments updates self.segments incrementally."""
    fs = 1000
    # 1. Create initial audio: DC Tone (0.1s)
    # Set min_segment_secs=0 to avoid filtering out short initial segment
    a1 = Audio(fs=fs, min_segment_secs=0.0)
    a1.samples = 0.5 * np.ones(100, dtype=np.float32)
    # Ensure it's active
    a1.select()

    # Run segmentation
    segs = a1.get_segments(min_silence_secs=0.01)
    assert len(segs) == 1
    assert segs[0].start == 0
    assert segs[0].stop == 100

    # Store ID of first segment
    id1 = segs[0].id

    # 2. Append Silence (0.2s) - long enough to be detected (min_silence=0.1s default)
    s1 = generate.silence(secs=0.2, fs=fs, channel=Channel.Left)
    a1.append(s1)

    # 3. Append Tone (0.1s)
    a2 = Audio(fs=fs)
    a2.samples = 0.5 * np.ones(100, dtype=np.float32)
    a2.select()
    a1.append(a2)

    # Total samples: 100 (Tone) + 200 (Silence) + 100 (Tone) = 400
    assert len(a1.samples) == 400

    # 4. Run segmentation again. logic should handle the existing segment + new data
    # The first segment might be re-evaluated if we pop it off.
    # Note: If logic re-evaluates from start of last segment, the first segment (index 0)
    # is the "last" segment if len=1.
    segs_updated = a1.get_segments(min_silence_secs=0.01)

    assert len(segs_updated) == 2
    assert segs_updated[0].start == 0
    assert segs_updated[0].stop == 100

    # Second segment
    assert segs_updated[1].start == 300
    assert segs_updated[1].stop == 400

    # 5. Incremental Check: Append more data that merges with last segment
    # Append another Tone (0.1s) immediately
    a3 = Audio(fs=fs)
    a3.samples = 0.5 * np.ones(100, dtype=np.float32)
    a3.select()
    a1.append(a3)
    # Total active: 200 + 100 = 300.

    segs_merged = a1.get_segments(min_silence_secs=0.01)
    assert len(segs_merged) == 2
    assert segs_merged[1].start == 300
    assert segs_merged[1].stop == 500
