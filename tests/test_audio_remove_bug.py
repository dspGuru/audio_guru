import numpy as np
import pytest
from audio import Audio
from segment import Segment


def test_remove_prior_updates_indices():
    """
    Test that Audio.remove with remove_prior=True correctly updates
    indices of subsequent segments.

    This reproduces the 'Segment stop must not be greater than audio length + 1'
    error seen when analyze.py processes multiple segments and removes processed ones.
    """
    fs = 1000.0
    # Create audio with enough length for multiple segments
    # 10 seconds of audio
    audio = Audio(fs=fs, name="Test Audio")
    audio.samples = np.zeros(int(10 * fs), dtype=np.float32)

    # Define 3 segments
    # Seg 1: 1s - 2s
    # Seg 2: 4s - 5s
    # Seg 3: 7s - 8s

    seg1 = Segment(fs, int(1.0 * fs), int(2.0 * fs))
    seg2 = Segment(fs, int(4.0 * fs), int(5.0 * fs))
    seg3 = Segment(fs, int(7.0 * fs), int(8.0 * fs))

    # Manually add to segments list
    audio.segments.append(seg1)
    audio.segments.append(seg2)
    audio.segments.append(seg3)

    expected_seg2_start = int(4.0 * fs)
    expected_seg2_stop = int(5.0 * fs)

    # 1. Select and remove seg1 with remove_prior=True
    # This should remove everything up to seg1.stop (2.0s) from the sample buffer.
    # The amount removed is seg1.stop = 2000 samples.
    # Current indices of seg2 are 4000-5000.
    # They should be shifted by 2000.
    # New valid indices for seg2 should be 2000-3000 relative to new buffer start.

    audio.select(seg1)
    audio.remove(seg1, remove_prior=True)

    # Check that seg2 is still valid within the new audio
    # New audio length should be 10000 - 2000 = 8000
    assert len(audio.samples) == 8000

    # Check seg2 indices
    current_seg2 = audio.segments[0]  # seg1 removed, seg2 is now first

    # The offset applied should be equal to the amount removed.
    # Amount removed = formerly seg1.stop (2000)
    # Original seg2.start = 4000. New seg2.start should be 2000.

    # Use validate to check if it raises the specific error
    # "Segment stop must not be greater than audio length + 1"
    try:
        audio.validate(current_seg2)
    except ValueError as e:
        pytest.fail(f"Validation failed for seg2: {e}")

    assert current_seg2.start == 2000
    assert current_seg2.stop == 3000
