import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from audio import Audio


def test_num_channels_1d():
    """Verify that a 1D sample buffer is correctly identified as mono (1 channel)."""
    a = Audio()

    # Force samples to be 1D explicitly logic check
    a.samples = np.zeros(100, dtype=np.float32)
    assert a.num_channels == 1


def test_filter_edge_cases():
    """Ensure that filtering with None or an empty list does not crash and
    behaves as a no-op."""
    a = Audio()

    # Filter with None
    a.filter(None)

    # Filter with empty list
    a.filter([])


def test_copy_defaults():
    """Test copy method defaults."""
    a = Audio()
    a.samples = np.array([1, 2, 3], dtype=a.DTYPE)
    # Copy with no args -> copies entire segment
    b = a.copy()

    assert len(b.samples) == 3
    assert np.all(b.samples == a.samples)
    assert b.fs == a.fs
    assert b.name == a.name


def test_remove_unselected():
    """Test remove_unselected method."""
    a = Audio(fs=1000)
    # 0.5s audio
    a.samples = np.zeros(500, dtype=a.DTYPE)

    # Select middle 0.1s: 200-300
    from segment import Segment

    seg = Segment(fs=1000, start=200, stop=300)
    a.segments.append(seg)
    a.select(seg)

    a.remove_unselected()

    # Should only have 100 samples left
    assert len(a.samples) == 100
    assert a.segment.start == 0
    assert a.segment.stop == 100


def test_read_block_overflow(capsys):
    a = Audio()
    # Mock open_device internals to setup a stream with overflow return
    with patch("sounddevice.InputStream") as mock_stream_cls:
        mock_stream = mock_stream_cls.return_value
        mock_stream.active = True
        mock_stream.channels = 2
        # Return data and overflow=True
        mock_stream.read.return_value = (np.zeros((100, 2), dtype=np.float32), True)

        a.open_device()

        # read_block will use the active stream
        assert a.read_block() is True

        assert "Audio input overflowed" in capsys.readouterr().out


def test_write_parent_mkdir(tmp_path):
    a = Audio()
    a.samples = np.zeros(100, dtype=np.float32)
    a.select()

    # Write to nested directory to force mkdir
    nested = tmp_path / "deep" / "dir" / "file.wav"
    assert not nested.parent.exists()

    # Mock soundfile.write so we don't actually write but verify logic reaches it
    with patch("soundfile.write"):
        a.write(str(nested))
        assert nested.parent.exists()


def test_get_segments_incremental_ids():
    # Construct audio with silence gaps for multiple segments
    fs = 1000
    # Set min_segment_secs=0 to allow short segments
    a = Audio(fs=fs, min_segment_secs=0.0)

    # Helper
    def tone():
        return np.sin(2 * np.pi * 100 * np.arange(200) / fs).astype(np.float32)

    def silence():
        return np.zeros(200, dtype=np.float32)

    # 1. First batch: Tone + Silence + Tone + Silence + Tone
    a.append(tone())  # Seg 1
    a.append(silence())  # Gap
    a.append(tone())  # Seg 2
    a.append(silence())  # Gap
    a.append(tone())  # Seg 3 (Last)

    # Run detection
    segs = a.get_segments(min_silence_secs=0.01)

    # Detection should find 3 segments
    assert len(segs) == 3
    ids = [s.id for s in segs]
    assert ids == [0, 1, 2]

    # 2. Second batch - append to extend last segment
    a.append(tone())

    # Call get_segments again
    segs = a.get_segments(min_silence_secs=0.01)

    # Verify 3 segments with sequential IDs
    assert len(segs) == 3
    # Verify IDs are sequential
    assert segs[0].id == 0
    assert segs[1].id == 1
    assert segs[2].id == 2


def test_remove_segment():
    # Setup audio: [Tone1 (100)] [Tone2 (100)] [Tone3 (100)]
    fs = 1000
    a = Audio(fs=fs)
    ones = np.ones(100, dtype=np.float32)

    a.append(ones * 1)  # Seg 1: 0-100
    a.append(ones * 2)  # Seg 2: 100-200
    a.append(ones * 3)  # Seg 3: 200-300

    # Manually create segments to test remove logic
    from segment import Segment

    s1 = Segment(fs, 0, 100)
    s2 = Segment(fs, 100, 200)
    s3 = Segment(fs, 200, 300)

    a.segments.extend([s1, s2, s3])

    # Remove s2 (middle)
    a.remove(s2)

    # Check samples length: 300 - 100 = 200
    assert len(a.samples) == 200

    # Check content: 1s then 3s. 2s gone.
    assert np.all(a.samples[:100] == 1)
    assert np.all(a.samples[100:] == 3)

    # s2 removed; check by identity
    assert a.segments[0] is s1
    assert a.segments[1] is s3

    # s1 unchanged
    assert s1.start == 0
    assert s1.stop == 100

    # s3 should be shifted left by 100: 200->100, 300->200
    assert s3.start == 100
    assert s3.stop == 200


def test_remove_segment_overlap():
    fs = 1000
    a = Audio(fs=fs)
    # Samples don't matter much for logic check
    a.samples = np.zeros(300, dtype=np.float32)

    # s1 covers 0-200
    from segment import Segment

    s1 = Segment(fs, 0, 200)

    # s_rem covers 50-100. It is physically inside s1's range (overlap).
    s_rem = Segment(fs, 50, 100)

    a.segments.extend([s1, s_rem])

    # Remove s_rem
    a.remove(s_rem)

    # s_rem removed
    assert s_rem not in a.segments
    assert len(a.segments) == 1
    assert a.segments[0] is s1

    # s1 logic: stop -= shift (50) -> 150
    assert s1.start == 0
    assert s1.stop == 150


def test_validate_edge_cases():
    """Verify Segment validation logic for mismatched FS, negative ranges, and out-of-bounds stops."""
    a = Audio(fs=1000)
    a.samples = np.zeros(100, dtype=np.float32)

    from segment import Segment

    # 1. FS mismatch
    s_bad_fs = Segment(500, 0, 100)
    with pytest.raises(ValueError, match="Segment fs must match audio fs"):
        a.validate(s_bad_fs)

    # 2. Negative start
    s_neg_start = Segment(1000, -1, 50)
    with pytest.raises(ValueError, match="Segment start must be non-negative"):
        a.validate(s_neg_start)

    # 3. Stop > len + 1
    # len=100. stop=102 is > 101.
    s_long = Segment(1000, 0, 102)
    with pytest.raises(
        ValueError, match="Segment stop .* must not be greater than audio length"
    ):
        a.validate(s_long)

    # 4. Stop <= start
    s_flipped = Segment(1000, 50, 40)
    with pytest.raises(ValueError, match="Segment stop must be greater than start"):
        a.validate(s_flipped)
