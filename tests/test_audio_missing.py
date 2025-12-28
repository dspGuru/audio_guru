import numpy as np
import pytest
from unittest.mock import patch

from audio import Audio
from segment import Segment
from util import Category
from constants import DEFAULT_FS, DEFAULT_FREQ
import generate


def test_min_max_properties():
    a = Audio(fs=1000)
    # Generate a simple signal: [1, -1, 0]
    a.samples = np.array([1.0, -1.0, 0.0], dtype=a.DTYPE)
    a.segment = Segment(fs=1000, start=0, stop=3)

    # Check max and min
    assert a.max == 1.0
    assert a.min == -1.0


def test_selected_samples():
    a = Audio(fs=1000)
    a.samples = np.arange(10, dtype=a.DTYPE)  # 0, 1, ..., 9

    # Select subset: indices 2 to 5 (exclusive)
    seg = Segment(fs=1000, start=2, stop=5)
    a.select(seg)

    selected = a.selected_samples
    assert len(selected) == 3
    assert np.array_equal(selected, np.array([2.0, 3.0, 4.0], dtype=a.DTYPE))

    # Check that .max and .min respect the selection
    assert a.max == 4.0
    assert a.min == 2.0


def test_audio_iteration():
    a = Audio(fs=1000, min_segment_secs=0.0)
    # Ensure audio has enough samples for the segments
    a.samples = np.zeros(20, dtype=a.DTYPE)
    # a.eof = True  # Mark as complete for iteration logic (Implicit with new property)

    # Manually set segments
    seg1 = Segment(fs=1000, start=0, stop=10, id=1, cat=Category.Tone)
    seg2 = Segment(fs=1000, start=10, stop=20, id=2, cat=Category.Silence)
    manual_segments = [seg1, seg2]

    # Patch get_segments so it doesn't overwrite our manual segments with
    # segments derived from empty/random samples.
    with patch.object(
        Audio, "get_segments", side_effect=lambda *args, **kwargs: list(manual_segments)
    ):
        # Testing iterator through list(a)
        # list() calls __iter__ which calls get_segments
        segments_list = list(a)

        assert len(segments_list) == 2
        assert segments_list[0] == seg1
        assert segments_list[1] == seg2

        # Test individual iteration (which sets selection)
        iterator = iter(a)

        s1 = next(iterator)
        assert s1 == seg1
        assert a.segment == seg1

        s2 = next(iterator)
        assert s2 == seg2
        assert a.segment == seg2

        with pytest.raises(StopIteration):
            next(iterator)


def test_inplace_addition():
    fs = DEFAULT_FS
    a1 = generate.sine(freq=DEFAULT_FREQ, secs=0.1, amp=0.5, fs=fs)
    a2 = generate.sine(freq=DEFAULT_FREQ, secs=0.1, amp=0.2, fs=fs)

    original_len = len(a1)

    # Check simple addition
    # Since samples are simple sine waves, a1 += a2 should result in summed amplitudes
    # but exact phase alignment matters.
    # However, we are primarily testing the mechanism, not the physics.

    # Just verify it runs and modifies a1
    a1 += a2

    assert len(a1) == original_len
    # Max value should likely increase (constructive interference at start)
    # 0.5 + 0.2 = 0.7
    assert a1.max > 0.5
    assert a1.max <= 0.7 + 1e-6  # allowing float precision


def test_read_failure():
    a = Audio()
    # Read non-existent file
    result = a.read("non_existent_file_xyz_123.wav")
    assert result is False


def test_iadd_errors():
    a = Audio(fs=1000)
    a.samples = np.zeros(100, dtype=a.DTYPE)
    a.segment = Segment(fs=1000, start=0, stop=100)

    # Test adding non-Audio object
    with pytest.raises(ValueError, match="Cannot add non-Audio object"):
        a += 123

    # Test adding Audio failure with differing sample rates
    b = Audio(fs=2000)
    with pytest.raises(ValueError, match="differing sample rates"):
        a += b

    # Test adding Audio with differing number of samples
    c = Audio(fs=1000)
    c.samples = np.zeros(50, dtype=c.DTYPE)  # Different length
    c.segment = Segment(fs=1000, start=0, stop=50)
    with pytest.raises(ValueError, match="different number of samples"):
        a += c


def test_append_errors():
    a = Audio(fs=1000)
    a.samples = np.zeros(100, dtype=a.DTYPE)

    # Test appending Audio with different FS
    b = Audio(fs=2000)
    with pytest.raises(ValueError, match="differing sample rates"):
        a.append(b)

    # Test appending array with incompatible dimensions
    # Current is 1D (100,)
    arr_2d = np.zeros((10, 10), dtype=a.DTYPE)
    with pytest.raises(ValueError, match="differing dimensions"):
        a.append(arr_2d)

    # Test appending array with compatible dimensions but incompatible channel shape
    # Try with 2D audio
    a_2d = Audio(fs=1000)
    a_2d.samples = np.zeros((100, 2), dtype=a.DTYPE)

    arr_different_channels = np.zeros((50, 3), dtype=a.DTYPE)
    with pytest.raises(ValueError, match="differing channel shapes"):
        a_2d.append(arr_different_channels)


def test_select_errors():
    a = Audio(fs=1000)
    a.samples = np.zeros(100, dtype=a.DTYPE)

    # Mismatched FS
    bad_seg_fs = Segment(fs=2000, start=0, stop=10)
    with pytest.raises(ValueError, match="Segment fs must match"):
        a.select(bad_seg_fs)

    # Negative start
    bad_seg_start = Segment(fs=1000, start=-1, stop=10)
    with pytest.raises(ValueError, match="negative"):
        a.select(bad_seg_start)

    # Stop out of bounds
    bad_seg_stop = Segment(fs=1000, start=0, stop=200)
    with pytest.raises(ValueError, match="greater than audio length"):
        a.select(bad_seg_stop)


def test_get_segments_edge_cases():
    a = Audio(fs=1000)
    # Empty audio
    a.samples = np.array([], dtype=a.DTYPE)
    segs = a.get_segments()
    assert len(segs) == 0

    # Audio with gap size issue (min_silence_secs resulting in <= 0 samples?)
    a.samples = np.zeros(100, dtype=a.DTYPE)
    # If we request min_silence_secs that is very small or negative?
    # If we request min_silence_secs that is very small or negative?
    with pytest.raises(ValueError, match="Invalid audio or min_silence_secs"):
        a.get_segments(min_silence_secs=-1.0)


def test_print_conditions(capsys):
    # Use low amplitude noise so a noise floor is found and printed
    a = generate.noise(amp=0.001, secs=0.3, fs=1000)

    # Print with metadata
    a.md.model = "TestModel"
    a.md.set(mfr="TestMfr", model="TestModel", desc="TestDesc")
    a.print()
    captured = capsys.readouterr()
    assert "TestModel" in captured.out
    assert "TestDesc" in captured.out

    # Print noise floor when silence is detected
    # silence = True for generate_silence
    assert "Noise Floor =" in captured.out

    # Print noise floor not found case
    # Create an audio that is NOT silence but has no clear "floor" or just test logic?
    # Actually get_noise_floor returns min average power.
    # If we have perfect silence, noise floor is 0.0 -> -inf dB?
    # Let's inspect get_noise_floor logic.
    # if peak is 0.0, get_noise_floor uses averages. min(0) is 0.
    # The print condition is: if nf > 0.0: print dBFS else: print "not found"

    a_silent = Audio(fs=1000)
    a_silent.samples = np.zeros(100, dtype=a_silent.DTYPE)
    a_silent.print()
    captured = capsys.readouterr()
    assert "Noise floor not found" in captured.out


def test_write_errors():
    a = generate.sine(secs=0.1, fs=1000)

    # No filename in metadata or arg
    a.md.pathname = ""
    assert a.write() is False

    # Empty audio
    a_empty = Audio(fs=1000)
    a_empty.samples = np.array([], dtype=a_empty.DTYPE)
    assert a_empty.write("test.wav") is False

    # Write exception (invalid path)
    # Using a directory as filename usually causes error, or invalid chars
    # On windows, * is invalid
    assert a.write("invalid*name.wav") is False
