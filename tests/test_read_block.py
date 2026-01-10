import soundfile as sf
import numpy as np
from unittest.mock import Mock
import generate
from audio import Audio


def test_read_block(tmp_path):
    # 1. Create a test file (1 second of tone)
    fname = tmp_path / "test_block.wav"
    fs = 48000
    a_orig = generate.sine(freq=440, secs=1.0, fs=fs)
    a_orig.write(str(fname))

    # Strip silence from a_orig to match read behavior (which strips initial silence)
    a_orig.get_segments()

    # 2. Open with soundfile
    f = sf.SoundFile(str(fname))

    # Set min_segment_secs=0 for 0.1s blocks
    a = Audio(fs=fs, min_segment_secs=0.0)
    a.samples = np.array([], dtype=np.float32)
    # Default blocksize 0.1s -> ~10 blocks
    a.blocksize = round(fs * 0.1)

    while a.read_block(f):
        # Check metadata update
        assert a.fs == fs
        # Verify segments extraction is happening (should be at least 1 segment found/updated)
        assert len(a.segments) > 0

    f.close()

    # 4. Verify
    assert len(a.samples) > 0

    # Compare with original
    assert np.allclose(a.samples, a_orig.samples, atol=1e-4)


def test_read_block_duck_typing():
    # Test read_block with a non-SoundFile object (Duck Typing)

    # Create a mock object that mimics SoundFile
    f_mock = Mock()
    f_mock.samplerate = 48000
    f_mock.name = "MockAudioSource"
    f_mock.closed = False
    f_mock.__len__ = Mock(return_value=1000)
    f_mock.tell.return_value = 0

    # Setup read to return one block then empty
    fs = 48000
    block_len = round(fs * 0.1)
    # Valid float32 data
    data_block = np.ones(block_len, dtype=np.float32)

    # side_effect: first call returns data, second returns empty (EOF)
    f_mock.read.side_effect = [data_block, np.array([], dtype=np.float32)]

    a = Audio(fs=fs)
    a.blocksize = block_len

    # First read (success)
    assert a.read_block(f_mock) is True
    assert len(a.samples) == block_len
    assert a.fs == 48000

    # Check that f.read was called with correct arguments
    f_mock.read.assert_called_with(block_len, dtype="float32", always_2d=False)

    # Second read (EOF)
    assert a.read_block(f_mock) is False


def test_read_block_max_samples_assertion():
    """Test that read_block asserts when samples exceed 2 * SEGMENT_MAX_SECS."""
    import pytest
    from constants import SEGMENT_MAX_SECS

    fs = 44100
    max_samples = int(2 * SEGMENT_MAX_SECS * fs)

    # Create a mock file that returns large blocks of non-silent data
    f_mock = Mock()
    f_mock.samplerate = fs
    f_mock.name = "MockLargeAudio"
    f_mock.closed = False
    f_mock.__len__ = Mock(return_value=max_samples * 2)
    f_mock.tell.return_value = 0

    # Large non-zero block to exceed limit
    large_block = np.full(max_samples + fs, 0.5, dtype=np.float32)

    f_mock.read.return_value = large_block

    a = Audio(fs=fs)
    a.blocksize = len(large_block)

    # Should raise AssertionError when samples exceed max
    with pytest.raises(AssertionError, match="Audio.samples exceeded maximum length"):
        a.read_block(f_mock)
