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

    # 2. Open with soundfile
    f = sf.SoundFile(str(fname))

    # 3. Read in blocks
    a = Audio(fs=fs)
    a.samples = np.array([], dtype=np.float32)
    # Default blocksize is 0.1s -> should get ~10 blocks
    a.blocksize = round(fs * 0.1)

    while a.read_block(f):
        # Check metadata update
        assert a.fs == fs

    f.close()

    # 4. Verify
    assert len(a.samples) > 0

    # Compare with original samples
    # Original might have slightly different length if write/read adds/removes anything? usually exact for wav.
    assert np.allclose(a.samples, a_orig.samples, atol=1e-4)


def test_read_block_duck_typing():
    # Test read_block with a non-SoundFile object (Duck Typing)

    # Create a mock object that mimics SoundFile
    f_mock = Mock()
    f_mock.samplerate = 48000
    f_mock.name = "MockAudioSource"

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
