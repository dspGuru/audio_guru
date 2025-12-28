import pytest
import soundfile as sf
import generate
from audio import Audio


def test_open_close_state(tmp_path):
    """Verify is_open property and close method."""
    fname = tmp_path / "test.wav"
    g = generate.sine(secs=0.1)
    g.write(str(fname))

    a = Audio()
    assert not a.is_open

    a.open(fname)
    assert a.is_open
    assert not a.eof

    a.close()
    assert not a.is_open


def test_read_until_eof(tmp_path):
    """Verify eof property is set when reading completes."""
    fname = tmp_path / "test_eof.wav"
    fs = 1000
    g = generate.sine(secs=0.1, fs=fs)  # 100 samples
    g.write(str(fname))

    a = Audio(fs=fs)
    a.open(fname)
    a.blocksize = 50  # 2 blocks

    # Read block 1
    assert a.read_block() is True
    assert not a.eof

    # Read block 2
    assert a.read_block() is True
    # With new logic, eof is True immediately if we reached end of file
    assert a.eof

    # Read block 3 (should be empty/EOF)
    assert a.read_block() is False
    assert a.eof is True

    a.close()
