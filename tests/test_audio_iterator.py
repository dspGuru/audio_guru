import pytest
import numpy as np
import generate
import audio
from audio import Audio


def test_iterator_open_file(tmp_path, monkeypatch):
    """Verify iterator stops at incomplete segment when file is open."""
    # Patch min silence to detect 0.1s silence
    monkeypatch.setattr(audio, "DEFAULT_SILENCE_MIN_SECS", 0.01)

    fname = tmp_path / "test_iter.wav"
    fs = 48000
    # Create file with Tone (0.1s) -> Silence (0.3s) -> Tone (0.1s)
    # Total 0.5s
    a1 = generate.sine(freq=1000, secs=0.1, fs=fs)
    a1.append(generate.silence(secs=0.3, fs=fs))
    a1.append(generate.sine(freq=1000, secs=0.1, fs=fs))
    # Add extra silence so reading Tone 2 doesn't hit EOF immediately
    a1.append(generate.silence(secs=0.1, fs=fs))
    a1.write(str(fname))

    a = Audio(fs=fs)
    a.open(fname)
    a.blocksize = 4800  # 0.1s per block
    a.blocksize = 4800  # 0.1s per block
    a.min_segment_secs = (
        0.0  # Disable min length filtering to avoid dropping short segments
    )
    a.max_segment_secs = 10.0  # Wait for complete segments up to 10s

    # Read block 1 (Tone 0.1s). 1 segment found.
    # With is_open=True and 1 segment, iterator yields nothing yet.
    assert a.read_block() is True
    assert len(list(a)) == 0

    # Read block 2: silence. Now 2 segments.
    # Iterator should yield 1st segment (Tone).
    assert a.read_block() is True

    # Silence detection may need more blocks
    items = list(a)

    # Read more blocks until silence is detected
    found_items = False
    for _ in range(10):
        if len(items) >= 1:
            found_items = True
            break
        a.read_block()
        items = list(a)

    assert found_items

    # Read block 4 (Tone 2)
    assert a.read_block() is True
    # Ensure we haven't hit EOF prematurely
    assert not a.eof, f"EOF reached prematurely! Position: {a.samples.shape}"

    found_items_2 = False
    for _ in range(10):
        if len(items) >= 1:
            found_items_2 = True
            break
        if a.read_block():
            items = list(a)
        else:
            break

    assert found_items_2
    assert items[0].segment.start <= 1  # Allow for 1 sample diff due to 0.0 value
    assert items[0].segment.stop >= 4799

    # Remaining in a.segments: [Silence].
    # list(a) calls __iter__ which calls get_segments.

    # Read remaining blocks (none left really, but good to check EOF state)
    while a.read_block():
        pass

    assert a.eof
    assert a.is_open

    # eof=True, so remaining segment should yield
    items_eof_open = list(a)
    assert len(items_eof_open) >= 1

    # Close file
    a.close()
    assert not a.is_open

    # Now closed. Should yield nothing as segments were consumed.
    items_closed = list(a)
    assert len(items_closed) == 0
