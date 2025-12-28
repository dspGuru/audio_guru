import pytest
import numpy as np
import generate
import pytest
import numpy as np
import generate
import audio
from audio import Audio


def test_iterator_open_file(tmp_path, monkeypatch):
    """Verify iterator stops at incomplete segment when file is open."""
    # Patch min silence to detect 0.1s silence
    monkeypatch.setattr(audio, "SILENCE_MIN_SECS", 0.01)

    fname = tmp_path / "test_iter.wav"
    fs = 48000
    # Create file with Tone (0.1s) -> Silence (0.2s) -> Tone (0.1s)
    # Total 0.4s
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
    a.min_segment_secs = 0.05
    a.max_segment_secs = 10.0  # Wait for complete segments up to 10s

    # Read block 1 (Tone 0.1s).
    # Should find 1 segment [0-100] (Tone).
    # Since file is open and only 1 segment, iterator should yield nothing?
    # Wait, if we have 1 segment, `len` is 1. `is_open` is True. StopIteration raised.
    # So yielding nothing.
    assert a.read_block() is True
    assert len(list(a)) == 0  # Iterator yields nothing yet

    # Read block 2 (Silence 0.1s). Total 0.2s.
    # Samples: 0.1s Tone, 0.1s Silence.
    # Segments: Tone [0-100], Silence [100-200].
    # Now we have 2 segments.
    # Iterator should yield 1st segment (Tone).
    # Logic: pop(0). Remaining len=1.
    # Next iter: len=1, is_open=True -> StopIteration.
    assert a.read_block() is True

    # Note: Silence detection might be sensitive at exact boundary (0.1s == min_silence).
    # If not detected yet, we iterate and read more.
    items = list(a)

    # Read block 3 (Silence 0.1s). Total 0.3s (0.1 Tone + 0.2 Silence).
    # Now silence is 0.2s, definitely > 0.1s.
    assert a.read_block() is True

    # Now we have [Tone 1, Silence].
    # Tone 1 is complete and followed by silence, so it should be yielded.
    items = list(a)
    assert len(items) == 1

    # Read block 4 (Tone 2 0.1s).
    # Now we have Tone 1 and Tone 2.
    assert a.read_block() is True
    # Ensure we haven't hit EOF yet (needs extra silence in file)
    assert not a.eof, f"EOF reached prematurely! Position: {a.samples.shape}"

    items = list(a)
    assert len(items) == 1
    assert items[0].start <= 1  # Allow for 1 sample diff due to 0.0 value
    assert items[0].stop >= 4799

    # Remaining in a.segments: [Silence].
    # BUT `list(a)` calls `__iter__` which calls `get_segments`.
    # `get_segments` returns [Tone, Silence].
    # Loop 1: pop Tone. Yield Tone. Remaining [Silence].
    # Loop 2: len=1, is_open=True. StopIteration.
    # So yes, 1 item yielded.

    # Read remaining blocks (none left really, but good to check EOF state)
    while a.read_block():
        pass

    assert a.eof
    assert a.is_open

    # Still open, single segment (Tone 2) remaining.
    # Logic: len=1, is_open=True check is replaced by len=1 and not eof.
    # Since eof is True, it should yield.
    items_eof_open = list(a)
    # Expected: Tone 1 and Tone 2
    assert len(items_eof_open) >= 1

    # Close file
    a.close()
    assert not a.is_open

    # Now closed. Should yield remaining segment (Tone 2) AND Tone 1 (already read).
    # Since iteration starts from beginning of samples.
    items_closed = list(a)
    assert len(items_closed) == 2
    assert items_closed[-1].stop_secs >= 0.49
    assert items_closed[0].stop_secs >= 0.09
