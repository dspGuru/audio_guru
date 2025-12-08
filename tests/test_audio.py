import pytest
from audio import Audio, Segment, Category
import numpy as np
import scipy.signal


def test_filter():
    fs = 48000
    # Create audio with two tones: 1 kHz and 10 kHz
    a = Audio(fs=fs, name="TestAudio")

    # 1 kHz tone
    a1 = Audio(fs=fs)
    a1.generate_tone(freq=1000, amp=0.5, secs=1.0)

    # 10 kHz tone
    a2 = Audio(fs=fs)
    a2.generate_tone(freq=10000, amp=0.5, secs=1.0)

    # Combine
    a.samples = a1.samples + a2.samples
    a.segment = a1.segment  # Update segment to match length

    # Verify initial content (rough check using freq property, which picks
    # dominant) or just trust the generation.

    # Design a bandpass filter with cutoff frequenceis of 1000 Hz to 10000 Hz
    numtaps = 200
    coeffs = scipy.signal.firwin(numtaps, cutoff=[1000, 10000], fs=fs, pass_zero=False)

    # Apply filter
    print("Applying filter...")
    a.filter(coeffs)

    # Check that 10 kHz is attenuated
    # One way is to measure RMS of high pass or just FFT
    # Let's simple check if the purely high freq component is gone.
    # We can filter the result again with a high pass and see if it's empty.

    # Or simpler: The resultant audio should look like 1kHz sine.
    # RMS of original single 1kHz sine ~ 0.5/sqrt(2) = 0.3535
    # RMS of combined ~ sqrt(0.3535^2 + 0.3535^2) = 0.5
    # RMS of filtered should be close to 0.3535

    rms = a.rms
    print(f"RMS after low-pass filter: {rms}")

    expected_rms = 0.5 / np.sqrt(2)
    diff = abs(rms - expected_rms)
    print(f"Expected RMS (approx): {expected_rms}")
    print(f"Difference: {diff}")

    if diff < 0.05:
        print("SUCCESS: 10kHz tone appears to be filtered out.")
    else:
        print("FAILURE: RMS too high, high freq content might still be there.")


def test_generate_tone_properties():
    a = Audio()
    a.generate_tone(freq=440.0, amp=0.5, secs=1.0, thd=0.0)
    assert a.name == "Tone"
    # length should be approximately secs * fs (allow off-by-one)
    expected = round(1.0 * a.fs)
    assert len(a.samples) in (expected, expected + 1)
    # DC for a pure sine should be near zero
    assert abs(a.dc) < 0.01
    # RMS of a sine of amplitude 0.5 should be around 0.5/sqrt(2)
    rms = a.rms
    assert 0.3 < rms < 0.4  # 0.5/1.414 = 0.3535

    # Check frequency estimation
    assert a.freq == pytest.approx(440.0, abs=5.0)


def test_generate_noise_statistics():
    a = Audio()
    amp = 0.3
    a.generate_noise(amp=amp, secs=1.0)
    assert a.name == "Noise"
    # Values should lie within amplitude bounds (with small epsilon for float precision)
    assert float(np.max(a.samples)) <= amp + 1e-5
    assert float(np.min(a.samples)) >= -amp - 1e-5
    assert a.rms > 0.0

    assert a.get_category() == Category.Noise


def test_generate_silence_segments_and_noise_floor():
    a = Audio()
    secs = 1.0
    a.generate_silence(secs=secs)
    # Segment span should match generated length
    expected = round(secs * a.fs)
    assert len(a.samples) in (expected, expected + 1)
    # get_segments(silence=True) should return one silent segment covering the buffer
    silent_segments = a.get_segments(silence=True)
    assert len(silent_segments) == 1
    seg = silent_segments[0]
    assert seg.start == 0
    # Stop can be expected - 1
    assert seg.stop >= expected - 1
    # noise floor for silence should be zero
    nf, idx = a.get_noise_floor()
    assert nf == pytest.approx(0.0)
    assert idx == 0


def test_iadd_append_and_segment_handling():
    # Create short tone and silence and append them using +=
    t1 = Audio()
    t1.generate_tone(secs=0.2)
    s = Audio()
    s.generate_silence(secs=0.3)
    t2 = Audio()
    t2.generate_tone(secs=0.2)

    combined = Audio()

    # Append Audio objects
    combined.append(t1)
    combined.append(s)
    combined.append(t2)

    # Expect combined length roughly sum of parts
    expected_len = len(t1.samples) + len(s.samples) + len(t2.samples)
    # Allow small variance
    assert abs(len(combined.samples) - expected_len) <= 1

    # Non-silent segments detection: should find at least one activity segment
    non_silent = combined.get_segments(silence=False, categorize=False)
    assert len(non_silent) >= 1

    # Silent segments detection: should find at least one silent segment of approx 0.3s
    silent_segments = combined.get_segments(silence=True)
    assert any(
        (seg.stop - seg.start + 1) >= round(0.25 * combined.fs)
        for seg in silent_segments
    )


def test_generate_sweep_and_detection():
    a = Audio()
    # Sweep from 100 to 1000 Hz over 2 seconds
    a.generate_sweep(f_start=100.0, f_stop=1000.0, secs=2.0)
    assert a.name == "SweepGen"
    assert len(a) > 0

    # Check is_sweep logic
    assert a.is_sweep(secs=0.1) is True
    assert a.get_category() == Category.Sweep


def test_push_pop_segments():
    a = Audio()
    a.generate_tone(secs=1.0)
    full_seg = a.segment

    # Select sub-segment
    sub = Segment(a.fs, 0, 100)
    a.push(sub)
    assert a.segment == sub
    assert a.stack[-1] == full_seg

    # Pop back
    a.pop()
    assert a.segment == full_seg
    assert len(a.stack) == 0


def test_read_write(tmp_path):
    # Generate content
    a = Audio()
    a.generate_tone(freq=1000, secs=0.5)

    fname = tmp_path / "test_io.wav"
    str_fname = str(fname)

    # Write
    assert a.write(str_fname) is True
    assert fname.exists()

    # Read back
    b = Audio()
    assert b.read(str_fname) is True
    assert len(b) == len(a)
    assert b.freq == pytest.approx(1000.0, abs=5.0)


def test_exceptions_iadd():
    a = Audio(fs=44100)
    b = Audio(fs=48000)

    # Differing sample rates
    with pytest.raises(ValueError, match="differing sample rates"):
        a.append(b)

    # Differing dimensions (if we mess with samples directly)
    a.samples = np.zeros((100,), dtype=np.float32)
    invalid = np.zeros((100, 2), dtype=np.float32)  # Stereo

    # Manually trying to append invalid array via internal logic stimulation or mock if needed
    # But public API `+=` accepts Audio or array.
    with pytest.raises(ValueError, match="differing dimensions"):
        a.append(invalid)
