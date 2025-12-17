import unittest.mock

import numpy as np
import pytest
import scipy.signal

from audio import Audio, Segment, Category
import generate


def test_filter():
    # Create audio with two tones: 1 kHz and 10 kHz

    amp = 0.5
    fs = 48000

    # Generate 500 Hz tone
    a1 = generate.sine(freq=500, amp=amp, secs=1.0, fs=fs)

    # Generate 20 kHz tone
    a2 = generate.sine(freq=20000, amp=amp, secs=1.0, fs=fs)

    # Combine the two tones
    a = Audio(fs=fs)
    a.samples = a1.samples + a2.samples
    a.segment = a1.segment  # Update segment to match length

    # Design a bandpass FIR filter with cutoff frequenceis of 1000 Hz to 10000
    # Hz
    numtaps = 200
    coeffs = scipy.signal.firwin(numtaps, cutoff=[1000, 10000], fs=fs, pass_zero=False)

    # Apply filter
    print("Applying filter...")
    a.filter(coeffs)

    # Verify that the filter attenuates the tones by the expected amount
    rms = a.rms
    max_rms = amp * 0.01  # for 40 dB expected attenuation
    assert rms < max_rms


def test_append_array_like_and_lengths():
    a = generate.silence(secs=0.1, fs=44100)
    initial_len = len(a.samples)

    # Append a numpy array chunk
    chunk = 0.1 * np.ones(round(0.05 * a.fs), dtype=np.float32)
    a.append(chunk)
    assert len(a.samples) == initial_len + len(chunk)

    # Append a Python list also works
    chunk2 = [0.0] * round(0.02 * a.fs)
    a.append(chunk2)
    assert len(a.samples) == initial_len + len(chunk) + len(chunk2)


def test_filter_noop_on_empty_coeffs_and_no_data():
    # No-op when filter coefficients empty
    a = generate.sine(secs=0.1)
    before = a.samples.copy()
    a.filter([])
    assert np.allclose(a.samples, before)

    # No-op when there's no data
    b = Audio()
    b.samples = np.array([], dtype=b.DTYPE)
    # Should not raise
    b.filter([0.1, 0.2, 0.1])


def test_generate_sine_short_duration_frequency_estimate():
    # Short duration sine still should give a reasonable freq estimate
    a = generate.sine(freq=1500.0, amp=0.3, secs=0.05, fs=48000)  # 50 ms
    # Frequency estimate may be coarse; allow wider tolerance
    assert a.freq == pytest.approx(1500.0, abs=15.0)


def test_append_and_segment_handling():
    # Create short tone and silence and append them
    t1 = generate.sine(secs=0.2)
    expected_len = len(t1.samples)

    s = generate.silence(secs=0.3)
    expected_len += len(s.samples)

    t2 = generate.sine(secs=0.1)
    expected_len += len(t2.samples)

    # Append Audio objects to create combined audio
    t1.append(s)
    t1.append(t2)

    # Length should match expected
    assert len(t1.samples) == expected_len

    # Non-silent segments detection: should find at least one activity segment
    non_silent = t1.get_segments(silence=False, categorize=False)
    assert len(non_silent) >= 1

    # Silent segments detection: should find at least one silent segment of approx 0.3s
    silent_segments = t1.get_segments(silence=True)
    assert any(
        (seg.stop - seg.start + 1) >= round(0.25 * t1.fs) for seg in silent_segments
    )


def test_generate_sweep_and_detection():
    # Sweep from 100 to 1000 Hz over 2 seconds
    a = generate.sweep(f_start=100.0, f_stop=1000.0, secs=2.0)
    assert a.name == "Sweep"
    assert len(a) > 0

    # Check is_sweep()
    assert a.is_sweep(secs=0.1) is True
    assert a.get_category() == Category.Sweep


def test_read_write(tmp_path):
    # Generate content
    a = generate.sine(freq=1000, secs=0.5)

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
    a.samples = np.zeros((100,), dtype=np.float32)  # Mono
    invalid = np.zeros((100, 2), dtype=np.float32)  # Stereo

    # Test append with incompatible dimensions.
    with pytest.raises(ValueError, match="differing dimensions"):
        a.append(invalid)


def test_init_invalid_fs():
    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        Audio(fs=0)
    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        Audio(fs=-44100)


def test_select_invalid_range():
    a = generate.sine(secs=1.0)
    # create invalid segment where stop <= start
    seg = Segment(a.fs, start=1000, stop=500)
    with pytest.raises(ValueError, match="Segment stop must be greater than start"):
        a.select(seg)


def test_generate_invalid_inputs():
    # generate_noise
    with pytest.raises(ValueError, match="Duration must be positive"):
        generate.noise(secs=0)
    with pytest.raises(ValueError, match="Amplitude must be between 0.0 and 1.0"):
        generate.noise(amp=1.5)

    # generate_silence
    with pytest.raises(ValueError, match="Duration must be positive"):
        generate.silence(secs=-1.0)

    # generate_sine
    with pytest.raises(ValueError, match="Frequency must be positive"):
        generate.sine(freq=0)
    with pytest.raises(ValueError, match="Duration must be positive"):
        generate.sine(secs=0)
    with pytest.raises(ValueError, match="Amplitude must be between 0.0 and 1.0"):
        generate.sine(amp=-0.1)

    # generate_sweep
    with pytest.raises(ValueError, match="Frequencies must be positive"):
        generate.sweep(f_start=0)
    with pytest.raises(ValueError, match="Duration must be positive"):
        generate.sweep(secs=0)
    with pytest.raises(ValueError, match="Amplitude must be between 0.0 and 1.0"):
        generate.sweep(amp=1.1)


def test_write_creates_dirs(tmp_path):
    a = generate.sine(secs=0.1)

    subdir = tmp_path / "subdir" / "nested"
    outfile = subdir / "test.wav"

    assert not subdir.exists()

    # Should create directories
    assert a.write(outfile) is True
    assert outfile.exists()


def test_get_segments_stereo():
    fs = 48000
    # Stereo audio: Left channel silence, Right channel tone
    # Mean will be (0 + tone)/2, which is essentially a tone with half amplitude
    a = Audio(fs=fs)
    a.samples = np.zeros((fs, 2), dtype=np.float32)

    t = np.arange(fs) / fs
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    a.samples[:, 1] = tone  # Right channel

    # This triggers the mono conversion in get_segments
    segs = a.get_segments()
    # It should find the tone (mean > threshold)
    assert len(segs) > 0


def test_get_segments_starts_with_silence():
    fs = 1000
    # Create audio: 0.2s silence, 0.2s tone
    a = Audio(fs=fs)
    silence = np.zeros(200, dtype=np.float32)
    tone = np.ones(200, dtype=np.float32)  # DC "tone"
    a.samples = np.concatenate((silence, tone))

    # This triggers "if below[0]" logic in get_segments
    # min_silence_secs default is 0.1, so 0.2s silence is detected
    silent_segs = a.get_segments(silence=True)
    assert len(silent_segs) > 0
    assert silent_segs[0].start == 0


def test_get_category_unknown():
    a = Audio()
    # Mock methods to force Unknown
    # is_silence -> False
    # bins.is_sine -> False
    # is_sweep -> False
    # bins.is_noise -> False

    # We can achieve this by mocking or by constructing a signal that fails all checks.
    # An impulse might work? Or just random data that isn't quite noise?
    # Or just mock.
    with unittest.mock.patch.object(a, "is_silence", return_value=False):
        with unittest.mock.patch.object(a, "get_bins") as mock_bins:
            mock_bins.return_value.is_sine.return_value = False
            mock_bins.return_value.is_noise.return_value = False

            with unittest.mock.patch.object(a, "is_sweep", return_value=False):
                assert a.get_category() == Category.Unknown


def test_get_noise_floor_edge_cases():
    # Empty audio
    a = Audio()
    nf, idx = a.get_noise_floor()
    assert nf == 0.0
    assert idx == 0

    # Stereo audio with -80 dBFS noise
    a2 = Audio(fs=1000)
    a2.samples = np.random.uniform(-1e-8, 1e-8, (1000, 2)).astype(np.float32)
    # this triggers abs_audio.ndim > 1

    nf2, _ = a2.get_noise_floor()
    assert nf2 > 0.0


def test_iadd_length_mismatch():
    a = Audio(fs=44100)
    a.samples = np.zeros(100, dtype=np.float32)
    a.select()  # segment covers 100

    b = Audio(fs=44100)
    b.samples = np.zeros(200, dtype=np.float32)
    b.select()  # segment covers 200

    # Line 97 check: shape mismatch
    with pytest.raises(ValueError, match="different number of samples"):
        a += b


def test_is_silence_direct():
    # Direct test for is_silence
    # Note: np.all returns np.bool_, which is not strictly 'is True'. Compare value.
    a = generate.silence(secs=0.1)
    assert a.is_silence() == True

    a = generate.sine(secs=0.1)
    assert a.is_silence() == False
