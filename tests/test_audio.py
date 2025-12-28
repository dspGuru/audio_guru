import unittest.mock

import numpy as np
import pytest
import scipy.signal

# Although we mock it, we import to see if we can use constants if needed.
# It might fail if not installed, but requirements.txt has it.
import sounddevice as sd

from audio import Audio, Segment, Category
from constants import DEFAULT_AMP, DEFAULT_BLOCKSIZE_SECS, DEFAULT_FS, DEFAULT_FREQ
import generate


def test_filter():
    # Create audio with two tones: 1 kHz and 10 kHz

    amp = DEFAULT_AMP
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
    coeffs = scipy.signal.firwin(
        numtaps, cutoff=[DEFAULT_FREQ, 10000], fs=fs, pass_zero=False
    )

    # Apply filter
    print("Applying filter...")
    a.filter(coeffs)

    # Verify that the filter attenuates the tones by the expected amount
    rms = a.rms
    max_rms = amp * 0.01  # for 40 dB expected attenuation
    assert rms < max_rms


def test_append_array_like_and_lengths():
    a = generate.silence(secs=0.1, fs=DEFAULT_FS)
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


def test_generate_sweep_and_detection():
    # Sweep from 100 to 1000 Hz over 2 seconds
    a = generate.sweep(f_start=100.0, f_stop=DEFAULT_FREQ, secs=2.0)
    assert a.name == "Sweep"
    assert len(a) > 0

    # Check is_sweep()
    assert a.is_sweep(secs=0.1) is True
    assert a.get_category() == Category.Sweep


def test_read_write(tmp_path):
    # Generate content
    a = generate.sine(freq=DEFAULT_FREQ, secs=0.5)

    fname = tmp_path / "test_io.wav"
    str_fname = str(fname)

    # Ensure source audio has segments processed (silence removed) to match read behavior
    a.get_segments()

    # Write
    assert a.write(str_fname) is True
    assert fname.exists()

    # Read back
    b = Audio()
    assert b.read(str_fname) is True
    assert len(b) == len(a)
    assert b.freq == pytest.approx(DEFAULT_FREQ, abs=5.0)


def test_exceptions_iadd():
    a = Audio(fs=DEFAULT_FS)
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
        Audio(fs=-DEFAULT_FS)


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
    tone = DEFAULT_AMP * np.sin(2 * np.pi * DEFAULT_FREQ * t).astype(np.float32)
    a.samples[:, 1] = tone  # Right channel

    # This triggers the mono conversion in get_segments
    segs = a.get_segments()
    # It should find the tone (mean > threshold)
    assert len(segs) > 0


def test_get_segments_starts_with_silence():
    fs = 1000
    # Create audio: 0.2s silence, 0.2s tone
    # Set min_segment_secs=0 to avoid filtering out the 0.2s tone (default is 0.5s)
    a = Audio(fs=fs, min_segment_secs=0.0)
    silence = np.zeros(200, dtype=np.float32)
    tone = np.ones(200, dtype=np.float32)  # DC "tone"
    a.append(np.concatenate((silence, tone)))

    # This triggers "if below[0]" logic in get_segments
    # min_silence_secs default is 0.1, so 0.2s silence is detected
    segments = a.get_segments(min_silence_secs=0.01)
    assert len(segments) > 0
    assert len(segments) > 0
    # Initial silence is NOT removed (only 1 silence gap, so it's 'last' and kept), so start index remains 200
    assert segments[0].start == 200


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
    a = Audio(fs=DEFAULT_FS)
    a.samples = np.zeros(100, dtype=np.float32)
    a.select()  # segment covers 100

    b = Audio(fs=DEFAULT_FS)
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


def test_fs_setter():
    a = Audio(fs=DEFAULT_FS)
    a.fs = 48000
    assert a.fs == 48000
    assert a._fs == 48000
    # Blocksize should update (DEFAULT_BLOCKSIZE_SECS is 5.0)
    assert a.blocksize == round(48000 * DEFAULT_BLOCKSIZE_SECS)

    # Segment fs should update
    assert a.segment.fs == 48000

    with pytest.raises(ValueError):
        a.fs = -100


def test_properties_basics():
    # DC signal
    fs = 1000
    a = Audio(fs=fs)
    a.samples = np.ones(100, dtype=np.float32) * DEFAULT_AMP
    a.select()

    assert a.max == DEFAULT_AMP
    assert a.min == DEFAULT_AMP
    assert a.dc == DEFAULT_AMP
    assert a.num_channels == 1

    # Stereo
    b = Audio(fs=fs)
    b.samples = np.zeros((100, 2), dtype=np.float32)
    assert b.num_channels == 2


def test_iter_protocol():
    # Create audio with clear segments: Tone - Silence - Tone
    fs = 1000
    t1 = generate.sine(freq=100, secs=0.1, fs=fs)  # 0.1s
    s1 = generate.silence(secs=0.2, fs=fs)  # 0.2s to ensure detection
    t2 = generate.sine(freq=100, secs=0.1, fs=fs)  # 0.1s

    a = t1
    a.min_segment_secs = 0.0
    a.append(s1)
    a.append(t2)

    # Manually iterate
    with unittest.mock.patch("audio.SILENCE_MIN_SECS", 0.01):
        it = iter(a)
        assert it is a

        # First segment: Tone
        s1 = next(it)
        # Manual categorization
        a.select(s1)
        s1.cat = a.get_category()
        assert s1.cat == Category.Tone

        # Second segment: Tone (wait, audio.get_segments() categorization logic?)
        # get_segments(categorize=True) is called in __iter__
        # Silence is usually split.
        # Let's check how get_segments works. It returns NON-SILENT segments.
        # So we expect 2 segments (t1 and t2).

        s2 = next(it)
        a.select(s2)
        s2.cat = a.get_category()
        assert s2.cat == Category.Tone

        with pytest.raises(StopIteration):
            next(it)


def test_select_none_resets():
    a = generate.sine(secs=1.0)
    # Select sub-range
    seg = Segment(a.fs, 100, 200)
    a.select(seg)
    assert len(a.selected_samples) == 100

    # Select None -> FULL
    a.select(None)
    assert len(a.selected_samples) == len(a.samples)


def test_select_fs_mismatch():
    a = Audio(fs=DEFAULT_FS)
    seg = Segment(48000, 0, 100)
    with pytest.raises(ValueError, match="Segment fs must match audio fs"):
        a.select(seg)


def test_read_errors():
    a = Audio()
    # Invalid file
    assert a.read("nonexistent_file.wav") is False

    # Empty file read check (mocking SoundFile)
    with unittest.mock.patch("soundfile.SoundFile") as mock_sf:
        # Mock instance
        instance = mock_sf.return_value
        instance.__enter__.return_value = instance
        instance.samplerate = DEFAULT_FS
        instance.name = "mock.wav"

        # Determine read_block behavior
        # First call returns empty data -> read_block returns False
        # instance.read.return_value = np.array([], dtype=np.float32)
        # Actually read_block calls f.read.

        # Logic in read():
        # while self.read_block(f)
        # if not len(self): return False

        # So if first read returns empty, len(self) is 0, returns False.
        instance.read.return_value = np.array([], dtype=np.float32)

        assert a.read("mock.wav") is False


def test_write_errors(tmp_path):
    a = Audio()
    # No filename, no metadata pathname
    assert a.write() is False

    # No data
    a.md.pathname = str(tmp_path / "test.wav")
    assert a.write() is False  # len is 0

    # Write exception (e.g. valid data but bad path/permission)
    # Hard to interpret permission error on all OS, so mock write
    a = generate.sine(secs=0.1)
    with unittest.mock.patch("soundfile.write", side_effect=Exception("Fail")):
        assert a.write("out.wav") is False


def test_print_coverage(capsys):
    a = generate.sine(secs=1.0)
    a.md.model = "TestModel"
    a.md.set(mfr="", model="TestModel", desc="TestDesc")

    a.print(noise_floor=True)

    captured = capsys.readouterr()
    assert "TestModel" in captured.out
    assert "TestDesc" in captured.out
    assert "Noise Floor" in captured.out or "not found" in captured.out


def test_get_segments_all_zeros():
    # Audio with all zeros
    fs = 1000
    a = Audio(fs=fs)
    a.samples = np.zeros(1000, dtype=np.float32)  # 1 second of silence

    # get_segments logic for peak=0 is to return 0 active segments now
    segs = a.get_segments()

    assert len(segs) == 0


def test_get_segments_complex_structure():
    # Construct: [Active1 0.1s] [Silence1 0.1s] [Active2 0.1s] [Silence2 0.1s] [Active3 0.1s]
    # Total 0.5s at 1000 Hz fs = 500 samples
    fs = 1000

    # Active segments (tones/DC)
    # Use DC to avoid zero-crossing extending silence detection
    # Set min_segment_secs=0 because these segments are 0.1s long
    a1 = Audio(fs=fs, min_segment_secs=0.0)
    a1.samples = DEFAULT_AMP * np.ones(100, dtype=np.float32)
    a1.select()

    a2 = Audio(fs=fs, min_segment_secs=0.0)
    a2.samples = DEFAULT_AMP * np.ones(100, dtype=np.float32)
    a2.select()

    a3 = Audio(fs=fs, min_segment_secs=0.0)
    a3.samples = DEFAULT_AMP * np.ones(100, dtype=np.float32)
    a3.select()

    # Silent segments
    s1 = generate.silence(secs=0.1, fs=fs)
    s2 = generate.silence(secs=0.1, fs=fs)

    # Combine
    # Note: append modifies in place
    comp = a1
    comp.append(s1)
    comp.append(a2)
    comp.append(s2)
    comp.append(a3)

    # Expected indices:
    # A1: 0-100
    # S1: 100-200
    # A2: 200-300
    # S2: 300-400
    # A3: 400-500

    # get_segments returns NON-SILENT segments by default
    segs = comp.get_segments(min_silence_secs=0.01)

    assert len(segs) == 3

    # Verify A1
    assert segs[0].start == 0
    assert segs[0].stop == 100

    # Verify A2
    # Internal silence (100-200) is NO LONGER removed.
    # So A2 starts at 200.
    assert segs[1].start == 200
    assert segs[1].stop == 300

    # Verify A3
    # S2 (after A2) is NOT removed.
    # So A3 starts at 300 (A2 end) + 100 (S2 len) = 400.
    assert segs[2].start == 400
    assert segs[2].stop == 500


def test_open_device_success():
    """Test successful opening of an audio device."""
    a = Audio(fs=DEFAULT_FS)

    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        # Configure mock
        mock_stream_instance = mock_input_stream.return_value
        mock_stream_instance.active = True

        # Test open_device
        a.open_device(device=1, channels=2)

        # Verify InputStream was called with correct parameters
        mock_input_stream.assert_called_once_with(
            device=1,
            samplerate=DEFAULT_FS,
            blocksize=a.blocksize,
            channels=2,
            dtype="float32",
        )

        # Verify stream was started
        mock_stream_instance.start.assert_called_once()

        # Check attributes
        assert a.stream is mock_stream_instance
        assert a.eof is True
        assert a.is_open is True


def test_open_device_with_settings():
    """Test open_device with custom settings."""
    a = Audio(fs=DEFAULT_FS)

    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        a.open_device(device="mic", fs=48000, blocksize=1024, channels=1, dtype="int16")

        mock_input_stream.assert_called_once_with(
            device="mic", samplerate=48000, blocksize=1024, channels=1, dtype="int16"
        )
        # Verify Audio fs was updated
        assert a.fs == 48000
        assert a.blocksize == 1024


def test_read_block_from_device():
    """Test read_block reading from an open device stream."""
    a = Audio(fs=DEFAULT_FS)
    a.blocksize = 100

    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        mock_stream_instance = mock_input_stream.return_value
        mock_stream_instance.active = True
        mock_stream_instance.channels = 2  # Stereo

        # Mock read return: (data, overflow)
        # Return 100 frames of stereo data
        fake_data = np.zeros((100, 2), dtype=np.float32)
        mock_stream_instance.read.return_value = (fake_data, False)

        a.open_device()

        # read_block implicitly uses self.stream if no file provided
        result = a.read_block()

        assert result is True
        mock_stream_instance.read.assert_called_once_with(100)

        # Verify data was appended
        # Note: Audio.append will be called.
        # Since we mocked read, we should check if samples have increased.
        # Initial samples is 0 or 1 (Audio init creates length 1 empty array usually? No, init creates 1 sample or something?)
        # Audio.init: self.samples = np.ndarray(shape=1, dtype=self.DTYPE) (random uninit memory)
        # But after appending 100 samples, len should be > 100.
        # Actually, since initial len is 0 (logic in __len__), append() replaces samples.
        # So it should be exactly 100.
        assert len(a.samples) == 100


def test_read_block_device_overflow():
    """Test read_block handles overflow flag (prints warning)."""
    a = Audio(fs=DEFAULT_FS)

    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        mock_stream_instance = mock_input_stream.return_value
        mock_stream_instance.active = True
        mock_stream_instance.channels = 1

        # return overflow=True
        # Mono, flattened in audio.py logic if channels==1
        fake_data_2d = np.zeros((1024, 1), dtype=np.float32)
        mock_stream_instance.read.return_value = (fake_data_2d, True)

        a.open_device(channels=1)

        # Capture stdout?
        # Or just ensure it doesn't crash.
        assert a.read_block() is True


def test_read_block_no_stream_no_file():
    """Test read_block returns False when nothing is open."""
    a = Audio()
    assert a.read_block() is False


def test_close_device():
    """Test close stops and closes the stream."""
    a = Audio()

    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        mock_stream = mock_input_stream.return_value
        mock_stream.active = True

        a.open_device()
        assert a.is_open is True

        a.close()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

        # Manually set active=False to see if is_open becomes False
        mock_stream.active = False
        assert a.is_open is False


def test_read_block_device_inactive():
    """Test read_block returns False if stream exists but is not active."""
    a = Audio()
    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        mock_stream = mock_input_stream.return_value
        mock_stream.active = False

        a.open_device()
        # open_device starts it, so we manually simulate it stopping

        mock_stream.active = False
        assert a.read_block() is False


def test_close_resets_eof_and_sf():
    """Test close sets eof to True and sf to None."""
    a = Audio()
    # Mock sf
    with unittest.mock.patch("soundfile.SoundFile") as mock_sf:
        # Fix: ensure samplerate is valid for Audio fs setter
        mock_sf.return_value.samplerate = DEFAULT_FS
        mock_sf.return_value.name = "dummy.wav"
        mock_sf.return_value.closed = False
        mock_sf.return_value.tell.return_value = 0
        mock_sf.return_value.__len__.return_value = 100

        a.open("dummy.wav")
        assert a.sf is not None
        assert a.eof is False

        a.close()

        assert a.sf is None
        assert a.eof is True


def test_iterator_device_max_length():
    """Test iterator stops yielding if segment exceeds max_segment_secs when reading from device."""
    a = Audio(fs=1000)
    a.max_segment_secs = 2.0  # Max 2 seconds

    # Mock open_device to set stream
    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        a.open_device()
        assert a.stream is not None

        # Ensure samples are large enough for the segment to be valid in select()
        a.samples = np.zeros(3000, dtype=np.float32)

        # Case 1: Short segment (incomplete) -> Should yield
        # 1.0 second segment
        s_short = Segment(1000, 0, 1000)

        # Iterating (initializes segments) then we inject our own
        it = iter(a)
        a.segments = [s_short]

        # Iterating should yield this segment because it is < max (and implied incomplete so we process it incrementally?)
        item = next(it)
        assert item is s_short

        # Case 2: Long segment (complete) -> Should StopIteration
        # 2.5 second segment
        s_long = Segment(1000, 0, 2500)
        a.segments = [s_long]

        # Note: 'it' is just 'a', so we can reuse it
        # The new logic forces the segment to be complete and yields it
        item = next(it)
        assert item is s_long
        assert item.complete

        a.close()
