import unittest.mock

import numpy as np
import pytest
import scipy.signal

# sounddevice import for mocking
import sounddevice as sd

from audio import Audio, Segment, Category
from constants import DEFAULT_AMP, DEFAULT_BLOCKSIZE_SECS, DEFAULT_FS, DEFAULT_FREQ
from util import Channel
import generate


def test_filter():
    # Create audio with two tones: 1 kHz and 10 kHz

    amp = DEFAULT_AMP
    fs = 48000

    # Generate 500 Hz tone (mono for filtering)
    a1 = generate.sine(freq=500, amp=amp, secs=1.0, fs=fs, channel=Channel.Left)

    # Generate 20 kHz tone (mono for filtering)
    a2 = generate.sine(freq=20000, amp=amp, secs=1.0, fs=fs, channel=Channel.Left)

    # Combine the two tones
    a = Audio(fs=fs)
    a.samples = a1.samples + a2.samples
    a.segment = a1.segment  # Update segment to match length

    # Design a bandpass FIR filter (1000-10000 Hz)
    numtaps = 200
    coeffs = scipy.signal.firwin(
        numtaps, cutoff=[DEFAULT_FREQ, 10000], fs=fs, pass_zero=False
    )

    # Apply a bandpass filter to verify suppression of frequencies outside the
    # 1-10 kHz range. The RMS of the combined 500 Hz + 20 kHz signal should
    # drop significantly.
    print("Applying filter...")
    a.filter(coeffs)

    # Verify attenuation: 500 Hz and 20 kHz are both outside the passband
    # [1k, 10k].
    rms = a.rms
    max_rms = amp * 0.01  # for 40 dB expected attenuation
    assert rms < max_rms


def test_append_array_like_and_lengths():
    a = generate.silence(secs=0.1, fs=DEFAULT_FS, channel=Channel.Left)
    initial_len = len(a.samples)

    # Append a numpy array chunk
    chunk = 0.1 * np.ones(round(0.05 * a.fs), dtype=np.float32)
    a.append(chunk)
    assert len(a.samples) == initial_len + len(chunk)

    # Append a Python list
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


def test_filter_stereo():
    """Filter on stereo audio applies to each channel individually."""
    fs = 48000
    amp = DEFAULT_AMP

    # Generate stereo audio with different tones per channel
    t = np.arange(fs) / fs  # 1 second
    left = amp * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    right = amp * np.sin(2 * np.pi * 20000 * t).astype(np.float32)

    a = Audio(fs=fs)
    a.samples = np.column_stack((left, right))
    a.select()

    # Design bandpass filter 1 kHz-10 kHz (attenuates both 500 Hz and 20 kHz)
    numtaps = 200
    coeffs = scipy.signal.firwin(
        numtaps, cutoff=[DEFAULT_FREQ, 10000], fs=fs, pass_zero=False
    )

    # Apply filter
    a.filter(coeffs)

    # Both channels should be attenuated
    left_rms = np.sqrt(np.mean(a.samples[:, 0] ** 2))
    right_rms = np.sqrt(np.mean(a.samples[:, 1] ** 2))

    # Original RMS ~0.566, expect 95% attenuation
    max_rms = amp * 0.05
    assert left_rms < max_rms
    assert right_rms < max_rms


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
    assert a.is_sweep() is True
    assert a.cat == Category.Sweep


def test_read_write(tmp_path):
    # Generate content
    a = generate.sine(freq=DEFAULT_FREQ, secs=0.5)

    fname = tmp_path / "test_io.wav"
    str_fname = str(fname)

    # Process segments to match read behavior
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
    # Stereo: Left=silence, Right=tone
    # Mean = tone/2 (half amplitude)
    a = Audio(fs=fs)
    a.samples = np.zeros((fs, 2), dtype=np.float32)

    t = np.arange(fs) / fs
    tone = DEFAULT_AMP * np.sin(2 * np.pi * DEFAULT_FREQ * t).astype(np.float32)
    a.samples[:, 1] = tone  # Right channel

    segs = a.get_segments()
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
    # Tone segment starts at 0
    assert segments[0].start == 0


def test_get_segments_removes_initial_silence():
    """Test that initial silence is removed from self.samples."""
    fs = 1000
    # Create audio: 0.5s silence (500 samples), 0.5s tone (500 samples)
    a = Audio(fs=fs, min_segment_secs=0.0)
    silence = np.zeros(500, dtype=np.float32)
    tone = np.ones(500, dtype=np.float32) * 0.5  # DC "tone"
    a.append(np.concatenate((silence, tone)))

    # Verify initial length
    assert len(a.samples) == 1000

    # Call get_segments - should remove the initial silence
    segments = a.get_segments(min_silence_secs=0.1)

    # Verify initial silence was removed from samples
    assert len(a.samples) == 500  # Only the tone remains

    # Verify n_samples_removed was updated
    assert a.n_samples_removed == 500

    # Verify the segment starts at 0 (adjusted after removal)
    assert len(segments) == 1
    assert segments[0].start == 0
    assert segments[0].stop == 500


def test_get_category_unknown():
    a = Audio()
    # Mock methods to force Unknown category
    with unittest.mock.patch.object(a, "is_silence", return_value=False):
        mock_bins = unittest.mock.Mock()
        mock_bins.is_sine.return_value = False
        mock_bins.is_noise.return_value = False
        # Manually set cached property
        a.bins = mock_bins

        with unittest.mock.patch.object(a, "is_sweep", return_value=False):
            assert a.cat == Category.Unknown


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
    # Test is_silence directly
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

    assert a.max == pytest.approx(DEFAULT_AMP)
    assert a.min == pytest.approx(DEFAULT_AMP)
    assert a.dc == pytest.approx(DEFAULT_AMP)
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
    with unittest.mock.patch("audio.DEFAULT_SILENCE_MIN_SECS", 0.01):
        it = iter(a)
        assert it is a

        # First segment: Tone
        s1 = next(it)
        # s1 is an Audio object, __next__ categorizes if Unknown
        assert s1.segment.cat == Category.Tone

        # Second segment: Tone (get_segments returns NON-SILENT segments)
        s2 = next(it)
        assert s2.segment.cat == Category.Tone

        with pytest.raises(StopIteration):
            next(it)


def test_select_none_resets():
    a = generate.sine(secs=1.0)
    # Select sub-range
    seg = Segment(a.fs, 100, 200)
    a.select(seg)
    assert len(a.selected_samples) == 100

    # Select None resets to the FULL sample buffer without any indexing sub-range.
    a.select(None)
    assert len(a.selected_samples) == len(a.samples)


def test_select_fs_mismatch():
    a = Audio(fs=DEFAULT_FS)
    seg = Segment(48000, 0, 100)
    with pytest.raises(ValueError, match="Segment fs must match audio fs"):
        a.select(seg)


def test_write_errors(tmp_path, capsys):
    a = Audio()
    # No filename, no metadata pathname
    assert a.write() is False

    # No data
    a.md.pathname = str(tmp_path / "test.wav")
    assert a.write() is False  # len is 0

    # Write exception - mock to simulate failure
    a = generate.sine(secs=0.1)
    with unittest.mock.patch("soundfile.SoundFile", side_effect=Exception("Fail")):
        assert a.write("out.wav") is False

    captured = capsys.readouterr()
    assert "Failed to write audio to 'out.wav': Fail" in captured.out


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

    # Active segments (DC to avoid zero-crossing extending silence detection)
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
    s1 = generate.silence(secs=0.1, fs=fs, channel=Channel.Left)
    s2 = generate.silence(secs=0.1, fs=fs, channel=Channel.Left)

    # Combine segments
    comp = a1
    comp.append(s1)
    comp.append(a2)
    comp.append(s2)
    comp.append(a3)

    # Expected: A1=0-100, S1=100-200, A2=200-300, S2=300-400, A3=400-500
    segs = comp.get_segments(min_silence_secs=0.01)

    assert len(segs) == 3

    # Verify A1
    assert segs[0].start == 0
    assert segs[0].stop == 100

    # A2: internal silence not removed
    assert segs[1].start == 200
    assert segs[1].stop == 300

    # A3: S2 not removed
    assert segs[2].start == 400
    assert segs[2].stop == 500


def test_open_device_success():
    """Test successful opening of an audio device."""
    a = Audio(fs=DEFAULT_FS)

    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        # Configure mock
        mock_stream_instance = mock_input_stream.return_value
        mock_stream_instance.active = True

        # Test open_device with quiet=True to avoid query_devices call
        a.open_device(device=1, channels=2, quiet=True)

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
        a.open_device(
            device="mic",
            fs=48000,
            blocksize=1024,
            channels=1,
            dtype="int16",
            quiet=True,
        )

        mock_input_stream.assert_called_once_with(
            device="mic", samplerate=48000, blocksize=1024, channels=1, dtype="int16"
        )
        # Verify Audio fs was updated
        assert a.fs == 48000
        assert a.blocksize == 1024


def test_open_device_quiet_suppresses_output(capsys):
    """Test that quiet=True suppresses 'Audio input' output."""
    a = Audio(fs=DEFAULT_FS)

    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        mock_stream_instance = mock_input_stream.return_value
        mock_stream_instance.active = True

        a.open_device(quiet=True)

    captured = capsys.readouterr()
    # "Audio input" should NOT appear in output
    assert "Audio input" not in captured.out


def test_open_device_not_quiet_prints_device_name(capsys):
    """Test that quiet=False prints 'Audio input' with device name."""
    a = Audio(fs=DEFAULT_FS)

    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        with unittest.mock.patch("sounddevice.query_devices") as mock_query:
            with unittest.mock.patch("sounddevice.default") as mock_default:
                mock_stream_instance = mock_input_stream.return_value
                mock_stream_instance.active = True

                mock_default.device = (0, 1)  # (input, output)
                mock_query.return_value = {"name": "Test Input Device"}

                a.open_device(quiet=False)

    captured = capsys.readouterr()
    # "Audio input" should appear in output
    assert "Audio input" in captured.out
    assert "Test Input Device" in captured.out


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

        a.open_device(quiet=True)

        # read_block implicitly uses self.stream if no file provided
        result = a.read_block()

        assert result is True
        mock_stream_instance.read.assert_called_once_with(100)

        # Verify data was appended (initial len 0, so should be 100)
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

        a.open_device(channels=1, quiet=True)

        # Ensure it doesn't crash
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

        a.open_device(quiet=True)
        assert a.is_open is True

        a.close()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

        # Set active=False to verify is_open becomes False
        mock_stream.active = False
        assert a.is_open is False


def test_read_block_device_inactive():
    """Test read_block returns False if stream exists but is not active."""
    a = Audio()
    with unittest.mock.patch("sounddevice.InputStream") as mock_input_stream:
        mock_stream = mock_input_stream.return_value
        mock_stream.active = False

        a.open_device(quiet=True)
        # open_device starts it, so we manually simulate it stopping

        mock_stream.active = False
        assert a.read_block() is False


def test_close_resets_eof_and_sf():
    """Test close sets eof to True and sf to None."""
    a = Audio()
    with unittest.mock.patch("soundfile.SoundFile") as mock_sf:
        # Ensure samplerate is valid for Audio fs setter
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
        # Use non-zero values so initial silence removal doesn't interfere
        a.samples = np.ones(5000, dtype=np.float32) * 0.5

        # Case 1: Short segment (incomplete) -> Should yield
        # 1.0 second segment
        s_short = Segment(1000, 0, 1000)

        # Iterating (initializes segments) then we inject our own
        it = iter(a)
        a.segments = [s_short]

        # Iterating should yield this segment (< max)
        item = next(it)
        # item is an Audio object wrapping the segment
        assert isinstance(item, Audio)
        assert item.segment == s_short

        # Case 2: Long segment (complete) -> Should StopIteration
        # 2.5 second segment
        s_long = Segment(1000, 0, 2500)
        a.segments = [s_long]

        # Reuse iterator; long segment is forced complete and yielded
        item = next(it)
        assert isinstance(item, Audio)
        assert item.segment == s_long
        assert item.segment.complete

        a.close()


# -----------------------------------------------------------------------------
# Channel-based Read/Write Tests
# -----------------------------------------------------------------------------


class TestReadChannelHandling:
    """Tests for Audio.read channel handling."""

    def test_read_mono_file_sets_left_channel(self, tmp_path):
        """Reading a mono file sets channel to Left."""
        # Generate mono audio and write it
        mono_audio = generate.sine(freq=1000, secs=0.1, channel=Channel.Left)
        fname = tmp_path / "mono_test.wav"
        assert mono_audio.write(fname) is True

        # Read it back
        a = Audio()
        assert a.read(fname) is True
        assert a.channel == Channel.Left

    def test_read_stereo_file_sets_stereo_channel(self, tmp_path):
        """Reading a stereo file sets channel to Stereo."""
        # Generate stereo audio and write it
        stereo_audio = generate.sine(freq=1000, secs=0.1, channel=Channel.Stereo)
        fname = tmp_path / "stereo_test.wav"
        assert stereo_audio.write(fname) is True

        # Read it back
        a = Audio()
        assert a.read(fname) is True
        assert a.channel == Channel.Stereo


class TestWriteChannelHandling:
    """Tests for Audio.write channel handling."""

    def test_write_invalid_channel_raises_valueerror(self, tmp_path):
        """Writing with invalid channel raises ValueError."""
        a = Audio(fs=1000, channel=Channel.Unknown)
        a.samples = np.ones(100, dtype=np.float32)
        a.select()

        fname = tmp_path / "invalid_channel.wav"
        with pytest.raises(ValueError, match="Channel must be Left, Right, or Stereo"):
            a.write(fname)

    def test_write_left_channel_extracts_from_stereo(self, tmp_path):
        """Writing Left channel extracts left from 2D stereo samples."""
        a = Audio(fs=1000, channel=Channel.Left)
        # Create 2D stereo samples: left=0.5, right=0.8
        left = np.ones(100, dtype=np.float32) * 0.5
        right = np.ones(100, dtype=np.float32) * 0.8
        a.samples = np.column_stack((left, right))
        a.select()

        fname = tmp_path / "left_extract.wav"
        assert a.write(fname) is True

        # Read back and verify it's mono with left channel values
        b = Audio()
        assert b.read(fname) is True
        assert b.channel == Channel.Left  # Mono file
        assert b.samples.ndim == 1 or b.samples.shape[1] == 1
        # Values should be approximately 0.5 (left channel)
        assert np.mean(np.abs(b.samples.flatten())) == pytest.approx(0.5, rel=0.01)

    def test_write_right_channel_extracts_from_stereo(self, tmp_path):
        """Writing Right channel extracts right from 2D stereo samples."""
        a = Audio(fs=1000, channel=Channel.Right)
        # Create 2D stereo samples: left=0.3, right=0.7
        left = np.ones(100, dtype=np.float32) * 0.3
        right = np.ones(100, dtype=np.float32) * 0.7
        a.samples = np.column_stack((left, right))
        a.select()

        fname = tmp_path / "right_extract.wav"
        assert a.write(fname) is True

        # Read back and verify it's mono with right channel values
        b = Audio()
        assert b.read(fname) is True
        assert b.channel == Channel.Left  # Mono file reads as Left
        # Values should be approximately 0.7 (right channel)
        assert np.mean(np.abs(b.samples.flatten())) == pytest.approx(0.7, rel=0.01)

    def test_write_left_channel_with_1d_samples(self, tmp_path):
        """Writing Left channel with already 1D samples works correctly."""
        a = Audio(fs=1000, channel=Channel.Left)
        a.samples = np.ones(100, dtype=np.float32) * 0.6
        a.select()

        fname = tmp_path / "mono_1d.wav"
        assert a.write(fname) is True

        # Read back and verify
        b = Audio()
        assert b.read(fname) is True
        assert np.mean(np.abs(b.samples.flatten())) == pytest.approx(0.6, rel=0.01)

    def test_write_stereo_channel_preserves_2d(self, tmp_path):
        """Writing Stereo channel preserves 2D samples."""
        a = Audio(fs=1000, channel=Channel.Stereo)
        # Create 2D stereo samples
        left = np.ones(100, dtype=np.float32) * 0.4
        right = np.ones(100, dtype=np.float32) * 0.9
        a.samples = np.column_stack((left, right))
        a.select()

        fname = tmp_path / "stereo_preserve.wav"
        assert a.write(fname) is True

        # Read back and verify it's stereo
        b = Audio()
        assert b.read(fname) is True
        assert b.channel == Channel.Stereo
        assert b.samples.ndim == 2
        assert b.samples.shape[1] == 2


# -----------------------------------------------------------------------------
# Quadrature Detection Tests
# -----------------------------------------------------------------------------


def test_is_quadrature_true():
    """A true quadrature signal (complex exponential) returns True."""
    fs = 1000
    t = np.arange(1000) / fs
    freq = 100.0
    amp = 0.8

    # Complex exponential: real part is cos, imaginary part is sin
    # |cos + j*sin| = 1 * amp for all samples
    ch0 = amp * np.cos(2 * np.pi * freq * t).astype(np.float32)
    ch1 = amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    a = Audio(fs=fs)
    a.samples = np.column_stack((ch0, ch1))
    a.select()

    assert a.is_quadrature is True


def test_is_quadrature_mono():
    """Mono audio returns False (not 2 channels)."""
    a = generate.sine(freq=1000, secs=0.5)
    a.samples = a.samples.flatten()  # Ensure 1D
    a.select()

    assert a.is_quadrature is False


def test_is_quadrature_stereo_non_quadrature():
    """Stereo audio with independent channels returns False."""
    fs = 1000
    t = np.arange(1000) / fs

    # Two signals with vastly different amplitudes - clearly not quadrature
    ch0 = 0.9 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
    ch1 = 0.1 * np.sin(2 * np.pi * 100 * t).astype(np.float32)

    a = Audio(fs=fs)
    a.samples = np.column_stack((ch0, ch1))
    a.select()

    assert a.is_quadrature is False


def test_is_quadrature_silence():
    """Silent stereo audio returns False."""
    fs = 1000
    a = Audio(fs=fs)
    a.samples = np.zeros((1000, 2), dtype=np.float32)
    a.select()

    assert a.is_quadrature is False


def test_phase_quadrature_wrapped():
    """Phase calculation on quadrature signal returns wrapped phase."""
    fs = 1000
    t = np.arange(1000) / fs
    freq = 100.0
    amp = 0.8

    # Complex exponential: phase should increase linearly
    ch0 = amp * np.cos(2 * np.pi * freq * t).astype(np.float32)
    ch1 = amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    a = Audio(fs=fs)
    a.samples = np.column_stack((ch0, ch1))
    a.select()

    phase = a.phase(unwrap=False)
    assert phase is not None
    assert len(phase) == 1000
    # Wrapped phase should be in [-pi, pi]
    assert np.all(phase >= -np.pi) and np.all(phase <= np.pi)


def test_phase_quadrature_unwrapped():
    """Unwrapped phase should be continuous and monotonic."""
    fs = 1000
    t = np.arange(1000) / fs
    freq = 100.0
    amp = 0.8

    ch0 = amp * np.cos(2 * np.pi * freq * t).astype(np.float32)
    ch1 = amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    a = Audio(fs=fs)
    a.samples = np.column_stack((ch0, ch1))
    a.select()

    phase = a.phase(unwrap=True)
    assert phase is not None
    # Unwrapped phase should be monotonically increasing
    diffs = np.diff(phase)
    assert np.all(diffs >= 0)


def test_phase_non_quadrature_returns_none():
    """Phase on mono signal returns None."""
    a = generate.sine(freq=1000, secs=0.5)
    a.samples = a.samples.flatten()
    a.select()

    assert a.phase() is None


def test_magnitude_quadrature():
    """Magnitude of quadrature signal is approximately constant."""
    fs = 1000
    t = np.arange(1000) / fs
    freq = 100.0
    amp = 0.8

    ch0 = amp * np.cos(2 * np.pi * freq * t).astype(np.float32)
    ch1 = amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    a = Audio(fs=fs)
    a.samples = np.column_stack((ch0, ch1))
    a.select()

    mag = a.magnitude
    assert mag is not None
    assert len(mag) == 1000
    # Magnitude should be approximately constant at amp
    assert np.all(np.abs(mag - amp) < 0.01)


def test_magnitude_non_quadrature_returns_none():
    """Magnitude on mono signal returns None."""
    a = generate.sine(freq=1000, secs=0.5)
    a.samples = a.samples.flatten()
    a.select()

    assert a.magnitude is None


def test_phase_error_quadrature_sine():
    """Phase error on quadrature sine is small."""
    from util import Channel

    a = generate.sine(freq=1000, secs=0.5, channel=Channel.Stereo)

    phase_err = a.phase_error()
    assert phase_err is not None
    # Generated sine should have very small phase error (near-perfect linear phase)
    assert phase_err < 0.01  # Less than 0.01 radians


def test_phase_error_mono_returns_none():
    """Phase error on mono signal returns None."""
    a = generate.sine(freq=1000, secs=0.5)
    a.samples = a.samples.flatten()
    a.select()

    assert a.phase_error() is None


def test_phase_error_non_sine_returns_none():
    """Phase error on quadrature noise returns None (not a sine)."""
    from util import Channel

    np.random.seed(42)
    a = generate.noise(secs=0.5, channel=Channel.Stereo)

    assert a.phase_error() is None
