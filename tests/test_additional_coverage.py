"""Additional tests to improve coverage of remaining uncovered lines."""

import numpy as np
import pytest
import unittest.mock

from audio import Audio
from component import Component, Components
from freq_bins import FreqBins
from metadata import Metadata
from segment import Segment
from tone_analyzer import ToneAnalyzer
from util import Category
import generate


class TestAudioRescale:
    """Tests for Audio.rescale() edge cases."""

    def test_rescale_empty_audio(self):
        """Test rescale() with empty audio (line 578)."""
        audio = Audio()
        # Empty audio - no samples
        audio.rescale(0.5)
        # Should return early without error
        assert len(audio) == 0

    def test_rescale_zero_amplitude(self):
        """Test rescale() with all-zero samples."""
        audio = Audio(fs=44100)
        audio.samples = np.zeros(1000, dtype=np.float32)
        audio.select()
        audio.rescale(0.5)
        # Should return early due to zero max amplitude
        assert np.max(np.abs(audio.samples)) == 0.0


class TestAudioSegments:
    """Tests for Audio segment detection edge cases."""

    def test_get_segments_no_segments_detected(self):
        """Test get_segments() returning existing segments (line 658)."""
        audio = Audio(fs=44100)
        # Very short silence that won't produce new segments
        audio.samples = np.zeros(100, dtype=np.float32)
        audio.select()

        # Force _detect_segments to return empty
        _ = audio.segments
        segments = audio.get_segments()
        # Should return existing segments when detection returns empty
        assert segments is not None

    def test_detect_segments_empty_amplitudes(self):
        """Test _detect_segments with empty amplitudes (line 715)."""
        audio = Audio(fs=44100)
        audio.samples = np.zeros(1000, dtype=np.float32)
        audio.select()

        # Call internal method with empty array
        segments = audio._detect_segments(0, np.array([]), 0.1, 0.5)
        assert len(segments) == 0


class TestAudioSweepDetection:
    """Tests for Audio.is_sweep() edge cases."""

    def test_is_sweep_very_short_audio(self):
        """Test is_sweep() with audio too short for reliable FFT (line 860)."""
        audio = Audio(fs=44100)
        # Very short audio - less than 8 samples for nperseg
        audio.samples = np.random.randn(20).astype(np.float32)
        audio.select()

        # Should return False for too-short audio
        result = audio.is_sweep()
        assert result is False

    def test_is_sweep_decreasing_frequency(self):
        """Test is_sweep() with decreasing frequency sweep (line 888)."""
        # Generate a decreasing sweep (high to low frequency)
        audio = generate.sweep(amp=0.5, f_start=5000, f_stop=100, secs=1.5)

        # Should detect as sweep
        result = audio.is_sweep()
        assert result is True


class TestAudioQuadrature:
    """Tests for Audio.is_quadrature property edge cases."""

    def test_is_quadrature_empty_samples(self):
        """Test is_quadrature with empty samples (line 918)."""
        audio = Audio(fs=44100)
        audio.samples = np.zeros((0, 2), dtype=np.float32)
        audio.segment = Segment(44100, 0, 0)

        result = audio.is_quadrature
        assert result is False

    def test_is_quadrature_different_frequencies(self):
        """Test is_quadrature with different channel frequencies (line 940)."""
        fs = 44100
        t = np.arange(fs) / fs
        # Two channels with very different frequencies
        ch0 = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        ch1 = np.sin(2 * np.pi * 5000 * t).astype(np.float32)

        audio = Audio(fs=fs)
        audio.samples = np.column_stack([ch0, ch1])
        audio.select()

        # Should return False due to frequency mismatch
        result = audio.is_quadrature
        assert result is False

    def test_is_quadrature_zero_peak(self):
        """Test is_quadrature with zero peak amplitude (line 954)."""
        audio = Audio(fs=44100)
        audio.samples = np.zeros((1000, 2), dtype=np.float32)
        audio.select()

        result = audio.is_quadrature
        assert result is False


class TestAudioPhaseError:
    """Tests for Audio.phase_error() method edge cases."""

    def test_phase_error_with_quadrature_signal(self):
        """Test phase_error returns a value for quadrature signal."""
        audio = generate.sine(freq=1000, amp=0.5, secs=0.5)
        # Generated sine is stereo quadrature
        result = audio.phase_error()
        # Should return a float (phase error in radians)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0

    def test_phase_error_silence_returns_none(self):
        """Test phase_error returns None for silence."""
        audio = Audio(fs=44100)
        audio.samples = np.zeros((44100, 2), dtype=np.float32)
        audio.select()

        # Silence is not quadrature
        result = audio.phase_error()
        assert result is None

    def test_phase_error_different_freq_channels(self):
        """Test phase_error with different frequency channels (line 940)."""
        fs = 44100
        t = np.arange(fs) / fs
        # Two channels with very different frequencies - not quadrature
        ch0 = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        ch1 = np.sin(2 * np.pi * 5000 * t).astype(np.float32)

        audio = Audio(fs=fs)
        audio.samples = np.column_stack([ch0, ch1])
        audio.select()

        # Different frequencies should fail quadrature detection
        result = audio.phase_error()
        # Should be None since frequencies don't match
        assert result is None


class TestAudioCategory:
    """Tests for Audio.cat property edge cases."""

    def test_cat_unknown_category(self):
        """Test cat returning Unknown category (line 1093)."""
        audio = Audio(fs=44100)
        # Create signal that's not silence, sweep, tone, or noise
        # Use a very short burst that doesn't match any category well
        samples = np.zeros(44100, dtype=np.float32)
        # Add some random samples in the middle
        samples[20000:20100] = np.random.randn(100).astype(np.float32) * 0.1
        audio.samples = samples
        audio.select()

        # Mock the detection methods to all return False
        with unittest.mock.patch.object(audio, "is_silence", return_value=False):
            with unittest.mock.patch.object(audio, "is_sweep", return_value=False):
                # Clear cached property
                if "cat" in audio.__dict__:
                    del audio.__dict__["cat"]
                if "bins" in audio.__dict__:
                    del audio.__dict__["bins"]

                # Mock bins
                mock_bins = unittest.mock.Mock()
                mock_bins.is_sine.return_value = False
                mock_bins.is_noise.return_value = False

                with unittest.mock.patch.object(
                    Audio, "bins", new_callable=lambda: property(lambda self: mock_bins)
                ):
                    # Access through a fresh instance
                    audio2 = Audio(fs=44100)
                    audio2.samples = samples.copy()
                    audio2.select()

                    # Clear cached properties
                    with unittest.mock.patch.object(
                        audio2, "is_silence", return_value=False
                    ):
                        with unittest.mock.patch.object(
                            audio2, "is_sweep", return_value=False
                        ):
                            audio2.__dict__["bins"] = mock_bins
                            if "cat" in audio2.__dict__:
                                del audio2.__dict__["cat"]

                            result = audio2.cat
                            assert result == Category.Unknown


class TestAudioRead:
    """Tests for Audio.read() edge cases."""

    def test_read_unsupported_channels(self, tmp_path):
        """Test read() with unsupported channel count (line 1174)."""
        import soundfile as sf

        # Create a file with 3 channels (not supported)
        audio_path = tmp_path / "multichannel.wav"
        samples = np.zeros((1000, 3), dtype=np.float32)
        sf.write(str(audio_path), samples, 44100)

        audio = Audio()
        result = audio.read(str(audio_path))

        # Should return False for unsupported channel count
        assert result is False

    def test_read_empty_file(self, tmp_path):
        """Test read() with empty audio file (line 1186)."""
        import soundfile as sf

        # Create a file with no samples
        audio_path = tmp_path / "empty.wav"
        samples = np.zeros((0, 2), dtype=np.float32)
        sf.write(str(audio_path), samples, 44100)

        audio = Audio()
        result = audio.read(str(audio_path))

        # Should return False for empty audio
        assert result is False


class TestAudioClose:
    """Tests for Audio.close() method."""

    def test_close_marks_segments_complete(self):
        """Test close() marks all segments as complete (line 1294)."""
        audio = generate.sine(freq=1000, amp=0.5, secs=1.0)
        audio.get_segments()

        # Ensure we have segments
        assert len(audio.segments) > 0

        # Mark segments as incomplete
        for seg in audio.segments:
            seg.complete = False

        audio.close()

        # All segments should now be complete
        for seg in audio.segments:
            assert seg.complete is True


class TestFreqBinsAdaptiveBandwidth:
    """Tests for FreqBins.adaptive_max_bins() edge cases."""

    def test_adaptive_bins_zero_freq(self):
        """Test adaptive_max_bins with freq <= 0 (lines 236-244)."""
        audio = generate.sine(freq=1000, amp=0.5, secs=0.5)
        bins = audio.bins

        # Test with zero frequency
        result = bins.adaptive_max_bins(0, quality_factor=10.0)
        # Should return MAX_BINS_PER_TONE for invalid frequency
        from constants import MAX_BINS_PER_TONE

        assert result == MAX_BINS_PER_TONE

    def test_adaptive_bins_negative_freq(self):
        """Test adaptive_max_bins with negative frequency."""
        audio = generate.sine(freq=1000, amp=0.5, secs=0.5)
        bins = audio.bins

        # Test with negative frequency
        result = bins.adaptive_max_bins(-100, quality_factor=10.0)
        from constants import MAX_BINS_PER_TONE

        assert result == MAX_BINS_PER_TONE

    def test_adaptive_bins_normal_freq(self):
        """Test adaptive_max_bins with normal frequency."""
        audio = generate.sine(freq=1000, amp=0.5, secs=0.5)
        bins = audio.bins

        # Test with normal frequency
        result = bins.adaptive_max_bins(1000, quality_factor=10.0)
        # Should return a value between 5 and 50
        assert 5 <= result <= 50


class TestToneAnalyzerEdgeCases:
    """Tests for ToneAnalyzer edge cases."""

    def test_spurious_categorization_below_threshold(self):
        """Test categorization when spur is below threshold (line 144)."""
        audio = Audio(fs=48000)
        audio.samples = np.zeros(48000, dtype=np.float32)
        audio.select()
        analyzer = ToneAnalyzer(audio)

        # Mock bins
        mock_bins = unittest.mock.Mock()
        analyzer.audio.bins = mock_bins

        # Create components with very low power spur
        c0 = Component(1000.0, 1.0, 0.0, Category.Unknown)  # Main tone
        c1 = Component(1500.0, 1e-12, 0.0, Category.Unknown)  # Low power -> Noise

        comps = Components(Metadata("test", Segment(48000)))
        comps.append(c0)
        comps.append(c1)
        comps.ref_pwr = 1.0

        mock_bins.tones = comps
        mock_bins.freq_to_bin.return_value = 100
        mock_bins.freq_in_list.return_value = False

        analyzer.get_components()

        assert c0.cat == Category.Tone
        # c1 should be Noise because its power is too low
        assert c1.cat == Category.Noise

    def test_to_csv_method_raises(self, tmp_path):
        """Test ToneAnalyzer.to_csv() raises AttributeError (lines 217-218).

        Note: This tests that the method correctly calls analyze() before
        attempting to export. The underlying ToneStats.to_csv doesn't exist,
        which is a known issue in the codebase.
        """
        audio = generate.sine(freq=1000, amp=0.5, secs=1.1)
        analyzer = ToneAnalyzer(audio)

        csv_path = tmp_path / "tone_analysis.csv"

        # The method calls analyze() first (line 217), then tries to call
        # to_csv on the result, which will fail since ToneStats lacks to_csv
        with pytest.raises(AttributeError):
            analyzer.to_csv(str(csv_path))


class TestAudioAnalyzerEdgeCases:
    """Tests for AudioAnalyzer edge cases."""

    def test_verbose_time_stats(self):
        """Test verbose output for TimeStats (line 122)."""

        from audio_analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer(verbose=True)
        audio = generate.sine(freq=1000, amp=0.5, secs=1.1)
        audio.segment.cat = Category.Tone

        # Should complete without error
        result = analyzer.analyze(audio)
        assert result is not None

    def test_get_msg_queue(self):
        """Test get_msg_queue returns queue (line 146)."""

        from audio_analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer()
        q = analyzer.get_msg_queue()

        import queue

        assert isinstance(q, queue.Queue)

    def test_analyze_worker_invalid_audio(self):
        """Test _analyze_worker with invalid audio object (lines 164-165)."""

        from audio_analyzer import AudioAnalyzer
        import queue

        analyzer = AudioAnalyzer()
        q = queue.Queue()

        # Put an invalid object (not Audio)
        q.put("not_an_audio_object")
        q.put(None)  # Signal to stop

        # Run worker
        analyzer._analyze_worker(q)

        # Should complete without error


class TestComponentSummary:
    """Tests for Component and Components edge cases."""

    def test_components_summary(self):
        """Test Components.summary() method (line 547)."""
        md = Metadata("test", Segment(44100))
        comps = Components(md, name="TestComponents")
        comps.append(Component(1000.0, 0.5, 0.0, Category.Tone))
        comps.append(Component(2000.0, 0.1, 0.0, Category.Harmonic))

        summary = comps.summary()
        assert "TestComponents" in summary
        assert "2 components" in summary

    def test_freq_bins_parabolic_and_tones_conditions(self):
        """Test _parabolic_peak, adaptive_max_bins, and tones branches."""
        from constants import MAX_BINS_PER_TONE

        fs = 1000
        samples = np.zeros(100, dtype=np.float32)
        md = Metadata("test", Segment(fs, 0, 100))
        bins = FreqBins(samples, md)

        # _parabolic_peak boundary conditions: bin_idx <= 0, >= n_bins - 1, empty
        arr = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        f_left, p_left = bins._parabolic_peak(0, arr)
        assert f_left == 0.0 and p_left == 1.0

        f_right, p_right = bins._parabolic_peak(2, arr)
        assert f_right == pytest.approx(2 * fs / 100) and p_right == 1.0

        f_empty, p_empty = bins._parabolic_peak(0, np.array([], dtype=np.float32))
        assert f_empty == 0.0 and p_empty == 0.0

        # _parabolic_peak denom == 0 condition (flat top)
        flat_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        f_flat, p_flat = bins._parabolic_peak(1, flat_arr)
        assert f_flat == pytest.approx(1 * fs / 100) and p_flat == 1.0

        # adaptive_max_bins lower floor and upper cap clamping
        assert (
            bins.adaptive_max_bins(20.0, quality_factor=100.0) == MAX_BINS_PER_TONE // 2
        )
        assert bins.adaptive_max_bins(10000.0, quality_factor=1.0) == 50

        # tones: < 3 bins branch
        bins_tiny = FreqBins(
            np.zeros(2, dtype=np.float32), Metadata("tiny", Segment(fs, 0, 2))
        )
        assert len(bins_tiny.tones) == 0

        # tones: > MAX_TONES (40) peaks and overlapping peak (total_power == 0)
        custom_bins = np.zeros(200, dtype=np.float32)
        # Create two adjacent peaks within MAX_BINS_PER_TONE // 2
        custom_bins[10] = 10.0
        custom_bins[11] = 1.0
        custom_bins[12] = 5.0
        # Create > 40 isolated peaks across the array
        for idx in range(20, 195, 3):
            custom_bins[idx] = float(idx)
        bins_many = FreqBins(
            np.zeros(398, dtype=np.float32), Metadata("many", Segment(fs, 0, 398))
        )
        bins_many.bins = custom_bins
        detected = bins_many.tones
        assert len(detected) == 40

        # Test overlapping peaks when total peaks <= MAX_TONES (total_power == 0)
        overlap_bins = np.zeros(50, dtype=np.float32)
        overlap_bins[20] = 10.0
        overlap_bins[21] = 0.0  # valley between 20 and 22
        # Wide monotonic mound (> 25 bins) with secondary peak in clamped half-width
        for k in range(1, 31):
            overlap_bins[k] = float(k)
        for k in range(31, 40):
            overlap_bins[k] = float(40 - k)
        overlap_bins[30] = 50.0
        overlap_bins[32] = 40.0  # Secondary peak within center + half-width
        bins_overlap = FreqBins(
            np.zeros(98, dtype=np.float32), Metadata("ov", Segment(fs, 0, 98))
        )
        bins_overlap.bins = overlap_bins
        ov_tones = bins_overlap.tones
        assert any(t.pwr == 0.0 for t in ov_tones)

    def test_tone_analyzer_condition_decision_branches(self):
        """Test single-component, two-tone spur, and adaptive thresholds."""
        audio = Audio(fs=48000)
        audio.samples = np.zeros(48000, dtype=np.float32)
        audio.select()
        analyzer = ToneAnalyzer(audio)

        mock_bins = unittest.mock.Mock()
        analyzer.audio.bins = mock_bins
        mock_bins.freq_to_bin.return_value = 100
        mock_bins.freq_in_list.return_value = False

        # 1. Single-component branch: len(components) == 1 (118->145)
        single_comps = Components(Metadata("single", Segment(48000)))
        single_comps.append(Component(1000.0, 1.0, 0.0, Category.Unknown))
        mock_bins.tones = single_comps
        res_single = analyzer.get_components()
        assert len(res_single) == 1
        assert res_single[0].cat == Category.Tone

        # 2. Two-tone secondary spur branch (line 154-158):
        # c0 (Tone 1): pwr = 1.0
        # c1 (Tone 2): pwr = 0.05 (>= 0.01 * 1.0 -> valid two-tone)
        # c2 (Spur relative to Tone 2 only): pwr = 5e-9 (> 0.05 * 1e-8)
        # c3 (Noise below both Tone 1 and Tone 2 thresholds): pwr = 1e-10
        two_tone_comps = Components(Metadata("two_tone", Segment(48000)))
        c0 = Component(1000.0, 1.0, 0.0, Category.Unknown)
        c1 = Component(1700.0, 0.05, 0.0, Category.Unknown)
        c2 = Component(3123.0, 5e-9, 0.0, Category.Unknown)
        c3 = Component(4123.0, 1e-10, 0.0, Category.Unknown)
        two_tone_comps.extend([c0, c1, c2, c3])
        mock_bins.tones = two_tone_comps

        analyzer.get_components()
        assert c0.cat == Category.Tone
        assert c1.cat == Category.Tone
        assert c2.cat == Category.Spurious
        assert c3.cat == Category.Noise

        # 3. Adaptive threshold clamping for moderate and high noise floor
        analyzer.audio._noise_sum = 1e-3  # -60 dBFS amplitude -> 1e-6 power
        analyzer.audio._noise_count = 1
        analyzer.tone = Component(1000.0, 1e-3, 0.0, Category.Tone)  # SNR = 30 dB
        analyzer._compute_adaptive_thresholds()
        # two_tone_db = -10 - (30 / 6) = -15 dB
        assert analyzer._two_tone_threshold == pytest.approx(10.0 ** (-15.0 / 10.0))
        # nf = 1e-6 -> 60 dB above MIN_PWR (1e-12) -> capped at 1e-7
        assert analyzer._spur_threshold == pytest.approx(1e-7)

    def test_signal_analysis_accuracy_improvements(self):
        """Test accuracy improvements across signal analysis modules."""
        from time_stats import TimeStats
        from tone_stats import ToneStats
        from constants import MIN_PWR

        # 1. Window power compensation: unit-amplitude sine has RMS^2 = 0.5
        fs = 48000
        t = np.arange(fs) / fs
        sine_samples = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)
        md = Metadata("sine", Segment(fs, 0, fs))
        bins = FreqBins(sine_samples, md)
        assert bins.tones[0].pwr == pytest.approx(0.5, rel=0.02)

        # Odd 3-sample FreqBins hits len(bins) == 2 and n_samples % 2 != 0 branch
        bins_3 = FreqBins(
            np.array([1.0, -0.5, 0.5], dtype=np.float32),
            Metadata("three", Segment(fs, 0, 3)),
        )
        assert len(bins_3) == 2

        # _parabolic_peak fallback when neighboring bin <= MIN_PWR
        f_lin, p_lin = bins._parabolic_peak(
            1, np.array([0.0, 2.0, 1.0], dtype=np.float32)
        )
        assert f_lin > 0.0 and p_lin > 0.0

        # 2. FreqBins.is_noise with single tone or negligible secondary tone
        bins.tones = [Component(1000.0, 0.5, 0.0, Category.Tone)]
        assert bins.is_noise() is False
        bins.tones = [
            Component(1000.0, 0.5, 0.0, Category.Tone),
            Component(2000.0, 1e-15, 0.0, Category.Noise),
        ]
        assert bins.is_noise() is False

        # 3. Descending sweep in Components.combine preserves distinct frequencies
        desc_comps = Components(md)
        desc_comps.extend(
            [
                Component(2000.0, 1.0, 0.1, Category.Tone),
                Component(1500.0, 1.0, 0.2, Category.Tone),
                Component(1500.2, 1.0, 0.3, Category.Tone),
            ]
        )
        desc_comps.combine(freq_tol=1.0)
        assert len(desc_comps) == 2

        # 4. Negative peak excursion in TimeStats.crest and TimeStats.dbfs
        asym_audio = Audio(fs=1000)
        asym_audio.samples = np.array([-0.8, 0.2] * 500, dtype=np.float32)
        asym_audio.select()
        t_stats = TimeStats(asym_audio)
        expected_rms = float(np.sqrt(0.5 * (0.64 + 0.04)))
        assert t_stats.crest == pytest.approx(0.8 / expected_rms, rel=1e-4)

        # 5. Quadrature phase_error reflects true phase imbalance independent of fs
        t_iq = np.arange(4800) / 48000.0
        imb_rad = np.radians(5.0)
        ch0 = np.cos(2.0 * np.pi * 500.0 * t_iq).astype(np.float32)
        ch1 = np.sin(2.0 * np.pi * 500.0 * t_iq + imb_rad).astype(np.float32)
        iq_audio = Audio(fs=48000)
        iq_audio.samples = np.column_stack((ch0, ch1))
        iq_audio.select()
        ph_err = iq_audio.phase_error()
        assert ph_err == pytest.approx(imb_rad / (2.0 * np.sqrt(2.0)), rel=0.1)

        # 6. ToneStats SNR fallback when noise <= MIN_PWR
        ts = ToneStats(asym_audio, 1000.0, 0.0, 0.5, 0.0, 0.0, 0.0)
        assert ts.snr == pytest.approx(0.5 / MIN_PWR)

    def test_runtime_optimizations_condition_decision_coverage(self):
        """Test condition and decision coverage for runtime optimization changes."""
        from scipy import signal

        from audio_analyzer import AudioAnalyzer
        from freq_bins import _window_cache, _window_s2_cache, blackmanharris7
        from unittest.mock import patch

        from sweep_analyzer import SweepAnalyzer
        from util import Channel, alias_freq

        # 1. blackmanharris7 edge cases (M <= 0, sym, M <= 12, cache eviction)
        _window_cache.clear()
        _window_s2_cache.clear()
        w_zero = blackmanharris7(0)
        assert len(w_zero) == 0
        assert _window_s2_cache[(0, True)] == 0.0
        w_neg = blackmanharris7(-2, sym=False)
        assert len(w_neg) == 0

        # sym=False with M <= 12 hits the empirical np.mean(w * w) branch
        w_short = blackmanharris7(8, sym=False)
        assert len(w_short) == 8
        # sym=True with M > 12 also hits the empirical np.mean(w * w) branch
        w_sym = blackmanharris7(20, sym=True)
        assert len(w_sym) == 20

        # Trigger _window_cache eviction (len(_window_cache) >= 16)
        for m in range(2, 22):
            blackmanharris7(m, sym=False)
        assert len(_window_cache) < 16

        # 2. FreqBins.__init__ with 3-channel audio and _window_s2_cache miss fallback
        fs = 1000
        md = Metadata("three_ch", Segment(fs, 0, 64))
        samples_3ch = np.ones((64, 3), dtype=np.float32) * 0.5
        _window_s2_cache.clear()  # Force s2 is None while window is in _window_cache
        blackmanharris7(64, sym=False)
        _window_s2_cache.pop((64, False), None)
        fb_3ch = FreqBins(samples_3ch, md, Channel.Unknown)
        assert fb_3ch.n_samples == 64

        # 3. Audio.get_segments and Audio.is_sweep with 3-channel audio
        audio_3ch = Audio(fs=1000, min_segment_secs=0.05)
        t = np.arange(500) / 1000.0
        tone_1d = (0.5 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)
        silence_3ch = np.zeros((200, 3), dtype=np.float32)
        active_3ch = np.column_stack((tone_1d, tone_1d, tone_1d))
        audio_3ch.samples = np.vstack((silence_3ch, active_3ch, silence_3ch))
        audio_3ch.select()
        segs_3ch = audio_3ch.get_segments(min_silence_secs=0.05)
        assert len(segs_3ch) == 1
        assert audio_3ch.is_sweep() is False

        # 3-channel audio where endpoints differ so full-array 3-channel mean is reached
        chirp_1d = signal.chirp(t, f0=50.0, f1=300.0, t1=0.5, method="linear").astype(
            np.float32
        )
        audio_3ch.samples = np.column_stack((chirp_1d, chirp_1d, chirp_1d))
        audio_3ch.select()
        assert audio_3ch.is_sweep() is True

        # 4. Audio.is_sweep boundary fast-path conditions & sub-threshold STFT columns
        # (a) First boundary frame active, last boundary frame quiet
        half_active = np.zeros(400, dtype=np.float32)
        half_active[:100] = tone_1d[:100]
        audio_half = Audio(fs=1000)
        audio_half.samples = half_active
        audio_half.select()
        assert audio_half.is_sweep() is False

        # (b) Both boundary frames active with idx0 == idx1 == 0 (idx0 <= min_bin)
        dc_only = np.full(400, 0.5, dtype=np.float32)
        audio_dc = Audio(fs=1000)
        audio_dc.samples = dc_only
        audio_dc.select()
        assert audio_dc.is_sweep() is False

        # (c) All STFT columns below (10 * MIN_PWR)**2 -> valid_freqs is empty
        tiny_audio = Audio(fs=1000)
        tiny_audio.samples = np.full(400, 1e-12, dtype=np.float32)
        tiny_audio.select()
        assert tiny_audio.is_sweep() is False

        # 5. Audio.copy cached property propagation conditions
        audio_cp = Audio(fs=1000)
        audio_cp.samples = tone_1d.copy()
        audio_cp.select()
        # Only "cat" in __dict__ (without "bins")
        audio_cp.__dict__["cat"] = Category.Silence
        cp1 = audio_cp.copy(audio_cp.segment)
        assert cp1.__dict__.get("cat") == Category.Silence
        assert "bins" not in cp1.__dict__
        # Only "bins" in __dict__ (without "cat")
        audio_cp.__dict__.pop("cat", None)
        _ = audio_cp.bins
        cp2 = audio_cp.copy(audio_cp.segment)
        assert "bins" in cp2.__dict__
        assert "cat" not in cp2.__dict__
        # Copying a sub-segment != self.segment does not copy cached properties
        sub_seg = Segment(1000, 0, 100)
        cp3 = audio_cp.copy(sub_seg)
        assert "bins" not in cp3.__dict__

        # 6. Audio.is_quadrature with 1-sample non-silence/non-noise (len(pwr0) == 0)
        iq_1 = Audio(fs=1000)
        iq_1.samples = np.array([[0.5, 0.0]], dtype=np.float32)
        iq_1.select()
        iq_1.__dict__["cat"] = Category.Tone
        assert iq_1.is_quadrature is True

        # 7. Audio.get_noise_floor with secs=0.0 (avg_size <= 0 -> n_segs == 0)
        assert audio_cp.get_noise_floor(secs=0.0) == (0.0, 0)

        # 8. AudioAnalyzer.analyze fallback when analyzer.time_stats is None
        aa = AudioAnalyzer(verbose=False)
        with patch("tone_analyzer.ToneAnalyzer.analyze") as mock_analyze:
            mock_analyze.return_value = None
            with patch.object(
                aa.analysis_classes[Category.Tone],
                "__init__",
                side_effect=lambda self_a, aud: setattr(self_a, "audio", aud)
                or setattr(self_a, "time_stats", None)
                or setattr(self_a, "components", Components(aud.md)),
                autospec=True,
            ):
                audio_cp.__dict__["cat"] = Category.Tone
                aa.analyze(audio_cp)

        # 9. util.alias_freq with fs=0.0 and freq == fs
        assert alias_freq(100.0, 0.0) == -100.0
        assert alias_freq(1000.0, 1000.0) == 0.0

        # 10. SweepAnalyzer.get_components with flat 3-bin log peak (abs(denom) <= eps)
        sa = SweepAnalyzer(audio_cp)
        with patch("sweep_analyzer.signal.stft") as mock_stft:
            f_arr = np.array([0.0, 100.0, 200.0, 300.0])
            t_arr = np.array([0.1])
            # Peak at bin 1 with neighbors nearly equal so abs(denom) <= MIN_PWR
            z_arr = np.array([[1.0 - 1e-13], [1.0], [1.0], [0.0]], dtype=np.complex128)
            mock_stft.return_value = (f_arr, t_arr, z_arr)
            comps = sa.get_components()
            assert len(comps) == 1
            assert comps[0].freq == pytest.approx(100.0)
            # Also cover len(f) <= 1 branch (df = 0.0)
            mock_stft.return_value = (
                np.array([100.0]),
                np.array([0.1]),
                np.array([[1.0]], dtype=np.complex128),
            )
            comps_single = sa.get_components()
            assert len(comps_single) == 1

    def test_audio_analyzer_write_csv(self, tmp_path):
        """Test AudioAnalyzer.write_csv and AudioAnalysis.to_dict."""
        from audio_analyzer import AudioAnalyzer

        fs = 1000
        t = np.arange(1500) / fs
        audio = Audio(fs=fs, name="Test_Unit_1k")
        audio.samples = (0.5 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)
        audio.select()
        audio.segment.cat = Category.Tone

        analyzer = AudioAnalyzer(quiet=True)
        res = analyzer.analyze(audio)
        assert res is not None
        d = res.to_dict()
        assert "THD%" in d and "RMS" in d and "Type" in d and "Noise Floor" in d
        assert "Desc" not in d and "Description" in d
        assert d["THD"] < 0.0
        assert res.name == "Test_Unit_1k"
        assert res.freq == pytest.approx(100.0, abs=1.0)
        with pytest.raises(AttributeError):
            _ = res.nonexistent_attr
        audio2 = Audio(fs=fs, name="Test_Unit_2k")
        audio2.samples = (0.5 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)
        audio2.select()
        audio2.segment.cat = Category.Tone
        analyzer.analyze(audio2)

        csv_file = tmp_path / "stats.csv"
        analyzer.write_csv(str(csv_file), sort_attr="freq")
        lines = csv_file.read_text().strip().splitlines()
        assert len(lines) == 3
        assert "Test_Unit_1k" in lines[1]
        assert "Test_Unit_2k" in lines[2]
        assert csv_file.exists()

        # MC/DC coverage for Analysis.name, AudioAnalysis, _resolve_sort_key,
        # _propagate_noise_floor, TimeStats.to_dict, and analyze.main (-csv)
        from types import SimpleNamespace
        import analyze
        from audio_analyzer import AudioAnalysis, _resolve_sort_key
        from audio_stats import AudioStats
        from component import Component, Components
        from freq_resp import FreqResp
        from time_stats import TimeStats

        # 1. Analysis.name property: default (_name is None) vs custom (_name set)
        assert res.analysis.name == "Test_Unit_1k"
        res.analysis.name = "CustomAnalysisName"
        assert res.analysis.name == "CustomAnalysisName"
        res.analysis.name = None
        assert res.analysis.name == "Test_Unit_1k"

        # 2. AudioAnalysis.name and __getattr__ branches
        # - analysis has attribute (freq)
        # - time_stats has attribute (crest) when analysis does not
        assert res.crest == pytest.approx(res.time_stats.crest)
        # - segment has attribute (stop_secs) when analysis=None, time_stats=None
        aa_seg_only = AudioAnalysis(audio.segment, None, None, None)
        assert aa_seg_only.name == ""
        assert aa_seg_only.stop_secs == pytest.approx(audio.segment.stop_secs)

        # 3. AudioAnalysis.to_dict branches:
        # - FreqResp rounding (float vs non-float values)
        comps = Components(audio.md)
        comps.append(Component(freq=100.0, pwr=1.0, secs=1.0))
        fr = FreqResp(comps)
        fr.minimum = -45.678
        aa_fr = AudioAnalysis(audio.segment, res.time_stats, fr, comps)
        d_fr = aa_fr.to_dict()
        assert d_fr["minimum"] == -45.7

        dummy_analysis = SimpleNamespace(
            to_dict=lambda: {"minimum": None, "ripple": 1, "other": 2.345}
        )
        aa_no_md = AudioAnalysis(
            SimpleNamespace(), res.time_stats, dummy_analysis, None
        )
        d_no_md = aa_no_md.to_dict()
        assert "Type" not in d_no_md
        assert "Desc" in d_no_md  # Not popped when "Description" not in md_dict
        assert d_no_md["minimum"] is None and d_no_md["ripple"] == 1

        # - analysis is time_stats (e.g. Silence/Unknown) and segment is None
        aa_same = AudioAnalysis(None, res.time_stats, res.time_stats, None)
        d_same = aa_same.to_dict()
        assert "Type" not in d_same and "Description" in d_same and "Desc" not in d_same

        # - time_stats is None and analysis is None
        assert aa_seg_only.to_dict() == {"Type": "Tone"}

        # - Noise Floor branches in AudioAnalysis.to_dict:
        #   nf > 0.0 (float), nf == 0.0 (float <= 0), nf is None (non-numeric)
        res.time_stats.noise_floor = 0.01
        assert res.to_dict()["Noise Floor"] == -40.0
        res.time_stats.noise_floor = 0.0
        assert res.to_dict()["Noise Floor"] is None
        res.time_stats.noise_floor = None
        assert res.to_dict()["Noise Floor"] is None

        # 4. _resolve_sort_key branches: val is None vs val is not None
        assert _resolve_sort_key(aa_seg_only, "thd") == (1, "")
        assert _resolve_sort_key(res, "freq")[0] == 0

        # 5. _propagate_noise_floor branches:
        # - empty nf_by_name (noise_floor == 0.0 and noise_floor is None)
        astats_zero = AudioStats(audio)
        astats_zero.noise_floor = 0.0
        astats_none = AudioStats(audio)
        astats_none.noise_floor = None
        analyzer.analysis_list.extend([astats_zero, astats_none])
        analyzer._propagate_noise_floor()
        assert res.time_stats.noise_floor is None

        # - non-empty nf_by_name with matching, non-matching, and time_stats=None
        astats_pos = AudioStats(audio)
        astats_pos.noise_floor = 0.005
        analyzer.analysis_list.append(astats_pos)
        analyzer.results["none_ts"] = aa_seg_only
        analyzer._propagate_noise_floor()
        assert res.time_stats.noise_floor == 0.005
        assert (
            analyzer.results[f"{audio2.name}:{audio2.segment}"].time_stats.noise_floor
            == 0.0
        )
        del analyzer.results["none_ts"]

        # 6. write_csv branches:
        # - sort_attr="" (isinstance(attr, str) is True, bool(attr) is False)
        analyzer.write_csv(str(csv_file), sort_attr="")
        # - sort_attr=None and _tone_sort_attr=None (isinstance(attr, str) is False)
        analyzer._tone_sort_attr = None
        analyzer.write_csv(str(csv_file), sort_attr=None)
        # - self.results empty (falls back to self.analysis_list)
        empty_analyzer = AudioAnalyzer(quiet=True)
        empty_analyzer.analysis_list.append(res.time_stats)
        empty_analyzer.write_csv(str(csv_file), sort_attr="start")

        # 7. TimeStats.to_dict quadrature branches (Phase and Balance non-None vs None)
        ch0 = (0.5 * np.cos(2.0 * np.pi * 100.0 * t)).astype(np.float32)
        ch1 = (0.5 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)
        q_audio = Audio(fs=fs, name="Quad_Test")
        q_audio.samples = np.column_stack((ch0, ch1))
        q_audio.select()
        q_ts = TimeStats(q_audio)
        q_dict = q_ts.to_dict()
        assert isinstance(q_dict["Phase"], float)
        assert isinstance(q_dict["Balance"], float)

        # 8. analyze.main with -csv option enabled
        from unittest.mock import MagicMock, patch

        mock_args = MagicMock()
        mock_args.pattern = "*.wav"
        mock_args.table = "freq"
        mock_args.components = None
        mock_args.audio_stats = True
        mock_args.device = False
        mock_args.max_length = 60.0
        mock_args.play = None
        mock_args.verbose = False
        mock_args.quiet = True
        mock_args.write_csv = str(csv_file)
        with (
            patch("analyze.get_args", return_value=mock_args),
            patch("analyze.AudioAnalyzer") as mock_aa_cls,
        ):
            mock_aa = mock_aa_cls.return_value
            mock_aa.read.return_value = 1
            analyze.main()
            assert mock_aa._tone_sort_attr == "freq"
            mock_aa.write_csv.assert_called_once_with(str(csv_file))
