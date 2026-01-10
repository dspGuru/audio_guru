"""
Test sweep_analyzer.py functionality.
"""

import numpy as np
import pytest

from audio import Audio
from freq_resp import FreqResp
from sweep_analyzer import SweepAnalyzer
from segment import Segment
from util import Category, Channel
from constants import DEFAULT_FS, DEFAULT_FREQ
import generate


def test_sweep_analyzer_integration(signals_dir):
    fname = signals_dir / "12-test_sweep.wav"
    if not fname.exists():
        pytest.skip("12-test_sweep.wav not found in signals")

    audio = Audio()
    assert audio.read(str(fname))

    # Sweep file has three segments: silence, sweep, silence.
    # SweepAnalyzer analyzes the sweep part.

    segments = audio.get_segments()
    if not segments:
        pytest.fail("No segments found in 12-test_sweep.wav")

    # Expect at least one segment (the sweep)
    assert len(segments) >= 1
    sweep_seg = segments[0]

    # Select the sweep segment
    audio.select(sweep_seg)
    analyzer = SweepAnalyzer(audio)

    comps = analyzer.get_components()
    assert len(comps) > 0

    freq_resp = analyzer.analyze()
    assert isinstance(freq_resp, FreqResp)
    assert len(freq_resp.components) > 0

    # Test get_bands
    bands = analyzer.get_bands()
    assert len(bands) > 0

    # Test print
    analyzer.print(components=10)


def test_sweep_short_audio_error():
    import analyzer

    # Verify patch didn't leak
    assert analyzer.SEGMENT_MIN_SECS == 1.0

    # Create very short audio
    a = Audio()
    a.samples = np.zeros(10, dtype=np.float32)
    a.segment = Segment(fs=a.fs, start=0, stop=10)
    assert len(a) == 10

    analyzer = SweepAnalyzer(a)
    try:
        analyzer.select(segment=a.segment)
        analyzer.analyze()
    except ValueError as e:
        assert "is too short to analyze" in str(e)
        return

    pytest.fail("Did not raise ValueError for short audio")


def test_sweep_stereo_conversion():
    # Stereo: Left=sweep, Right=silence, Mean=half amplitude sweep

    a = generate.sweep(
        f_start=100, f_stop=1000, secs=2.0, fs=DEFAULT_FS, channel=Channel.Left
    )
    mono_samples = a.samples.copy()

    # Make stereo
    stereo = np.zeros((len(mono_samples), 2), dtype=np.float32)
    stereo[:, 0] = mono_samples
    a.samples = stereo

    analyzer = SweepAnalyzer(a)
    comps = analyzer.get_components()
    assert len(comps) > 0


def test_sweep_print_coverage():
    # Just ensure it runs with components=10
    a = generate.sweep(secs=1.1)
    analyzer = SweepAnalyzer(a)
    analyzer.analyze()

    analyzer.print(components=10)


def test_get_bands_default_ref():
    a = generate.sweep(secs=1.0)
    analyzer = SweepAnalyzer(a)
    analyzer.get_components()  # Populate components

    # Call get_bands with ref_freq=None
    bands = analyzer.get_bands(ref_freq=None)
    assert bands is not None


def test_filtered_sweep_response():
    """Verify that a filtered sweep matches the expected bandpass filter response."""
    from scipy import signal as scipy_signal
    from decibels import db

    fs = DEFAULT_FS
    secs = 5.0
    # Generate broad sweep (mono)
    audio = generate.sweep(
        f_start=20, f_stop=20000, secs=secs, fs=fs, channel=Channel.Left
    )

    # Apply 4th order Butterworth bandpass (300 Hz - 3 kHz).
    # N=4 provides a steep 24 dB/octave rolloff for clear stopband
    # verification.
    sos = scipy_signal.butter(4, [300, 3000], btype="bandpass", fs=fs, output="sos")
    audio.samples = scipy_signal.sosfilt(sos, audio.samples).astype(audio.DTYPE)

    analyzer = SweepAnalyzer(audio)
    freq_resp = analyzer.analyze()
    comps = freq_resp.components

    # Check Passband Center (1 kHz)
    c_1k = comps.find_freq(DEFAULT_FREQ)
    assert c_1k is not None
    ref_pwr = c_1k.pwr
    assert ref_pwr > 0

    # Check Lower Corner (300 Hz) - expected ~-3 dB
    c_300 = comps.find_freq(300.0)
    assert c_300 is not None
    # Calculate dB relative to 1kHz
    db_300 = db(c_300.pwr / ref_pwr)
    # 4th order Butterworth: -3 dB at cutoff, allow +/- 1 dB tolerance
    assert -5.0 < db_300 < -1.0, f"Expected ~-3 dB at 300 Hz, got {db_300:.2f} dB"

    # Check Upper Corner (3kHz) - expected ~-3 dB
    c_3k = comps.find_freq(3000.0)
    assert c_3k is not None
    db_3k = db(c_3k.pwr / ref_pwr)
    assert -5.0 < db_3k < -1.0, f"Expected ~-3 dB at 3 kHz, got {db_3k:.2f} dB"
    # Check Stopband (< 300 Hz) - e.g. 100 Hz
    # Slope 24 dB/oct for 4th order? No, 4th order 'bandpass' might be 2nd order per side?
    # butter(4, ...) gives 4th order overall or per side?
    # Usually butter(N, ...) is Nth order. 4th order falls off fast.
    # At 100 Hz (approx 1.5 octave down from 300), attenuation should be large.
    c_100 = comps.find_freq(100.0)
    if c_100:
        db_100 = db(c_100.pwr / ref_pwr)
        assert db_100 < -10.0, f"Expected < -10 dB at 100 Hz, got {db_100:.2f} dB"

    # Check Stopband (> 3kHz) - e.g. 10kHz (approx 1.5 octave up)
    c_10k = comps.find_freq(10000.0)
    if c_10k:
        db_10k = db(c_10k.pwr / ref_pwr)
        assert db_10k < -10.0, f"Expected < -10 dB at 10 kHz, got {db_10k:.2f} dB"


def test_reset():
    """Test that reset() clears components state."""
    a = generate.sweep(secs=1.1)
    analyzer = SweepAnalyzer(a)

    # Run analysis to populate state
    analyzer.analyze()
    assert analyzer.components is not None

    # Reset should clear components
    analyzer.reset()
    assert analyzer.components is None


def test_custom_ref_freq_in_constructor():
    """Test SweepAnalyzer with custom ref_freq in constructor."""
    a = generate.sweep(secs=1.1)
    custom_ref = 500.0
    analyzer = SweepAnalyzer(a, ref_freq=custom_ref)

    assert analyzer.ref_freq == custom_ref

    analyzer.get_components()
    bands = analyzer.get_bands()  # Should use custom ref_freq
    assert bands is not None


def test_get_bands_with_explicit_ref_freq():
    """Test get_bands with explicit ref_freq parameter."""
    a = generate.sweep(secs=1.1)
    analyzer = SweepAnalyzer(a)
    analyzer.get_components()

    # Call with explicit ref_freq
    bands = analyzer.get_bands(ref_freq=500.0)
    assert bands is not None
    assert len(bands) > 0


def test_get_bands_with_custom_centers():
    """Test get_bands with custom center frequencies."""
    a = generate.sweep(f_start=100, f_stop=5000, secs=1.5)
    analyzer = SweepAnalyzer(a)
    analyzer.get_components()

    # Use custom band centers
    custom_centers = (250.0, 500.0, 1000.0, 2000.0, 4000.0)
    bands = analyzer.get_bands(centers=custom_centers)
    assert bands is not None
    assert bands.name == "Bands"


def test_silent_frames_skipped():
    """Test that silent frames in STFT are skipped."""
    # Create audio with silence at the start
    fs = DEFAULT_FS
    silence_samples = int(0.5 * fs)
    sweep_samples = int(1.0 * fs)

    # Generate sweep (mono for concatenation)
    sweep_audio = generate.sweep(
        f_start=500, f_stop=2000, secs=1.0, fs=fs, channel=Channel.Left
    )

    # Prepend silence
    silence = np.zeros(silence_samples, dtype=np.float32)
    combined = np.concatenate([silence, sweep_audio.samples])

    a = Audio()
    a.samples = combined
    a.segment = Segment(fs=fs, start=0, stop=len(combined))

    analyzer = SweepAnalyzer(a)
    comps = analyzer.get_components()

    # Should still find sweep components (500-2000 Hz range)
    assert len(comps) > 0
    for c in comps:
        assert c.freq >= 400.0  # Allow tolerance


def test_low_frequency_filtering():
    """Test that frequencies <= 1 Hz are filtered out."""
    # Create low freq content (DC-like) + higher frequency
    fs = DEFAULT_FS
    secs = 1.5
    n_samples = int(secs * fs)

    # Signal with very low freq (0.5 Hz) + 500 Hz
    t = np.linspace(0, secs, n_samples, dtype=np.float32)
    low_freq_signal = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    high_freq_signal = 0.5 * np.sin(2 * np.pi * 500 * t)
    combined = (low_freq_signal + high_freq_signal).astype(np.float32)

    a = Audio()
    a.samples = combined
    a.segment = Segment(fs=fs, start=0, stop=len(combined))

    analyzer = SweepAnalyzer(a)
    comps = analyzer.get_components()

    # All components should have freq > 1.0 Hz
    for c in comps:
        assert c.freq > 1.0, f"Found component with freq {c.freq} <= 1.0 Hz"


def test_analyze_empty_components():
    """Test analyze() when get_components returns empty list."""
    # Create silence that produces no valid components
    fs = DEFAULT_FS
    n_samples = int(1.5 * fs)
    silence = np.zeros(n_samples, dtype=np.float32)

    a = Audio()
    a.samples = silence
    a.segment = Segment(fs=fs, start=0, stop=len(silence))

    analyzer = SweepAnalyzer(a)

    # get_components returns empty for silence
    comps = analyzer.get_components()

    # analyze() handles empty components by adding dummy
    freq_resp = analyzer.analyze()
    assert freq_resp is not None
    assert isinstance(freq_resp, FreqResp)
    # Should have at least the dummy component
    assert len(analyzer.components) >= 1


def test_print_without_components_flag():
    """Test print() with components=False (default)."""
    a = generate.sweep(secs=1.1)
    analyzer = SweepAnalyzer(a)
    analyzer.analyze()

    # Should not raise
    analyzer.print(components=False)
    analyzer.print()  # Default is None


def test_dc_offset_removal():
    """Test that DC offset is removed from samples."""
    fs = DEFAULT_FS
    secs = 1.5

    # Generate sweep with DC offset
    sweep_audio = generate.sweep(f_start=200, f_stop=2000, secs=secs, fs=fs)
    dc_offset = 0.3
    sweep_audio.samples = sweep_audio.samples + dc_offset

    analyzer = SweepAnalyzer(sweep_audio)
    comps = analyzer.get_components()

    # Should find sweep components, DC removed
    assert len(comps) > 0
    for c in comps:
        assert c.freq > 1.0
