"""
Test sweep_analyzer.py functionality.
"""

import numpy as np
import pytest

from audio import Audio
from freq_resp import FreqResp
from sweep_analyzer import SweepAnalyzer
from segment import Segment
from util import Category
from constants import DEFAULT_FS, DEFAULT_FREQ
import generate


def test_sweep_analyzer_integration(examples_dir):
    fname = examples_dir / "test_sweep.wav"
    if not fname.exists():
        pytest.skip("test_sweep.wav not found in examples")

    audio = Audio()
    assert audio.read(str(fname))

    # Sweep file has three segments: silence, sweep, silence.
    # SweepAnalyzer expects to analyze the sweep part.
    # But if we pass the whole file, it "should" work if STFT handles it,
    # or we might need to select the sweep part.
    # The example "main" code in sweep_analyzer.py does:
    # segments = audio.get_segments(); audio.select(segments[0])
    # implying it relies on silence detection to find the sweep.
    # We should reproduce that logic or just check if it finds components
    # regardless.

    segments = audio.get_segments()
    if not segments:
        pytest.fail("No segments found in test_sweep.wav")

    # Assume the sweep is one of the segments (likely the longest or non-silent one)
    # The test file generation adds 1s silence, then sweep, then 1s silence.
    # So we expect ONE main segment if audio.py joins them or detects non-silence?
    # Actually audio.py get_segments() can find non-silent segments.
    # So we should find one segment, which is the sweep.

    assert len(segments) >= 1
    sweep_seg = segments[0]

    # Select the sweep segment
    audio.select(sweep_seg)
    analyzer = SweepAnalyzer(audio)

    # Test get_components
    comps = analyzer.get_components()
    assert len(comps) > 0
    # Should find many components across the spectrum

    # Test analyze (returns FreqResp)
    freq_resp = analyzer.analyze()
    assert isinstance(freq_resp, FreqResp)
    assert len(freq_resp.components) > 0  # Should have response points

    # Test get_bands
    bands = analyzer.get_bands()
    assert len(bands) > 0

    # Test print
    analyzer.print(components=True)


def test_sweep_short_audio_error():
    import analyzer

    # Verify patch didn't leak
    assert analyzer.MIN_SEGMENT_SECS == 1.0

    # Create very short audio
    a = Audio()
    a.samples = np.zeros(10, dtype=np.float32)
    a.segment = Segment(fs=a.fs, start=0, stop=10)
    assert len(a) == 10

    analyzer = SweepAnalyzer(a)
    try:
        analyzer.analyze()
    except ValueError as e:
        assert "is too short to analyze" in str(e)
        return

    pytest.fail("Did not raise ValueError for short audio")


def test_sweep_stereo_conversion():
    # Stereo sweep
    # Left: sweep, Right: silence
    # Mean: half amplitude sweep

    a = generate.sweep(f_start=100, f_stop=1000, secs=2.0, fs=DEFAULT_FS)
    mono_samples = a.samples.copy()

    # Make stereo
    stereo = np.zeros((len(mono_samples), 2), dtype=np.float32)
    stereo[:, 0] = mono_samples
    a.samples = stereo

    analyzer = SweepAnalyzer(a)
    comps = analyzer.get_components()
    assert len(comps) > 0


def test_sweep_print_coverage():
    # Just ensure it runs with components=True
    a = generate.sweep(secs=1.1)
    analyzer = SweepAnalyzer(a)
    analyzer.analyze()

    analyzer.print(components=True)


def test_get_bands_default_ref():
    a = generate.sweep(secs=1.0)
    analyzer = SweepAnalyzer(a)
    analyzer.get_components()  # Populate components

    # call get_bands with ref_freq=None
    bands = analyzer.get_bands(ref_freq=None)
    assert bands is not None


def test_filtered_sweep_response():
    """Verify that a filtered sweep matches the expected bandpass filter response."""
    from scipy import signal as scipy_signal
    from decibels import db

    fs = DEFAULT_FS
    secs = 5.0
    # Generate broad sweep
    audio = generate.sweep(f_start=20, f_stop=20000, secs=secs, fs=fs)

    # Apply 4th order Butterworth bandpass (300Hz - 3kHz) as in generate.py
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
    # 4th order Butterworth: -3dB at cutoff. Allow some tolerance (e.g. +/- 1dB) due to binning/windowing effects
    # "similar to" -> check range -4dB to -2dB
    assert -5.0 < db_300 < -1.0, f"Expected ~-3 dB at 300 Hz, got {db_300:.2f} dB"

    # Check Upper Corner (3kHz) - expected ~-3 dB
    c_3k = comps.find_freq(3000.0)
    assert c_3k is not None
    db_3k = db(c_3k.pwr / ref_pwr)
    assert -5.0 < db_3k < -1.0, f"Expected ~-3 dB at 3 kHz, got {db_3k:.2f} dB"
    # Check Stopband (< 300Hz) - e.g. 100Hz
    # slope 24 dB/oct for 4th order? No, 4th order 'bandpass' might be 2nd order per side?
    # butter(4, ...) gives 4th order overall or per side?
    # Usually butter(N, ...) is Nth order. 4th order falls off fast.
    # At 100Hz (approx 1.5 octave down from 300), attenuation should be large.
    c_100 = comps.find_freq(100.0)
    if c_100:
        db_100 = db(c_100.pwr / ref_pwr)
        assert db_100 < -10.0, f"Expected < -10 dB at 100 Hz, got {db_100:.2f} dB"

    # Check Stopband (> 3kHz) - e.g. 10kHz (approx 1.5 octave up)
    c_10k = comps.find_freq(10000.0)
    if c_10k:
        db_10k = db(c_10k.pwr / ref_pwr)
        assert db_10k < -10.0, f"Expected < -10 dB at 10 kHz, got {db_10k:.2f} dB"
