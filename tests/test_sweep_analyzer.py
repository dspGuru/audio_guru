"""
Test sweep_analyzer.py functionality.
"""

import numpy as np
import pytest

from audio import Audio
from freq_resp import FreqResp
from sweep_analyzer import SweepAnalyzer
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
    # Create very short audio
    a = Audio()
    a.samples = np.zeros(10, dtype=np.float32)
    analyzer = SweepAnalyzer(a)

    with pytest.raises(RuntimeError, match="Not enough audio samples"):
        analyzer.get_components()


def test_sweep_stereo_conversion():
    # Stereo sweep
    # Left: sweep, Right: silence
    # Mean: half amplitude sweep

    a = generate.sweep(f_start=100, f_stop=1000, secs=2.0, fs=48000)
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
    a = generate.sweep(secs=1.0)
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
