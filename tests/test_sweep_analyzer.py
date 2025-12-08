"""
Test sweep_analyzer.py functionality.
"""

import pytest

from audio import Audio
from freq_resp import FreqResp
from sweep_analyzer import SweepAnalyzer


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
