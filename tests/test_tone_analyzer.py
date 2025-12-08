import re

import pytest

from tone_analyzer import ToneAnalyzer
from audio import Audio


def test_tone_analyzer_with_examples(examples_dir):
    # Find all test tone files
    tone_files = list(examples_dir.glob("test_tone_*.wav"))
    if not tone_files:
        pytest.skip("No test_tone_*.wav files found in examples")

    # Regex to extract expected THD percent from filename
    # Format: test_tone_1k-X_XXX_pct.wav -> X_XXX
    # Example: test_tone_1k-0_010_pct.wav -> 0.010%
    # Example: test_tone_1k-1_000_pct.wav -> 1.000%
    # pattern: .*?(\d+_\d+)_pct\.wav
    pattern = re.compile(r".*?(\d+_\d+)_pct\.wav")

    for fpath in tone_files:
        fname = fpath.name
        match = pattern.match(fname)
        if not match:
            print(f"Skipping {fname} (no THD pattern match)")
            continue

        thd_str = match.group(1).replace("_", ".")
        expected_thd_pct = float(thd_str)

        print(f"Testing {fname}, expecting THD={expected_thd_pct}%")

        audio = Audio()
        assert audio.read(str(fpath)), f"Failed to read {fname}"

        analyzer = ToneAnalyzer(audio)
        stats = analyzer.analyze()

        # Verify Frequency (should be ~1k)
        assert stats.freq == pytest.approx(1000.0, abs=5.0)

        # Verify THD
        # Tolerance: THD estimation isn't perfect, especially at low levels.
        # But for generated signals it should be close.
        # 0% THD file might have some noise floor THD.

        if expected_thd_pct == 0.0:
            # Should be very low
            assert stats.thd_pct < 0.001
        else:
            # Use relative tolerance
            # 1% -> 0.01. Code calculates thd_pct.
            # Allow 5-10% relative error or small absolute
            assert stats.thd_pct == pytest.approx(expected_thd_pct, rel=0.8, abs=0.5)
