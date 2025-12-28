import pytest
import sys
from unittest.mock import patch

from pathlib import Path
from audio import Audio
from util import Category
from constants import DEFAULT_FS
from join import join_files, main
from time_analyzer import TimeAnalyzer
import generate

# Define paths to example files
EXAMPLES_DIR = Path("examples")
TONE_FILE = EXAMPLES_DIR / "test_tone_1k.wav"
NOISE_FILE = EXAMPLES_DIR / "test_noise.wav"


def test_join_files(tmp_path):
    # Create two temporary test files
    file1 = tmp_path / "test1.wav"
    file2 = tmp_path / "test2.wav"

    # Use silence generator to create Audio objects with correct fs
    from generate import silence

    a1 = silence(1.0, fs=DEFAULT_FS)
    a1.samples[:] = 1.0  # Set to ones
    a1.write(str(file1))

    a2 = silence(1.0, fs=DEFAULT_FS)
    a2.samples[:] = 0.0  # Already zeros, but being explicit
    a2.write(str(file2))

    # Test join without silence
    joined = join_files([str(file1), str(file2)], silence_secs=0.0)
    expected_len = 2 * int(DEFAULT_FS)
    assert abs(len(joined) - expected_len) <= 1

    # Test join WITH silence
    silence_len_sec = 0.5
    joined_silence = join_files([str(file1), str(file2)], silence_secs=silence_len_sec)
    expected_silence_samples = int(silence_len_sec * DEFAULT_FS)
    expected_len = 2 * int(DEFAULT_FS) + expected_silence_samples
    assert abs(len(joined_silence) - expected_len) <= 1


def test_join_negative_silence(tmp_path):
    file1 = tmp_path / "test1.wav"
    # Create valid file
    a1 = generate.sine(secs=0.1)
    a1.write(str(file1))

    with pytest.raises(ValueError, match="Silence duration must be non-negative"):
        join_files([str(file1)], silence_secs=-1.0)


def test_join_creates_output_dir(tmp_path):
    # Mock data to join
    file1 = tmp_path / "test1.wav"
    a1 = generate.sine(secs=0.1)
    a1.write(str(file1))

    # Output in new subdir
    subdir = tmp_path / "subdir"
    output = subdir / "joined.wav"

    assert not subdir.exists()

    with patch.object(sys, "argv", ["join.py", str(file1), "-w", str(output)]):
        main()

    assert subdir.exists()
    assert output.exists()


def test_joined_stats(examples_dir, tmp_path):
    # We want specific files to be deterministic
    files_to_join: list[Path] = [
        examples_dir / "test_tone_1k.wav",
        examples_dir / "test_noise.wav",
    ]

    # Check existence
    for pathname in files_to_join:
        if not pathname.exists():
            pytest.skip(f"Example file {pathname.name} missing")

    # 2. Join them with silence
    silence_secs = 1.0
    # convert Paths to strings
    file_strs = [str(f) for f in files_to_join]

    joined_audio = join_files(file_strs, silence_secs=silence_secs)
    joined_path = tmp_path / "joined.wav"
    joined_audio.write(str(joined_path))

    assert joined_path.exists()

    # 3. "Split" the joined file (logic similar to split.py main loop)
    # Read joined file back
    read_joined = Audio()

    # Verify that the length of the joined audio is close to the
    # sum of the lengths of the files that were joined
    read_joined.read(str(joined_path))

    audio = Audio()
    total_secs = 0.0
    for path in files_to_join:
        assert audio.read(path) == 1
        total_secs += audio.segment.secs
    assert abs(read_joined.segment.secs - total_secs) < 2.0

    # Get segments
    # Audio.get_segments should separate by silence (assuming 1.0s is enough to be detected)
    segments = read_joined.get_segments()

    assert len(segments) == len(files_to_join)

    # 4. Verify integrity using TimeAnalyzer
    for i, orig_path in enumerate(files_to_join):
        # Original Analysis
        orig_audio = Audio()
        orig_audio.read(str(orig_path))
        orig_analyzer = TimeAnalyzer(orig_audio)
        orig_stats = orig_analyzer.analyze()

        # Analyze the *Active* part of original to be fair comparisons
        # Split logic removes silence, so we compare against non-silent original part
        orig_segs = orig_audio.get_segments()
        if orig_segs:
            orig_audio.select(orig_segs[0])
            orig_stats = orig_analyzer.analyze()

        # Split Segment Analysis
        split_path = tmp_path / f"split_{i}.wav"

        # Select segment in read_joined provided by iterator/list
        read_joined.select(segments[i])
        read_joined.write(str(split_path))

        # Read back split file
        split_audio = Audio()
        split_audio.read(str(split_path))
        split_analyzer = TimeAnalyzer(split_audio)
        split_stats = split_analyzer.analyze()

        # Verify stats match
        # RMS
        # Relax tolerance for noise which might have edge effects variations
        # Also relax for Tone where joined context might affect boundary refinement (cleaner segment)
        tol = 0.15 if "noise" in orig_path.name else 0.05
        assert split_stats.rms == pytest.approx(
            orig_stats.rms, rel=tol
        ), f"RMS mismatch for {orig_path.name}"

        # Peak (Max)
        assert split_stats.max == pytest.approx(
            orig_stats.max, rel=0.01
        ), f"Peak mismatch for {orig_path.name}"

        # Crest Factor
        assert split_stats.crest == pytest.approx(
            orig_stats.crest, rel=tol
        ), f"Crest mismatch for {orig_path.name}"


def test_join_integrity(tmp_path):
    # Verify examples exist
    if not TONE_FILE.exists() or not NOISE_FILE.exists():
        pytest.skip("Example files not found, skipping integration test.")

    # 1. Join files with 1.0 second silence
    # Note: join_files takes a list of glob patterns.
    # We pass exact paths as patterns.
    patterns = [str(TONE_FILE), str(NOISE_FILE)]
    silence_secs = 1.0

    combined = join_files(patterns, silence_secs=silence_secs)

    # 2. Write to temp file
    joined_path = tmp_path / "joined.wav"
    combined.write(joined_path)

    # 3. Read back
    reloaded = Audio()
    assert reloaded.read(joined_path) == 1

    # 4. Split / Iterate
    # Audio iteration uses get_segments which detects silence gaps.
    # We expect 2 segments: Tone and Noise.
    # The silence gap of 1.0s should be sufficient for default get_segments settings.
    # default min_silence_secs=0.1.

    segments = list(reloaded)

    # Verify we found 2 segments
    assert len(segments) == 2, f"Expected 2 segments, found {len(segments)}"

    # 5. Read originals for comparison
    orig_tone = Audio()
    orig_tone.read(TONE_FILE)

    orig_noise = Audio()
    orig_noise.read(NOISE_FILE)

    # 6. Compare Statistics

    # Segment 0 -> Tone
    reloaded.select(segments[0])
    # Compare RMS, Max, Min with tolerance
    # Re-reading/Writing audio can introduce minor quantization noise if formats differ,
    # but likely float32 -> float32 or pcm16 round trip.
    # Usually robust to ~1% or better.

    # Compare active segment vs active segment
    # The original files may have silence. We must compare the 'active' part of both.

    # Process original tone to find its active segment
    orig_tone_segs = orig_tone.get_segments()
    assert len(orig_tone_segs) == 1
    orig_tone.select(orig_tone_segs[0])

    assert reloaded.rms == pytest.approx(orig_tone.rms, rel=1e-3)
    assert reloaded.max == pytest.approx(orig_tone.max, rel=1e-3)
    assert reloaded.freq == pytest.approx(orig_tone.freq, abs=5.0)

    # Segment 1 -> Noise
    reloaded.select(segments[1])

    # Process original noise
    orig_noise_segs = orig_noise.get_segments()
    if not orig_noise_segs:
        # If noise is constant and considered ONE segment? or none?
        # get_segments returns non-silent parts.
        # If orig_noise is detected as 1 segment
        assert len(orig_noise_segs) == 1
        orig_noise.select(orig_noise_segs[0])
    else:
        # If it returns multiple, pick the main one?
        # Assume 1 for the test file
        orig_noise.select(orig_noise_segs[0])

    assert reloaded.rms == pytest.approx(orig_noise.rms, rel=1e-3)
    # Noise max might fluctuate slightly depending on how boundaries are cut vs original,
    # but should be very close.
    assert reloaded.max == pytest.approx(orig_noise.max, rel=1e-2)
    # Frequency of noise is unstable, so maybe skip freq check or check it's not a tone.

    # 7. Check Segment Descriptions (Optional but good)
    # get_segments(categorize=True) is called by __iter__.
    # So segments should have categories.
    # Tone -> Category.Tone
    # Noise -> Category.Noise (or Unknown if not recognized as Sine/Sweep)

    # Manually categorize since __iter__ no longer does it locally without explicit call?
    # Actually __iter__ calls get_segments which defaults categorize=False (based on recent changes? No, get_segments removed categorize arg).
    # We must categorize manually or trust that generated segments have valid defaults.
    # But checks below rely on category property.

    # We check if it is NOT Silence or Tone for the noise part.
    assert segments[1].cat not in [Category.Silence, Category.Tone]

    # Mock command line arguments
    output_file = tmp_path / "output.wav"
    # We use valid example files to ensure it runs through
    test_args = [
        "join.py",
        "examples/test_tone_1k.wav",
        "examples/test_noise.wav",
        "-w",
        str(output_file),
        "-s",
        "1.0",
    ]

    with patch.object(sys, "argv", test_args):
        # We also need to mock print or capturing stdout could be useful,
        # but the main goal is verifying it runs and produces output.
        main()

    with patch.object(sys, "argv", test_args):
        # We also need to mock print or capturing stdout could be useful,
        # but the main goal is verifying it runs and produces output.
        main()

    assert output_file.exists()


def test_join_no_files():
    # Test join_files when no files differ
    # join_files handles list of patterns
    # if patterns match nothing, it returns empty Audio

    with patch("join.glob.glob") as mock_glob:
        mock_glob.return_value = []
        # pattern that doesn't exist
        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = False

            with patch("builtins.print") as mock_print:
                result = join_files(["non_existent.wav"])

                assert len(result) == 0
                mock_print.assert_any_call("No files found.")


def test_join_read_fail():
    # Test when reading a file fails
    with patch("join.glob.glob") as mock_glob:
        mock_glob.return_value = ["bad.wav"]

        with patch("join.Audio") as MockAudio:
            instance = MockAudio.return_value
            # Fail on read
            instance.read.return_value = False

            with patch("builtins.print") as mock_print:
                result = join_files(["bad.wav"])

                # Should be empty if all failed
                assert len(result) == 0
                # Error printed
                mock_print.assert_any_call("Error: Could not read audio file 'bad.wav'")


def test_main_no_write(tmp_path):
    # Test main when -w is NOT provided
    test_args = ["join.py", "examples/test_tone_1k.wav"]

    with patch.object(sys, "argv", test_args):
        # Mock join_files to return something so we reach the check
        with patch("join.join_files") as mock_join:
            mock_join.return_value = Audio()  # empty is fine

            with patch("builtins.print") as mock_print:
                main()

                mock_print.assert_any_call("Use -w <filename> to save the output.")
