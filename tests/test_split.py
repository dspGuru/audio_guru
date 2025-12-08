import pytest
from audio import Audio
from join import join_files
from time_analyzer import TimeAnalyzer


def test_split_and_verify_with_time_analyzer(examples_dir, tmp_path):
    # 1. Select Example Files
    # matching pattern to ensure we get a few expected files
    pattern = str(examples_dir / "test_*.wav")

    # We want specific files to be deterministic
    files_to_join = [examples_dir / "test_tone_1k.wav", examples_dir / "test_noise.wav"]

    # Check existence
    for f in files_to_join:
        if not f.exists():
            pytest.skip(f"Example file {f.name} missing")

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


def test_split_and_verify_with_time_analyzer(examples_dir, tmp_path):
    # 1. Select Example Files
    # matching pattern to ensure we get a few expected files
    pattern = str(examples_dir / "test_*.wav")

    # We want specific files to be deterministic
    files_to_join = [examples_dir / "test_tone_1k.wav", examples_dir / "test_noise.wav"]

    # Check existence
    for f in files_to_join:
        if not f.exists():
            pytest.skip(f"Example file {f.name} missing")

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
    read_joined.read(str(joined_path))

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
        # Split logic removes silence, so we should compare against non-silent original part
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

        print(f"Comparing {orig_path.name} vs Segment {i}")

        # Verify stats match
        # RMS
        # Relax tolerance for noise which might have edge effects variations
        tol = 0.15 if "noise" in orig_path.name else 0.01
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
