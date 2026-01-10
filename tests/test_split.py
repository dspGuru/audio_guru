import pytest

from unittest.mock import patch
import sys
from audio import Audio
from join import join_files
from split import split_file, main
from time_analyzer import TimeAnalyzer
import generate


def test_split(signals_dir, tmp_path):
    # We want specific files to be deterministic
    files_to_join = [
        signals_dir / "05-test_tone_1k.wav",
        signals_dir / "13-test_noise.wav",
    ]

    # Check existence
    for pathname in files_to_join:
        if not pathname.exists():
            pytest.skip(f"Example file {pathname.name} missing")

    # Join the example files with silence
    silence_secs = 1.0
    joined_audio = join_files(files_to_join, silence_secs=silence_secs)
    joined_path = tmp_path / "joined.wav"
    joined_audio.write(str(joined_path))

    assert joined_path.exists()

    # Silence preserved -> 2 segments
    split_files = split_file(str(joined_path))
    assert len(split_files) == 2

    # Get the segments of the joined audio for comparision to the audio from
    # the split files
    orig_segs = joined_audio.get_segments()
    assert len(orig_segs) == len(split_files)

    # Verify integrity using TimeAnalyzer
    joined_analyzer = TimeAnalyzer(joined_audio)
    for i, split_path in enumerate(split_files):
        # Original Analysis
        split_audio = Audio()
        split_audio.read(str(split_path))

        split_analyzer = TimeAnalyzer(split_audio)
        split_stats = split_analyzer.analyze()

        # Analyze non-silent segments of original since splitting removes silence
        joined_analyzer.select(orig_segs[i])
        joined_stats = joined_analyzer.analyze()
        print(f"Comparing {split_path} to joined segment {i}")

        # Verify time statistics match within a tolerance

        # Verify RMS
        tol = 0.15 if "noise" in str(split_path) else 0.05
        assert split_stats.rms == pytest.approx(
            joined_stats.rms, rel=tol
        ), f"RMS mismatch for {split_path}"

        # Verify peak (max)
        assert split_stats.max == pytest.approx(
            joined_stats.max, rel=0.01
        ), f"Peak mismatch for {split_path}"

        # Verify crest factor
        assert split_stats.crest == pytest.approx(
            joined_stats.crest, rel=tol
        ), f"Crest mismatch for {split_path}"


def test_split_no_extension(tmp_path):
    # Create a file without extension
    file1 = tmp_path / "testfile"
    a = generate.sine(secs=2.0)
    # Audio.write might need format; write with extension then rename
    file_orig = tmp_path / "testfile.wav"
    a.write(str(file_orig))
    file_orig.rename(file1)

    # Run split
    split_files = split_file(str(file1))

    # Should produce testfile_1.wav
    assert len(split_files) > 0
    assert "testfile_1.wav" in split_files[0]
    assert (tmp_path / "testfile_1.wav").exists()


def split_file_not_found():
    with pytest.raises(
        FileNotFoundError, match="Could not read audio file 'non_existent_file.wav'"
    ):
        split_file("non_existent_file.wav")


def test_main(tmp_path):
    # Mock command line arguments
    test_args = ["split.py", "examples/*.wav"]

    with patch.object(sys, "argv", test_args):
        with patch("split.glob.glob") as mock_glob:
            with patch("split.split_file") as mock_split_file:
                # Mock glob to return some files
                mock_glob.return_value = ["file1.wav", "file2.wav"]

                # Mock Audio read so it returns True
                with patch("split.Audio") as MockAudio:
                    instance = MockAudio.return_value
                    instance.read.return_value = True

                    main()

                    # Verify glob called with pattern
                    mock_glob.assert_called_with("examples/*.wav")

                    # Verify split_file called for each file
                    assert mock_split_file.call_count == 2
                    mock_split_file.assert_any_call("file1.wav")
                    mock_split_file.assert_any_call("file2.wav")


def test_main_no_files():
    # Test when glob returns no files
    test_args = ["split.py", "examples/*.wav"]

    with patch.object(sys, "argv", test_args):
        with patch("split.glob.glob") as mock_glob:
            # Return empty list
            mock_glob.return_value = []

            with patch("builtins.print") as mock_print:
                main()

                # Verify usage message printed
                mock_print.assert_any_call(
                    "Usage: python split.py <audio_file_pattern>"
                )


def test_main_read_fail():
    # Test when Audio.read() fails
    test_args = ["split.py", "examples/*.wav"]

    with patch.object(sys, "argv", test_args):
        with patch("split.glob.glob") as mock_glob:
            mock_glob.return_value = ["bad.wav"]

            with patch("split.Audio") as MockAudio:
                instance = MockAudio.return_value
                # Simulate read failure
                instance.read.return_value = False

                with patch("split.split_file") as mock_split_file:
                    with patch("builtins.print") as mock_print:
                        main()

                        # Verify split_file NOT called
                        mock_split_file.assert_not_called()

                        # Verify error message
                        mock_print.assert_any_call(
                            "Error: Could not read audio file 'bad.wav'"
                        )
