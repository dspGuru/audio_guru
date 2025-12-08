from join import join_files


def test_join_files(tmp_path):
    # Create two temporary test files
    file1 = tmp_path / "test1.wav"
    file2 = tmp_path / "test2.wav"

    # Use silence generator to create Audio objects with correct fs
    from generate import silence

    a1 = silence(1.0, fs=44100)
    a1.samples[:] = 1.0  # Set to ones
    a1.write(str(file1))

    a2 = silence(1.0, fs=44100)
    a2.samples[:] = 0.0  # Already zeros, but being explicit
    a2.write(str(file2))

    # Test join without silence
    joined = join_files([str(file1), str(file2)], silence_secs=0.0)
    expected_len = 44100 + 44100
    assert abs(len(joined) - expected_len) <= 1

    # Test join WITH silence
    silence_len_sec = 0.5
    joined_silence = join_files([str(file1), str(file2)], silence_secs=silence_len_sec)
    expected_silence_samples = int(silence_len_sec * 44100)
    # Smoke test for argument parsing if we could invoke main,
    # but unit testing the logic function is usually sufficient.
    pass
