from audio_analyzer import AudioAnalyzer


class TestAudioAnalyzer:
    def test_init(self):
        """Test initial state of AudioAnalyzer."""
        analyzer = AudioAnalyzer()
        assert analyzer.analyzers == []
        assert analyzer.results == {}
        assert analyzer.unit_ids == set()
        assert analyzer.audio is not None

    def test_read_no_files(self, tmp_path):
        """Test reading when no files match the pattern."""
        analyzer = AudioAnalyzer()
        # Use a pattern that definitely doesn't match anything in tmp_path
        pattern = str(tmp_path / "*.nonexistent")
        count = analyzer.read(pattern)
        assert count == 0
        assert len(analyzer.analyzers) == 0
        assert len(analyzer.results) == 0

    def test_read_single_file(self, examples_dir):
        """Test reading a single known existing file."""
        fname = examples_dir / "test_tone_1k.wav"
        fname_str = str(fname)

        analyzer = AudioAnalyzer()
        count = analyzer.read(fname_str)

        assert count == 1
        assert len(analyzer.unit_ids) == 1
        # The unit_id logic in Audio.read (or wherever) usually derives from file or metadata
        # We just verify something was added.

        # We expect at least one analyzer to be created (ToneAnalyzer likely for sine wave)
        # Note: Depending on how 'test_tone_1k.wav' is classified (Tone vs Unknown).
        # Based on test_audio_analysis.py, it should be Tone.
        assert len(analyzer.analyzers) >= 1

        # Verify results structure
        # keys are f"{fname}:{segment}"
        assert len(analyzer.results) >= 1
        key = list(analyzer.results.keys())[0]
        assert str(fname) in key or fname.name in key  # fname in key is full path

        analysis = analyzer.results[key]
        assert analysis.segment is not None
        assert analysis.result is not None
        assert analysis.components is not None

    def test_read_multiple_files(self, examples_dir):
        """Test reading multiple files via wildcard."""
        # We assume examples_dir has at least test_tone_1k.wav and some others.
        # We'll use a specific pattern if possible, or just *.wav and limit checks

        pattern = str(examples_dir / "test_*.wav")
        analyzer = AudioAnalyzer()
        count = analyzer.read(pattern)

        assert count > 0
        assert len(analyzer.unit_ids) > 0

    def test_print(self, examples_dir, capsys):
        """Test print method output."""
        fname = examples_dir / "test_tone_1k.wav"
        analyzer = AudioAnalyzer()
        analyzer.read(str(fname))

        analyzer.print(sort_stat_attr="thd", components=True, noise_floor=True)

        captured = capsys.readouterr()
        # We expect some output
        assert len(captured.out) > 0
        # Usually headers or stats are printed.
        # "Tone Statistics Summary:" is printed if ToneAnalyzer was used and sort_stat_attr is set
        # But only if ToneStatsList is not empty.

        # If the file is 1k tone, it should produce ToneStats
        # Since we know test_tone_1k.wav is a tone, we expect the summary
        assert "Tone Statistics Summary:" in captured.out
