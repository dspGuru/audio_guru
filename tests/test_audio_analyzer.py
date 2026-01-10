from audio_analyzer import AudioAnalyzer
import unittest.mock


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
        # Use a pattern that won't match
        pattern = str(tmp_path / "*.nonexistent")
        count = analyzer.read(pattern)
        assert count == 0
        assert len(analyzer.analyzers) == 0
        assert len(analyzer.results) == 0

    def test_read_single_file(self, signals_dir):
        """Test reading a single known existing file."""
        fname = signals_dir / "05-test_tone_1k.wav"
        fname_str = str(fname)

        analyzer = AudioAnalyzer()
        with unittest.mock.patch("analyzer.SEGMENT_MIN_SECS", 0.0):
            count = analyzer.read(fname_str)

        assert count == 1
        assert len(analyzer.unit_ids) == 1
        # unit_id is derived from either the filename or embedded metadata in
        # the WAV file.

        # Expect at least one analyzer (likely ToneAnalyzer for sine wave)
        assert len(analyzer.analyzers) >= 1

        # Verify results structure (keys are f"{fname}:{segment}")
        assert len(analyzer.results) >= 1
        key = list(analyzer.results.keys())[0]
        assert str(fname) in key or fname.name in key

        analysis = analyzer.results[key]
        assert analysis.segment is not None
        assert analysis.analysis is not None
        assert analysis.components is not None

    def test_read_multiple_files(self, signals_dir):
        """Test reading multiple files via wildcard."""
        # Assume signals_dir has test_tone_1k.wav and some others

        pattern = str(signals_dir / "*-test_*.wav")
        analyzer = AudioAnalyzer()
        with unittest.mock.patch("analyzer.SEGMENT_MIN_SECS", 0.0):
            count = analyzer.read(pattern)

        assert count > 0
        assert len(analyzer.unit_ids) > 0

    def test_print(self, signals_dir, capsys):
        """Test print method output."""
        fname = signals_dir / "05-test_tone_1k.wav"
        analyzer = AudioAnalyzer()
        analyzer.read(str(fname))

        analyzer.print(tone_sort_attr="thd", components=10, audio_stats=True)

        captured = capsys.readouterr()
        # Expect some output
        assert len(captured.out) > 0

        # ToneStats printed if ToneAnalyzer was used
        assert "Tone Statistics" in captured.out

    def test_read_quiet_suppresses_output(self, signals_dir, capsys):
        """Test that quiet=True suppresses 'Reading' output."""
        fname = signals_dir / "05-test_tone_1k.wav"
        analyzer = AudioAnalyzer()

        with unittest.mock.patch("analyzer.SEGMENT_MIN_SECS", 0.0):
            count = analyzer.read(str(fname), quiet=True)

        captured = capsys.readouterr()
        # "Reading" should NOT appear in output
        assert "Reading" not in captured.out
        assert count == 1

    def test_read_not_quiet_prints_reading(self, signals_dir, capsys):
        """Test that quiet=False (default) prints 'Reading' output."""
        fname = signals_dir / "05-test_tone_1k.wav"
        analyzer = AudioAnalyzer()

        with unittest.mock.patch("analyzer.SEGMENT_MIN_SECS", 0.0):
            count = analyzer.read(str(fname), quiet=False)

        captured = capsys.readouterr()
        # "Reading" should appear in output
        assert "Reading" in captured.out
        assert count == 1

    def test_read_device_quiet_parameter(self):
        """Test that read_device accepts quiet parameter and passes it to open_device."""
        # Mock the Audio class to track open_device calls
        with unittest.mock.patch("audio_analyzer.Audio") as MockAudio:
            mock_audio_instance = MockAudio.return_value

            # Set up context manager for open_device
            mock_context = unittest.mock.MagicMock()
            mock_audio_instance.open_device.return_value = mock_context
            mock_context.__enter__.return_value = mock_audio_instance
            mock_context.__exit__.return_value = False

            mock_audio_instance.read_block.return_value = False  # No data read
            mock_audio_instance.fs = 44100
            mock_audio_instance.blocksize = 44100

            analyzer = AudioAnalyzer()
            result = analyzer.read_device(duration=0.1, quiet=True)

            # Verify open_device was called with quiet=True
            mock_audio_instance.open_device.assert_any_call(quiet=True)
