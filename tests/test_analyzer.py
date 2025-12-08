import pytest
import pandas as pd
from analyzer import Analyzer
from audio import Audio
from component import Components


# Concrete implementation for testing
class ConcreteAnalyzer(Analyzer):
    def analyze(self, segment=None):
        self.select(segment)
        # Return a dummy stats object or just the result of select
        # The base implementation does self.select(segment),
        # then self.time_stats = TimeStats(self.audio),
        # then self.components = self.get_components()
        # Since TimeStats needs audio, we should probably stick to standard helper behavior if possible,
        # or mock it.
        # But for 'analyze' return, Analyzer.analyze returns self.time_stats.
        # Let's call super if we wanted full behavior, but Abstract method in Analyzer gives implementation!
        # wait, Analyzer.analyze IS NOT empty?
        # Abstract methods can have implementation in Python, but usually you override.
        # Analyzer.analyze docstring says "Abstract method...".
        # But it HAS a body:
        # self.select(segment)
        # self.time_stats = TimeStats(self.audio)
        # self.components = self.get_components()
        # return self.time_stats
        # So we can just call super().analyze(segment) if we provide get_components.
        return super().analyze(segment)

    def get_components(self):
        # Return empty components or mock
        return Components(self.audio.md)

    def print(self, **kwargs):
        super().print(**kwargs)
        print("CONCRETE_PRINT_CALLED")

    def reset(self):
        super().reset()


# Integration Tests with Real Example Files
def test_analyzer_integration_with_examples(examples_dir):
    # Use a real file
    fname = examples_dir / "test_tone_1k.wav"
    if not fname.exists():
        pytest.skip("test_tone_1k.wav not found in examples")

    audio = Audio()  # Empty audio
    analyzer = ConcreteAnalyzer(audio)

    # Test read
    assert analyzer.read(str(fname))

    # Test select
    # Analyzer.select enforces min length and truncation.
    # Default select(max_secs=10.0) -> trunc(1.0).
    # If we pass a small segment, it might be expanding it or ignoring?
    # Analyzer.select code:
    #   segment.trunc(1.0)
    #   segment.limit(max_secs)
    #   audio.select(segment)
    # If segment is 0.1s. trunc(1.0) might do nothing if it's start/stop indices?
    # Or does it round down to nearest second?
    # Let's rely on standard select which defaults to something safe.
    analyzer.select(max_secs=1.0)
    # Check that SOME selection happened and it's not empty
    assert len(analyzer.audio) > 0
    # And it ideally respects max_secs if audio is long enough
    # If audio is < 10s, it plays full.
    # We just want to ensure select works without crashing.

    # Test analyze
    stats = analyzer.analyze()
    assert stats is not None
    assert stats.rms > 0

    # Test print
    analyzer.print()

    # Test get_dataframe
    df = analyzer.get_dataframe()
    assert isinstance(df, pd.DataFrame)


def test_analyzer_to_csv(examples_dir, tmp_path):
    fname = examples_dir / "test_tone_1k.wav"
    if not fname.exists():
        pytest.skip("file missing")

    audio = Audio()
    audio.read(str(fname))
    analyzer = ConcreteAnalyzer(audio)

    out_file = tmp_path / "output.csv"
    analyzer.to_csv(str(out_file))
    assert out_file.exists()


# Mock Tests from _test_analyzer.py (Unit Tests)
def test_mock_instantiation_and_fs_property():
    class FakeAudio:
        def __init__(self, length=48000, fs=48000):
            self._len = length
            self.fs = fs

            class MD:
                pathname = None
                segment = None

            self.md = MD()

        def __len__(self):
            return self._len

        def reset(self):
            pass

        # Analyzer calls audio.reset? No, Analyzer calls self.reset

    fa = FakeAudio(length=96000, fs=48000)
    ta = ConcreteAnalyzer(fa)
    assert ta.fs == 48000
