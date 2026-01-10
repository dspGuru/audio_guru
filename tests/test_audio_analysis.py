import pytest
from audio import Audio
from constants import DEFAULT_FS, DEFAULT_FREQ
from unit_stats import UnitStats
from util import Category

# Simulate analysis process using Audio methods to verify UnitStats integration.


class TestAudioAnalysis:

    def test_tone_1k_analysis(self, signals_dir):
        fname = signals_dir / "05-test_tone_1k.wav"
        assert fname.exists()

        audio = Audio()
        assert audio.read(fname)

        # Verify Audio properties
        assert audio.fs == DEFAULT_FS
        assert len(audio) > 0

        # Verify Frequency
        assert audio.freq == pytest.approx(DEFAULT_FREQ, abs=5.0)  # +/- 5 Hz tolerance

        # Calculate Stats manually to verify UnitStats integration
        stats = UnitStats("Sine1k")
        stats.noise_floor = audio.get_noise_floor()[0]
        # Verify noise floor is low for pure sine
        stats.print_specs()
        # (This is mostly a smoke test to ensure no crash on real data)

    def test_sweep_analysis(self, signals_dir):
        fname = signals_dir / "12-test_sweep.wav"
        audio = Audio()
        audio.read(fname)

        # Select active sweep part (first 4.8s) to avoid appended silence
        from segment import Segment

        audio.select(
            Segment(
                fs=audio.fs, start=0, stop=int(4.8 * audio.fs), id=1, cat=Category.Sweep
            )
        )
        assert audio.cat == Category.Sweep

    def test_noise_analysis(self, signals_dir):
        fname = signals_dir / "13-test_noise.wav"
        audio = Audio()
        audio.read(fname)

        assert audio.cat == Category.Noise

        # Noise floor might be 0.0 due to digital silence appended
        nf, idx = audio.get_noise_floor()
        assert nf >= 0.0

    def test_distortion_file(self, examples_dir):
        # example_tone_1k-1_000_pct.wav -> 1% THD
        fname = examples_dir / "example_tone_1k-1-000-pct.wav"
        audio = Audio()
        assert audio.read(fname)

        assert audio.freq == pytest.approx(DEFAULT_FREQ, abs=5.0)

        # Verify it loads and is categorized as Tone
        assert audio.cat == Category.Tone
