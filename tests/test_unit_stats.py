import pytest
from unit_stats import UnitStats
from component import Component
from util import Category
from decibels import db_to_pwr_ratio


class TestUnitStats:
    def test_initialization(self):
        stats = UnitStats("TestUnit")
        assert stats.unit_id == "TestUnit"
        assert stats.num_tests == 0
        assert stats.noise_floor == 0.0
        assert stats.thd == 0.0
        assert stats.snr == 0.0
        assert len(stats.freq_response) == 0

    def test_aggregation(self):
        stats1 = UnitStats("U1")
        stats1.num_tests = 1
        stats1.noise_floor = 0.001
        stats1.thd = 0.01
        stats1.snr = 100.0

        stats2 = UnitStats("U2")
        stats2.num_tests = 2
        stats2.noise_floor = 0.002  # Higher (worse?) or just max amplitude
        stats2.thd = 0.005
        stats2.snr = 200.0

        stats1.aggregate(stats2)

        assert stats1.num_tests == 3
        assert stats1.noise_floor == 0.002  # Max
        assert stats1.thd == 0.01  # Max
        assert stats1.snr == 200.0  # Max

    def test_freq_response_aggregation(self):
        stats1 = UnitStats("U1")
        c1 = Component(freq=1000.0, pwr=1.0, cat=Category.Tone)
        stats1.freq_response.append(c1)

        stats2 = UnitStats("U2")
        c2 = Component(freq=1000.0, pwr=0.5, cat=Category.Tone)  # Different power
        stats2.freq_response.append(c2)

        c3 = Component(freq=2000.0, pwr=0.1, cat=Category.Tone)  # New freq
        stats2.freq_response.append(c3)

        stats1.aggregate(stats2)

        assert len(stats1.freq_response) == 2

        # Check averaged 1kHz component
        comp_1k = stats1.freq_response.find_freq(1000.0)
        assert comp_1k is not None
        assert comp_1k.pwr == pytest.approx((1.0 + 0.5) / 2)  # Magnitude averaged?
        # Wait, the code averages magnitude: comp.magnitude = (comp.magnitude + other_comp.magnitude) / 2
        # Component doesn't have .magnitude, it has .pwr!
        # Let's check the code implementation again.
        # The `aggregate` method in `unit_stats.py` uses `.magnitude` and `.phase` which might be incorrect if Component uses `pwr`.
        # I need to fix `unit_stats.py` if that's the case.
        # `view_file` of `component.py` shows it has `pwr` but NOT `magnitude`.
        # This is a BUG in `unit_stats.py`. I will fix it in EXECUTION.

    def test_print_specs_output(self, capsys):
        stats = UnitStats("TestUnit")
        stats.noise_floor = 10 ** (-90.0 / 20.0)
        stats.thd = 0.0005
        stats.snr = db_to_pwr_ratio(96)

        stats.print_specs()

        captured = capsys.readouterr()
        output = captured.out

        assert "AUDIO UNIT SPECIFICATIONS" in output
        assert "Noise Floor ............... -90.0 dBFS" in output
        assert "Total Harmonic Distortion ... 0.0500 %" in output
        assert "Signal-to-Noise Ratio ....... 96.0 dB" in output
