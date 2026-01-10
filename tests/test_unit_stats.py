import pytest
from unit_stats import UnitStats
from component import Component
from util import Category
from constants import DEFAULT_FREQ
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
        stats2.noise_floor = 0.002
        stats2.thd = 0.005
        stats2.snr = 200.0

        stats1.aggregate(stats2)

        assert stats1.num_tests == 3
        assert stats1.noise_floor == 0.002  # Max
        assert stats1.thd == 0.01  # Max
        assert stats1.snr == 200.0  # Max

    def test_freq_response_aggregation(self):
        stats1 = UnitStats("U1")
        c1 = Component(freq=DEFAULT_FREQ, pwr=1.0, cat=Category.Tone)
        stats1.freq_response.append(c1)

        stats2 = UnitStats("U2")
        c2 = Component(freq=DEFAULT_FREQ, pwr=0.5, cat=Category.Tone)  # Different power
        stats2.freq_response.append(c2)

        c3 = Component(freq=2000.0, pwr=0.1, cat=Category.Tone)  # New freq
        stats2.freq_response.append(c3)

        stats1.aggregate(stats2)

        assert len(stats1.freq_response) == 2

        # Check averaged 1kHz component
        comp_1k = stats1.freq_response.find_freq(DEFAULT_FREQ)
        assert comp_1k is not None
        assert comp_1k.pwr == pytest.approx((1.0 + 0.5) / 2)

    def test_aggregate_overlapping_results(self):
        # Verify that result values are summed if a test name repeat across
        # aggregated units.
        stats1 = UnitStats("U1")
        stats1.results["T1"] = 1.0

        stats2 = UnitStats("U2")
        stats2.results["T1"] = 2.0

        stats1.aggregate(stats2)

        # Should sum overlapping results
        assert stats1.results["T1"] == 3.0

    def test_print_specs_coverage(self, capsys):
        stats = UnitStats("U1")

        # Verify frequency response logic when components are present.
        c1 = Component(100.0, 1.0, 0.0, Category.Tone)
        stats.freq_response.append(c1)

        # Multiple components for ripple calc
        c2 = Component(DEFAULT_FREQ, 0.5, 0.0, Category.Tone)
        stats.freq_response.append(c2)

        stats.print_specs()

        captured = capsys.readouterr()
        assert "Frequency Response" in captured.out
        assert "+/-" in captured.out

    def test_manager_add_result(self):
        from unit_stats import UnitStatsManager

        mgr = UnitStatsManager()

        # Verify automatic creation of UnitStats if the unit_id is new.
        mgr.add_result("U1", "Test1", 10.0)

        assert "U1" in mgr.units
        stats = mgr.units["U1"]
        assert stats.num_tests == 1
        assert stats.results["Test1"] == 10.0

        # Add another result to existing unit (implicit else path, but covers
        # logic flow)
        mgr.add_result("U1", "Test2", 20.0)
        assert stats.num_tests == 2

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

    def test_aggregate_noise_floor_zero(self):
        # Test case where one unit has 0 noise floor (uninitialized or perfectly silent?)
        stats1 = UnitStats("U1")
        stats1.noise_floor = 0.0

        stats2 = UnitStats("U2")
        stats2.noise_floor = 0.001

        stats1.aggregate(stats2)
        assert stats1.noise_floor == 0.001

    def test_aggregate_missing_snr(self):
        # Gracefully handle missing attributes (e.g. snr) in the unit being
        # aggregated.
        stats1 = UnitStats("U1")
        stats1.snr = 10.0

        stats2 = UnitStats("U2")
        # Delete snr from stats2 to trigger AttributeError
        del stats2.snr

        # Should not raise exception
        stats1.aggregate(stats2)
        assert stats1.snr == 10.0

    def test_print_specs_empty_band_components(self, capsys):
        # Create components outside ripple band (20 Hz-20 kHz)
        stats = UnitStats("U1")
        c1 = Component(10.0, 1.0, 0.0, Category.Tone)  # Below 20
        c2 = Component(30000.0, 1.0, 0.0, Category.Tone)  # Above 20k
        stats.freq_response.append(c1)
        stats.freq_response.append(c2)

        stats.print_specs()
        captured = capsys.readouterr()
        # Should show dev 0.0
        assert "+/- 0.0 dB" in captured.out
