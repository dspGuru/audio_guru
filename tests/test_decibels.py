import pytest
import math
from decibels import db, dbv, db_str, db_to_pwr_ratio
from constants import MIN_PWR, MIN_DB


class TestDecibels:
    def test_db(self):
        # Power ratio 1.0 -> 0 dB
        assert db(1.0) == 0.0
        # Power ratio 10.0 -> 10 dB
        assert db(10.0) == 10.0
        # Power ratio 0.5 -> -3.0 dB (approx)
        assert db(0.5) == -3.0

        # Test rounding
        assert db(0.5, ndigits=2) == -3.01

        # MIN_PWR clamping
        min_db_val = round(10.0 * math.log10(MIN_PWR), 1)
        assert db(0.0) == min_db_val
        assert db(-1.0) == min_db_val  # Input validation

    def test_dbv(self):
        # Voltage 1.0 -> 0 dB
        assert dbv(1.0) == 0.0
        # Voltage 10.0 -> 20 dB
        assert dbv(10.0) == 20.0
        # Voltage 0.001 -> -60 dB
        assert dbv(0.001) == -60.0

        # Test negative voltage (should be squared)
        assert dbv(-1.0) == 0.0

        # Test zero/small voltage
        min_db_val = round(10.0 * math.log10(MIN_PWR), 1)
        assert dbv(0.0) == min_db_val

    def test_db_str(self):
        assert db_str(1.0).strip() == "0.0"
        assert db_str(0.5).strip() == "-3.0"

        # Test suffix
        assert db_str(1.0, suffix=" dB").strip() == "0.0 dB"

        # Input < MIN_PWR returns "-----"
        assert "----" in db_str(MIN_PWR * 0.1)

    def test_db_to_pwr_ratio(self):
        assert db_to_pwr_ratio(0.0) == 1.0
        assert db_to_pwr_ratio(10.0) == 10.0
        assert db_to_pwr_ratio(20.0) == 100.0
        assert db_to_pwr_ratio(-10.0) == 0.1

        # Values below MIN_DB are clamped
        val = db_to_pwr_ratio(MIN_DB - 10.0)
        expected = 10 ** (MIN_DB / 10.0)
        assert val == pytest.approx(expected)
