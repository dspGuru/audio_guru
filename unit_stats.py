"""Unit performance statistics."""

from component import Components
from constants import DEFAULT_FS
from decibels import db, dbv
from metadata import Metadata
from segment import Segment


class UnitStats:
    """Class to hold unit statistics."""

    def __init__(self, unit_id: str) -> None:
        self.unit_id = unit_id
        self.num_tests = 0
        self.results: dict[str, float] = {}
        self.noise_floor: float = 0.0
        self.thd: float = 0.0
        self.snr: float = 0.0
        self.freq_response: Components = Components(Metadata("", Segment(DEFAULT_FS)))

    def aggregate(self, other: "UnitStats") -> None:
        """Aggregate statistics from another UnitStats instance."""
        self.num_tests += other.num_tests
        for test_name, result in other.results.items():
            if test_name in self.results:
                self.results[test_name] += result
            else:
                self.results[test_name] = result

        self.noise_floor = max(self.noise_floor, other.noise_floor)

        self.thd = max(self.thd, other.thd)

        try:
            self.snr = max(self.snr, other.snr)
        except AttributeError:
            pass

        # Combine frequency response components
        for other_comp in other.freq_response:
            my_comp = self.freq_response.find_freq(other_comp.freq)
            if my_comp is None:
                self.freq_response.append(other_comp)
            else:
                # Average the power if frequency already exists
                my_comp.pwr = (my_comp.pwr + other_comp.pwr) / 2

    def print_specs(self) -> None:
        """Print the unit specifications in a manual-like format."""

        print("AUDIO UNIT SPECIFICATIONS")
        print("--------------------------------------------------")

        nf_db = dbv(self.noise_floor) if self.noise_floor > 0 else -999.0
        print(f"Noise Floor ............... {nf_db:0.1f} dBFS")

        print(f"Total Harmonic Distortion ... {self.thd * 100.0:0.4f} %")

        if self.snr > 0:
            snr_db = db(self.snr)
            print(f"Signal-to-Noise Ratio ....... {snr_db:0.1f} dB")

        if len(self.freq_response) > 0:
            self.freq_response.normalize_pwr(1000.0)
            low, high = self.freq_response.band_edges(20.0, 20000.0, attn_db=3.0)

            # Calculate ripple in dB
            comps_in_band = [c for c in self.freq_response if low <= c.freq <= high]
            if comps_in_band:
                vals_db = [dbv(c.pwr) for c in comps_in_band]
                ripple_db = max(vals_db) - min(vals_db)
                dev = ripple_db / 2.0
            else:
                dev = 0.0

            print(
                f"Frequency Response .......... {low:0.0f} Hz - {high:0.0f} Hz (+/- {dev:0.1f} dB)"
            )

        print("--------------------------------------------------")


class UnitStatsManager:
    """Class to manage multiple unit statistics."""

    def __init__(self) -> None:
        self.units: dict[str, UnitStats] = {}

    def add_result(self, unit_id: str, test_name: str, result: float) -> None:
        """Add a test result for a specific unit."""
        if unit_id not in self.units:
            self.units[unit_id] = UnitStats(unit_id)
        unit_stats = self.units[unit_id]
        unit_stats.num_tests += 1
        unit_stats.results[test_name] = result
