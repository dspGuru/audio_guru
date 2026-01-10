#!/usr/bin/env python3

import unittest
import numpy as np
from scipy import signal, interpolate

# Adjust import path assuming tests are run from root
from min_sidelobe_windows import min_sidelobe_window, MINIMUM_SIDELOBE_WINDOWS


class TestMinSidelobeWindows(unittest.TestCase):

    def calculate_params(self, n_terms, M=1024 * 32):
        """Calculate window parameters for a given number of terms."""
        # Use sym=False for spectral analysis consistency (large M makes this negligible though)
        w = min_sidelobe_window(n_terms, M, sym=False)

        # 1. ENBW
        # Formula: M * sum(w^2) / (sum(w)^2)
        enbw = M * np.sum(w**2) / (np.sum(w) ** 2)

        # 2. Coherent Gain (dB)
        # Loss formula: -20 * log10( sum(w)/M )
        cg_linear = np.sum(w) / M
        cg_db = -20.0 * np.log10(cg_linear)

        # Spectrum calc
        # oversample=32 provides high frequency resolution to accurately find
        # bandwidth crossings.
        oversample = 32
        fft_size = oversample * M
        W = np.fft.fft(w, n=fft_size)
        W_mag = np.abs(W)
        peak = np.max(W_mag)
        W_db = 20.0 * np.log10(W_mag / peak + 1e-15)

        # 3. Scallop Loss
        # Gain at half-bin offset (bin 0.5)
        # In oversampled grid, bin spacing is 1/oversample.
        # Half bin is index = oversample / 2.
        idx_half_bin = int(oversample / 2)
        scallop_val = W_mag[idx_half_bin]
        scallop_loss = 20.0 * np.log10(peak / scallop_val)

        # 4. Highest Sidelobe Level
        # Find peaks
        peaks, properties = signal.find_peaks(W_db)
        if len(peaks) > 0:
            # Sort by height, descending
            peak_heights = W_db[peaks]
            sorted_peaks = np.sort(peak_heights)[::-1]
            # First is main lobe (0 dB). Second is max sidelobe.
            # Safety check: if multiple peaks at 0 (unlikely), skip them.
            sidelobes = sorted_peaks[sorted_peaks < -0.001]
            if len(sidelobes) > 0:
                highest_sidelobe = -sidelobes[0]  # Return as positive attenuation
            else:
                highest_sidelobe = 0.0
        else:
            highest_sidelobe = 0.0

        # 5. Bandwidths (3 dB and 6 dB)
        # Use interpolation for better precision
        # Only look at main lobe area
        # We know main lobe is at DC (index 0) and symmetric.
        # Find index crossing -3 dB and -6 dB
        half_spectrum = W_db[: fft_size // 2]
        freqs_bins = np.arange(len(half_spectrum)) / oversample

        # Interpolate to find exact crossing
        # Create a simple linear interpolator for the main lobe slope
        # Find where it drops below -10 dB to capture the main lobe width safely
        search_limit = np.argmax(half_spectrum < -10)
        if search_limit == 0:
            search_limit = 100.0 * oversample  # Fallback

        f_interp = interpolate.interp1d(
            half_spectrum[:search_limit], freqs_bins[:search_limit]
        )

        try:
            bw_3db = 2 * f_interp(-3.0)
            bw_6db = 2 * f_interp(
                -6.0
            )  # Actually usually defined at -6.0 or -6.02? Sticking to -6.0 as implied by name.
        except ValueError:
            bw_3db = 0
            bw_6db = 0

        return {
            "enbw_bins": enbw,
            "coherent_gain_db": cg_db,
            "scallop_loss_db": scallop_loss,
            "highest_sidelobe_level_db": highest_sidelobe,
            "bandwidth_3db_bins": bw_3db,
            "bandwidth_6db_bins": bw_6db,
        }

    def test_window_parameters(self):
        """Verify calculated parameters match the dictionary values."""

        # Tolerances
        # Bandwidths: 2% tolerance (due to definitions of 6 dB vs 6.02 dB etc)
        # DB Params: 0.1 dB
        # ENBW: 0.01 bins

        for n_terms, data in MINIMUM_SIDELOBE_WINDOWS.items():
            expected = data["params"]
            calculated = self.calculate_params(n_terms)

            with self.subTest(n_terms=n_terms):
                print(f"Testing {n_terms}-term window...")

                # ENBW
                self.assertAlmostEqual(
                    calculated["enbw_bins"],
                    expected["enbw_bins"],
                    delta=0.01,
                    msg="ENBW mismatch",
                )

                # Coherent Gain
                self.assertAlmostEqual(
                    calculated["coherent_gain_db"],
                    expected["coherent_gain_db"],
                    delta=0.01,
                    msg="Coherent Gain mismatch",
                )

                # Scallop Loss
                self.assertAlmostEqual(
                    calculated["scallop_loss_db"],
                    expected["scallop_loss_db"],
                    delta=0.01,
                    msg="Scallop Loss mismatch",
                )

                # Sidelobe Level
                # For extremely low sidelobes (> 200 dB), numerical noise floor
                # (float64 ~ -300 dB) and FFT precision becomes a factor.
                # We relax tolerance there.
                expected_sl = expected["highest_sidelobe_level_db"]
                if expected_sl > 250.0:
                    delta_sl = 5.0
                elif expected_sl > 200.0:
                    delta_sl = 2.0
                else:
                    delta_sl = 0.5

                self.assertAlmostEqual(
                    calculated["highest_sidelobe_level_db"],
                    expected["highest_sidelobe_level_db"],
                    delta=delta_sl,
                    msg="Sidelobe Level mismatch",
                )

                # 3 dB Bandwidth
                self.assertAlmostEqual(
                    calculated["bandwidth_3db_bins"],
                    expected["bandwidth_3db_bins"],
                    delta=0.1,
                    msg="3-dB BW mismatch",
                )

                # 6 dB Bandwidth
                self.assertAlmostEqual(
                    calculated["bandwidth_6db_bins"],
                    expected["bandwidth_6db_bins"],
                    delta=0.1,
                    msg="6-dB BW mismatch",
                )


if __name__ == "__main__":
    unittest.main()
