"""Audio tone statistics module"""

import math

from analysis import Analysis
from audio import Audio
from constants import MIN_DB, MIN_PWR
from decibels import db_str

__all__ = ["ToneStats"]


class ToneStats(Analysis):
    """Audio Time and Frequency Statistics

    Attributes
    ----------
        freq : float
            First tone frequency, in Hz.
        freq2 : float
            Second tone frequency, in Hz.
        sig : float
            Signal (first tone) power.
        snr : float
            Signal-to-noise ratio.
        sfdr : float
            Spurious-free dynamic range.
        thd : float
            Total harmonic distortion.
        thdn : float
            Total harmonic distortion plus noise.
    """

    def __init__(
        self,
        audio: Audio,
        freq: float,
        freq2: float,
        sig: float,
        noise: float,
        spur: float,
        dist: float,
    ):
        """Initialize statistics from audio data and tone analysis.

        Parameters
        ----------
        audio : Audio
            Audio data to analyze.
        freq : float
            First tone frequency, in Hz.
        freq2 : float
            Second tone frequency, in Hz.
        sig : float
            Signal (first tone) power.
        noise : float
            Noise power.
        spur : float
            Spurious power.
        dist : float
            Distortion power.
        """
        # Initialize super-class
        super().__init__(audio.md)

        # Set frequency stats
        self.freq = freq
        self.freq2 = freq2

        # Set basic power stats
        self.sig = sig
        if noise > MIN_PWR:
            self.snr = sig / noise
        else:
            self.snr = -MIN_DB

        # Calculate SFDR
        spur = max(spur, dist)
        if spur > MIN_PWR:
            self.sfdr = sig / spur
        else:
            self.sfdr = 0.0

        # Calculate distortion-related values if signal is present
        if sig > 0.0:
            self.thd = dist / sig
            self.thdn = (dist + noise) / sig
        else:
            # No signal present: flag distortion stats as invalid
            self.thd = 0.0
            self.thdn = 0.0

    def __str__(self):
        """Return a multi-line string representation of the stats."""
        lines: list[str] = []

        lines.append(f"{self.md.segment.desc}")
        lines.append("---------------------------------------")

        # Frequency stats
        lines.append(f"Freq  = {self.freq:0.1f} Hz")
        if self.freq2 > 0.0:
            lines.append(f"Freq2 = {self.freq2:0.1f} Hz")

        # Power stats
        lines.append(f"Sig   = {db_str(self.sig, ' dBFS')}")
        lines.append(f"SNR   = {db_str(self.snr, ' dB')}")
        lines.append(f"SFDR  = {db_str(self.sfdr, ' dB')}")
        if self.thdn > 0.0:
            lines.append(f"THDN  = {db_str(self.thdn, ' dB')}")
            lines.append(f"THD   = {db_str(self.thd, ' dB')}")
            lines.append(f"THD%  = {self.thd_pct:.3f}%")

        return "\n".join(lines)

    def print(self) -> None:
        """Print the statistics."""
        print(self.__str__())

    @property
    def thd_pct(self) -> float:
        """Return total harmonic distortion as a percentage."""
        return math.sqrt(self.thd) * 100.0

    # @override
    def to_dict(self) -> dict[str, float]:
        """Return an ordered dictionary of statistics."""
        stats_list = [
            ("THD%", self.thd_pct),
            ("THD", self.thd),
            ("SNR", self.snr),
            ("THDN", self.thdn),
            ("SFDR", self.sfdr),
            ("F1", self.freq),
            ("F2", self.freq2),
            ("Secs", self.secs),
        ]
        stats = self.md.to_dict() | dict(stats_list)
        return stats

    @staticmethod
    def summary_header() -> str:
        """Return statistics summary header"""
        return (
            "Tone Statistics:\n"
            "Unit                Description          THD%    THD   THDN   SFDR     F1     F2"
        )

    # @override
    def summary(self) -> str:
        """Return statistics summary string."""
        # Description
        s = f"{self.md.unit_id:19} {self.md.desc:18}"

        # THD, THDN
        if self.thd <= 0.0:
            s = s + "                     "
        else:
            s = s + f" {self.thd_pct:6.3f} {db_str(self.thd):6} {db_str(self.thdn):6}"

        # SFDR, F1, F2
        s = s + f" {db_str(self.sfdr):6}  {round(self.freq):5.0f}"
        if self.freq2 > 0.0:
            s = s + f"  {round(self.freq2):5.0f}"
        else:
            s = s + "       "

        return s
