"""Audio tone statistics class"""

import math
from operator import attrgetter
from pathlib import Path

import pandas as pd

from audio import Audio
from constants import MIN_DB, MIN_PWR
from decibels import dbv, db_str
from time_stats import TimeStats

__all__ = ["ToneStats", "ToneStatsList"]


class ToneStats(TimeStats):
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
        Note: Additional attributes are inherited from TimeStats.
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
        dc : float
            DC (zero Hz) power relative to signal.
        sig : float
            Signal (first tone) power.
        noise : float
            Noise power.
        spur : float
            Spurious power.
        dist : float
            Distortion power.
        """
        # Set sample and time stats
        super().__init__(audio)

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

        lines.append(f"Analysis ({self.md.segment.desc})")
        lines.append("-------------------------------")

        # Frequency stats
        lines.append(f"Freq  = {self.freq:0.1f} Hz")
        if self.freq2 > 0.0:
            lines.append(f"Freq2 = {self.freq2:0.1f} Hz")

        lines += super().__str__().splitlines()[3:]

        # Power stats
        lines.append(f"Sig   = {db_str(self.sig, ' dB')}")
        lines.append(f"SNR   = {db_str(self.snr, ' dB')}")
        lines.append(f"SFDR  = {db_str(self.sfdr, ' dB')}")
        if self.thdn > 0.0:
            lines.append(f"THDN  = {db_str(self.thdn)}")
            lines.append(f"THD   = {db_str(self.thd)}")
            lines.append(f"THD%  = {self.thd_pct:.3f}%")

        return "\n".join(lines)

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
            ("RMS", self.rms),
            ("Crest", self.crest),
            ("Secs", self.secs),
        ]
        stats = self.md.to_dict() | dict(stats_list)
        return stats

    @staticmethod
    def print_summary_header() -> None:
        """Print statistics summary header"""
        print(
            "Description                            THD%    THD   THDN   SFDR     F1     F2   dBFS"
        )
        print(
            "-------------------------------------------------------------------------------------"
        )

    # @override
    def summary(self) -> str:
        """Return statistics summary string."""
        # Description
        s = f"{self.md.mfr:8} {self.md.model:10} {self.md.desc:16}"

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

        # dBFS
        s = s + f"{dbv(self.max):7.1f}"
        return s


class ToneStatsList(list[ToneStats]):
    """Audio tone statistics list class

    A convenience container for collecting, organizing and exporting Audio Stats.
    This class subclasses list[Stats] and provides high-level operations commonly
    used when working with collections of audio analysis results:
    - generating test signals at multiple THD values and collecting their stats,
    - loading statistics from one or more data files produced by an AudioTest,
    - producing a pandas.DataFrame representation for downstream analysis, and
    - writing the collected statistics to CSV.

    Note: This class expects cooperating types/functions to be available at runtime:
    Stats, AudioTest, and a stats object API that exposes .get_stats(), .md.get(),
    and .print_summary(). It also depends on pandas for DataFrame creation and
    pathlib for CSV output.
    """

    def print(self, sort_stat_attr: str = "thd") -> None:
        """
        Print the list sorted on the specified statistic attribute.

        Parameters
        ----------
        sort_stat_attr : str, optional
            Name of the statistic attribute to sort by (default is 'thd').
        """
        ToneStats.print_summary_header()
        self.sort(key=attrgetter(sort_stat_attr))
        for tone_stats in self:
            print(tone_stats.summary())

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representing the list using Stats.get_stats() for
        each stat in the list.
        """
        records = [tone_stats.to_dict() for tone_stats in self]
        df = pd.DataFrame.from_records(records)
        return df

    def to_csv(
        self, filepath: str, index: bool = False, print_head: bool = False
    ) -> None:
        """
        Write the object's DataFrame to the specified CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        index : bool, optional
            Whether to write row indices to the CSV file (default is False).
        """
        df = self.get_dataframe()
        p = Path(filepath)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        # Print the data frame header if specified
        if print_head:
            print(df.head())

        # Write the DataFrame to CSV with specified options
        df.to_csv(p, header=True, index=index, float_format="%.1f")
