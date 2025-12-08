"""Audio Time-Domain Statistics."""

from copy import copy
from operator import attrgetter

from audio import Audio
from decibels import dbv
from analysis import Analysis

__all__ = ["TimeStats", "TimeStatsList"]


class TimeStats(Analysis):
    """Audio Time-Domain Statistics

    Attributes
    ----------
        dc : float
            DC offset.
        min : float
            Minimum sample value.
        max : float
            Maximum sample value.
        dbfs : float
            Signal dB below full scale.
        rms : float
            Signal RMS value.
        start : float
            Start time of segment in seconds.
        secs : float
            Duration of segment in seconds.
        crest : float
            Crest factor.
        noise_floor : float
            Estimated noise floor.
    """

    def __init__(self, audio: Audio) -> None:
        """
        Set audio sample time-domain statistics.

        Parameters
        ----------
        audio : Audio
            Audio data to analyze.
        """
        super().__init__(audio.md)
        self.dc = audio.dc
        self.min = audio.min
        self.max = audio.max
        self.dbfs = dbv(self.max)
        self.rms = audio.rms
        self.start = self.md.segment.start_secs
        self.secs = self.md.segment.secs
        self.crest = self.max / self.rms if self.rms > 0.0 else 0.0
        self.noise_floor, _ = audio.get_noise_floor()

    def __str__(self) -> str:
        """Return a multi-line string representation of the stats."""
        lines: list[str] = []

        lines.append(f"Time Stats ({self.md.segment})")
        lines.append("-------------------------------")
        lines.append(f"Fs    = {self.md.fs} Hz")
        lines.append(f"Start = {self.start:0.3f}s")
        lines.append(f"Secs  = {self.secs:0.3f}s")
        lines.append(f"DC    = {self.dc:0.3f} = {dbv(self.dc):0.1f} dBFS")
        lines.append(f"Max   = {self.max:0.3f} = {dbv(self.max):0.1f} dBFS")
        lines.append(f"Min   = {self.min:0.3f} = {dbv(self.min):0.1f} dBFS")
        lines.append(f"RMS   = {self.rms:0.3f} = {dbv(self.rms):0.1f} dBFS")
        lines.append(f"Crest = {self.crest:0.3f} = {dbv(self.crest):0.1f} dB")
        lines.append(
            f"Noise = {self.noise_floor:0.3f} = {dbv(self.noise_floor):0.1f} dBFS"
        )

        return "\n".join(lines)

    # @override
    def to_dict(self) -> dict[str, float]:
        """Return an ordered dictionary of statistics."""
        stats_list: list[tuple[str, float]] = [
            # TODO: more here?
            ("RMS", self.rms),
            ("DC", self.dc),
            ("Crest", self.crest),
            ("Secs", self.secs),
        ]
        stats = dict(stats_list)
        return stats

    # @override
    def summary(self) -> str:
        """Return statistics summary string."""
        return (
            f"{self.md.desc:35s} "
            f"{self.secs:7.3f}s "
            f"{self.rms:8.4f} "
            f"{self.dc:8.4f} "
            f"{self.crest:8.4f}"
        )


class TimeStatsList(list[TimeStats]):

    def print(self, sort_attr: str = "start") -> None:
        """
        Print the list sorted on the specified statistic attribute.

        Parameters
        ----------
        sort_attr : str, optional
            Name of the statistic attribute to sort by (default is 'start').
        """
        self.sort(key=attrgetter(sort_attr))
        for time_stats in self:
            print(str(time_stats))
