"""Audio time-domain statistics module."""

from analysis import Analysis
from audio import Audio
from decibels import dbv

__all__ = ["TimeStats"]


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
        rms : float
            Signal RMS value.
        noise_floor : float
            Estimated noise floor.
        noise_floor_idx : int
            Index of the noise floor in samples.
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
        self.rms = audio.rms
        self.noise_floor, self.noise_floor_idx = audio.get_noise_floor()

    @property
    def noise_floor_secs(self) -> float:
        """Return the time in seconds of the noise floor index."""
        return float(self.noise_floor_idx) / self.md.fs

    def __str__(self) -> str:
        """Return a multi-line string representation of the stats."""
        lines: list[str] = []

        lines.append(f"Time Stats ({self.md.segment})")
        lines.append("---------------------------------------")
        lines.append(f"Fs    = {self.md.fs} Hz")
        lines.append(f"DC    = {self.dc:0.3f} = {dbv(self.dc):0.1f} dBFS")
        lines.append(f"Max   = {self.max:0.3f} = {dbv(self.max):0.1f} dBFS")
        lines.append(f"Min   = {self.min:0.3f} = {dbv(self.min):0.1f} dBFS")
        lines.append(f"RMS   = {self.rms:0.3f} = {dbv(self.rms):0.1f} dBFS")
        lines.append(f"Crest = {self.crest:0.3f} = {dbv(self.crest):0.1f} dB")
        lines.append(
            f"Noise = {self.noise_floor:0.3f} = {dbv(self.noise_floor):0.1f} dBFS"
        )

        return "\n".join(lines)

    @property
    def crest(self) -> float:
        """Return the crest factor (max/RMS)."""
        return self.max / self.rms if self.rms > 0.0 else 0.0

    @property
    def dbfs(self) -> float:
        """Return the maximum signal level in dBFS."""
        return dbv(self.max)

    def print(self) -> None:
        """Print the statistics."""
        print(self.__str__())

    # @override
    def to_dict(self) -> dict[str, float]:
        """Return an ordered dictionary of statistics."""
        stats_list: list[tuple[str, float]] = [
            ("Desc", self.md.desc),
            ("Fs", self.md.fs),
            ("Secs", self.md.secs),
            ("RMS", self.rms),
            ("DC", self.dc),
            ("Crest", self.crest),
        ]
        stats = dict(stats_list)
        return stats

    @staticmethod
    def summary_header() -> str:
        """Return statistics summary header"""
        return (
            "Time Statistics:\n"
            "Unit                Description        Type    Secs    RMS  Crest  dBFS     DCFS"
        )

    # @override
    def summary(self) -> str:
        """Return statistics summary string."""
        return (
            f"{self.md.unit_id:19} "
            f"{self.md.desc:18} "
            f"{self.md.segment.cat.name:6} "
            f"{self.md.secs:5.1f} "
            f"{self.rms:6.3f} "
            f"{self.crest:6.3f}"
            f"{dbv(self.rms):6.1f} "
            f"{dbv(abs(self.dc)):8.1f}"
        )
