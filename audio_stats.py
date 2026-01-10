"""Audio statistics module."""

from analysis import Analysis
from audio import Audio
from decibels import dbv


__all__ = ["AudioStats"]


class AudioStats(Analysis):
    """Audio statistics.

    Attributes
    ----------
    noise_floor : float
        Noise floor in volts.
    num_channels : int
        Number of channels.
    """

    def __init__(self, audio: Audio):
        super().__init__(audio.md)

        self.noise_floor = audio.avg_noise_floor
        self.num_channels = audio.num_channels

    # @override
    def to_dict(self) -> dict:
        """Convert the statistics result to a dictionary."""
        stats_list: list[tuple[str, float | int | bool]] = [
            ("Noise Floor", self.noise_floor),
            ("Num Channels", self.num_channels),
        ]
        return dict(stats_list)

    # @override
    def summary(self) -> str:
        """Return a one-line summary of the analyzer result."""
        if self.noise_floor > 0.0:
            nf_str = f"{dbv(self.noise_floor):10.3f} dBFS "
        else:
            nf_str = "   Not found"

        return (
            f"{self.md.unit_id:19} "
            f"{self.md.desc:18} "
            f"{self.md.fs:6.0f} "
            f"{self.num_channels:7}   "
            f"{nf_str}"
        )

    @staticmethod
    def summary_header() -> str:
        """Return statistics summary header"""
        return (
            "Audio Statistics\n"
            "Unit                Description            Fs   Chnls   Noise Floor"
        )
