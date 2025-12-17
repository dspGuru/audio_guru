"""Audio statistics module."""

from analysis import Analysis
from audio import Audio
from decibels import dbv


__all__ = ["AudioStats"]


class AudioStats(Analysis):

    def __init__(self, audio: Audio):
        super().__init__(audio.md)

        self.noise_floor, self.noise_floor_idx = audio.get_noise_floor()
        self.num_channels = audio.num_channels()

    @property
    def noise_floor_secs(self) -> float:
        """Return the time in seconds of the noise floor index."""
        return float(self.noise_floor_idx) / self.md.fs

    # @override
    def to_dict(self) -> dict:
        """Convert the statistics result to a dictionary."""
        stats_list: list[tuple[str, float | int]] = [
            ("Noise Floor", self.noise_floor),
            ("Noise Floor Secs", self.noise_floor_secs),
            ("Num Channels", self.num_channels),
        ]
        return dict(stats_list)

    # @override
    def summary(self) -> str:
        """Return a one-line summary of the analyzer result."""
        if self.noise_floor > 0.0:
            nf_str = (
                f"{dbv(self.noise_floor):10.3f} dBFS at {self.noise_floor_secs:5.1f}s"
            )
        else:
            nf_str = "   Not found"

        return (
            f"{self.md.unit_id:19} "
            f"{self.md.desc:18} "
            f"{self.num_channels:8} "
            f"{nf_str}"
        )

    @staticmethod
    def summary_header() -> str:
        """Return statistics summary header"""
        return (
            "Audio Statistics:\n"
            "Unit                Description        Channels    Noise Floor"
        )
