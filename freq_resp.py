"""Audio frequency response module."""

from analysis import Analysis
from component import Components

__all__ = ["FreqResp"]


class FreqResp(Analysis):
    """Class to represent a frequency response."""

    def __init__(self, components: Components):
        super().__init__(components.md)

        self.components = components
        self.minimum: float = min([comp.pwr for comp in components])

        # Calculate ripple and band edges relative to an audio
        # analysis band above and below the audible range
        lower, upper = (10, 22000)
        self.ripple: float = components.ripple(2.0 * lower, 0.5 * upper)
        self.lower_edge, self.upper_edge = components.band_edges(lower, upper, 3.0)

    def __str__(self) -> str:
        """Return a multi-line string representation of the frequency response."""
        lines: list[str] = []

        lines.append(f"{self.md.segment.desc}")
        lines.append("---------------------------------------")
        lines.append(f"Lower Edge = {self.lower_edge:.1f} Hz")
        lines.append(f"Upper Edge = {self.upper_edge:.1f} Hz")
        lines.append(f"Ripple     = {self.ripple:.1f} dB")
        lines.append(f"Minimum    = {self.minimum:.1f} dB")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        """Return frequency response as a dictionary."""
        return {
            "minimum": self.minimum,
            "ripple": self.ripple,
            "lower_edge": self.lower_edge,
            "upper_edge": self.upper_edge,
        }

    # @override
    @staticmethod
    def summary_header() -> str:
        return (
            "Frequency Response:\n"
            "Unit                Description        Type      Lower   Upper  Ripple   Minimum"
        )

    def summary(self) -> str:
        """Provide a summary of the frequency response."""
        return (
            f"{self.md.unit_id:19} "
            f"{self.md.desc:18} "
            f"{self.md.segment.cat.name:6} "
            f"{self.lower_edge:8.0f}"
            f"{self.upper_edge:8.0f}"
            f"{self.ripple:8.1f}"
            f"{self.minimum:10.1f}"
        )
