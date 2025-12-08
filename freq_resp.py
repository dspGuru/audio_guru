"""Audio frequency response class"""

from analysis import Analysis
from component import Components

__all__ = ["FreqResp"]


class FreqResp(Analysis):
    """Class to represent a frequency response."""

    def __init__(self, components: Components):
        super().__init__(components.md)

        self.components = components
        self.noise_floor: float = min([comp.pwr for comp in components])

        # Calculate ripple and band edges relative to a nominal audio center
        # band of 300 Hz to 3 kHz
        lower, upper = (300, 3000)
        self.ripple: float = components.ripple(lower, upper)
        self.lower_edge, self.upper_edge = components.band_edges(lower, upper, 3.0)

    def to_dict(self) -> dict[str, float]:
        """Return frequency response as a dictionary."""
        return {
            "noise_floor": self.noise_floor,
            "ripple": self.ripple,
            "lower_edge": self.lower_edge,
            "upper_edge": self.upper_edge,
        }

    def summary(self) -> str:
        """Provide a summary of the frequency response."""
        return (
            f"Noise Floor={self.noise_floor:.1f} dB, "
            f"Ripple={self.ripple:.1f} dB, "
            f"Lower Edge={self.lower_edge:.1f} Hz, "
            f"Upper Edge={self.upper_edge:.1f} Hz"
        )
