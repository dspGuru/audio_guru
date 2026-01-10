"""Audio noise analyzer."""

from analysis import Analysis
from analyzer import Analyzer
from audio import Audio
from component import Components
from constants import EQ_BANDS, REF_FREQ, STD_BANDS
from freq_resp import FreqResp
from util import Channel

__all__ = ["NoiseAnalyzer"]


class NoiseAnalysis(Analysis):
    """Noise analysis class."""

    def __init__(self, components: Components, bands: Components):
        super().__init__(components.md)
        self.components = components
        self.freq_resp = FreqResp(self.components)
        self.bands = bands

    # @override
    def to_dict(self) -> dict:
        """Convert the analyzer result to a dictionary."""
        return self.freq_resp.to_dict()

    # @override
    def summary(self) -> str:
        """Return a one-line summary of the analyzer result."""
        return self.freq_resp.summary()

    # @override
    @staticmethod
    def summary_header() -> str:
        """Return the header for the summary table."""
        return (
            "Noise Frequency Response\n"
            "Unit                Description        Type      Lower   Upper  Ripple   Minimum"
        )


class NoiseAnalyzer(Analyzer):
    """
    Audio noise analysis class.

    Attributes
    ----------
        audio : Audio
            Audio data to analyze.
        components : Components
            Frequency components identified in the audio.
        max_components : int
            Maximum number of frequency components to identify.
    """

    def __init__(self, audio: Audio, max_components: int = 256):
        """
        Initialize the noise analyzer.

        Parameters
        ----------
        audio : Audio
            Audio data to analyze.
        max_components : int
            Maximum number of frequency components to identify (default 256).
        """
        super().__init__(audio)
        self.max_components = max_components

    # @override
    def reset(self) -> None:
        """Reset the analyzer state."""
        super().reset()

    # @override
    def get_components(self) -> Components:
        """Get noise frequency components."""
        # If audio is a sine, use channel difference for noise analysis
        if self.audio.bins.is_sine():
            self.audio.select(Channel.Difference)

        # Get and return noise components
        self.components = self.audio.bins.get_components(self.max_components)
        self.components.name = "Noise Components"
        return self.components

    # @override
    def analyze(self) -> Components:
        """
        Analyze noise in the specified audio segment.

        Parameters
        ----------
        segment : Segment | None
            Segment of audio to analyze. If None, use the entire audio.

        Returns
        -------
        NoiseAnalysis
            Noise analysis.
        """
        super().analyze()
        self.bands = self.get_bands(STD_BANDS)
        self.bands.name = "Noise Bands"
        self.analysis = NoiseAnalysis(self.components, self.bands)

        return self.analysis

    def get_bands(self, centers=EQ_BANDS, ref_freq: float = REF_FREQ) -> Components:
        """
        Get noise frequency components on equalizer band centers.

        Parameters
        ----------
        centers : list[float]
            List of center frequencies for bands (default EQ_BANDS).
        ref_freq : float
            Reference frequency for normalization (default REF_FREQ).
        """
        # If audio is a sine, use channel difference for noise analysis
        if self.audio.bins.is_sine():
            self.audio.select(Channel.Difference)

        # Get and return noise bands
        bands = self.audio.bins.get_bands(centers)
        bands.normalize_pwr(ref_freq)
        bands.name = "Noise Bands"

        return bands
