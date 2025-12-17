"""Audio noise analyzer."""

from analyzer import Analyzer
from audio import Audio
from component import Components
from constants import EQ_BANDS, STD_BANDS
from segment import Segment
from util import Channel

__all__ = ["NoiseAnalyzer"]


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
        super().__init__(audio)
        self.max_components = max_components

    # @override
    def reset(self) -> None:
        """Reset the analyzer state."""
        super().reset()

    # @override
    def get_components(self) -> Components:
        """Get noise frequency components."""
        bins = self.audio.get_bins()

        # If audio is a sine, use channel difference for noise analysis
        if bins.is_sine():
            bins = self.audio.get_bins(Channel.Difference)
        self.components = bins.get_components(self.max_components)
        self.components.name = "Noise Components"
        return self.components

    # @override
    def analyze(self, segment: Segment | None = None) -> Components:
        """
        Analyze noise in the specified audio segment.

        Parameters
        ----------
        segment : Segment | None
            Segment of audio to analyze. If None, use the entire audio.

        Returns
        -------
        Components
            Noise frequency components on equalizer band centers.
        """
        super().analyze(segment)
        self.analysis = self.get_bands(STD_BANDS)
        self.analysis.name = "Noise Bands"

        return self.analysis

    def get_bands(self, centers=EQ_BANDS, ref_freq: float = 1000.0) -> Components:
        """Get noise frequency components on equalizer band centers."""
        bins = self.audio.get_bins()

        # If audio is a sine, use channel difference for noise analysis
        if bins.is_sine():
            bins = self.audio.get_bins(Channel.Difference)

        bands = bins.get_bands(centers)
        bands.normalize_pwr(ref_freq)
        bands.name = "Noise Bands"

        return bands
