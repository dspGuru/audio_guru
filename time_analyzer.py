"""Audio time-domain analyzer."""

from analyzer import Analyzer
from audio import Audio
from component import Component, Components
from segment import Segment
from time_stats import TimeStats


class TimeAnalyzer(Analyzer):
    """
    Audio time-domain analysis class.

    Attributes
    ----------
        audio : Audio
            Audio data to analyze.
    """

    def __init__(self, audio: Audio):
        """
        Initialize the time analyzer.

        Parameters
        ----------
        audio : Audio
            Audio data to analyze.
        """
        super().__init__(audio)

    # @override
    def reset(self) -> None:
        """Resets/creates members which are calculated from audio"""
        super().reset()

    # @override
    def get_components(self, **kwargs) -> Components:
        """Return components, which consists of a single component representing
        the primary frequency of the audio."""

        # Create empty Components object
        self.components = Components(self.audio.md, name="Frequency")

        # Create a single component representing the primary audio frequency
        # and the signal's RMS power
        component = Component(freq=self.audio.freq, pwr=self.audio.rms)
        self.components.append(component)

        return self.components

    # @override
    def analyze(self, segment: Segment | None = None) -> TimeStats:
        """
        Abstract method to calculate and return statistics from audio analysis.

        Parameters
        ----------
        Parameters
        ----------
        segment : Segment | None
            Segment of audio to analyze. If None, use the entire audio.

        Returns
        -------
        TimeStats
            Time-domain statistics.
        """
        super().analyze(segment)  # Select segment and get time statistics
        self.analysis = self.time_stats

        return self.analysis
