"""Audio time-domain analyzer."""

from analyzer import Analyzer
from audio import Audio
from component import Component, Components
from time_stats import TimeStats


class TimeAnalyzer(Analyzer):
    """
    Audio time-domain analysis class.

    Attributes
    ----------
        audio : Audio
            Audio data to analyze.
    """

    # Note: __init__() is inherited from super class

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
    def analyze(self) -> TimeStats:
        """
        Calculate and return time-domain statistics.

        Returns
        -------
        TimeStats
            Time-domain statistics.
        """
        super().analyze()  # Gets time statistics
        self.analysis = self.time_stats

        return self.analysis
