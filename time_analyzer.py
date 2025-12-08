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
        seg : Segment | None
            Segment of audio to analyze. If None, use the entire audio.

        Returns
        -------
        TimeStats
            Time-domain statistics.
        """
        super().analyze(segment)  # Select segment and get time statistics
        self.analysis = self.time_stats

        return self.analysis

    # @override
    def print(self, **kwargs) -> None:
        """
        Print the analysis results sorted on the specified statistic attribute.

        Parameters
        ----------
        kwargs
            Keyword arguments.
        """
        super().print(**kwargs)

        print_components: bool = kwargs.get("components", False)
        if print_components:
            self.components.print()
            print()

        print(str(self.time_stats))


def main():
    """Main function for testing."""
    from analyzer_args import get_args

    audio = Audio()
    args = get_args()
    if args.pattern:
        if not audio.read(args.pattern):
            print(f"Error: Could not read audio file '{args.pattern}'")
            return
    elif args.generate:
        audio.generate_tone()
        audio.write("test_tone.wav")
        print("Generated test tone audio 'test_tone.wav'")
    else:
        print(
            "No input audio specified."
            " Use --pattern to specify an audio file"
            " or --generate to create a test tone."
        )
        return

    analyzer = TimeAnalyzer(audio)
    time_stats = analyzer.analyze()
    print(str(time_stats))


if __name__ == "__main__":
    main()
