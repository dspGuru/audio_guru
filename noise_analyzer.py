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

    # @override
    def print(self, **kwargs) -> None:
        """
        Print the specified analysis data.

        Parameters
        ----------
        args
            Command-line arguments.
        """
        super().print(**kwargs)

        # Analyze noise and get the bands
        bands = self.analyze()

        # Print the components
        print(f"Noise Components ({len(self.components)} found):")
        self.components.print()
        print()

        bands.print()
        print()


def main():
    """Main function for testing."""
    import sys

    from generate import noise, silence

    audio = Audio()
    try:
        fname = sys.argv[1]
        if not audio.read(fname):
            print(f"Error: Could not read audio file '{fname}'")
            return
    except IndexError:
        # Generate test noise audio with leading and trailing silence
        audio = silence(1.0)
        audio.append(noise(silence_secs=1.0))
        audio.write("test_noise.wav")
        print("Wrote noise audio to 'test_noise.wav'")

    analyzer = NoiseAnalyzer(audio, max_components=25)
    components = analyzer.get_components()
    components.print()
    print()

    bands = analyzer.analyze()
    bands.print()


if __name__ == "__main__":
    main()
