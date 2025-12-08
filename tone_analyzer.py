"""Audio tone test."""

from analyzer import Analyzer
from audio import Audio
from component import Component, Components
from constants import BINS_PER_TONE, MAX_TONES, MIN_AUDIO_LEN
from segment import Segment
from tone_stats import ToneStats
from util import Category, alias_freq, is_harmonic

__all__ = ["ToneAnalyzer"]


class ToneAnalyzer(Analyzer):
    """
    Tone test

    Analyzes single-tone or two-tone audio signals by identifying tones,
    distortion products, spurs, and noise, and calculating the resulting
    audio statistics.

    Attributes
    ----------
        audio : Audio
            Audio data to analyze.
        tone : Component
            Primary tone component.
        tone2 : Component
            Secondary tone component (for two-tone signals).
        components : Components
            All identified components in the audio signal.
    """

    """Static Constants"""

    MAX_DIST_ORDER = 3
    """Maximum distortion order to consider"""

    MAX_HARM_ORDER = 5
    """Maximum harmonic order to consider"""

    MIN_SPUR_RATIO = 1e-8
    """Threshold for distinguishing spurs from noise (-80 dB)"""

    MIN_TWO_TONE_RATIO = 1e-2
    """Minimum ratio for two-tone detection (-20 dB). Second tone must be at
    least this large relative to the first tone."""

    # Note: __init__() is inherited from BaseAnalyzer

    # @override
    def reset(self) -> None:
        """Resets/creates members which are calculated from audio"""
        super().reset()
        self.tone = Component()
        self.tone2 = Component()

    # @override
    def get_components(self) -> Components:
        """
        Calculate and return audio tone components.

        Returns
        -------
        Components
            Calculated components.
        """

        # Raise ValueError if audio sample is too short to calculate meaningful
        # statistics
        if len(self.audio) < MIN_AUDIO_LEN:
            raise ValueError("Audio sample is too short to analyze")

        # Get frequency components from frequency bins
        bins = self.audio.get_bins()
        components = bins.get_tones(BINS_PER_TONE, MAX_TONES)
        # TODO?        components.combine(1.0, 1e-4)  # combine components that are within 2 Hz

        # Initialize distortion frequency list
        dist_freqs: list[float] = []

        # The fundamental signal frequency is the bin with the largest power,
        # which is at index zero
        components[0].cat = Category.Tone
        self.tone = components[0]
        start = 1

        # If there is more than one component, compare power and frequency of
        # the second-largest power with those of the first to determine if this
        # is a two-tone signal
        if len(components) > 1:
            harmonic = alias_freq(2.0 * components[0].freq, self.fs)
            if (components[1].pwr < self.MIN_TWO_TONE_RATIO * components[0].pwr) or (
                bins.freq_to_bin(abs(components[1].freq - harmonic)) < BINS_PER_TONE
            ):
                # Second-largest is a small or a harmonic: single tone signal
                dist_freqs = [
                    alias_freq(n * self.tone.freq, self.fs) for n in range(10)
                ]
                self.tone2 = Component(0.0, 0.0)  # flag as invalid
            else:
                # Second-largest is a large and not a harmonic: two-tone signal
                components[1].cat = Category.Tone
                self.tone2 = components[1]
                start = 2

                # Swap tones if tone2 has a lower frequency than the first
                if self.tone2.freq < self.tone.freq:
                    (self.tone, self.tone2) = (self.tone2, self.tone)

                # Calculate distortion frequencies for two-tone signal
                (f1, f2) = (self.tone.freq, self.tone2.freq)
                for n in range(self.MAX_DIST_ORDER):
                    for n2 in range(self.MAX_DIST_ORDER):
                        dist_freqs.append(n * f1 + n2 * f2)
                        dist_freqs.append(n * f1 - n2 * f2)
                        dist_freqs.append(-n * f1 + n2 * f2)

        # Categorize remaining components
        for comp in components[start:]:
            if (
                is_harmonic(comp.freq, self.tone.freq, self.fs, self.MAX_HARM_ORDER)
            ) or (
                is_harmonic(comp.freq, self.tone2.freq, self.fs, self.MAX_HARM_ORDER)
            ):
                comp.cat = Category.Harmonic
            elif bins.freq_in_list(comp.freq, dist_freqs, BINS_PER_TONE):
                comp.cat = Category.Distortion
            elif comp.pwr > self.tone.pwr * self.MIN_SPUR_RATIO:
                comp.cat = Category.Spurious
            elif (
                self.tone2.freq > 0.0
                and comp.pwr > self.tone2.pwr * self.MIN_SPUR_RATIO
            ):
                comp.cat = Category.Spurious
            else:
                comp.cat = Category.Noise

        # Recategorize noise components as spurs which are harmonics of spur components
        spur_freqs = [tone.freq for tone in components if tone.cat == Category.Spurious]
        for idx, comp in enumerate(components):
            if comp.cat == Category.Noise:
                for freq in spur_freqs:
                    if is_harmonic(comp.freq, freq, self.fs, self.MAX_HARM_ORDER):
                        components[idx].cat = Category.Spurious

        self.components = components
        return components

    # @override
    def analyze(self, segment: Segment | None = None) -> ToneStats:
        """
        Calculate and return statistics from audio analysis.

        Parameters
        ----------
        segment : Segment | None
            Segment of audio to analyze. If None, use the entire audio.

        Returns
        -------
        ToneStats
            Audio tone statistics result.
        """

        # Select segment and get components via base class analyze()
        super().analyze(segment)

        # Calculate spurious, distortion, and noise from tones, based on their
        # categories and frequencies
        spur_pwr = 0.0
        dist_pwr = 0.0
        noise_pwr = 0.0
        for component in self.components:
            match component.cat:
                case Category.Spurious:
                    spur_pwr = max(spur_pwr, component.pwr)
                case Category.Harmonic | Category.Distortion:
                    dist_pwr += component.pwr
                case Category.Noise:
                    noise_pwr += component.pwr
                case _:
                    pass

        # Set frequency statistics from powers and return stats
        self.analysis = ToneStats(
            self.audio,
            self.tone.freq,
            self.tone2.freq,
            self.tone.pwr,
            noise_pwr,
            spur_pwr,
            dist_pwr,
        )
        return self.analysis

    # @override
    def print(self, **kwargs) -> None:
        """
        Print the specified analysis data.

        Parameters
        ----------
        kwargs
            Keyword arguments.
        """
        # Call base class print() method to print header
        super().print(**kwargs)

        # Print components if specified
        print_components: bool = kwargs.get("components", False)
        if print_components:
            self.components.print(self.tone.pwr)
            print()

        # Print the result summary
        print(str(self.analysis))
        print()

    def to_csv(
        self, filepath: str, index: bool = False, print_head: bool = False
    ) -> None:
        """
        Export analysis results to a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        index : bool
            Whether to include the index in the CSV output (default is False).
        print_head : bool
            Whether to print the CSV header (default is False).
        """
        tone_stats: ToneStats = self.analyze()
        tone_stats.to_csv(filepath, index, print_head)


def main():
    """Main function for testing the ToneAnalyzer class."""
    import sys

    audio = Audio()
    try:
        fname = sys.argv[1]
        if not audio.read(fname):
            print(f"Error: Could not read audio file '{fname}'")
            return
    except IndexError:
        audio.generate_tone(silence_secs=1.0)
        audio.write("test_tone.wav")
        print("Generated test tone audio 'test_tone.wav'")

    analyzer = ToneAnalyzer(audio)
    stats = analyzer.analyze()
    print(stats)


if __name__ == "__main__":
    main()
