"""Audio sweep test."""

import numpy as np
from scipy import signal

from analyzer import Analyzer
from audio import Audio
from component import Component, Components
from constants import DEFAULT_AMP, EQ_BANDS, MIN_AUDIO_LEN, MIN_PWR
from freq_resp import FreqResp
from generate import silence, sweep
from segment import Segment
from util import Category


class SweepAnalyzer(Analyzer):
    """
    Audio sweep test class.

    Attributes
    ----------
        audio : Audio
            Audio data to analyze.
        ref_freq : float
            Reference frequency for normalization.
        components: Components
            Frequency components identified in the audio.
    """

    def __init__(self, audio: Audio, ref_freq: float = 1000.0):
        super().__init__(audio)
        self.ref_freq: float = ref_freq

    # @override
    def reset(self) -> None:
        """Reset the analyzer state."""
        super().reset()

    # @override
    def get_components(self) -> Components:
        """Get sweep frequency components.

        Returns
        -------
        Components
            List of components.
        """
        components = Components(self.audio.md)

        # Use the short-time Fourier transform (STFT) to analyze the sweep

        # Collect samples and sample rate from the Audio object
        samples = self.audio.selected_samples

        if len(samples) < MIN_AUDIO_LEN:
            raise RuntimeError("Not enough audio samples found for analysis.")

        # Ensure samples are a numpy array
        samples = np.asarray(samples)

        # Convert to mono if necessary
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        # Remove DC offset
        samples = samples - np.mean(samples)

        # STFT parameters — reasonable defaults for sweep analysis
        nperseg = 4096
        noverlap = int(nperseg * 0.75)
        f, t, z = signal.stft(
            samples,
            fs=self.audio.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            padded=False,
            boundary=None,
        )
        mags = np.abs(z)

        components = Components(self.audio.md)
        eps = MIN_PWR
        for i in range(mags.shape[1]):
            col = mags[:, i]
            if np.all(col <= eps):
                continue
            idx = int(np.argmax(col))
            freq_hz = float(f[idx])
            pwr = float(np.sum(np.abs(col) ** 2.0))  # power in this frame
            secs = float(t[i])

            # If the frequency is greater than 1 Hz, append the frequency bin
            # as a component
            if freq_hz > 1.0:
                component = Component(freq_hz, pwr, secs, Category.Tone)
                components.append(component)

        # Combine consecutive duplicate frequency entries (common for long frames)
        components.combine()
        self.components = components
        return self.components

    # @override
    def analyze(self, segment: Segment | None = None) -> FreqResp:
        """Get sweep statistics for frequency bands.

        Parameters
        ----------
        segment : Segment | None
            Segment of audio to analyze. If None, use the entire audio.

        Returns
        -------
        Frequency bands as Components
        """
        super().analyze(segment)
        self.analysis = FreqResp(self.components)
        return self.analysis

    def get_bands(self, centers=EQ_BANDS, ref_freq: float | None = None) -> Components:
        """Get sweep frequency components on band centers.

        Parameters
        ----------
        centers : list[float]
            List of center frequencies for bands.
        ref_freq : float | None
            Reference frequency for normalization. If None, use self.ref_freq.
        """
        if ref_freq is None:
            ref_freq = self.ref_freq
        bands = self.components.get_bands(centers)
        bands.normalize_pwr(ref_freq)
        bands.name = "Bands"
        return bands

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
            print(f"Sweep Components ({len(self.components)} found):")
            self.components.print()
            print()

        # Print the frequency response
        freq_resp = self.analyze()
        print(f"\nSweep Frequency Response")
        print(freq_resp.summary())
        print()


def main():
    """Main function for testing the SweepTest class."""

    # Generate a sweep from 20 Hz to 20 kHz with one second of silence before
    # and after the sweep
    audio = silence(1.0)
    audio.append(sweep(f_start=20.0, f_stop=20000.0, secs=10.0, amp=DEFAULT_AMP))
    audio.append(silence(secs=1.0, fs=audio.fs))
    audio.write("test_sweep.wav")
    print("Wrote sweep audio to 'test_sweep.wav'")

    # Create analyzer instance
    analyzer = SweepAnalyzer(audio)

    # Get the sweep segments from the audio and verify that exactly one is found
    segments = audio.get_segments()
    assert len(segments) == 1

    # Analyze the sweep segment
    audio.select(segments[0])
    components = analyzer.get_components()
    components.print()

    # Analyze and print the sweep frequency response
    freq_resp = analyzer.analyze()
    print(freq_resp.summary())

    # Print the sweep components
    analyzer.print()


if __name__ == "__main__":
    main()
