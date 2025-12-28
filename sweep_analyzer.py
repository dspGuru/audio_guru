"""Audio sweep test."""

import numpy as np
from scipy import signal

from analyzer import Analyzer
from audio import Audio
from component import Component, Components
from constants import EQ_BANDS, MIN_PWR, REF_FREQ
from freq_resp import FreqResp
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

    def __init__(self, audio: Audio, ref_freq: float = REF_FREQ):
        """
        Initialize the sweep analyzer.

        Parameters
        ----------
        audio : Audio
            Audio data to analyze.
        ref_freq : float
            Reference frequency for normalization (default REF_FREQ).
        """
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
        components.name = "Sweep Components"
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

        # If no components were found, create a dummy component to allow
        # FreqResp to be created
        if not self.components:
            c = Component(freq=0.0, pwr=MIN_PWR, secs=0.0, cat=Category.Unknown)
            self.components.append(c)

        self.analysis = FreqResp(self.components)
        self.analysis.name = "Sweep Frequency Response"

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
            self.components.print()
            print()

        # Print the frequency response
        print("\nSweep Frequency Response")
        print(self.analysis.summary())
        print()
