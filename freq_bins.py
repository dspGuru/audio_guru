"""Audio Frequency Bins Class"""

from copy import copy

import numpy as np
from scipy.fft import rfft
from scipy import signal

from component import Component, Components
from constants import BINS_PER_TONE, MAX_TONES, MIN_PWR
from decibels import db
from metadata import Metadata
from util import alias_freq, Category, Channel

__all__ = ["FreqBins"]


class FreqBins:
    """
    Store and manipulate audio frequency data aka 'bins'.

    Attributes
    ----------
    bins : np.ndarray
        Frequency bins of the audio signal.
    channel : Channel
        Audio channel being analyzed.
    dc : float
        DC offset of the audio signal.
    noise : float
        Noise power in the audio signal.
    md : Metadata
        Metadata of the audio being analyzed.
    """

    def __init__(
        self, samples: np.ndarray, md: Metadata, channel: Channel = Channel.Mean
    ):
        """Initialize frequency bins.

        Parameters
        ----------
        samples : np.ndarray
            Audio samples to analyze.
        md : Metadata
            Metadata of the audio being analyzed.
        """
        # Initialize attributes
        assert isinstance(md, Metadata)
        self.md = copy(md)
        self.channel = channel
        self.bins: np.ndarray = np.ndarray(shape=1, dtype=samples.dtype)
        self.dc = 0.0
        self.noise = 0.0

        # Get audio and convert multi-channel audio to mono as needed
        x = copy(samples)
        if x.ndim > 1:
            match self.channel:
                case Channel.Left:
                    x = x[:, 0]
                case Channel.Right:
                    x = x[:, 1]
                case Channel.Mean:
                    x = np.mean(x, axis=1)
                case Channel.Difference:
                    x = x[:, 0] - x[:, 1]
                case Channel.Unknown:
                    x = np.mean(x, axis=1)
        else:
            self.channel = Channel.Unknown

        # Calculate DC and remove it
        self.dc = float(np.mean(x))
        x -= self.dc

        # Window the signal
        w = signal.windows.blackmanharris(len(x))
        xw = x * w

        # Compute normalized real FFT
        yf = rfft(xw, norm="forward")
        self.bins = yf * yf.conjugate().real

    def __len__(self):
        """Return the length of the entire audio data."""
        return len(self.bins)

    @property
    def fs(self) -> int:
        """Return the sampling frequency in Hz."""
        return self.md.fs

    @property
    def max(self) -> float:
        """Return the maximum value of the selected audio."""
        return float(np.max(self.bins)) if len(self) else 0.0

    @property
    def min(self) -> float:
        """Return the minimum value of the selected audio."""
        return float(np.min(self.bins)) if len(self) else 0.0

    def mean(self, f1: float, f2: float) -> float:
        """
        Return the mean power in the specified frequency range.

        Parameters
        ----------
        f1 : float
            Start frequency, in Hz.
        f2 : float
            Stop frequency, in Hz.

        Returns
        -------
        float
            Mean power in the specified frequency range.
        """
        # Swap frequencies if out of order
        if f1 > f2:
            f1, f2 = f2, f1

        # Get start and stop bins
        bin1 = self.freq_to_bin(f1)
        bin2 = self.freq_to_bin(f2)
        bin2 = min(bin2, len(self.bins) - 1)

        # Return mean power in the specified range
        if bin2 > bin1 and bin1 >= 0 and bin2 < len(self.bins):
            a = np.abs(self.bins[bin1 : bin2 + 1])
            m = np.mean(a)
            return float(m)
        elif bin1 >= 0 and bin1 < len(self.bins):
            return float(np.abs(self.bins[bin1]))
        else:
            raise ValueError("Frequency range is out of bounds.")

    def ripple(self, f1: float, f2: float) -> float:
        """
        Return the ripple (max - min) power in the specified frequency range.

        Parameters
        ----------
        f1 : float
            Start frequency, in Hz.
        f2 : float
            Stop frequency, in Hz.

        Returns
        -------
        float
            Ripple power in the specified frequency range.
        """
        # Get start and stop bins
        bin1 = self.freq_to_bin(f1)
        bin2 = self.freq_to_bin(f2)

        # Return ripple power in the specified range
        if bin2 > bin1 and bin1 >= 0 and bin2 < len(self.bins):
            return float(np.max(self.bins[bin1:bin2]) - np.min(self.bins[bin1:bin2]))
        else:
            return 0.0

    def get_band_edges(self, tol_db: float = 3.0) -> tuple[float, float]:
        """
        Return the -tol_db band edges of the frequency bins.

        Parameters
        ----------
        tol_db : float
            Tolerance in dB below the maximum to define band edges.
        Returns
        -------
        tuple[float, float]
            The lower and upper band edges in Hz.
        """
        a = np.abs(self.bins)
        max_val = np.max(a)
        threshold = max_val * 10 ** (-tol_db / 20)

        # Find indices where bins are above the threshold
        above_threshold = np.where(a >= threshold)[0]

        if max_val == 0.0 or len(above_threshold) == 0:
            return (0.0, 0.0)

        # Get the lower and upper band edges
        lower_edge = self.bin_to_freq(above_threshold[0])
        upper_edge = self.bin_to_freq(above_threshold[-1])

        return (lower_edge, upper_edge)

    def get_bands(self, centers: tuple[float]) -> Components:
        """Get frequency bands as components.

        Parameters
        ----------
        centers : tuple[float]
            Center frequencies of the bands, in Hz.

        Returns
        -------
        Components
            List of band components.
        """
        bands = Components(self.md)

        sqrt2 = 2**0.5
        for center in centers:
            lower = center / sqrt2
            upper = min(center * sqrt2, 0.5 * self.fs)
            pwr = self.mean(lower, upper)
            band = Component(center, pwr, 0.0, Category.Band, lower, upper)
            bands.append(band)

        return bands

    def get_components(self, max_components: int = 256) -> Components:
        """Get frequency components.

        Returns
        -------
        Components
            List of frequency components.
        """
        components = Components(self.md)

        low_freq = 0.0
        nyquist = 0.5 * self.fs
        step = nyquist / max_components
        high_freq = step - 1
        while low_freq < nyquist:
            freq = 0.5 * (low_freq + high_freq)
            pwr = self.mean(low_freq, high_freq)
            component = Component(freq, pwr, 0.0, Category.Band, low_freq, high_freq)
            components.append(component)
            low_freq += step
            high_freq += step

        return components

    def get_tones(
        self, w_tol: int = BINS_PER_TONE, max_components: int = MAX_TONES
    ) -> Components:
        """
        Find the audio's components, DC, and noise.

        Parameters
        ----------
        w_tol : int
            Tolerance for component detection, in bins.
        max_components : int
            Maximum number of components to identify.

        Returns
        -------
        Components
            The tone components found in the audio.
        """
        tones = Components(self.md)

        # Get absolute value of bins
        bins = np.absolute(self.bins)

        # Get components
        last_bin = len(bins) - 1
        while len(tones) < max_components:
            center = int(np.where(bins == bins.max())[0][0])
            first = max(center - w_tol, 0)
            last = min(center + w_tol, last_bin)
            freq = self.bin_to_freq(center)
            pwr = float(sum(bins[first:last]))

            # Recalculate center frequency as power-weighted average
            weighted_sum = 0.0
            total_power = 0.0
            for bin in range(first, last):
                bin_freq = self.bin_to_freq(bin)
                bin_pwr = bins[bin]
                weighted_sum += bin_freq * bin_pwr
                total_power += bin_pwr
            freq = weighted_sum / total_power if total_power > 0 else freq

            # Create tone component
            tone = Component(
                freq,
                pwr,
                0.0,
                Category.Tone,
                self.bin_to_freq(first),
                self.bin_to_freq(last),
            )

            # Set the tone's bins to zero to exclude them from the noise
            # calculation below
            bins[first:last] = 0.0
            tones.append(tone)

        # Get noise as the sum of the remaining non-tone bins
        self.noise = sum(bins)
        return tones

    def is_noise(self, threshold_db: float = 10.0) -> bool:
        """
        Determine if the audio is noise.

        Parameters
        ----------
        threshold_db : float
            SNR threshold in dB below which the audio is considered noise.

        Returns
        -------
        bool
            True if the audio is noise.
        """
        # Get tones
        tones = self.get_tones()

        # If no components found, it is noise
        if not tones:
            return True

        # Get power of largest component
        max_comp_pwr = tones[0].pwr

        # Calculate average tone power excluding the largest component
        avg_tone_pwr = sum([tone.pwr for tone in tones]) / (len(tones) - 1)
        snr_db = db(max_comp_pwr / avg_tone_pwr)

        # Return True if SNR is less than the threshold
        return snr_db < threshold_db

    def is_sine(self, tol_db: float = 3.0) -> bool:
        """
        Determine if the audio is a sine wave.

        Parameters
        ----------
        tol_db : float
            Tolerance in dB above the noise floor to consider as sine wave.

        Returns
        -------
        bool
            True if the audio is a sine wave.
        """
        # Get components
        tones = self.get_tones()

        # If no components found, it is not a sine wave
        if not tones:
            return False

        # Get power of largest component
        max_comp_pwr = tones[0].pwr if tones else 0.0

        # If no noise, it is a sine wave
        if self.noise <= MIN_PWR:
            return True

        # Calculate SNR in dB
        snr_db = db(max_comp_pwr / self.noise)

        # Return True if SNR is within tolerance
        return snr_db >= tol_db

    def bin_to_freq(self, bin: int) -> float:
        """Return the frequency of the bin number."""
        return self.fs * 0.5 * bin / len(self.bins)

    def freq_to_bin(self, freq: float) -> int:
        """Return the bin number of the frequency."""
        if freq < 0.0:
            raise ValueError("Frequency must be non-negative")
        return round(freq / (self.fs * 0.5) * len(self.bins))

    def freq_in_list(
        self, freq: float, freqs: list[float], bins_per_component: int
    ) -> bool:
        """
        Determine if the specified frequency is in the list of frequencies.

        Parameters
        ----------
        freq : float
            Frequency to test.
        freqs : list[float]
            List of frequencies to search.

        Returns
        -------
        bool
            True if the inputs are valid and the specified frequency is in the
            list of frequencies.
        """

        # Return False if any input is invalid
        if freq <= 0.0 or not freqs:
            return False

        # Check if frequency is within tolerance of any frequency in the list
        for target_freq in freqs:
            target_freq = alias_freq(target_freq, self.fs)
            bin_diff = self.freq_to_bin(abs(freq - target_freq))
            if bin_diff <= bins_per_component:
                return True  # frequency is in the list

        # Frequency not found in the list: return False
        return False
