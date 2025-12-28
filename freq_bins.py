"""Audio Frequency Bins Class"""

from copy import copy

import numpy as np
from scipy.fft import rfft
from scipy import signal

from component import Component, Components
from constants import MAX_BINS_PER_TONE, MAX_TONES, MIN_PWR
from decibels import db
from metadata import Metadata
from util import alias_freq, Category, Channel

__all__ = ["FreqBins"]


def blackmanharris7(M, sym=True):
    """Return a minimum 7-term Blackman-Harris window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    "The Use of DFT Windows in Signal-to-Noise Ratio and Harmonic Distortion
    Computations", IEEE Transactions on Instrumentation and Measurement,
    April 1994, vol. 43, pp 194-199, by O. M. Solomon, Jr.
    """
    a = np.asarray(
        [
            0.271051400693424,
            0.433297939234485,
            0.21812299954311,
            0.06592544638803099,
            0.010811742098371,
            0.000776584825226,
            0.000013887217352,
        ],
        dtype=np.float64,
    )
    return signal.windows.general_cosine(M, a, sym=sym)


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
    tones : Components
        Tone components found in the audio.
    n_samples : int
        Number of samples in the audio signal.
    """

    def __init__(
        self,
        samples: np.ndarray,
        md: Metadata,
        channel: Channel = Channel.Unknown,
    ) -> None:
        """
        Initialize the frequency bins.

        Parameters
        ----------
        samples : np.ndarray
            Audio samples.
        md : Metadata
            Metadata associated with the audio.
        channel : Channel
            Channel for which to compute bins (default is Channel.Unknown).
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

        # Store number of audio samples being analyzed
        self.n_samples = len(x)

        # Calculate DC and remove it
        self.dc = float(np.mean(x))
        x -= self.dc

        # Window the signal
        w = blackmanharris7(len(x))
        xw = x * w

        # Compute normalized real FFT
        yf = rfft(xw, norm="forward")
        self.bins = (yf * yf.conjugate()).real

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

    def get_components(self, max_components: int = 10) -> Components:
        """
        Identify frequency components in the audio.

        Parameters
        ----------
        max_components : int
            Maximum number of components to return (default 10).

        Returns
        -------
        Components
            List of identified frequency components.
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

    def get_tones(self, max_components: int = MAX_TONES) -> Components:
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

            # Find the first bin, which is the first before the center which
            # has monotonically decreasing power
            first = center
            while first > 0 and bins[first - 1] <= bins[first]:
                first -= 1

            # Find the last bin, which is the last after the center which has
            # monotonically decreasing power
            last = center
            while last < last_bin and bins[last + 1] <= bins[last]:
                last += 1
            last += 1

            # If the number of bins spanned is greater than MAX_BINS_PER_TONE,
            # limit the span to MAX_BINS_PER_TONE, with clipping at the edges
            if last - first > MAX_BINS_PER_TONE:
                first = max(0, center - MAX_BINS_PER_TONE // 2)
                last = min(last_bin + 1, center + MAX_BINS_PER_TONE // 2)

            # Calculate the center frequency as power-weighted average of the
            # tone's bins
            weighted_sum = 0.0
            total_power = 0.0
            for bin in range(first, last):
                bin_freq = self.bin_to_freq(bin)
                bin_pwr = bins[bin]
                weighted_sum += bin_freq * bin_pwr
                total_power += bin_pwr
            freq = weighted_sum / total_power if total_power > 0 else freq

            # Calculate the power of the tone as the sum of the tone's bins
            pwr = float(sum(bins[first:last]))

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

        NOTES
        -----
        The function does not check for significant skirt energy, which
        might occur if the signal was clipped, such as with a sweep signal.
        Therefore, the signal should be checked as a possible sweep before
        using this function.
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
        return self.fs * bin / self.n_samples

    def freq_to_bin(self, freq: float) -> int:
        """Return the bin number of the frequency."""
        if freq < 0.0:
            raise ValueError("Frequency must be non-negative")
        return round(freq / self.fs * self.n_samples)

    def freq_in_list(
        self, freq: float, freq_list: list[float], bins_per_component: int = 3
    ) -> bool:
        """
        Check if the specified frequency is present in the list of frequencies.

        Parameters
        ----------
        freq : float
            Frequency to check.
        freq_list : list[float]
            List of frequencies to check against.
        bins_per_component : int
            Number of bins to consider around the target frequency (default 3).

        Returns
        -------
        bool
            True if the frequency is in the list (within tolerance).
        """

        # Return False if any input is invalid
        if freq <= 0.0 or not freq_list:
            return False

        # Check if frequency is within tolerance of any frequency in the list
        for target_freq in freq_list:
            target_freq = alias_freq(target_freq, self.fs)
            bin_diff = self.freq_to_bin(abs(freq - target_freq))
            if bin_diff <= bins_per_component:
                return True  # frequency is in the list

        # Frequency not found in the list: return False
        return False
