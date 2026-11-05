"""Audio Frequency Bins Class"""

from copy import copy
from functools import cached_property

import numpy as np
from scipy.fft import rfft

from component import Component, Components
from constants import MAX_BINS_PER_TONE, MAX_TONES, MIN_PWR
from decibels import db
from metadata import Metadata
from util import alias_freq, Category, Channel

__all__ = ["FreqBins"]

# Module-level cache for window functions by (length, symmetry)
_window_cache: dict[tuple[int, bool], np.ndarray] = {}
_window_s2_cache: dict[tuple[int, bool], float] = {}

# Seven-term minimum sidelobe Blackman-Harris coefficients from reference below
BH7_COEFFS = np.asarray(
    [
        0.271051400693424,
        0.433297939234485,
        0.21812299954311,
        0.06592544638803099,
        0.010811742098371,
        0.000776584825226,
        0.000013887217352,
    ],
    dtype=np.float32,
)


def blackmanharris7(M: int, sym: bool = True) -> np.ndarray:
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
    # Check cache first
    cache_key = (M, sym)
    if cache_key in _window_cache:
        return _window_cache[cache_key]

    # Calculate window
    # Calculate window using Chebyshev recurrence (1 cos evaluation instead of 6)
    if M <= 0:
        w = np.array([], dtype=np.float32)
        s2 = 0.0
    elif M == 1:
        w = np.ones(1, dtype=np.float32)
        s2 = 1.0
    else:
        n_pts = M if sym else M + 1
        c1 = np.linspace(-np.pi, np.pi, n_pts, dtype=np.float32)[:M]
        np.cos(c1, out=c1)
        a = BH7_COEFFS
        p6 = np.float32(32.0 * a[6])
        p5 = np.float32(16.0 * a[5])
        p4 = np.float32(8.0 * a[4] - 48.0 * a[6])
        p3 = np.float32(4.0 * a[3] - 20.0 * a[5])
        p2 = np.float32(2.0 * a[2] - 8.0 * a[4] + 18.0 * a[6])
        p1 = np.float32(a[1] - 3.0 * a[3] + 5.0 * a[5])
        p0 = np.float32(a[0] - a[2] + a[4] - a[6])
        w = np.full_like(c1, p6)
        for pk in (p5, p4, p3, p2, p1, p0):
            w *= c1
            w += pk
        if not sym and M > 12:
            s2 = float(a[0] * a[0] + 0.5 * np.sum(a[1:] * a[1:]))
        else:
            s2 = float(np.mean(w * w))
    _window_s2_cache[cache_key] = s2

    # Cache and return (bound cache size to avoid unbounded memory use)
    if len(_window_cache) >= 16:
        _window_cache.clear()
    _window_cache[cache_key] = w
    return w


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

        # Get audio and convert multi-channel audio to mono as needed. Use a
        # view where possible to avoid redundant copying.
        x = np.asarray(samples)
        if x.ndim > 1:
            match self.channel:
                case Channel.Left:
                    x = x[:, 0]
                case Channel.Right:
                    x = x[:, 1]
                case Channel.Stereo | Channel.Unknown:
                    if x.shape[1] == 2:
                        x = (x[:, 0] + x[:, 1]) * np.float32(0.5)
                    else:
                        x = np.mean(x, axis=1)
                case Channel.Difference:
                    x = x[:, 0] - x[:, 1]
        else:
            self.channel = Channel.Unknown

        # Store number of audio samples being analyzed
        self.n_samples = len(x)

        # Calculate DC and remove it
        self.dc = float(np.mean(x))
        if self.dc != 0.0:
            # Create a copy only if we need to modify x and it's a view
            if x.base is not None or not x.flags.writeable:
                x = x - self.dc
            else:
                x -= self.dc

        # Window the signal
        w = blackmanharris7(len(x), sym=False)
        xw = x * w

        # Compute normalized real FFT and convert to power spectral density
        yf = rfft(xw, norm="forward")
        bins = yf.real * yf.real + yf.imag * yf.imag
        if len(w) > 0:
            s2 = _window_s2_cache.get((len(x), False))
            if s2 is None:
                s2 = float(np.mean(w * w))
            if s2 > 0.0:
                bins *= 1.0 / s2
                if len(bins) > 2:
                    bins[1:-1] *= 2.0
                elif len(bins) == 2 and self.n_samples % 2 != 0:
                    bins[1:] *= 2.0
        self.bins = bins

    def __len__(self):
        """Return the length of the entire audio data."""
        return len(self.bins)

    def _parabolic_peak(self, bin_idx: int, bins: np.ndarray) -> tuple[float, float]:
        """Estimate peak frequency and power using parabolic interpolation.

        Fits a parabola through three bins (left, center, right) around the peak
        to provide sub-bin frequency accuracy and power correction.

        Parameters
        ----------
        bin_idx : int
            Center bin index.
        bins : np.ndarray
            Power spectrum bins.

        Returns
        -------
        tuple[float, float]
            Interpolated frequency (in Hz) and corrected power.
        """
        n_bins = len(bins)
        if bin_idx <= 0 or bin_idx >= n_bins - 1:
            freq = bin_idx * self.fs / self.n_samples
            return freq, max(float(bins[bin_idx]), 0.0) if n_bins > 0 else 0.0

        y0 = float(bins[bin_idx - 1])
        y1 = float(bins[bin_idx])
        y2 = float(bins[bin_idx + 1])

        # Parabolic peak position relative to center bin
        # Log-parabolic interpolation when all 3 bins are above MIN_PWR
        if y0 > MIN_PWR and y1 > MIN_PWR and y2 > MIN_PWR:
            ly0, ly1, ly2 = float(np.log(y0)), float(np.log(y1)), float(np.log(y2))
            denom = 2.0 * ly1 - ly0 - ly2
        else:
            ly0, ly1, ly2 = y0, y1, y2
            denom = 2.0 * y1 - y0 - y2
        if abs(denom) > MIN_PWR:
            delta_bin = float(np.clip(0.5 * (ly2 - ly0) / denom, -0.5, 0.5))
        else:
            delta_bin = 0.0

        # Interpolated peak power
        peak_power = y1 - 0.25 * (y0 - y2) * delta_bin

        # Convert to frequency
        interpolated_bin = bin_idx + delta_bin
        freq = interpolated_bin * self.fs / self.n_samples

        return freq, max(peak_power, 0.0)

    def adaptive_max_bins(self, freq: float, quality_factor: float = 10.0) -> int:
        """Compute adaptive bin tolerance based on frequency and signal quality.

        Parameters
        ----------
        freq : float
            Center frequency in Hz.
        quality_factor : float
            Quality factor Q (bandwidth = freq / Q). Default 10 assumes narrow tone.

        Returns
        -------
        int
            Number of bins to use for component extraction (minimum 5, maximum 50).
        """
        if freq <= 0:
            return MAX_BINS_PER_TONE
        # Bins needed to span bandwidth = freq / Q
        bandwidth = freq / quality_factor
        bins_for_bandwidth = int(np.ceil(bandwidth * self.n_samples / self.fs))
        # Clamp to reasonable range
        return max(5, min(50, max(bins_for_bandwidth, MAX_BINS_PER_TONE // 2)))

    @property
    def fs(self) -> int:
        """Return the sampling frequency in Hz."""
        return self.md.fs

    @cached_property
    def max(self) -> float:
        """Return the maximum value of the selected audio."""
        return float(np.max(self.bins)) if len(self) else 0.0

    @cached_property
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
            return float(
                np.max(self.bins[bin1 : bin2 + 1]) - np.min(self.bins[bin1 : bin2 + 1])
            )
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
        threshold = max_val * 10 ** (-tol_db / 10)

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

        # Calculate octave-spaced frequency bands (1/sqrt(2) to sqrt(2) range
        # around center)
        sqrt2 = 2.0**0.5
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

        # Divide frequency range into equal-width bands from zero to Nyquist
        low_freq = 0.0
        nyquist = 0.5 * self.fs
        step = nyquist / max_components
        high_freq = step

        # Iterate through each frequency band and compute its mean power
        while low_freq < nyquist:
            freq = 0.5 * (low_freq + high_freq)
            pwr = self.mean(low_freq, high_freq)
            component = Component(freq, pwr, 0.0, Category.Band, low_freq, high_freq)
            components.append(component)
            low_freq += step
            high_freq += step

        return components

    @cached_property
    def tones(self) -> Components:
        """
        Find the audio's components, DC, and noise.

        Returns
        -------
        Components
            The tone components found in the audio.
        """
        tones = Components(self.md)

        # Get absolute value of bins
        bins = np.absolute(self.bins)
        n_bins = len(bins)

        # Vectorized peak detection: faster than scipy.signal.find_peaks
        # Find local maxima by comparing with neighbors
        if n_bins < 3:
            self.noise = float(np.sum(bins))
            return tones

        # Vectorized approach: bins where gradient changes from positive to
        # negative
        # Much faster than scipy.signal.find_peaks for large arrays
        diffs = np.diff(bins)
        # Left derivative (positive = increasing)
        is_peak = np.empty(n_bins, dtype=bool)
        is_peak[0] = (diffs[0] <= 0) and (bins[0] > 0)
        is_peak[-1] = (diffs[-1] >= 0) and (bins[-1] > 0)
        is_peak[1:-1] = (diffs[:-1] >= 0) & (diffs[1:] <= 0) & (bins[1:-1] > 0)
        # Right derivative (negative = decreasing)

        # Local maxima: where left derivative >= 0 AND right derivative <= 0

        peak_indices = np.where(is_peak)[0]

        if len(peak_indices) == 0:
            self.noise = float(np.sum(bins))
            return tones

        # Get top MAX_TONES peaks by power (vectorized, single-pass)
        peak_powers = bins[peak_indices]
        n_top = min(MAX_TONES, len(peak_indices))

        # Use argpartition for O(n) partial sort instead of O(n log n) full sort
        if len(peak_powers) <= MAX_TONES:
            # If fewer peaks than max, use all and sort
            top_peak_order = np.argsort(-peak_powers)  # Descending order
            sorted_peak_indices = peak_indices[top_peak_order]
        else:
            # Partition to find top N, then sort just those
            top_local_indices = np.argpartition(-peak_powers, n_top - 1)[:n_top]
            # Sort the top N by power
            top_peak_order = top_local_indices[
                np.argsort(-peak_powers[top_local_indices])
            ]
            sorted_peak_indices = peak_indices[top_peak_order]

        # Pre-compute frequency scale once (avoids repeated bin_to_freq calls)
        freq_scale = self.fs / self.n_samples

        # Pre-compute gradient change indices to vectorize monotonic bound
        # detection
        # valley_left: where bins decrease (potential left tone boundary)
        # valley_right: where bins increase (potential right tone boundary)
        valley_left = np.where(diffs < 0)[0]
        valley_right = np.where(diffs > 0)[0]

        # Batch find monotonic bounds for all peaks using binary search
        left_indices = np.searchsorted(valley_left, sorted_peak_indices)
        right_indices = np.searchsorted(valley_right, sorted_peak_indices)

        # Process top MAX_TONES peaks with pre-computed bounds
        last_bin = n_bins - 1
        for i, center in enumerate(sorted_peak_indices):
            center = int(center)

            # Left bound: find where bins begin decreasing toward center
            l_idx = left_indices[i]
            first = int(valley_left[l_idx - 1] + 1) if l_idx > 0 else 0

            # Right bound: find where bins begin increasing away from center
            r_idx = right_indices[i]
            last = int(valley_right[r_idx]) if r_idx < len(valley_right) else last_bin

            # Ensure right bound extends past center and is valid
            last = max(center + 1, last + 1)

            # Limit span to MAX_BINS_PER_TONE
            if last - first > MAX_BINS_PER_TONE:
                half = MAX_BINS_PER_TONE // 2
                first = max(0, center - half)
                last = min(last_bin + 1, center + half)

            # Use parabolic interpolation for sub-bin frequency accuracy
            interp_freq, _ = self._parabolic_peak(center, bins)

            # Calculate power-weighted centroid frequency for robustness
            bin_pwrs = bins[first:last]
            total_power = bin_pwrs.sum()
            if total_power > 0:
                # Blend parabolic (50%) and weighted centroid (50%) for
                # robustness
                c_first = max(first, center - 3)
                c_last = min(last, center + 4)
                core_pwrs = bins[c_first:c_last]
                core_sum = float(core_pwrs.sum())
                centroid_freq = float(
                    (np.arange(c_first, c_last) * freq_scale * core_pwrs).sum()
                    / core_sum
                )
                freq = 0.5 * interp_freq + 0.5 * centroid_freq
            else:
                freq = interp_freq

            # Create tone component
            tone = Component(
                freq,
                float(total_power),
                0.0,
                Category.Tone,
                first * freq_scale,
                (last if last < n_bins else n_bins - 1) * freq_scale,
            )

            # Zero out tone bins for noise calculation
            bins[first:last] = 0.0
            tones.append(tone)

        # Get noise as the sum of the remaining non-tone bins
        self.noise = float(bins.sum())

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
        tones = self.tones

        # If no components found, it is noise
        if not tones:
            return True

        # Peak tone power is the first element (tones are sorted by power)
        max_comp_pwr = tones[0].pwr

        # Compare peak to average of remaining tones to assess signal dominance
        if len(tones) == 1:
            return False
        rem_pwr = sum([tone.pwr for tone in tones[1:]])
        if rem_pwr <= MIN_PWR:
            return False
        avg_tone_pwr = rem_pwr / (len(tones) - 1)
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
        tones = self.tones

        # If no components found, it is not a sine wave
        if not tones:
            return False

        # Peak tone power determines signal strength
        max_comp_pwr = tones[0].pwr if tones else 0.0

        # Pure tone with negligible noise floor confirms sine wave
        if self.noise <= MIN_PWR:
            return True

        # Compare tone component power-to-noise ratio (signal-to-noise ratio)
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

        # Return True if frequency is within tolerance of any frequency in the
        # list
        for target_freq in freq_list:
            # Handle aliasing (frequencies folded back at Nyquist)
            target_freq = alias_freq(target_freq, self.fs)

            # Calculate bin difference
            bin_diff = self.freq_to_bin(abs(freq - target_freq))

            # Return True if frequency is within tolerance of any frequency in
            # the list
            if bin_diff <= bins_per_component:
                return True

        # Frequency not found in the list
        return False
