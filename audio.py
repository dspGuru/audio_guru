"""Audio Data Class"""

from copy import copy
from functools import cached_property
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy import signal
from scipy.fft import rfft, rfftfreq
import sounddevice as sd
import soundfile as sf

from constants import (
    DEFAULT_BLOCKSIZE_SECS,
    DEFAULT_FS,
    DEFAULT_MAX_NF_DBFS,
    DEFAULT_SILENCE_MIN_SECS,
    DEFAULT_SILENCE_THRESH_RATIO,
    MIN_FREQ,
    MIN_PWR,
    QUADRATURE_MAGNITUDE_TOLERANCE,
    QUADRATURE_MIN_FREQ_RATIO,
    QUADRATURE_SAMPLE_RATIO,
    SEGMENT_MAX_SECS,
    SEGMENT_MIN_SECS,
    SILENCE_MIN_PEAK,
    SILENCE_MIN_THRESHOLD,
    SILENCE_WINDOW_SECS,
    SWEEP_SEGMENT_SECS,
)
from decibels import dbv
from freq_bins import FreqBins
from metadata import Metadata
from segment import Segment, Segments
from util import Category, Channel

__all__ = ["Audio"]


class Audio:
    """
    Read, store and write audio data.

    Attributes and Properties
    -------------------------
    samples : np.ndarray
        Audio signal samples.
    name : str
        Name of the audio.
    segment : Segment
        Selected audio segment.
    md : Metadata
        Audio metadata.
    avg_noise_floor : float
        Running average noise floor.
    max_segment_secs : float
        Maximum segment length in seconds.
    next_id : int
        Next segment identifier.
    sf : soundfile.SoundFile
        Soundfile object for reading audio.
    stream : sounddevice.InputStream
        Sounddevice stream for reading audio.
    blocksize : int
        Block size for reading audio.
    min_segment_secs : float
        Minimum segment length in seconds.
    """

    DTYPE = np.float32
    """Audio sample data type"""

    def __init__(
        self,
        fs: float = DEFAULT_FS,
        name: str = "",
        min_segment_secs: float = SEGMENT_MIN_SECS,
        max_segment_secs: float = SEGMENT_MAX_SECS,
        channel: Channel = Channel.Stereo,
    ):
        """Initialize audio

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        name : str
            Name of the audio.
        min_segment_secs : float
            Minimum segment length in seconds.
        max_segment_secs : float
            Maximum segment length in seconds.
        channel : Channel
            Channel of the audio.

        Attributes
        ----------
        fs : float
            Sampling frequency in Hz.
        samples : np.ndarray
            Audio signal samples.
        name : str
            Name of the audio.
        segment : Segment
            Selected audio segment.
        segments : Segments
            Segments of audio.
        md : Metadata
            Audio metadata.
        sf : soundfile.SoundFile
            Soundfile object for reading audio.
        stream : sounddevice.InputStream
            Sounddevice stream for reading audio.
        min_segment_secs : float
            Minimum segment length in seconds.
        next_id : int
            Next segment identifier.
        """
        self.fs = fs
        self.min_segment_secs: float = min_segment_secs
        self.max_segment_secs: float = max_segment_secs
        self.channel = channel

        self.blocksize = round(fs * DEFAULT_BLOCKSIZE_SECS)
        self.segment = Segment(fs)
        self.samples: np.ndarray = np.ndarray(shape=1, dtype=self.DTYPE)
        self.n_samples_removed: int = 0
        self.segments = Segments()
        self.md = Metadata(name, self.segment)
        self.sf = None
        self.stream = None
        self.cat = Category.Unknown
        self.next_id: int = 0
        self._noise_sum: float = 0.0
        self._noise_count: int = 0

    @property
    def eof(self) -> bool:
        """Return True if the audio source has been exhausted or is closed."""
        if self.sf is not None and not self.sf.closed:
            return self.sf.tell() >= len(self.sf)
        if self.stream is not None and self.stream.active:
            return True
        return True

    @property
    def fs(self) -> float:
        """Sampling frequency in Hz.

        Raises
        ------
        ValueError
            If the sampling frequency is not positive on setter.
        """
        return self._fs

    @fs.setter
    def fs(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Sampling frequency must be positive")
        self._fs = value
        self.blocksize = round(value * DEFAULT_BLOCKSIZE_SECS)
        if hasattr(self, "segment"):
            self.segment.fs = value

    @property
    def name(self) -> str:
        """Get the audio file name."""
        return self.md.name

    @name.setter
    def name(self, value: str) -> None:
        """Set the audio file name."""
        self.md.name = value

    def __iadd__(self, other: "Audio") -> "Audio":
        """In-place addition operator to add audio data.

        Parameters
        ----------
        other : Audio
            Audio to add.

        Returns
        -------
        Audio
            The updated audio.

        Raises
        ------
        ValueError
            If `other` is not an Audio object, has a different sample rate, or
            has a different number of samples.
        """
        # Check if other is an Audio object
        if not isinstance(other, Audio):
            raise ValueError("Cannot add non-Audio object: " + str(type(other)))

        # Check if sample rates match
        if other.fs != self.fs:
            raise ValueError("Cannot add audio with differing sample rates")

        if np.shape(other.selected_samples) != np.shape(self.selected_samples):
            raise ValueError(
                "Cannot add audio with a different number of samples than the current audio"
            )

        # Add the samples
        self.samples[self.segment.slice] += other.samples[other.segment.slice]
        return self

    def append(self, other: "Audio") -> "Audio":
        """Append audio data.

        Parameters
        ----------
        other : Audio | array-like
            Audio or array-like buffer to append.

        Returns
        -------
        Audio
            The updated audio.

        Raises
        ------
        ValueError
            If `other` has a different sample rate, differing dimensions, or
            differing channel shapes.
        """
        # Accept either another Audio or an array-like buffer
        if isinstance(other, Audio):
            # Require matching sample rates
            if other.fs != self.fs:
                raise ValueError("Cannot append audio with differing sample rates")
            new_samples = np.asarray(other.samples, dtype=self.DTYPE)
        else:
            new_samples = np.asarray(other, dtype=self.DTYPE)

        # Concatenate the new samples to self.samples
        if len(self):
            # Ensure compatible shapes (time axis is axis 0)
            if self.samples.ndim != new_samples.ndim:
                raise ValueError("Cannot append arrays with differing dimensions")
            if (
                self.samples.ndim > 1
                and self.samples.shape[1:] != new_samples.shape[1:]
            ):
                raise ValueError("Cannot append arrays with differing channel shapes")

            # Concatenate the new samples to self.samples
            self.samples = np.concatenate((self.samples, new_samples), axis=0)
        else:
            # Set self.samples to the new samples
            self.samples = new_samples

        # Update segment and metadata to span the whole buffer
        self.segment = Segment(self.fs, 0, len(self.samples))
        self.md.segment = copy(self.segment)

        return self

    def __len__(self):
        """Return the length of the entire audio data."""
        return len(self.samples) if len(self.samples) > 1 else 0

    def __iter__(self):
        """Return an iterator over the audio segments."""
        self.segments = self.get_segments()
        return self

    def __next__(self):
        """Select and return the next audio segment in the iteration.

        Raises
        ------
        StopIteration
            If there are no more segments to iterate over.
        """
        # Stop if no segments have been found
        if not self.segments:
            raise StopIteration

        # Mark the next segment as complete if we are at EOF or the segment
        # exceeds maximum length
        if self.eof or (
            self.max_segment_secs > 0 and self.segments[0].secs >= self.max_segment_secs
        ):
            self.segments[0].complete = True

        # Stop iteration if the next segment is incomplete
        if not self.segments[0].complete:
            raise StopIteration

        # The next segment is complete: pop and select it
        segment = self.segments.pop(0)
        self.select(segment)

        # Determine category if unknown
        if self.segment.cat == Category.Unknown:
            self.segment.cat = self.cat

        # Copy the segment into a new Audio object, remove the segment,
        # and return the copied audio
        audio = self.copy(self.segment)
        self.remove(self.segment, True)

        return audio

    def __enter__(self) -> "Audio":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context manager."""
        self.close()

    @property
    def selected_samples(self) -> np.ndarray:
        """Return the slice of the selected segment."""
        return self.samples[self.segment.slice]

    @cached_property
    def max(self) -> float:
        """Return the maximum value of the selected audio."""
        return float(np.max(self.selected_samples))

    @cached_property
    def min(self) -> float:
        """Return the minimum value of the selected audio."""
        return float(np.min(self.selected_samples))

    @cached_property
    def dc(self) -> float:
        """Return the DC offset of the selected audio."""
        a = self.selected_samples
        dc = float(np.mean(a))
        return dc

    @cached_property
    def rms(self) -> float:
        """Return the RMS value of the selected audio."""
        a = self.selected_samples
        return float(la.norm(a) / np.sqrt(a.size))

    @property
    def avg_noise_floor(self) -> float:
        """Return the running average noise floor."""
        if self._noise_count == 0:
            return 0.0
        return self._noise_sum / self._noise_count

    @property
    def freq(self) -> float:
        """
        Return the estimated fundamental frequency of the selected audio.

        Returns
        -------
        float
            The estimated fundamental frequency.
        """
        tones = self.bins.tones
        freq = tones[0].freq if tones else 0.0
        return freq

    @property
    def num_channels(self) -> int:
        """Return the number of channels in the audio data."""
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[1]

    def copy(self, segment: Segment | None = None) -> "Audio":
        """
        Return a copy of the specified audio segment.

        Parameters
        ----------
        segment : Segment
            Segment of audio to copy.

        Returns
        -------
        Audio
            Copy of the specified audio segment.
        """
        if segment is None:
            segment = Segment(self.fs, 0, len(self.samples))
        else:
            self.validate(segment)

        # Create a new Audio object, copy the segment's samples
        # and its metadata, and select the entire copied audio
        audio = Audio(self.fs, self.name, self.min_segment_secs, self.max_segment_secs)
        audio.append(self.samples[segment.slice])
        audio.md = copy(self.md)
        audio.segment = copy(segment)
        audio.segment.start = 0
        audio.segment.stop = len(audio.samples)
        audio.segment.channel = segment.channel
        audio.select(audio.segment)

        return audio

    def remove_unselected(self) -> None:
        """Remove all audio data except for the selected segment."""
        before_segment = Segment(self.fs, 0, self.segment.start)
        self.remove(before_segment, remove_prior=True)
        after_segment = Segment(self.fs, self.segment.stop, len(self.samples))
        self.remove(after_segment, remove_prior=False)

    def validate(self, segment: Segment) -> None:
        """Validate that the segment is compatible with the audio.

        Raises
        ------
        ValueError
            If the segment has a different sample rate, a negative start index,
            a stop index greater than the audio length + 1, or a stop index
            less than or equal to the start index.
        """
        # Raise ValueError if segment is incompatible with audio or otherwise
        # invalid
        if segment.fs != self.fs:
            raise ValueError("Segment fs must match audio fs")
        if segment.start < 0:
            raise ValueError("Segment start must be non-negative")
        if segment.stop > len(self.samples) + 1:
            raise ValueError(
                f"Segment stop ({segment.stop}) must not be greater than audio length ({len(self.samples)}) + 1"
            )
        if segment.stop <= segment.start:
            raise ValueError("Segment stop must be greater than start")

    def remove(self, segment: Segment, remove_prior: bool = False) -> None:
        """Remove the specified segment from the audio.

        Parameters
        ----------
        segment : Segment
            The segment to remove.
        remove_prior : bool
            If True, also remove the audio prior to the segment.
        """
        # Validate the segment and set convenience variables
        self.validate(segment)
        start = segment.start
        stop = segment.stop

        # Remove the segment from the audio
        if remove_prior:
            self.samples = self.samples[stop:]
        else:
            self.samples = np.concatenate((self.samples[:start], self.samples[stop:]))

        # Update the start and stop indices of all segments after the segment
        if remove_prior:
            offset = stop
            self.n_samples_removed += stop
        else:
            offset = stop - start
            self.n_samples_removed += stop - start

        for s in self.segments:
            if s.start >= stop:
                s.offset(offset)

            elif s.stop > stop:
                # Handle simplified case where s might overlap end
                s.stop -= offset

        # Remove the segment from the list
        try:
            self.segments.remove(segment)
        except ValueError:
            pass

    def select(self, selection: Segment | Channel | None = None) -> Segment:
        """
        Select the specified segment or channel for subsequent operations.

        Parameters
        ----------
        selection : Segment | Channel | None
            Segment or channel to select. If None, select the entire audio.

        Returns
        -------
        Segment
            The previous segment, for possible later restoration.
        """
        # Save the previous segment
        prev_segment = self.segment

        # If no segment is specified, select the entire audio below. Otherwise,
        # validate the segment.
        if selection is None:
            selection = Segment(self.fs, 0, len(self.samples))
        elif isinstance(selection, Channel):
            selection = Segment(self.fs, 0, len(self.samples), channel=selection)
        else:
            self.validate(selection)

        # Select the segment and update metadata
        self.segment = selection
        self.md.segment = copy(selection)

        # Delete cached properties
        for prop in (
            "bins",
            "cat",
            "dc",
            "is_quadrature",
            "magnitude",
            "max",
            "min",
            "rms",
        ):
            try:
                delattr(self, prop)
            except AttributeError:
                pass

        # Return the previous segment
        return prev_segment

    def filter(self, fir_h: list[float] | np.ndarray) -> None:
        """
        Filter the selected audio data using the specified FIR filter.

        For multi-channel audio, each channel is filtered individually.

        Parameters
        ----------
        fir_h : list[float] | np.ndarray
            The FIR filter coefficients.
        """
        # Return if there are no coefficients
        if fir_h is None or len(fir_h) == 0:
            return

        # Return if there is no audio data
        if not len(self):
            return

        # Get the selected samples and original shape
        samples = self.selected_samples
        original_shape = samples.shape

        # Treat mono as single-channel 2D array
        samples_2d = np.atleast_2d(samples.T).T

        # Filter each channel individually
        filtered = np.empty_like(samples_2d)
        for ch in range(samples_2d.shape[1]):
            filtered[:, ch] = signal.convolve(
                samples_2d[:, ch], fir_h, mode="same", method="auto"
            )

        # Restore original shape and update audio data
        self.samples[self.segment.slice] = filtered.reshape(original_shape).astype(
            self.DTYPE
        )

    def rescale(self, amplitude: float) -> None:
        """
        Rescale the selected audio data to the specified amplitude.

        Parameters
        ----------
        amplitude : float
            The target maximum amplitude.
        """
        # Return if there is no audio data
        if not len(self):
            return

        # Get the current maximum absolute amplitude
        current_max = np.max(np.abs(self.selected_samples))

        # Return if the current maximum is zero (avoid division by zero)
        if current_max == 0.0:
            return

        # Calculate the gain
        gain = amplitude / current_max

        # Apply the gain to the selected samples
        self.samples[self.segment.slice] *= gain

    def get_segments(
        self,
        silence_thresh_ratio: float | None = None,
        min_silence_secs: float | None = None,
    ) -> Segments:
        """
        Get a list of audio segments whose absolute amplitude is above
        (peak * ratio), or below it (if silence=True) for at least the
        specified time.

        Parameters
        ----------
        silence_thresh_ratio : float | None
            Ratio of peak amplitude to use as threshold for segment detection.
        min_silence_secs : float | None
            Minimum duration in seconds for a segment to be included.

        Raises
        ------
        ValueError
            If `min_silence_secs` results in non-positive number of samples.
        """
        # Set default values if needed
        if silence_thresh_ratio is None:
            silence_thresh_ratio = DEFAULT_SILENCE_THRESH_RATIO
        if min_silence_secs is None:
            min_silence_secs = DEFAULT_SILENCE_MIN_SECS

        # Convert min_silence_secs to samples
        min_silence_n = round(min_silence_secs * self.fs)
        if min_silence_n <= 0:
            raise ValueError("Invalid audio or min_silence_secs")

        # Handle incremental updates
        start_sample = 0
        if self.segments:
            last_seg = self.segments.pop()
            self.next_id = last_seg.id
            start_sample = last_seg.start

        # Get the samples from the start sample to the end. Avoid copying if
        # possible.
        samples = np.asarray(self.samples[start_sample:], copy=False)

        # Return if there are no amplitudes
        if not len(samples):
            return Segments()

        # Get amplitudes of samples (mono or peak across channels)
        if samples.ndim > 1:
            # Efficiently get peak amplitude across channels without full mean
            amplitudes = np.max(np.abs(samples), axis=1)
        else:
            amplitudes = np.abs(samples)

        # Calculate silence threshold
        peak = float(max(np.max(amplitudes), SILENCE_MIN_PEAK))
        thresh = peak * float(DEFAULT_SILENCE_THRESH_RATIO)
        thresh = max(thresh, SILENCE_MIN_THRESHOLD)

        # Detect segments from raw audio
        segments = self._detect_segments(
            start_sample, amplitudes, thresh, min_silence_secs
        )
        if not segments:
            return self.segments

        # Refine bounds and mark completed segments
        self._refine_segment_bounds(
            segments, start_sample, amplitudes, thresh, min_silence_n
        )

        # Remove initial silence from self.samples if at least min_silence_n
        if (
            len(segments) > 1
            and segments[0].cat == Category.Silence
            and len(segments[0]) >= min_silence_n
        ):
            initial_silence = segments[0]
            offset = initial_silence.stop
            self.remove(initial_silence, remove_prior=True)
            segments.pop(0)
            # Offset remaining segments (remove() only offsets self.segments,
            # not local segments)
            for seg in segments:
                seg.offset(offset)

        # Merge new segments with existing segments
        segments = Segments([s for s in segments if s.cat != Category.Silence])
        self.segments.merge(segments)

        # Remove short segments
        if self.min_segment_secs > 0:
            self.segments = Segments(
                [s for s in self.segments if s.secs >= self.min_segment_secs]
            )

        # Assign IDs to segments
        self.next_id = self.segments.assign_ids(self.next_id)

        # Validate all segments
        for segment in self.segments:
            self.validate(segment)

        return self.segments

    def _detect_segments(
        self,
        start_sample: int,
        amplitudes: np.ndarray,
        thresh: float,
        min_silence_secs: float,
    ) -> Segments:
        """Detect segment boundaries using amplitude threshold."""
        # Detect segments using subsegments
        segments = Segments()
        start_secs = self.n_samples_removed / self.fs
        step = max(1, int(self.fs * SILENCE_WINDOW_SECS))

        # Vectorized subsegment max calculation
        n = len(amplitudes)
        if n == 0:
            return segments

        # Calculate number of complete subsegments and pad if needed
        n_windows = (n + step - 1) // step
        padded_len = n_windows * step
        if padded_len > n:
            # Pad with zeros to get complete subsegments
            padded = np.zeros(padded_len, dtype=amplitudes.dtype)
            padded[:n] = amplitudes
        else:
            padded = amplitudes

        # Reshape to (n_windows, step) and compute max per subsegment
        reshaped = padded.reshape(n_windows, step)
        subsegment_maxes = np.max(reshaped, axis=1)

        # Create segments from vectorized results
        for i, max_amp in enumerate(subsegment_maxes):
            subsegment_start = i * step
            subsegment_stop = min(subsegment_start + step, n)
            cat = Category.Unknown if max_amp >= thresh else Category.Silence
            segment = Segment(
                self.fs,
                start_sample + subsegment_start,
                start_sample + subsegment_stop,
                cat=cat,
                start_secs=start_secs,
                complete=False,
            )
            segments.append(segment)

        # Combine same-category segments
        segments.combine()

        # Recategorize short silence as Unknown
        for segment in segments:
            if segment.cat == Category.Silence and segment.secs < min_silence_secs:
                segment.cat = Category.Unknown

        # Combine recategorized segments
        segments.combine()

        self._update_noise_floor(segments)

        # Mark Unknown segments which are followed by Silence as complete
        for i in range(len(segments) - 1):
            if (
                segments[i].cat != Category.Silence
                and segments[i + 1].cat == Category.Silence
            ):
                segments[i].complete = True

        return segments

    def _refine_segment_bounds(
        self,
        segments: Segments,
        start_sample: int,
        amplitudes: np.ndarray,
        thresh: float,
        min_silence_n: int,
    ) -> None:
        """Refine segment boundaries to active samples with padding."""
        pad = round(self.fs * 0.001)  # 1 ms padding

        # Refine segment boundaries
        for seg in segments:
            # Skip silence segments and segments which precede start sample
            if seg.cat == Category.Silence or seg.start < start_sample:
                continue

            # Get segment samples
            seg_a = amplitudes[seg.start - start_sample : seg.stop - start_sample]

            # Find active samples
            active = np.where(seg_a >= thresh)[0]
            if active.size > 0:
                first = int(max(0, active[0] - pad))
                last = int(min(len(seg_a) - 1, active[-1] + pad))
                seg.start += first
                seg.stop = seg.start + (last - first) + 1
                self.validate(seg)

            # Mark segment as complete if it is followed by silence or EOF
            if seg.stop < len(self.samples) - min_silence_n or self.eof:
                seg.complete = True

    def _update_noise_floor(self, segments: Segments) -> None:
        """Update the statistics for the running average noise floor.

        Updates the noise floor statistics using all segments which are marked
        as silence.
        """
        for seg in segments:
            if seg.cat == Category.Silence:
                # Calculate the sum of the amplitude of the noise samples
                noise = float(np.sum(np.abs(self.samples[seg.slice])))

                # Update the running average noise floor
                self._noise_sum += noise
                self._noise_count += len(seg)

    @cached_property
    def bins(self) -> FreqBins:
        """Return the frequency bins of the selected audio.

        Returns
        -------
        FreqBins
            Frequency bins of the selected audio.
        """
        bins = FreqBins(self.selected_samples, self.md, self.segment.channel)
        return bins

    def is_silence(self) -> bool:
        """
        Determine if the selected audio is silence.

        Returns
        -------
        bool
            True if the selected audio is silence.
        """
        return np.all(self.selected_samples == 0)

    def is_sweep(self) -> bool:
        """
        Determine if the audio is a sweep signal.

        Returns
        -------
        bool
            True if the audio is a sweep signal.
        """
        # Get mono samples for analysis
        samples = self.selected_samples
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        # Use STFT to estimate frequency changes over time. Use a segment
        # length corresponding to SWEEP_SEGMENT_SECS but capped to 1/4 of total
        # samples to ensure at least 4 frames.
        n_samples = len(samples)
        nperseg = min(round(SWEEP_SEGMENT_SECS * self.fs), n_samples // 4)
        if nperseg < 8:  # Minimum window for reliable FFT
            return False

        f, _, magnitude = signal.stft(
            samples, self.fs, nperseg=nperseg, noverlap=nperseg // 2
        )
        mags = np.abs(magnitude)

        # Find peak frequency for each time frame
        freqs = []
        for i in range(mags.shape[1]):
            col = mags[:, i]
            # Use a slightly higher threshold than MIN_PWR to ignore graininess
            if np.max(col) > 10.0 * MIN_PWR:
                idx = np.argmax(col)
                freqs.append(float(f[idx]))

        # Filter out very low frequencies which might be noise
        valid_freqs = [f for f in freqs if f > 0.5 * MIN_FREQ]

        # If all subsegments have increasing or decreasing frequencies it is a
        # sweep
        if len(valid_freqs) >= 4:
            # Allow for small fluctuations/jitter (e.g. constant freq for one
            # frame)
            diffs = np.diff(valid_freqs)
            if np.all(diffs >= 0) and np.any(diffs > 0):
                return True
            if np.all(diffs <= 0) and np.any(diffs < 0):
                return True

        # Sweep not detected
        return False

    @cached_property
    def is_quadrature(self) -> bool:
        """
        Determine if the audio signal is in quadrature (I/Q format).

        A quadrature signal has two channels where paired samples represent
        the real and imaginary components of a complex waveform with
        consistent magnitude.

        Returns
        -------
        bool
            True if the audio is a two-channel quadrature signal.
        """
        # Quadrature requires exactly two channels
        if self.num_channels != 2:
            return False

        cat = self.cat
        if cat == Category.Silence:
            return False

        # Get channel samples
        samples = self.selected_samples
        if len(samples) == 0:
            return False
        ch0 = samples[:, 0]
        ch1 = samples[:, 1]

        # For non-noise signals, return False if the frequencies of the two
        # channels differ significantly
        if cat != Category.Noise:
            # Compute FFT magnitude spectrum of each channel
            fft0 = np.abs(rfft(ch0))
            fft1 = np.abs(rfft(ch1))
            freqs = rfftfreq(len(ch0), 1.0 / self.fs)

            # Find peak frequency of each channel, ignoring DC bin (index 0)
            peak_bin0 = int(np.argmax(fft0[1:]) + 1)
            peak_bin1 = int(np.argmax(fft1[1:]) + 1)
            freq0 = freqs[peak_bin0]
            freq1 = freqs[peak_bin1]

            # Reject if frequencies differ by more than the allowed ratio
            if freq0 > 0.0 and freq1 > 0.0:
                freq_ratio = min(freq0, freq1) / max(freq0, freq1)
                if freq_ratio < QUADRATURE_MIN_FREQ_RATIO:
                    return False

        # Calculate complex magnitudes and check for magnitude consistency

        # Compute complex magnitudes: |ch0 + j*ch1|
        magnitudes = np.sqrt(ch0**2 + ch1**2)

        # Get peak amplitude of each channel
        peak0 = np.max(np.abs(ch0))
        peak1 = np.max(np.abs(ch1))
        peak = max(peak0, peak1)

        # Avoid division by zero for silent audio
        if peak == 0.0:
            return False

        # Check how many samples have magnitude close to peak (constant
        # envelope)
        tolerance = QUADRATURE_MAGNITUDE_TOLERANCE * peak
        consistent = np.abs(magnitudes - peak) <= tolerance
        ratio = np.sum(consistent) / len(magnitudes)

        # Signal is quadrature if sufficient samples have consistent
        # magnitude
        return bool(ratio >= QUADRATURE_SAMPLE_RATIO)

    def phase(self, unwrap: bool = True) -> np.ndarray | None:
        """
        Calculate the instantaneous phase of a quadrature signal.

        Parameters
        ----------
        unwrap : bool
            If True, unwrap the phase to be continuous (no 2*pi jumps).

        Returns
        -------
        np.ndarray | None
            Instantaneous phase in radians, or None if not a quadrature signal.
        """
        if not self.is_quadrature:
            return None

        # Get channel samples (I = ch0, Q = ch1)
        samples = self.selected_samples
        re = samples[:, 0]
        im = samples[:, 1]

        # Calculate phase using arctan2(Q, I) for proper quadrant handling
        phase = np.arctan2(im, re)

        if unwrap:
            phase = np.unwrap(phase)

        return phase

    @cached_property
    def magnitude(self) -> np.ndarray | None:
        """
        Calculate the instantaneous magnitude of a quadrature signal.

        Returns
        -------
        np.ndarray | None
            Instantaneous magnitude, or None if not a quadrature signal.
        """
        if not self.is_quadrature:
            return None

        # Get channel samples (I = re, Q = im)
        samples = self.selected_samples
        re = samples[:, 0]
        im = samples[:, 1]

        # Calculate magnitude: |re + j*im| = sqrt(re^2 + im^2)
        return np.sqrt(re**2 + im**2)

    def phase_error(self) -> float | None:
        """
        Calculate the phase error of a quadrature sine signal.

        Uses a robust algorithm based on instantaneous frequency (phase
        differences) to avoid cumulative errors from noise-induced phase jumps
        that confuse standard unwrapping.

        Returns
        -------
        float | None
            RMS phase error in radians, or None if not a quadrature sine signal.
        """
        # Must be quadrature and a sine wave
        if not self.is_quadrature:
            return None

        # Check if it's a sine wave using FreqBins
        from freq_bins import FreqBins

        bins = FreqBins(self.selected_samples, self.md)
        if not bins.is_sine():
            return None

        # Get wrapped phase (no unwrapping to avoid noise issues)
        samples = self.selected_samples
        re = samples[:, 0]
        im = samples[:, 1]
        phase = np.arctan2(im, re)

        if len(phase) < 3:
            return None

        # Calculate phase differences (instantaneous frequency)
        # Wrap differences to [-π, π] to handle wraparound correctly
        phase_diff = np.diff(phase)
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # The ideal signal has constant instantaneous frequency
        # Phase error = RMS deviation of phase_diff from its mean
        mean_diff = np.mean(phase_diff)
        error = phase_diff - mean_diff

        # Wrap errors to [-π, π] for proper RMS calculation
        error = np.arctan2(np.sin(error), np.cos(error))

        rms_error = float(np.sqrt(np.mean(error**2)))

        return rms_error

    @cached_property
    def cat(self) -> Category:
        """
        Determine the category of the selected audio.

        Returns
        -------
        Category
            The audio category: Silence, Tone, Sweep, Noise, or Unknown.
        """
        # Check for silence
        if self.is_silence():
            return Category.Silence

        # Check for sweep (before checking for sine)
        if self.is_sweep():
            return Category.Sweep

        # Check for sine
        if self.bins.is_sine():
            return Category.Tone

        # Check for noise
        if self.bins.is_noise():
            return Category.Noise

        return Category.Unknown

    # TODO: Remove?
    def get_noise_floor(
        self, secs: float = 0.25, max_noise_db: float = DEFAULT_MAX_NF_DBFS
    ) -> tuple[float, int]:
        """
        Get the audio's noise floor and the index at which it was found.

        The noise floor is calculated as the minimum average absolute
        amplitude over segments of the specified duration.

        Parameters
        ----------
        secs : float
            Duration in seconds of each segment over which the average
            absolute amplitude is calculated.
        max_noise_db : float
            Maximum noise level in dBFS.

        Returns
        -------
        tuple(noise floor, index)
            The audio's noise floor and the index at which it was
            found.
        """
        # Return zero if the audio is empty
        if not len(self):
            return (0.0, 0)

        # Calculate the absolute value of the audio data and average it
        # across channels if necessary
        abs_audio = np.abs(self.samples)
        if abs_audio.ndim > 1:
            abs_audio = np.mean(abs_audio, axis=1)

        # Calculate the average of each segment of the audio
        avg_size = int(self.fs * secs)
        averages = np.array([], dtype=self.DTYPE)
        for i in range(0, len(abs_audio) - avg_size + 1, avg_size):
            segment = abs_audio[i : i + avg_size]
            avg = np.mean(segment)
            averages = np.append(averages, avg)

        # Return zero if no averages were calculated
        if averages.size == 0:
            return (0.0, 0)

        # Calculate the noise floor as the minimum value of the averages
        noise_floor = float(np.min(averages))
        if dbv(noise_floor) > max_noise_db:
            return (0.0, 0)

        # Return the noise floor and the index at which it was found
        index = int(np.where(averages == noise_floor)[0][0] * avg_size)
        return (noise_floor, index)

    def read(self, fname: str | Path) -> bool:
        """
        Read audio data from file.

        Parameters
        ----------
        fname : str | Path
            File name to read.

        Returns
        -------
        bool
            True if successful, False if no data or unsupported channel count.
        """
        # Convert possible Path object to string
        fname = str(fname)

        with sf.SoundFile(fname) as f:
            # Set channel based on file's channel count
            if f.channels == 1:
                channel = Channel.Left
            elif f.channels == 2:
                channel = Channel.Stereo
            else:
                return False

            self.__init__(fs=f.samplerate, name=f.name, channel=channel)
            self.sf = f

            while self.read_block(f):
                pass

            # Clean up reference before checking length to maintain consistency
            self.sf = None

            if not len(self):
                return False

            # Update segment to cover full length
            self.select()

        return True

    def open(self, fname: str | Path) -> None:
        """
        Open an audio file.

        Parameters
        ----------
        fname : str
            File name to open.

        Raises
        ------
        ValueError
            If `other` has a different sample rate, differing dimensions, or
            differing channel shapes.
        """
        # Convert possible Path object to string
        fname = str(fname)

        # Open the file
        self.sf = sf.SoundFile(fname)
        self.fs = float(self.sf.samplerate)
        self.name = fname
        self.cat = Category.Unknown
        self.md.source_data = self.sf.copy_metadata()

        # Reset segment ID counter
        self.next_id = 0

        return self

    def open_device(
        self,
        device: int | str | None = None,
        fs: float | None = None,
        blocksize: int | None = None,
        channels: int = 1,
        dtype: str = "float32",
        quiet: bool = False,
    ) -> None:
        """
        Open an audio device for input.

        Parameters
        ----------
        device : int | str | None
            Device ID or substring of device name.
        fs : float | None
            Sampling frequency in Hz.
        blocksize : int | None
            Number of frames per block.
        channels : int
            Number of input channels.
        dtype : str
            Data type of the audio samples.
        quiet : bool
            If False, print the audio input device name (default: False).
        """
        if fs is not None:
            self.fs = fs

        if blocksize is not None:
            self.blocksize = blocksize

        # Print the input device name if not quiet
        if not quiet:
            # Get the device info (use default input if device is None)
            dev_id = device if device is not None else sd.default.device[0]
            device_info = sd.query_devices(dev_id, "input")
            print(f"Audio input: {device_info['name']}")

        # Open the input stream
        self.stream = sd.InputStream(
            device=device,
            samplerate=self.fs,
            blocksize=self.blocksize,
            channels=channels,
            dtype=dtype,
        )
        self.stream.start()

        # Reset segment ID counter
        self.next_id = 0

        return self

    def close(self) -> None:
        """Close the audio file or device stream."""

        # Close sound file if it is open
        if self.sf is not None and not self.sf.closed:
            self.sf.close()
            self.sf = None

        # Close device stream if it is open
        if self.stream is not None and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Mark all segments as complete
        for segment in self.segments:
            segment.complete = True

    @property
    def is_open(self) -> bool:
        """Return True if the audio file or device stream is open."""
        return (self.sf is not None and not self.sf.closed) or (
            self.stream is not None and self.stream.active
        )

    def read_block(self, f=None) -> bool:
        """
        Read a block of audio data from an open file or device.

        Parameters
        ----------
        f : sf.SoundFile | None
            Open sound file to read from. If None, use self.sf or self.stream.

        Returns
        -------
        bool
            True if data was read, False if EOF.
        """
        data = []

        # Use the provided file or the internal sound file if available
        if f is None:
            f = self.sf

        # Check for open file
        if f is not None and not f.closed:
            # Read a block of audio data from file
            data = f.read(self.blocksize, dtype="float32", always_2d=False)

        # Check for open stream
        elif self.stream is not None and self.stream.active:
            # Read a block of audio data from stream
            data, overflowed = self.stream.read(self.blocksize)
            if overflowed:
                print("Audio input overflowed")

            # Reduce to 1D if single channel
            if self.stream.channels == 1:
                data = data.flatten()

        # Return False if nothing is open
        else:
            return False

        # Append the read data to the samples
        if len(data) > 0:
            self.append(data)
            self.get_segments()

            # Verify samples buffer doesn't grow unbounded
            max_samples = int(2 * SEGMENT_MAX_SECS * self.fs)
            assert len(self.samples) <= max_samples, (
                f"Audio.samples exceeded maximum length: "
                f"{len(self.samples)} > {max_samples}"
            )

        # Return False if no data was read
        if len(data) == 0:
            return False

        return True

    def write(self, fname: str | Path = "", quiet: bool = False) -> bool:
        """
        Write the selected segment of audio data to file.

        Parameters
        ----------
        fname : str | Path
            File pathname to write.

        Returns
        -------
        bool
            True if successful.

        Raises
        ------
        ValueError
            If the audio's channel is not Left, Right, or Stereo.
        """
        # Get the file name to write. Use the metadata pathname if none is
        # specified.
        if not fname:
            fname = self.md.pathname
        elif isinstance(fname, Path):
            fname = str(fname)

        # Return False if no file name is available.
        if not fname:
            return False

        # Return False if there is no data
        if not len(self):
            return False

        # Validate channel type
        if self.channel not in (Channel.Left, Channel.Right, Channel.Stereo):
            raise ValueError("Channel must be Left, Right, or Stereo")

        # Prepare data based on channel type
        samples = self.selected_samples
        if self.channel == Channel.Stereo:
            # Write stereo: use samples as-is (should be 2D)
            data = np.asarray(samples, dtype=self.DTYPE)
        else:
            # Write mono for Left or Right channel
            if samples.ndim > 1:
                # Extract the appropriate channel (0=left, 1=right)
                channel_idx = 0 if self.channel == Channel.Left else 1
                data = np.asarray(samples[:, channel_idx], dtype=self.DTYPE)
            else:
                # Already 1D mono
                data = np.asarray(samples, dtype=self.DTYPE)

        # Print message if not quiet
        if not quiet:
            print(f"Writing audio to '{fname}'")

        # Write the file and update the file name
        p = Path(fname)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            if not quiet:
                print(f"Created directory '{p.parent}'")

        try:
            # Open a soundfile object for writing and set its metadata from  the
            # audio object
            channels = data.ndim
            with sf.SoundFile(
                fname, "w", samplerate=int(self.fs), channels=channels
            ) as f:
                self.md.info.set_sf_metadata(f)
                f.write(data)

            self.md.pathname = fname
            if not quiet:
                print(f"Wrote audio to '{fname}'")
            return True
        except Exception as e:
            if not quiet:
                print(f"Failed to write audio to '{fname}': {e}")
            return False

    def print(self, noise_floor: bool = True) -> None:
        """
        Print analysis results to the console.

        Parameters
        ----------
        noise_floor : bool
            Whether to print the noise floor (default is True).
        """
        # Print header with audio metadata if available
        if self.md.model or self.md.desc:
            print(f"Data from {self.md.pathname}:")
            if self.md.model:
                print(f"  Model: {self.md.model}")
            if self.md.desc:
                print(f"  Description: {self.md.desc}")
            print()

        # Print noise floor if requested
        if noise_floor:
            nf, index = self.get_noise_floor()
            secs = index / self.fs
            if nf > 0.0:
                nf_dbfs = dbv(nf)
                print(f"Noise Floor = {nf_dbfs:0.1f} dBFS at {secs:0.3f} secs")
            else:
                print("Noise floor not found: no silence detected.")

            print()
