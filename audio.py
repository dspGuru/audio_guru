"""Audio Data Class"""

from copy import copy
import math
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy import signal
import soundfile as sf

from component import Component
from constants import DEFAULT_AMP, DEFAULT_FS
from decibels import dbv, MIN_PWR
from freq_bins import FreqBins
from metadata import Metadata
from segment import Segment, Segments
from util import Category, Channel

__all__ = ["Audio"]


class Audio:
    """
    Read, write, generate and store audio data.

    Attributes
    ----------
    samples : np.ndarray
        Audio signal samples.
    name : str
        Name of the audio.
    segment : Segment
        Selected audio segment.
    md : Metadata
        Audio metadata.
    dc : float
        DC offset of the audio signal.
    noise : float
        Noise power in the audio signal.
    """

    DTYPE = np.float32
    """Audio sample data type"""

    def __init__(self, fs: float = DEFAULT_FS, name: str = ""):
        """Initialize audio

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        name : str
            Name of the audio.
        """
        self.samples: np.ndarray = np.ndarray(shape=1, dtype=self.DTYPE)
        self.name: str = name
        self.stack: list[Segment] = []
        self.segment: Segment = Segment(fs)
        self.segments: Segments = Segments()
        self.segment_index: int = 0
        self.md: Metadata = Metadata(self.name, self.segment)

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
        other : Audio
            Audio to append.

        Returns
        -------
        Audio
            The updated audio.

        Parameters
        ----------
        other : Audio | array-like
            Audio or array-like buffer to append.
        """
        # Accept either another Audio or an array-like buffer
        if isinstance(other, Audio):
            # Require matching sample rates
            if other.fs != self.fs:
                raise ValueError("Cannot append audio with differing sample rates")
            new_samples = np.asarray(other.samples, dtype=self.DTYPE)
        else:
            new_samples = np.asarray(other, dtype=self.DTYPE)

        # If current samples is empty, just set to new_samples
        if self.samples is None or self.samples.size == 0:
            self.samples = new_samples
        else:
            # Ensure compatible shapes (time axis is axis 0)
            if self.samples.ndim != new_samples.ndim:
                raise ValueError("Cannot append arrays with differing dimensions")
            if (
                self.samples.ndim > 1
                and self.samples.shape[1:] != new_samples.shape[1:]
            ):
                raise ValueError("Cannot append arrays with differing channel shapes")
            self.samples = np.concatenate((self.samples, new_samples), axis=0)

        # Update segment and metadata to span the whole buffer
        self.segment = Segment(self.fs, 0, len(self.samples) - 1)
        self.md.segment = copy(self.segment)

        return self

    def __len__(self):
        """Return the length of the entire audio data."""
        return len(self.samples)

    def __iter__(self):
        """Return an iterator over the audio segments."""
        self.segments = self.get_segments(categorize=True)
        self.segment_index = 0
        return self

    def __next__(self):
        """Select and return the next audio segment in the iteration."""
        if self.segment_index >= len(self.segments):
            raise StopIteration

        segment = self.segments[self.segment_index]
        self.segment_index += 1
        self.select(segment)

        return segment

    @property
    def fs(self) -> float:
        """Return the sampling frequency in Hz."""
        return self.segment.fs

    @property
    def selected_samples(self) -> np.ndarray:
        """Return the slice of the selected segment."""
        return self.samples[self.segment.slice]

    @property
    def max(self) -> float:
        """Return the maximum value of the selected audio."""
        return float(np.max(self.selected_samples)) if len(self) else 0.0

    @property
    def min(self) -> float:
        """Return the minimum value of the selected audio."""
        return float(np.min(self.selected_samples)) if len(self) else 0.0

    @property
    def dc(self) -> float:
        """Return the DC offset of the selected audio."""
        a = self.selected_samples
        if a.size == 0:
            return 0.0
        dc = float(np.mean(a))
        return dc

    @property
    def rms(self) -> float:
        """Return the RMS value of the selected audio."""
        a = self.selected_samples
        if a.size == 0:
            return 0.0
        rms = float(la.norm(a) / np.sqrt(a.size))
        return rms

    @property
    def freq(self) -> float:
        """
        Get the fundamental frequency of the selected audio.

        Returns
        -------
        float
            The fundamental frequency.
        """
        bins = self.get_bins()
        tones = bins.get_tones(max_components=1)
        freq = tones[0].freq if tones else 0.0
        return freq

    def select(self, segment: Segment | None = None) -> Segment:
        """
        Select the specified segment for subsequent operations.

        Parameters
        ----------
        segment : Segment | None
            Segment to select. If None, select the entire audio.

        Returns
        -------
        Segment
            The previous segment, for possible later restoration.
        """
        prev_segment = self.segment

        # Select the entire audio if no segment is specified
        if segment is None:
            segment = Segment(self.fs, 0, len(self.samples) - 1)

        # Ensure segment is within bounds
        if segment.start < 0:
            segment = Segment(self.fs, 0, segment.stop)
        if segment.stop >= len(self.samples):
            segment = Segment(self.fs, segment.start, len(self.samples) - 1)

        self.segment = segment
        self.md.segment = copy(segment)

        return prev_segment

    def push(self, segment: Segment) -> None:
        """
        Push the current segment onto the stack and select the specified
        segment.

        Parameters
        ----------
        segment : Segment
            Segment to select.
        """
        self.stack.append(self.segment)
        self.select(segment)

    def pop(self) -> None:
        """Pop the last segment from the stack and select it."""
        segment = self.stack.pop()
        self.select(segment)

    def filter(self, fir_h: list[float] | np.ndarray) -> None:
        """
        Filter the selected audio data using the specified FIR filter.

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

        # Perform the convolution using the efficient 'auto' method (FFT or direct)
        # mode='same' returns output of the same size as input
        filtered = signal.convolve(
            self.selected_samples, fir_h, mode="same", method="auto"
        )

        # Update the audio data with the filtered samples, casting back to original type
        self.samples[self.segment.slice] = filtered.astype(self.DTYPE)

    def generate_tone(
        self,
        freq: float = 1000.0,
        amp: float = 0.6,
        secs: float = 10.0,
        thd: float = MIN_PWR,
    ) -> None:
        """
        Generate tone (sine) audio with optional first-harmonic distortion.

        Parameters
        ----------
        freq : float
            Frequency of the tone in Hz.
        amp : float
            Peak amplitude) of the tone (0.0 to 1.0).
        secs : float
            Duration of the generated audio in seconds.
        thd : float
            Total harmonic distortion as a fraction (not percentage)of the
            fundamental. If less than or equal to MIN_PWR, no distortion is
            generated.
        """

        # Initialize name and segment
        self.name = "Tone"
        self.segment = Segment(self.fs, 0, round(secs * self.fs) - 1)

        # Generate tone at the specified frequency
        t = Component(freq, amp)
        self.samples = t.generate(secs, self.fs)

        # If specified, generate distortion tone at twice the nominal frequency
        if thd > MIN_PWR:
            t2 = Audio()
            t2.generate_tone(freq=freq * 2.0, amp=thd, secs=secs)
            self += t2
            thd_str = f"{(thd * 100.0):0.3f}%"
        else:
            thd_str = ""

        # Set metadata
        desc = f"{freq:0.1f}" if freq < 1000.0 else f"{math.trunc(freq / 1000.0)}k"
        self.md.set(mfr=self.name, model=thd_str, desc=desc)

    def generate_noise(self, amp: float = 0.6, secs: float = 10.0) -> None:
        """
        Generate white noise audio.

        Parameters
        ----------
        amp : float
            Peak amplitude of the noise (0.0 to 1.0).
        secs : float
            Duration of the generated noise in seconds.
        """

        # Initialize name and segment
        self.name = "Noise"
        self.segment = Segment(self.fs, 0, round(secs * self.fs) - 1)

        # Generate white noise
        n_samples = round(secs * self.fs)
        self.samples = np.random.uniform(low=-amp, high=amp, size=n_samples).astype(
            self.DTYPE
        )

        # Set metadata
        desc = f"White Noise {secs:0.1f}s"
        self.md.set(mfr=self.name, model="", desc=desc)

    def generate_silence(self, secs: float = 10.0) -> None:
        """
        Generate silence audio.

        Parameters
        ----------
        secs : float
            Duration of the generated silence in seconds.
        """

        # Initialize name and segment
        self.name = "Silence"
        self.segment = Segment(self.fs, 0, round(secs * self.fs) - 1)

        # Generate silence
        self.samples = np.zeros(round(secs * self.fs), dtype=self.DTYPE)

        # Set metadata
        desc = f"Silence {secs:0.1f}s"
        self.md.set(mfr=self.name, model="", desc=desc)

    def generate_sweep(
        self,
        f_start: float = 20.0,
        f_stop: float = 20000.0,
        amp: float = DEFAULT_AMP,
        secs: float = 10.0,
        method: str = "logarithmic",
    ) -> None:
        """
        Generate a frequency sweep audio signal.

        Parameters
        ----------
        f_start : float
            Starting frequency of the sweep in Hz.
        f_stop : float
            Ending frequency of the sweep in Hz.
        amp : float
            Peak amplitude of the sweep (0.0 to 1.0).
        secs : float
            Duration of the generated sweep in seconds.
        """
        # Initialize name and segment
        self.name = "SweepGen"
        self.segment = Segment(self.fs, 0, round(secs * self.fs) - 1)

        # Generate sweep signal
        t = np.arange(secs * self.fs) / self.fs
        self.samples = amp * signal.chirp(
            t, f0=f_start, t1=secs, f1=f_stop, method=method
        )

        # Set metadata
        desc = f"Sweep {f_start:0.1f}Hz to {f_stop:0.1f}Hz"
        self.md.set(mfr=self.name, model="", desc=desc)

    def get_segments(
        self,
        silence_thresh_ratio: float = 1e-3,
        min_silence_secs: float = 0.1,
        silence: bool = False,
        categorize: bool = False,
    ) -> Segments:
        """
        Get a list of audio segments whose absolute amplitude is above
        (peak * ratio), or below it (if silence=True) for at least the
        specified time.

        Parameters
        ----------
        ratio : float
            Ratio of peak amplitude to use as threshold for segment detection.
        secs : float
            Minimum duration in seconds for a segment to be included.
        silence : bool
            If True, return segments where the signal is below the threshold.
            Otherwise, return segments where the signal is above the threshold.
        """

        # Get number of samples for signal gap detection. Return an empty
        # list if the audio is empty or the gap size is zero or less.
        min_silence_secs_n = min_silence_secs * self.fs
        a = np.asarray(self.samples)
        n = a.size // a.ndim
        if n == 0 or min_silence_secs_n <= 0:
            return Segments()

        # Convert audio to mono for subsequent processing
        if a.ndim > 1:
            a = np.mean(a, axis=1)

        # Get the absolute value of audio
        a = np.abs(a)

        # Smooth the absolute audio using a moving average filter
        window_size = int(self.fs * 0.01)  # 10 ms window
        window = np.ones(window_size) / window_size
        a = np.convolve(a, window, mode="same")

        # Find silent segments where amplitude is below threshold
        silent_segments = Segments()

        # Determine peak amplitude
        peak = float(np.max(a))
        if peak == 0.0:
            # Entire signal is zero -> single segment spanning whole array if
            # nsamples satisfied
            silent_segments.append(Segment(self.fs, 0, n - 1, Category.Silence))
            return silent_segments

        # Calculate threshold
        thresh = peak * float(silence_thresh_ratio)

        # Create boolean array where condition is true (below threshold)
        below = a <= thresh

        # Find run starts and ends.
        # Diff will be 1 where False->True (start), -1 where True->False (end)
        d = np.diff(below.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1

        # Handle case where run starts at index 0
        if below[0]:
            starts = np.concatenate(([0], starts))
        # Handle case where run continues to end
        if below[-1]:
            ends = np.concatenate((ends, [n]))

        # Add each segment which has sufficient length to segments
        pairs = [(int(p[0]), int(p[1])) for p in zip(starts, ends)]
        for s, e in pairs:
            if (e - s) >= min_silence_secs_n:
                silent_segments.append(
                    Segment(self.fs, s, e, len(silent_segments), Category.Silence)
                )

        # if silence is True, return the segments that have been found
        if silence:
            return silent_segments

        # Otherwise, calculate the non-silent segments between the silent ones
        start = 0
        segments = Segments()
        if silent_segments:
            for segment in silent_segments:
                if segment.start - start >= min_silence_secs_n:
                    segments.append(
                        Segment(self.fs, start, segment.start - 1, len(segments))
                    )
                start = segment.stop + 1
        else:
            # No silent segments found: return entire audio as a single segment
            segments.append(Segment(self.fs, 0, n - 1))

        # Categorize the non-silent segments if specified
        if categorize:
            for segment in segments:
                self.push(segment)
                segment.cat = self.get_category()
                self.pop()

        return segments

    def get_bins(self, channel: Channel = Channel.Mean) -> FreqBins:
        """Return the frequency bins of the selected audio."""
        bins = FreqBins(self.selected_samples, self.md, channel)
        return bins

    def is_sweep(self, secs: float) -> bool:
        """
        Determine if the audio is a sweep signal.

        Parameters
        ----------
        secs : float
            Duration in seconds of subsegments to analyze for frequency
            changes.

        Returns
        -------
        bool
            True if the audio is a sweep signal.
        """
        # Estimate frequency changes over subsegments to check for sweep
        nsamps = round(secs * self.fs)
        freqs: list[float] = []
        for start in range(self.segment.start, self.segment.stop, nsamps):
            # Define subsegment
            stop = start + nsamps - 1
            if stop >= len(self.samples):
                stop = len(self.samples) - 1
            subsegment = Segment(self.fs, start, stop)

            # Append the primary frequency of the subsegment
            self.push(subsegment)
            freqs.append(self.freq)
            self.pop()

        # If enough frequencies have been collected, remove first and last
        # to avoid edge effects
        if len(freqs) >= 4:
            freqs = freqs[1:-1]

        # If all subsegments have increasing or decreasing frequencies, it
        # likely is a sweep.
        if len(freqs) >= 2:
            diffs = np.diff(freqs)
            if not len(diffs):
                return False

            if np.all(diffs >= 0) or np.all(diffs <= 0):
                return True

        # Sweep not detected: return False
        return False

    def get_category(self, secs: float = 0.1) -> Category:
        """
        Determine the category of the selected audio.

        Returns
        -------
        Category
            The audio category: Tone, Sweep, Noise, or Unknown.
        """
        # Check for silence
        if np.all(self.selected_samples == 0):
            return Category.Silence

        # Get frequency bins
        bins = self.get_bins()

        # Check for sine
        if bins.is_sine():
            return Category.Tone

        # Check for sweep
        if self.is_sweep(secs=secs):
            return Category.Sweep

        # Check for noise
        if bins.is_noise():
            return Category.Noise

        return Category.Unknown

    def get_noise_floor(self, secs: float = 0.25) -> tuple[float, int]:
        """
        Get the audio's noise floor and the index at which it was found.

        The noise floor is calculated as the minimum average absolute
        amplitude over segments of the specified duration.

        Parameters
        ----------
        secs : float
            Duration in seconds of each segment over which the average
            absolute amplitude is calculated.

        Returns
        -------
        tuple(noise floor, index)
        Return the audio's noise floor and the index at which it was
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

        # Calculate the noise floor as the minimum value of the averages
        noise_floor = np.min(averages)
        index = np.where(averages == noise_floor)[0][0] * avg_size
        return (float(noise_floor), index)

    def read(self, fname: str | Path) -> bool:
        """
        Read audio data from file.

        Parameters
        ----------
        fname : str
            File name to read.

        Returns
        -------
        bool
            True if successful.
        """
        # Convert possible Path object to string
        fname = str(fname)

        try:
            data, fs = sf.read(fname, dtype="float32")
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return False

        # Store audio data and metadata
        self.samples = np.asarray(data, dtype=self.DTYPE)
        self.segment = Segment(fs, 0, len(self) - 1)
        self.name = fname
        self.md = Metadata(self.name, self.segment)

        # Get segments
        self.segments = self.get_segments(categorize=True)

        return len(self) > 0

    def write(self, fname: str | Path = "") -> bool:
        """
        Write the selected segment of audio data to file.

        Parameters
        ----------
        fname : str | Path
            File name to write.

        Returns
        -------
        bool
            True if successful.
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

        # Write the file and update the file name
        data = np.asarray(self.selected_samples, dtype=self.DTYPE)
        try:
            sf.write(fname, data, int(self.fs))
            self.md.pathname = fname
            return True
        except Exception:
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

        if noise_floor:
            nf, index = self.get_noise_floor()
            secs = index / self.fs
            if nf > 0.0:
                nf_dbfs = dbv(nf)
                print(f"Noise Floor = {nf_dbfs:0.1f} dBFS at {secs:0.3f} secs")
            else:
                print("Noise floor not found: no silence detected.")

            print()
