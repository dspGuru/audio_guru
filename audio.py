"""Audio Data Class"""

from copy import copy
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy import signal
import soundfile as sf

from constants import (
    DEFAULT_BLOCKSIZE_SECS,
    DEFAULT_FS,
    DEFAULT_MAX_NF_DBFS,
)
from decibels import dbv
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
        md : Metadata
            Audio metadata.
        name : str
            Name of the audio.
        segment : Segment
            Selected audio segment.
        segment_index : int
            Index of the selected segment.
        segments : Segments
            Segments of audio.
        stack : list[Segment]
            Stack of audio segments.
        """
        self.fs = fs
        self.samples: np.ndarray = np.ndarray(shape=1, dtype=self.DTYPE)
        self.name: str = name
        self.segment: Segment = Segment(self.fs)
        self.segments: Segments = Segments()
        self.md = Metadata(self.name, self.segment)
        self.eof = False

    @property
    def fs(self) -> float:
        """Sampling frequency in Hz."""
        return self._fs

    @fs.setter
    def fs(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Sampling frequency must be positive")
        self._fs = value
        self.blocksize = round(value * DEFAULT_BLOCKSIZE_SECS)
        if hasattr(self, "segment"):
            self.segment.fs = value

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
        self.segments = self.get_segments(categorize=True)
        return self

    def __next__(self):
        """Select and return the next audio segment in the iteration."""
        if len(self.segments) == 0:
            raise StopIteration

        segment = self.segments.pop(0)
        self.select(segment)

        return segment

    @property
    def selected_samples(self) -> np.ndarray:
        """Return the slice of the selected segment."""
        return self.samples[self.segment.slice]

    @property
    def max(self) -> float:
        """Return the maximum value of the selected audio."""
        return float(np.max(self.selected_samples))

    @property
    def min(self) -> float:
        """Return the minimum value of the selected audio."""
        return float(np.min(self.selected_samples))

    @property
    def dc(self) -> float:
        """Return the DC offset of the selected audio."""
        a = self.selected_samples
        dc = float(np.mean(a))
        return dc

    @property
    def rms(self) -> float:
        """Return the RMS value of the selected audio."""
        a = self.selected_samples
        rms = float(la.norm(a) / np.sqrt(a.size))
        return rms

    @property
    def freq(self) -> float:
        """
        Return the estimated fundamental frequency of the selected audio.

        Returns
        -------
        float
            The estimated fundamental frequency.
        """
        bins = self.get_bins()
        tones = bins.get_tones(max_components=1)
        freq = tones[0].freq if tones else 0.0
        return freq

    def num_channels(self) -> int:
        """Return the number of channels in the audio data."""
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[1]

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
        # Save the previous segment
        prev_segment = self.segment

        # If no segment is specified, select the entire audio below
        if segment is None:
            segment = Segment(self.fs, 0, len(self.samples))

        # Raise ValueError if segment is incompatible with audio or invalid
        if segment.fs != self.fs:
            raise ValueError("Segment fs must match audio fs")
        if segment.start < 0:
            raise ValueError("Segment start must be non-negative")
        if segment.stop > len(self.samples) + 1:
            raise ValueError("Segment stop must not be greater than audio length + 1")
        if segment.stop <= segment.start:
            raise ValueError("Segment stop must be greater than start")

        # Select the segment and update metadata
        self.segment = segment
        self.md.segment = copy(segment)

        # Return the previous segment
        return prev_segment

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

    def get_segments(
        self,
        start_sample: int = 0,
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
        silence_thresh_ratio : float
            Ratio of peak amplitude to use as threshold for segment detection.
        min_silence_secs : float
            Minimum duration in seconds for a segment to be included.
        silence : bool
            If True, return segments where the signal is below the threshold.
            Otherwise, return segments where the signal is above the threshold.
        """

        # Get number of samples for signal gap detection. Return an empty
        # list if the audio is empty or the gap size is zero or less.
        min_silence_secs_n = min_silence_secs * self.fs
        a = np.asarray(self.samples[start_sample:])
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
            silent_segments.append(Segment(self.fs, 0, n, Category.Silence))
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
                        Segment(self.fs, start, segment.start, len(segments))
                    )
                start = segment.stop + 1
        else:
            # No silent segments found: return entire audio as a single segment
            segments.append(Segment(self.fs, 0, n))

        # Categorize the non-silent segments if specified
        if categorize:
            for segment in segments:
                old_segment = self.select(segment)
                segment.cat = self.get_category()
                self.select(old_segment)

        # Adjust start and stop to be relative to the start of the audio
        for segment in segments:
            segment.start += start_sample
            segment.stop += start_sample

        return segments

    def get_bins(self, channel: Channel = Channel.Mean) -> FreqBins:
        """Return the frequency bins of the selected audio."""
        bins = FreqBins(self.selected_samples, self.md, channel)
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
            stop = start + nsamps
            if stop >= len(self.samples):
                stop = len(self.samples)
            subsegment = Segment(self.fs, start, stop)

            # Append the primary frequency of the subsegment
            old_segment = self.select(subsegment)
            freqs.append(self.freq)
            self.select(old_segment)

        # If enough frequencies have been collected, remove first and last
        # to avoid edge effects
        if len(freqs) >= 4:
            freqs = freqs[1:-1]

        # If all subsegments have increasing or decreasing frequencies it
        # likely is a sweep.
        if len(freqs) >= 2:
            diffs = np.diff(freqs)
            if np.all(diffs >= 0) or np.all(diffs <= 0):
                return True

        # Sweep not detected
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
        if self.is_silence():
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

        if averages.size == 0:
            return (0.0, 0)

        # Calculate the noise floor as the minimum value of the averages
        noise_floor = float(np.min(averages))
        if dbv(noise_floor) > max_noise_db:
            return (0.0, 0)

        index = int(np.where(averages == noise_floor)[0][0] * avg_size)
        return (noise_floor, index)

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
            with sf.SoundFile(fname) as f:

                self.__init__(fs=f.samplerate, name=f.name)

                while self.read_block(f):
                    pass

                if not len(self):
                    return False

                # Update segment to cover full length
                self.select()

                return True

        except Exception as e:
            print(f"Error reading audio file: {e}")
            return False

    def open(self, fname: str | Path) -> None:
        """
        Open an audio file.

        Parameters
        ----------
        fname : str
            File name to open.

        Returns
        -------
        bool
            True if successful.
        """
        self.sf = sf.SoundFile(fname)
        self.fs = self.sf.samplerate
        self.name = self.sf.name

    def read_block(self, f=None) -> bool:
        """
        Read a block of audio data from an open file.

        Parameters
        ----------
        f : sf.SoundFile | None
            Open sound file to read from. If None, use self.sf.

        Returns
        -------
        bool
            True if data was read, False if EOF.
        """
        if f is None:
            f = self.sf

        data = f.read(self.blocksize, dtype="float32", always_2d=False)
        if len(data) == 0:
            return False

        self.append(data)

        return True

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
        p = Path(fname)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
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
