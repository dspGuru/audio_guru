"""Audio Data Class"""

from copy import copy
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la
from scipy import signal
import sounddevice as sd
import soundfile as sf

from constants import (
    DEFAULT_BLOCKSIZE_SECS,
    DEFAULT_FS,
    DEFAULT_MAX_NF_DBFS,
    MIN_SEGMENT_SECS,
    MAX_SEGMENT_SECS,
    MIN_FREQ,
    MIN_SILENCE_PEAK,
    SILENCE_MIN_SECS,
    SILENCE_THRESH_RATIO,
    SILENCE_WINDOW_SECS,
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
    noise : float
        Noise power in the audio signal.
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
        min_segment_secs: float = MIN_SEGMENT_SECS,
        max_segment_secs: float = MAX_SEGMENT_SECS,
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
        self.name: str = name
        self.min_segment_secs: float = min_segment_secs
        self.max_segment_secs: float = max_segment_secs

        self.blocksize = round(fs * DEFAULT_BLOCKSIZE_SECS)
        self.segment = Segment(fs)
        self.samples: np.ndarray = np.ndarray(shape=1, dtype=self.DTYPE)
        self.n_samples_removed: int = 0
        self.segments = Segments()
        self.md = Metadata(self.name, self.segment)
        self.sf = None
        self.stream = None
        self.next_id: int = 0

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
        self.segments = self.get_segments()
        return self

    def __next__(self):
        """Select and return the next audio segment in the iteration."""
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

        # The next segment is complete: pop, select, and return it
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

    @property
    def num_channels(self) -> int:
        """Return the number of channels in the audio data."""
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[1]

    def validate(self, segment: Segment) -> None:
        # Raise ValueError if segment is incompatible with audio or otherwise
        # invalid
        if segment.fs != self.fs:
            raise ValueError("Segment fs must match audio fs")
        if segment.start < 0:
            raise ValueError("Segment start must be non-negative")
        if segment.stop > len(self.samples) + 1:
            raise ValueError("Segment stop must not be greater than audio length + 1")
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
        self.n_samples_removed += stop - start

        offset = stop - start
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

        # If no segment is specified, select the entire audio below. Otherwise,
        # validate the segment.
        if segment is None:
            segment = Segment(self.fs, 0, len(self.samples))
        else:
            self.validate(segment)

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
        """
        # Set default values if needed
        if silence_thresh_ratio is None:
            silence_thresh_ratio = SILENCE_THRESH_RATIO
        if min_silence_secs is None:
            min_silence_secs = SILENCE_MIN_SECS

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

        # Get the samples from the start sample to the end
        samples = np.asarray(self.samples[start_sample:])

        # Convert to mono
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        # Return if there are no amplitudes
        if not len(samples):
            return Segments()

        # Get amplitudes of samples
        amplitudes = np.abs(samples)

        # Calculate silence threshold
        peak = float(max(np.max(amplitudes), MIN_SILENCE_PEAK))
        thresh = peak * float(SILENCE_THRESH_RATIO)

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
        # Detect segments using short windows
        segments = Segments()
        start_secs = self.n_samples_removed / self.fs
        step = max(1, int(self.fs * SILENCE_WINDOW_SECS))

        # Categorize segments
        n = len(amplitudes)
        for i in range(0, n, step):
            stop = min(i + step, n)
            max_amp = np.max(amplitudes[i:stop])
            cat = Category.Unknown if max_amp >= thresh else Category.Silence
            seg = Segment(
                self.fs,
                start_sample + i,
                start_sample + stop,
                cat=cat,
                start_secs=start_secs,
                complete=False,
            )
            segments.append(seg)

        # Combine same-category segments
        segments.combine()

        # Recategorize short silence as Unknown
        for seg in segments:
            if seg.cat == Category.Silence and seg.secs < min_silence_secs:
                seg.cat = Category.Unknown

        # Combine segments again to merge recategorized segments
        segments.combine()

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
                seg.start = seg.start - first
                seg.stop = seg.start + last + 1

            # Mark segment as complete if it is followed by silence or EOF
            if seg.stop < len(self.samples) - min_silence_n or self.eof:
                seg.complete = True

    def get_bins(self, channel: Channel = Channel.Mean) -> FreqBins:
        """Return the frequency bins of the selected audio.

        Parameters
        ----------
        channel : Channel
            Channel for which to compute bins (default is Channel.Mean).
        """
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

        # Filter out very low frequencies which might be noise
        valid_freqs = [f for f in freqs if f > 0.5 * MIN_FREQ]

        # If all subsegments have increasing or decreasing frequencies it
        # is a sweep.
        if len(valid_freqs) >= 2:
            diffs = np.diff(valid_freqs)
            # Check for strict monotonicity: not just >= 0, but > 0 to avoid
            # constant freq
            if np.all(diffs > 0) or np.all(diffs < 0):
                return True

        # Sweep not detected
        return False

    def get_category(self, secs: float = 0.1) -> Category:
        """
        Determine the category of the selected audio.

        Parameters
        ----------
        secs : float
            Duration in seconds for sweep detection. Default is 0.1 seconds.

        Returns
        -------
        Category
            The audio category: Silence, Tone, Sweep, Noise, or Unknown.
        """
        # Check for silence
        if self.is_silence():
            return Category.Silence

        # Get frequency bins
        bins = self.get_bins()

        # Check for sweep (before checking for sine)
        if self.is_sweep(secs=secs):
            return Category.Sweep

        # Check for sine
        if bins.is_sine():
            return Category.Tone

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
        max_noise_db : float
            Maximum noise level in dBFS.

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
                self.sf = f

                while self.read_block(f):
                    pass

                # Cleanup reference before checking length to maintain consistency
                self.sf = None

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
        # Ensure Path objects are converted to strings for soundfile
        if isinstance(fname, Path):
            fname = str(fname)

        # Open the file
        self.sf = sf.SoundFile(fname)
        self.fs = self.sf.samplerate
        self.name = self.sf.name
        self.next_id = 0

    def open_device(
        self,
        device: int | str | None = None,
        fs: float | None = None,
        blocksize: int | None = None,
        channels: int = 1,
        dtype: str = "float32",
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
        """
        if fs is not None:
            self.fs = fs

        if blocksize is not None:
            self.blocksize = blocksize

        # Open the input stream
        self.stream = sd.InputStream(
            device=device,
            samplerate=self.fs,
            blocksize=self.blocksize,
            channels=channels,
            dtype=dtype,
        )
        self.stream.start()
        self.next_id = 0

    def close(self) -> None:
        """Close the audio file or device stream."""
        if self.sf is not None and not self.sf.closed:
            self.sf.close()
            self.sf = None

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

        # Return False if no data was read
        if len(data) == 0:
            return False

        return True

    def write(self, fname: str | Path = "") -> bool:
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
            sf.write(fname, data, int(self.fs), subtype="FLOAT")
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
