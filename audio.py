"""Audio data class"""

import math

import numpy as np
import scipy.linalg as la
from scipy import signal
import soundfile as sf

from metadata import Metadata
from segment import Segment
from segment import Segments
from tone import Tone
from db import *

class Audio:
    """Audio data"""

    DTYPE = np.float32      # audio sample data type

    def __init__(self, fs: int=44100, name: str=''):
        self.data = np.ndarray(shape=1, dtype=self.DTYPE)
        self.segment = Segment(fs)
        self.md = Metadata(name, self.segment)        


    def __len__(self):
        return len(self.data)


    @property
    def fs(self) -> int:
        """Return the audio's sampling frequency."""
        return self.segment.fs


    @property
    def max(self) -> float:
        """Return the maximum value of the selected audio."""
        return float(np.max(self.data[self.segment.slice])) if len(self) else 0.0
    

    @property
    def min(self) -> float:
        """Return the minimum value of the selected audio."""
        return float(np.min(self.data[self.segment.slice])) if len(self) else 0.0


    @property
    def rms(self) -> float:
        """Return the RMS value of the audio."""
        a = self.data[self.segment.slice]
        if a.size == 0:
            return 0.0
        rms = la.norm(a) / np.sqrt(a.size)
        return float(rms)


    @property
    def freq(self) -> float:
        """Return the audio's fundamental frequency from its zero-crossings."""
        # Get zero-crossings
        zc = np.where(np.diff(np.sign(self.data[self.segment.slice])))[0]

        # Remove the last zero-crossing, if necessary, to span an integer
        # number of cycles
        if len(zc) % 2 == 0:
            zc = zc[:-1]

        # Calculate the signal frequency from the zero-crossings
        nzc = len(zc)
        freq = 0.5 * (nzc - 1) / float( zc[-1] - zc[0] ) * self.fs
        return freq


    def select(self, segment:Segment | None=None) -> None:
        """Select the specified segment for subsequent operations."""
        if segment is None:
            # Select the entire audio
            self.segment = Segment(self.fs, 0, len(self.data) - 1)
        else:
            self.segment = segment


    def append(self, other) -> None:
        """Append the other audio to self"""
        # Support appending from another Audio or any array-like audio buffer
        if isinstance(other, Audio):
            arr = other.audio
        else:
            arr = other

        arr = np.asarray(arr, dtype=self.DTYPE)
        if arr.size == 0:
            return

        # If current audio is empty, just copy; otherwise concatenate
        if getattr(self.data, "size", 0) == 0:
            self.data = arr.copy()
        else:
            self.data = np.concatenate((np.asarray(self.data, dtype=self.DTYPE), arr))


    def generate(self, freq: float=1000.0, thd: float=MIN_PWR,
                 secs: float=10.0) -> None:
        """Generate audio."""

        # Generate tone at the specified frequency
        pwr = 0.6
        t = Tone(freq, pwr)
        self.data = t.generate(secs, self.fs)

        # If specified, generate distortion tone at twice the nominal frequency
        if thd > MIN_PWR:
            t2 = Tone(2.0 * freq, thd*pwr)
            self.data += t2.generate(secs, self.fs)
            thd_str = f'{(thd * 100.0):0.3f}%'
        else:
            thd_str = ''
        freq_str = f'{freq:0.1f}' if freq < 1000.0 else f'{math.trunc(freq / 1000.0)}k'
        self.md.set(mfr='Gen', desc=freq_str, model=thd_str)


    def get_segments(self, ratio: float=1e-3, secs: float=0.1,
                     silent: bool=False) -> Segments:
        """Return a list of audio segments where the signal's absolute amplitude
        is above (peak * ratio) or below (silent) for at least the specified
        time.
        """

        # Initialize return value
        segments = Segments()        

        # Get number of samples for signal gap detection. Return an empty
        # list if the audio is empty or the gap size is zero or less.
        secs_n = secs*self.fs
        a = np.asarray(self.data)
        n = a.size // a.ndim
        if n == 0 or secs_n <= 0:
            return segments
        
        # Convert audio to mono for subsequent processing
        if a.ndim > 1:
            a = np.mean(a, axis=1)

        peak = float(np.max(np.abs(a)))
        if peak == 0.0:
            # Entire signal is zero -> single segment spanning whole array if
            # nsamples satisfied
            segments.append(Segment(self.fs, 0, n - 1))
            return segments

        thresh = peak * float(ratio)

        # Create boolean array where condition is true (below threshold)
        below = np.abs(a) <= thresh

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
        start = 0
        for s, e in zip(starts, ends):
            s, e = int(s), int(e)
            if (e - s) >= secs_n:            
                if silent:
                    segments.append(Segment(self.fs, s, e))
                else:                         
                    if (s - start) >= secs_n:
                        segments.append(Segment(self.fs, start, s))
                        start = e

        # If no segments were found when silent is False, add a segment for
        # the entire sample
        if not silent and not segments:
            segments.append(Segment(self.fs, start, n - 1))

        return segments


    def get_noise_floor(self, secs: float=0.25) -> (float, int):
        """Return the audio's noise floor and the index at which it was
        found."""
        if not len(self):
            return 0.0

        # Calculate the absolute value of the audio data and average it
        # across channels if necessary
        abs_audio = np.abs(self.data)
        if abs_audio.ndim > 1:
            abs_audio = np.mean(abs_audio, axis=1)

        # Calculate the average of each segment of the audio
        avg_size = int(self.fs * secs)        
        averages = np.array([], dtype=self.DTYPE)
        for i in range(0, len(abs_audio) - avg_size + 1, avg_size):
            segment = abs_audio[i:i + avg_size]
            avg = np.mean(segment)
            averages = np.append(averages, avg)

        # Calculate the noise floor as the minimum value of the averages
        noise_floor = np.min(averages)
        index = np.where(averages == noise_floor)[0][0] * avg_size
        return (float(noise_floor), index)


    def read(self, fname: str) -> bool:
        """Read audio data from file. Retun True if successful."""
        try:
            data, fs = sf.read(fname, dtype='float32')
        except Exception:
            # On failure, leave audio empty
            self.data = np.array([], dtype=self.DTYPE)
            return len(self) > 0

        self.data = np.asarray(data, dtype=self.DTYPE)
        self.segment = Segment(fs, 0, len(self) - 1)
        self.md = Metadata(fname, self.segment)

        return len(self) > 0


    def write(self, fname: str='') -> bool:
        """Write the selected segment of audio data to file. Return True if
          successful."""
        # Write audio data using soundfile module
        if not fname:
            fname = self.md.name
        if not fname:
            return False

        # Return False if there is nothing to write
        if not len(self):
            return False

        # Write the file and update the file name
        try:
            sf.write(fname, np.asarray(self.data[self.segment.slice], dtype=self.DTYPE), int(self.fs))
            self.md.name = fname
            return True
        except Exception:
            return False

