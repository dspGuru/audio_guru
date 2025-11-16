"""Audio segment classes"""

from dataclasses import dataclass
from enum import Enum

class Category(Enum):
    """Enumerates categories assigned to audio signals.

    Members
    -------
    Unknown
        The category is not determined or does not fit any other class.
    Tone
        Narrowband, near-sinusoidal signal tone content.
    Harmonic
        Harmonic of signal tone content.
    Distortion
        Harmonically related artifacts produced by nonlinear processing or
        clipping.
    Spurious
        Tone which is not harmonically related to any signal tones.
    Noise
        Noise-like content.
    """
    Unknown = 0
    Tone = 1
    Harmonic = 2
    Distortion = 3
    Spurious = 4
    Noise = 5


@dataclass
class Segment:
    """
    Audio data segment

    Attributes
    ----------
        fs: Sampling frequency, in Hz
        start: Start index
        stop: Stop index
        cat: Category
    """
    fs: int                         # sampling frequency, in Hz
    start: int=0                    # start index
    stop: int=1                     # stop index
    cat: Category=Category.Unknown  # category

    def __len__(self):
        return self.stop - self.start


    def __str__(self) -> str:
        """Return a description of the segment."""
        return f'{self.start_secs:0.1f}s - {self.stop_secs:0.1f}s'


    @property
    def secs(self) -> float:
        """Return the length of the segment, in seconds."""
        return float(len(self)) / float(self.fs) if self.fs > 0 else 0.0


    @property
    def slice(self) -> slice:
        """Return a slice for the segment."""
        return slice(self.start, self.stop)


    @property
    def start_secs(self) -> float:
        """Return the start seconds of the segment."""
        return float(self.start) / float(self.fs) if self.fs > 0 else 0.0


    @property
    def stop_secs(self) -> float:
        """Return the stop seconds of the segment."""
        return float(self.stop) / float(self.fs) if self.fs > 0 else 0.0


    def trunc(self, secs: float=1.0) -> 'Segment':
        """Return a segment truncated to span a multiple of the specified
        number of seconds.
        
        Parameters
        ----------
        secs : float
            Number of seconds to which the segment is truncated.
            Default is 1.0 (one second).
        
        Returns
        -------
        Segment
            Truncated segment.
        """

        # If the segment is less than one second, return it unchanged
        n = len(self)
        if n < self.fs:
            return self

        # Initialize locals
        stop = self.stop
        length = len(self)
        sec_samples = round(secs * self.fs)

        if length > sec_samples:
            # Shorten to an integral number of seconds (floor)
            whole_secs = length // sec_samples
            stop = self.start + whole_secs*sec_samples

        # Return the truncated segment  
        seg = Segment(self.fs, self.start, stop)
        return seg


class Segments(list[Segment]):
    """List of Audio Segments."""

    def __str__(self) -> str:
        """Return a multi-line string representation of the segment list."""
        lines = []
        for segment in self:
            lines.append(str(segment))
        return '\n'.join(lines)