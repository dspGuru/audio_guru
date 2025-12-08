"""Audio segment classes"""

from dataclasses import dataclass
import math

from constants import DEFAULT_FS
from util import Category


@dataclass
class Segment:
    """
    Audio data segment, which is used to describe a portion of an audio
    signal.

    Attributes
    ----------
        fs: Sampling frequency, in Hz
        start: Start index
        stop: Stop index
        id: Segment identifier
        cat: Audio category of the segment
    """

    fs: float = DEFAULT_FS
    start: int = 0
    stop: int = 1
    id: int = 0
    cat: Category = Category.Unknown

    def __len__(self):
        return self.stop - self.start + 1

    def __str__(self) -> str:
        """Return a string description of the segment."""
        return f"{self.id}: {self.start_secs:4.1f}-{self.stop_secs:0.1f}s"

    @property
    def desc(self) -> str:
        """Return a description of the segment, including its category."""
        return f"Segment {self.id}: {self.start_secs:0.1f}-{self.stop_secs:0.1f}s {self.cat.name}"

    @property
    def secs(self) -> float:
        """Return the length of the segment, in seconds."""
        return float(len(self)) / float(self.fs) if self.fs > 0 else 0.0

    @property
    def slice(self) -> slice:
        """Return a slice for the segment."""
        return slice(self.start, self.stop + 1)

    @property
    def start_secs(self) -> float:
        """Return the start seconds of the segment."""
        return float(self.start) / float(self.fs) if self.fs > 0 else 0.0

    @property
    def stop_secs(self) -> float:
        """Return the stop seconds of the segment."""
        return float(self.stop) / float(self.fs) if self.fs > 0 else 0.0

    @property
    def title(self) -> str:
        """Return a title for the segment."""
        start = round(self.start_secs)
        stop = round(self.stop_secs)
        return f"{self.cat.name}_{start}-{stop}s"

    def limit(self, max_secs: float) -> None:
        """Return a segment limited to the specified maximum number of seconds.

        Parameters
        ----------
        max_secs : float
            Maximum number of seconds for the segment. If 0.0, no limit is
            applied.
        """

        # Return if the limit is invalid or the segment is already within the
        # limit
        if max_secs == 0.0 or self.secs <= max_secs:
            return

        # Set the new stop index
        new_length = math.trunc(max_secs * self.fs)
        self.stop = self.start + new_length

    def trunc(self, secs_multiple: float = 1.0) -> None:
        """Truncate the segment to span a multiple of the specified number of
        seconds.

        Parameters
        ----------
        secs_multiple : float
            Number of seconds to which the segment is truncated.
            Default is 1.0 (one second).
        """

        # If the segment is less than one second, return with it unchanged
        n = len(self)
        if n < self.fs:
            return

        # Truncate to an integral number of seconds
        n_secs = round(secs_multiple * self.fs)
        if n > n_secs:
            # Shorten to an integral number of seconds (floor)
            multiple = n // n_secs
            self.stop = self.start + multiple * n_secs


class Segments(list[Segment]):
    """List of Audio Segments."""

    def __str__(self) -> str:
        """Return a multi-line string representation of the segment list."""
        lines: list[str] = []
        for segment in self:
            lines.append(str(segment))
        return "\n".join(lines)
