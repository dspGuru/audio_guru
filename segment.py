"""Audio segment classes"""

from dataclasses import dataclass
import math

from constants import DEFAULT_FS
from util import Category


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
        start_delta: Delta applied to start index to account for audio sample
         truncation after segment initialization

    Properties
    ----------
        desc: Description of the segment
        secs: Length of the segment, in seconds
        slice: Slice for the segment
        start_secs: Start time of the segment, in seconds
        stop_secs: Stop time of the segment, in seconds
        title: Title for the segment

    Notes
    -----
    As with Python's slice(), the stop index is exclusive, i.e. the segment
    spans start to stop-1.
    """

    def __init__(
        self,
        fs: float = DEFAULT_FS,
        start: int = 0,
        stop: int = 1,
        id: int | None = None,
        cat: Category = Category.Unknown,
        start_secs: float = 0.0,
        complete: bool = False,
    ):
        """
        Initialize the audio segment.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz (default DEFAULT_FS).
        start : int
            Start index of the segment (default 0).
        stop : int
            Stop index of the segment (exclusive) (default 1).
        id : int | None
            Identifier for the segment. If None, the segment will be described as
            "All".
        cat : Category
            Category of the audio content (default Category.Unknown).
        start_secs : float
            Start time of the file in seconds (default 0.0).
        """
        self.fs: float = fs
        self.start: int = start
        self.stop: int = stop
        self.id: int | None = id
        self.cat: Category = cat
        self.complete: bool = complete
        self._start_secs: float = start_secs

    def __len__(self):
        return self.stop - self.start

    def __str__(self) -> str:
        """Return a string description of the segment."""
        id_str = "All" if self.id is None else str(self.id)
        return f"{id_str}: {self.start_secs:4.1f}-{self.stop_secs:0.1f}s"

    @property
    def desc(self) -> str:
        """Return a description of the segment, including its category."""
        if self.id is None:
            return f"{self.start_secs:0.1f}-{self.stop_secs:0.1f}s {self.cat.name}"

        return f"Segment {self.id}: {self.start_secs:0.1f}-{self.stop_secs:0.1f}s {self.cat.name}"

    @property
    def secs(self) -> float:
        """Return the length of the segment, in seconds."""
        return float(len(self)) / float(self.fs)

    @property
    def slice(self) -> slice:
        """Return a slice for the segment."""
        return slice(self.start, self.stop)

    @property
    def start_secs(self) -> float:
        """Return the start seconds of the segment."""
        return self._start_secs + (float(self.start) / self.fs)

    @property
    def stop_secs(self) -> float:
        """Return the stop seconds of the segment."""
        return self.start_secs + self.secs

    @property
    def title(self) -> str:
        """Return a title for the segment."""
        start = round(self.start_secs)
        stop = round(self.stop_secs)
        return f"{self.cat.name}_{start}-{stop}s"

    def limit(self, max_secs: float) -> None:
        """Limit the segment to the specified maximum number of seconds.

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

    def offset(self, offset: int) -> None:
        """Offset the segment by the specified number of samples.

        Parameters
        ----------
        offset : int
            Number of samples to offset the segment by.

        Notes
        -----
        This is primarily used to adjust the segment's start and stop indices
        and the start time of the segment when the audio data is truncated.
        In such cases, the offset is the number of samples that were truncated
        from the original start of the audio data.
        """
        self.start -= offset
        assert self.start >= 0
        self.stop -= offset
        self._start_secs += float(offset) / self.fs

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
        if n <= self.fs:
            return

        # Truncate to an integral number of seconds
        n_secs = round(secs_multiple * self.fs)
        if n > n_secs:
            # Shorten to an integral number of seconds (floor). However, if
            # the segment is very close (within 1 ms) to the next integral
            # number of seconds, snap to that instead to avoid discarding
            # large amounts of data due to minor trimming (e.g. from silence
            # detection).
            tolerance = max(1, round(self.fs * 0.001))
            remainder = n % n_secs
            if remainder > 0 and (n_secs - remainder) <= tolerance:
                multiple = (n // n_secs) + 1
            else:
                multiple = n // n_secs

            self.stop = self.start + (multiple * n_secs)


class Segments(list[Segment]):
    """List of Audio Segments."""

    def __str__(self) -> str:
        """Return a multi-line string representation of the segment list."""
        lines: list[str] = []
        for segment in self:
            lines.append(str(segment))
        return "\n".join(lines)

    def combine(self) -> None:
        """Combine consecutive segments which have the same category."""
        i = 0
        while i < len(self) - 1:
            if self[i].cat == self[i + 1].cat and self[i].stop == self[i + 1].start:
                self[i].stop = self[i + 1].stop
                self.pop(i + 1)
            else:
                i += 1

    def merge(self, segments: "Segments") -> None:
        """Merge segments into this list."""
        if self and segments:
            if (
                not self[-1].complete
                and self[-1].stop == segments[0].start
                and self[-1].cat == segments[0].cat
            ):
                self[-1].stop = segments[0].stop
                self[-1].complete = segments[0].complete
                segments.pop(0)

        self.extend(segments)

    def assign_ids(self, next_id: int = 0) -> int:
        """Assign contiguous IDs to segments."""
        # Return next_id if the list is empty
        if not self:
            return next_id

        # Assign ID to the first segment if it has no ID
        if self[0].id is None:
            self[0].id = next_id

        # Assign IDs to the remaining segments
        for i, seg in enumerate(self[1:], start=1):
            seg.id = self[i - 1].id + 1

        # Return the next available ID
        return self[-1].id + 1

    def offset(self, offset_samples: int) -> None:
        """Offset the segment by the specified number of samples.

        Parameters
        ----------
        offset_samples : int
            Number of samples to offset the segment by.
        """
        for segment in self:
            segment.offset(offset_samples)

    def trunc(self, secs_multiple: float = 1.0) -> None:
        """Truncate all segments to span a multiple of the specified number of
        seconds.

        Parameters
        ----------
        secs_multiple : float
            Number of seconds to which the segments are truncated.
            Default is 1.0 (one second).
        """
        for segment in self:
            segment.trunc(secs_multiple)
