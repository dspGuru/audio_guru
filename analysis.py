"""Abstract base class for analyzer result."""

from abc import ABC, abstractmethod
from copy import copy

from metadata import Metadata
from segment import Segment


__all__ = ["Analysis"]


class Analysis(ABC):
    """Abstract base class for analyzer result.

    Attributes
    ----------
        md : Metadata
            Metadata for the analyzer result.
    """

    @staticmethod
    @abstractmethod
    def summary_header(self) -> str:
        """Return the header for the summary table."""
        pass

    def __init__(self, md: Metadata) -> None:
        """Initialize the analyzer result."""
        self.md = copy(md)

    @property
    def cat(self) -> str:
        """Return the category of the segment."""
        return self.md.segment.cat.name

    @property
    def fs(self) -> int:
        """Return the sample rate."""
        return self.md.fs

    @property
    def secs(self) -> float:
        """Return the duration of the segment."""
        return self.md.segment.secs

    @property
    def segment(self) -> Segment:
        """Return the segment."""
        return self.md.segment

    @property
    def start(self) -> float:
        """Return the start time of the segment."""
        return self.md.segment.start_secs

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the analyzer result to a dictionary."""
        pass

    @abstractmethod
    def summary(self) -> str:
        """Return a one-line summary of the analyzer result."""
        pass
