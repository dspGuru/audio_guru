"""Abstract base class for analyzer result."""

from abc import ABC, abstractmethod
from copy import copy

from metadata import Metadata


__all__ = ["Analysis"]


class Analysis(ABC):
    """Abstract base class for analyzer result."""

    def __init__(self, md: Metadata) -> None:
        """Initialize the analyzer result."""
        self.md = copy(md)

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the analyzer result to a dictionary."""
        pass

    @abstractmethod
    def summary(self) -> str:
        """Provide a summary of the analyzer result."""
        pass
