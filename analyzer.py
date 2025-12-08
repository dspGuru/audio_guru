"""Abstract base class for audio testing and analysis."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from audio import Audio
from component import Components
from constants import MIN_AUDIO_LEN
from segment import Segment
from time_stats import TimeStats

__all__ = ["Analyzer"]


class Analyzer(ABC):
    """Abstract base class for audio signal analysis.

    Attributes
    ----------
        audio : Audio
            Audio data to analyze.
        components : Components
            Frequency components identified in the audio.

    Abstract Methods
    ----------------
        analyze(segment: Segment | None = None) -> object
            Calculate and return statistics from audio analysis.
        get_components() -> Components
            Get audio components from analysis.
        print(**kwargs) -> None
            Print the analysis results sorted on the specified statistic attribute.
        reset() -> None
            Resets/creates members which are calculated from audio.
    """

    def __init__(self, audio: Audio) -> None:
        """
        Initialize AudioTester with an Audio object.

        Parameters
        ----------
        audio : Audio
            The Audio object to be tested.
        """
        self.audio = audio
        self.reset()

    @property
    def fs(self) -> float:
        """Return the sampling frequency in Hz."""
        return self.audio.fs

    @abstractmethod
    def analyze(self, segment: Segment | None = None) -> TimeStats:
        """
        Abstract method to calculate and return statistics from audio analysis.
        Selects the specified segment of audio for analysis by derived classes
        and calls the get_components() method overridden by the derived class.
        Sets and returns the time-domain statistics.

        Parameters
        ----------
        seg : Segment | None
            Segment of audio to analyze. If None, use the entire audio.

        Returns
        -------
        TimeStats
            Time-domain statistics.
        """
        self.select(segment)
        self.time_stats = TimeStats(self.audio)
        self.components = self.get_components()
        return self.time_stats

    @abstractmethod
    def get_components(self) -> Components:
        """
        Abstract method to get audio components from analysis.

        Returns
        -------
        Components
            List of audio components.
        """
        return Components(self.audio.md)

    @abstractmethod
    def print(self, **kwargs) -> None:
        """
        Print the analysis results sorted on the specified statistic attribute.

        Parameters
        ----------
        kwargs
            Keyword arguments.
        """

        if self.audio.md.pathname:
            print(f"Data from {self.audio.md.pathname}:")
            print(f"{self.audio.md.segment}")
            print()

    @abstractmethod
    def reset(self) -> None:
        """Resets/creates members which are calculated from audio"""
        self.components = None
        self.analysis = None

    def read(self, fname: str) -> bool:
        """Read audio data from file. Retun True if successful."""
        self.reset()
        return self.audio.read(fname)

    def get_segments(self) -> list[Segment]:
        """
        Get the list of audio segments from the Audio object.

        Returns
        -------
        list[Segment]
            List of audio segments.
        """
        return self.audio.get_segments()

    def select(self, segment: Segment | None = None, max_secs: float = 10.0) -> None:
        """
        Select a segment of audio for analysis.

        Parameters
        ----------
        seg : Segment
            The audio segment to select. If None, select the entire audio.
        """
        if segment is None:
            segment = Segment(self.fs, 0, len(self.audio))

        # Truncate the slice to a whole number of seconds
        segment.trunc(1.0)
        segment.limit(max_secs)

        # Raise ValueError if the selected audio sample is too short for
        # meaningful analysis
        if len(segment) < MIN_AUDIO_LEN:
            raise ValueError(f"Audio segment {segment} is too short to analyze")

        self.audio.select(segment)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representing the list using Stats.get_stats() for
        each stat in the list.
        """
        components = self.get_components()
        return components.get_dataframe()

    def to_csv(
        self, filepath: str, index: bool = False, print_head: bool = False
    ) -> None:
        """
        Write the object's DataFrame to the specified CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        index : bool, optional
            Whether to write row indices to the CSV file (default is False).
        """
        df = self.get_dataframe()
        p = Path(filepath)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        # Print the data frame header if specified
        if print_head:
            print(df.head())

        # Write the DataFrame to CSV with specified options
        df.to_csv(p, header=True, index=index, float_format="%.1f")
