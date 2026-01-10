"""Abstract base class for audio signal analysis."""

from abc import ABC, abstractmethod
from copy import copy
from pathlib import Path

import pandas as pd

from audio import Audio
from component import Components
from constants import DEFAULT_MAX_ANALYSIS_SECS, SEGMENT_MIN_SECS
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
        analyze() -> object
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
    def analyze(self) -> TimeStats:
        """
        Abstract method to calculate and return statistics from audio analysis.
        Calls the get_components() method overridden by the derived class.
        Sets and returns the time-domain statistics.

        Returns
        -------
        TimeStats
            Time-domain statistics.
        """
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

    def print(self, **kwargs) -> None:
        """
        Print the analysis results sorted on the specified statistic attribute.

        Parameters
        ----------
        kwargs
            Keyword arguments.
        """
        max_components: int | None = kwargs.get("components", None)
        if max_components is not None:
            self.components.print(max_components=max_components)
            print()

    @abstractmethod
    def reset(self) -> None:
        """Resets/creates members which are calculated from audio"""
        self.components = None

    def read(self, fname: str) -> bool:
        """
        Read audio data from file. Retun True if successful.

        Parameters
        ----------
        fname : str
            File name to read.
        """
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

    def select(
        self,
        segment: Segment | None = None,
        max_secs: float = DEFAULT_MAX_ANALYSIS_SECS,
    ) -> None:
        """
        Select a segment of audio for analysis.

        Parameters
        ----------
        segment : Segment | None
            The audio segment to select. If None, select the entire audio.
        max_secs : float
            Maximum duration of the selection in seconds (default
            DEFAULT_MAX_ANALYSIS_SECS).

        Raises
        ------
        ValueError
            If the audio segment is too short to analyze.
        """
        if segment is None:
            segment = Segment(self.fs, 0, len(self.audio))
        else:
            segment = copy(segment)

        # Truncate the slice to a whole number of seconds and limit to max_secs
        segment.trunc(1.0)
        segment.limit(max_secs)

        # Raise ValueError if the selected audio segment is too short for
        # meaningful analysis
        if segment.secs < SEGMENT_MIN_SECS:
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
        print_head : bool, optional
            Whether to print the CSV head (default is False).
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
