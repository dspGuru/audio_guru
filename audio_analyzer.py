"""Audio analyzer class."""

from dataclasses import dataclass
import glob

from analysis import Analysis
from audio import Audio
from component import Components
from noise_analyzer import NoiseAnalyzer
from segment import Segment
from sweep_analyzer import SweepAnalyzer
from time_analyzer import TimeAnalyzer
from tone_analyzer import ToneAnalyzer
from tone_stats import ToneStats, ToneStatsList
from util import Category

__all__ = ["AudioAnalyzer", "AudioAnalysis"]


@dataclass
class AudioAnalysis:
    """Base class for analysis results."""

    segment: Segment
    result: Analysis
    components: Components

    def __str__(self) -> str:
        """Return string representation of analysis result."""
        return f"Analysis(segment={self.segment}, result={self.analysis}, components={self.components})"


class AudioAnalyzer:
    """Audio analyzer class."""

    analyzer_classes = {
        Category.Noise: NoiseAnalyzer,
        Category.Sweep: SweepAnalyzer,
        Category.Tone: ToneAnalyzer,
        Category.Unknown: TimeAnalyzer,
    }

    def __init__(self) -> None:
        self.audio = Audio(name="Analyzer Audio")
        self.analyzers: list[object] = []
        self.results: dict[str, AudioAnalysis] = {}
        self.unit_ids: set[str] = set()

    def read(self, pattern: str) -> int:
        """
        Read and analyze the specified sample data files.

        Parameters
        ----------
        pattern : str
            File name pattern (may include wildcards) to match input files.

        Returns
        -------
        int
            Number of files successfully read.
        """

        num_read = 0
        fnames: list[str] = glob.glob(pattern)
        for fname in fnames:
            if not self.audio.read(fname):
                continue

            num_read += 1
            self.unit_ids.add(self.audio.md.unit_id)

            for segment in self.audio:
                # Get the analyzer class for the segment's category
                analyzer_class = self.analyzer_classes.get(segment.cat)
                if analyzer_class is None:
                    continue

                # Create the analyzer
                analyzer = analyzer_class(self.audio)
                self.analyzers.append(analyzer)

                # Analyze the segment and store the results
                result = analyzer.analyze(segment)
                components: Components = analyzer.components
                key = f"{fname}:{segment}"
                analysis = AudioAnalysis(segment, result, components)
                self.results[key] = analysis

        return num_read

    def print(
        self,
        sort_stat_attr: str | None = "thd",
        components: bool = True,
        noise_floor: bool = True,
    ) -> None:
        """
        Print the analysis results sorted on the specified statistic attribute.

        Parameters
        ----------
        sort_stat_attr : str
            Name of the statistic attribute to sort by (default is 'start').
        components : bool
            Whether to print component details for each audio segment.
        noise_floor : bool
            Whether to print noise floor details for each audio segment.
        """
        # Print analysis results for each analyzer
        for analyzer in self.analyzers:
            analyzer.print(
                sort_stat_attr=sort_stat_attr,
                components=components,
                noise_floor=noise_floor,
            )

        # Collect tone statistics results
        tone_stats_list = ToneStatsList()
        for analysis in self.results.values():
            if isinstance(analysis.result, ToneStats):
                tone_stats_list.append(analysis.result)

        # Print tone statistics table if specified
        if tone_stats_list and sort_stat_attr:
            print("Tone Statistics Summary:")
            tone_stats_list.print(sort_stat_attr=sort_stat_attr)


def main():
    """Main function to run the audio analyzer from command line."""
    import sys

    try:
        pattern = sys.argv[1]
    except IndexError:
        pattern = "*.wav"

    audio_analyzer = AudioAnalyzer()
    if not audio_analyzer.read(pattern):
        print(f"Error: No audio files found matching pattern '{pattern}'")
        return

    audio_analyzer.print(sort_stat_attr="start", components=True, noise_floor=True)


if __name__ == "__main__":
    main()
