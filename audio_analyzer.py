"""Audio analyzer class."""

from dataclasses import dataclass
import glob

from analysis import Analysis
from analysis_list import AnalysisList
from audio import Audio
from audio_stats import AudioStats
from component import Components
from freq_resp import FreqResp
from noise_analyzer import NoiseAnalyzer
from segment import Segment
from sweep_analyzer import SweepAnalyzer
from time_analyzer import TimeAnalyzer
from time_stats import TimeStats
from tone_analyzer import ToneAnalyzer
from tone_stats import ToneStats
from util import Category

__all__ = ["AudioAnalysis", "AudioAnalyzer"]


@dataclass
class AudioAnalysis:
    """Audio analysis results class."""

    segment: Segment
    time_stats: TimeStats
    analysis: Analysis
    components: Components

    def __str__(self) -> str:
        """Return string representation of analysis result."""
        return f"Analysis(segment={self.segment}, result={self.analysis}, components={self.components})"


class AudioAnalyzer:
    """Audio analyzer class."""

    analysis_classes = {
        Category.Noise: FreqResp,
        Category.Sweep: FreqResp,
        Category.Tone: ToneStats,
        Category.Unknown: TimeStats,
    }

    analyzer_classes = {
        Category.Noise: NoiseAnalyzer,
        Category.Sweep: SweepAnalyzer,
        Category.Tone: ToneAnalyzer,
        Category.Unknown: TimeAnalyzer,
    }

    def __init__(self) -> None:
        self.audio = Audio(name="Analyzer Audio")
        self.analyzers: list[object] = []
        self.analysis_list = AnalysisList()
        self.results: dict[str, AudioAnalysis] = {}
        self.unit_ids: set[str] = set()

    def analyze(self, segment: Segment) -> AudioAnalysis | None:
        # Get the analyzer class for the segment's category
        analyzer_class = self.analyzer_classes.get(segment.cat)
        if analyzer_class is None:
            return None

        # Create the analyzer
        analyzer = analyzer_class(self.audio)
        self.analyzers.append(analyzer)

        # Get the segment's time stats
        time_stats = TimeStats(self.audio)
        self.analysis_list.append(time_stats)

        # Analyze the segment
        analysis = analyzer.analyze(segment)
        self.analysis_list.append(analysis)

        components: Components = analyzer.components
        audio_analysis = AudioAnalysis(segment, time_stats, analysis, components)

        # Store the analysis results in the results dictionary
        key = f"{self.audio.name}:{segment}"
        self.results[key] = audio_analysis

        # Store the units analyzed
        self.unit_ids.add(self.audio.md.unit_id)

        return audio_analysis

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
            Number of files read successfully.
        """
        num_read = 0
        try:
            fnames = glob.glob(pattern)
            for fname in fnames:
                self.audio = Audio(name=fname)

                print(f"Reading {fname}")

                num_blocks = 0
                self.audio.open(fname)
                while self.audio.read_block():
                    num_blocks += 1

                if num_blocks > 0:
                    num_read += 1
                else:
                    continue

                # Get the audio statistics object and append it to the list
                audio_stats = AudioStats(self.audio)
                self.analysis_list.append(audio_stats)

                # Analyze each audio segment
                for segment in self.audio:
                    print(f"Analyzing {fname} segment: {segment}")
                    self.analyze(segment)

        except Exception as e:
            print(f"Error reading audio file: {e}")

        return num_read

    def print(
        self,
        tone_sort_attr: str | None = "thd",
        time_sort_attr: str | None = "start",
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
                sort_tone_attr=tone_sort_attr,
                sort_time_attr=time_sort_attr,
                components=components,
                noise_floor=noise_floor,
            )

        # Print the audio statistics table if specified
        if noise_floor:
            audio_stats_list = self.analysis_list.filtered(__class__=AudioStats)
            if audio_stats_list:
                print(audio_stats_list.summary("noise_floor"))
                print()

        # Print time statistics table if specified
        if time_sort_attr:
            time_stats_list = self.analysis_list.filtered(__class__=TimeStats)
            if time_stats_list:
                print(time_stats_list.summary(time_sort_attr))
                print()

        # Print frequency response statistics table
        freq_resp_list = self.analysis_list.filtered(__class__=FreqResp)
        if freq_resp_list:
            print(freq_resp_list.summary("lower_edge"))
            print()

        # Print tone statistics table if specified
        if tone_sort_attr:
            tone_stats_list = self.analysis_list.filtered(__class__=ToneStats)
            if tone_stats_list:
                print(tone_stats_list.summary(tone_sort_attr))
