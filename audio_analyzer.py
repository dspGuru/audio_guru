"""Audio analyzer class."""

from dataclasses import dataclass
import glob

from analysis import Analysis
from analysis_list import AnalysisList
from audio import Audio
from audio_stats import AudioStats
from component import Components
from constants import MIN_SEGMENT_SECS, MAX_SEGMENT_SECS
from freq_resp import FreqResp
from noise_analyzer import NoiseAnalysis
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
        Category.Silence: TimeAnalyzer,
        Category.Sweep: SweepAnalyzer,
        Category.Tone: ToneAnalyzer,
        Category.Unknown: TimeAnalyzer,
    }

    def __init__(self, max_segment_secs: float = MAX_SEGMENT_SECS) -> None:
        """
        Initialize the analyzer.

        Parameters
        ----------
        max_segment_secs : float
            Maximum segment length in seconds. Default is 60 seconds.
        """
        self.audio = Audio(name="Analyzer Audio")
        self.analyzers: list[object] = []
        self.analysis_list = AnalysisList()
        self.results: dict[str, AudioAnalysis] = {}
        self.unit_ids: set[str] = set()

    def analyze(self, segment: Segment) -> AudioAnalysis | None:
        """
        Analyze the specified audio segment.

        Parameters
        ----------
        segment : Segment
            Audio segment to analyze.

        Returns
        -------
        AudioAnalysis | None
            Analysis results or None if analysis failed.
        """
        if segment.secs < MIN_SEGMENT_SECS:
            print(f"Segment {segment.desc} is too short to analyze")
            return None

        # Get the category of the audio segment if it is unknown
        if segment.cat == Category.Unknown:
            self.audio.select(segment)
            segment.cat = self.audio.get_category()

        # Select the segment in the audio object to update the segment's
        # possible new category
        self.audio.select(segment)

        print(f"Analyzing segment: {segment.desc}")

        # Get the analyzer class for the segment's category
        analyzer_class = self.analyzer_classes.get(segment.cat)
        if analyzer_class is None:
            return None

        # Create the analyzer
        analyzer = analyzer_class(self.audio)
        self.analyzers.append(analyzer)

        # Analyze the segment
        analysis = analyzer.analyze(segment)
        self.analysis_list.append(analysis)

        # Get segment time stats. Use the analysis result if it is a TimeStats
        # object. Otherwise, create a new TimeStats object for the segment.
        if isinstance(analysis, TimeStats):
            time_stats = analysis
        else:
            time_stats = TimeStats(self.audio)
            self.analysis_list.append(time_stats)

        # Creat an audio analysis object to store the complete analysis results
        components = analyzer.components
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

                # Read the audio file in blocks
                num_blocks = 0
                self.audio.open(fname)
                while self.audio.read_block():
                    num_blocks += 1

                    # Analyze each audio segment
                    for segment in self.audio:
                        self.analyze(segment)
                        self.audio.remove(segment, True)

                if num_blocks > 0:
                    num_read += 1
                else:
                    continue

                # Get the audio statistics object and append it to the list
                audio_stats = AudioStats(self.audio)
                self.analysis_list.append(audio_stats)

        except Exception as e:
            print(f"Error reading audio file: {e}")

        return num_read

    def read_device(self, duration: float = 0.0) -> bool:
        """
        Read and analyze audio from the default device.

        Parameters
        ----------
        duration : float
            Maximum duration to record in seconds. If 0, record indefinitely.

        Returns
        -------
        bool
            True if successful.
        """
        try:
            self.audio = Audio(name="Device Input")
            print("Reading from default audio device...")

            self.audio.open_device()

            # Calculate max blocks if duration is set
            max_blocks = None
            if duration > 0:
                # Calculate blocks: (duration * fs) / blocksize
                # Since fs might trigger update of blocksize, ensure we use current values
                max_blocks = int(duration * self.audio.fs / self.audio.blocksize) + 1

            num_blocks = 0
            while self.audio.read_block():
                num_blocks += 1
                print(f".", flush=True, end="")

                # Analyze each audio segment
                for segment in self.audio:
                    print(f"Analyzing segment: {segment}", flush=True)
                    analysis = self.analyze(segment)
                    self.analysis_list.append(analysis)
                    print(analysis.analysis.summary(), flush=True)

                if max_blocks and num_blocks >= max_blocks:
                    print(f"Reached maximum duration of {duration}s")
                    break

            self.audio.close()

            if num_blocks > 0:
                # Get the audio statistics object and append it to the list
                audio_stats = AudioStats(self.audio)
                self.analysis_list.append(audio_stats)
                return True

        except KeyboardInterrupt:
            print("\nStopping recording...")
            self.audio.close()
            if num_blocks > 0:
                audio_stats = AudioStats(self.audio)
                self.analysis_list.append(audio_stats)
                self.print()
            return True

        except Exception as e:
            print(f"Error reading device: {e}")
            return False

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
        tone_sort_attr : str | None
            Name of the tone statistic attribute to sort by (default is 'thd').
        time_sort_attr : str | None
            Name of the time statistic attribute to sort by (default is 'start').
        components : bool
            Whether to print component details for each audio segment.
        noise_floor : bool
            Whether to print noise floor details for each audio segment.
        """
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

        # Print noise frequency response statistics table
        noise_freq_resp_list = self.analysis_list.filtered(__class__=NoiseAnalysis)
        if True:  # noise_freq_resp_list:
            print(noise_freq_resp_list.summary("freq_resp.lower_edge"))
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
