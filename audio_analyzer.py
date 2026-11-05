"""Audio analyzer class."""

from dataclasses import dataclass
import glob
import queue
import threading

from analysis import Analysis
from analysis_list import AnalysisList
from audio import Audio
from audio_stats import AudioStats
from component import Components
from constants import SEGMENT_MAX_SECS
from decibels import dbv
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

    @property
    def name(self) -> str:
        """Return the file name (pathname) associated with this analysis."""
        return self.analysis.md.name if hasattr(self.analysis, "md") else ""

    def __getattr__(self, name: str):
        """Delegate attribute lookup to underlying analysis, time_stats, or segment."""
        for obj in (self.analysis, self.time_stats, self.segment):
            if obj is not None and hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(f"'AudioAnalysis' object has no attribute '{name}'")

    def __str__(self) -> str:
        """Return string representation of analysis result."""
        return (
            f"Analysis(segment={self.segment}, result={self.analysis}, "
            f"components={self.components})"
        )

    def to_dict(self) -> dict:
        """Convert the audio analysis result to a dictionary."""
        md_dict = self.analysis.md.to_dict() if hasattr(self.analysis, "md") else {}
        type_dict = (
            {"Type": self.segment.cat.name}
            if self.segment is not None and hasattr(self.segment, "cat")
            else {}
        )
        analysis_dict = (
            dict(self.analysis.to_dict())
            if self.analysis is not None and self.analysis is not self.time_stats
            else {}
        )
        for k in ("minimum", "ripple", "lower_edge", "upper_edge"):
            if k in analysis_dict and isinstance(analysis_dict[k], float):
                analysis_dict[k] = round(analysis_dict[k], 1)

        time_dict = (
            dict(self.time_stats.to_dict()) if self.time_stats is not None else {}
        )
        if "Description" in md_dict:
            time_dict.pop("Desc", None)

        if self.time_stats is not None:
            nf = getattr(self.time_stats, "noise_floor", 0.0)
            time_dict["Noise Floor"] = (
                round(dbv(nf), 1) if isinstance(nf, (int, float)) and nf > 0.0 else None
            )

        return md_dict | type_dict | analysis_dict | time_dict


def _resolve_sort_key(item: object, attr: str) -> tuple:
    """Return a comparable sort key tuple for an Analysis or AudioAnalysis item."""
    val = getattr(item, attr, None)
    if val is None:
        return (1, "")
    return (0, val)


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

    def __init__(
        self,
        max_segment_secs: float = SEGMENT_MAX_SECS,
        quiet: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the analyzer.

        Parameters
        ----------
        max_segment_secs : float
            Maximum segment length in seconds. Default is 60 seconds.
        """
        self.audio = Audio(name="Analyzer Audio", max_segment_secs=max_segment_secs)
        self.analyzers: list[object] = []
        self.analysis_list = AnalysisList()
        self.results: dict[str, AudioAnalysis] = {}
        self.unit_ids: set[str] = set()
        self.quiet = quiet
        self.verbose = verbose
        self._tone_sort_attr: str | None = "thd"
        self._q: queue.Queue = queue.Queue()

    def analyze(self, audio: Audio) -> AudioAnalysis | None:
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
        if not self.quiet:
            print(f"    Analyzing {audio.segment.desc}")

        # Get the analyzer class for the segment's category
        analyzer_class = self.analyzer_classes.get(audio.segment.cat)
        if analyzer_class is None:
            return None

        # Create the analyzer
        analyzer = analyzer_class(audio)
        self.analyzers.append(analyzer)

        # Analyze the segment
        analysis = analyzer.analyze()
        self.analysis_list.append(analysis)

        # Get segment time stats. Use the analysis result if it is a TimeStats
        # object. Otherwise, create a new TimeStats object for the segment.
        if isinstance(analysis, TimeStats):
            time_stats = analysis
        else:
            time_stats = getattr(analyzer, "time_stats", None) or TimeStats(audio)
            self.analysis_list.append(time_stats)

        if self.verbose:
            time_stats.print()

        # Creat an audio analysis object to store the complete analysis results
        components = analyzer.components
        audio_analysis = AudioAnalysis(audio.segment, time_stats, analysis, components)

        # Store the analysis results in the results dictionary
        key = f"{audio.name}:{audio.segment}"
        self.results[key] = audio_analysis

        # Store the units analyzed
        self.unit_ids.add(audio.md.unit_id)

        return audio_analysis

    def get_msg_queue(self) -> queue.Queue:
        """
        Get the message queue for audio segments to analyze.

        Returns
        -------
        queue.Queue
            Queue for audio analysis.
        """
        return self._q

    def _analyze_worker(self, q: queue.Queue) -> None:
        """
        Worker thread to analyze audio segments from the queue.

        Parameters
        ----------
        q : queue.Queue
            Queue containing Audio segments to analyze.
        """
        while True:
            audio = q.get()
            if audio is None:
                q.task_done()
                break

            if not hasattr(audio, "segment"):
                print("Analyzing file: ", audio)
                continue

            try:
                analysis = self.analyze(audio)
                if analysis:
                    self.analysis_list.append(analysis)
            except Exception as e:
                print(f"Error in analysis worker: {e}")
                raise
            finally:
                q.task_done()

    def _propagate_noise_floor(self) -> None:
        """Propagate file-level AudioStats noise floor to segment TimeStats."""
        nf_by_name = {
            s.name: s.noise_floor
            for s in self.analysis_list.filtered(__class__=AudioStats)
            if isinstance(getattr(s, "noise_floor", None), (int, float))
            and s.noise_floor > 0.0
        }
        if not nf_by_name:
            return
        for res in self.results.values():
            if res.time_stats is not None and res.time_stats.name in nf_by_name:
                res.time_stats.noise_floor = nf_by_name[res.time_stats.name]

    def read(self, pattern: str, quiet: bool = False) -> int:
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
        self.quiet = quiet

        # Get the list of files matching the pattern
        fnames = glob.glob(pattern)
        if not fnames:
            return 0

        # Start the analysis worker thread
        worker = threading.Thread(target=self._analyze_worker, args=(self._q,))
        worker.start()

        num_read = 0
        for fname in fnames:
            self.audio = Audio(name=fname)

            # Print the file name if not quiet
            if not quiet:
                print(f"Reading {fname}")

            # Read the audio file in blocks
            num_blocks = 0
            with self.audio.open(fname):
                while self.audio.read_block():
                    num_blocks += 1

                    # Analyze each audio segment
                    for audio in self.audio:
                        self._q.put(audio)

            # If we read any blocks, increment the count. Otherwise, skip
            # analysis below and continue to the next file.
            if num_blocks > 0:
                num_read += 1
            else:
                continue

            # Get the audio statistics object and append it to the list
            audio_stats = AudioStats(self.audio)
            self.analysis_list.append(audio_stats)

        # Signal the worker to stop and wait for it to finish
        self._q.put(None)
        worker.join()
        self._propagate_noise_floor()

        return num_read

    def read_device(self, duration: float = 0.0, quiet: bool = False) -> bool:
        """
        Read and analyze audio from the default device.

        Parameters
        ----------
        duration : float
            Maximum duration to record in seconds. If 0, record indefinitely.

        Returns
        -------
        bool
            True if recording was successful, False otherwise.
        """
        self.quiet = quiet
        try:
            # Create the audio object and print status
            self.audio = Audio(name="Device Input")
            if not quiet:
                print("Reading from default audio device...")

            # Initialize analysis queue and worker thread
            q: queue.Queue = queue.Queue()
            worker = threading.Thread(target=self._analyze_worker, args=(q,))
            worker.start()

            # Determine the number of blocks to record based on duration
            max_blocks = None
            if duration > 0:
                # Calculate blocks: (duration * fs) / blocksize. Since fs might
                # trigger update of blocksize, ensure we use current values.
                max_blocks = int(duration * self.audio.fs / self.audio.blocksize) + 1

            # Read the audio file in blocks
            num_blocks = 0
            with self.audio.open_device(quiet=quiet):
                while self.audio.read_block():
                    num_blocks += 1
                    if not quiet:
                        print(".", flush=True, end="")

                    # Analyze each audio segment
                    for audio in self.audio:
                        if not quiet:
                            print(f"Analyzing {audio.segment}", flush=True)
                        q.put(audio)

                    if max_blocks and num_blocks >= max_blocks:
                        if not quiet:
                            print(f"Reached maximum duration of {duration}s")
                        break

            # Signal worker to stop
            q.put(None)
            worker.join()

            # If we read any blocks, get the audio statistics object, append it
            # to the list and return True
            if num_blocks > 0:
                audio_stats = AudioStats(self.audio)
                self.analysis_list.append(audio_stats)
                self._propagate_noise_floor()
                return True

            # No blocks were read
            return False

        except KeyboardInterrupt:
            print("\nStopping recording...")
            # Signal worker to stop
            q.put(None)
            worker.join()

            # If we read any blocks, get the audio statistics object, append it
            # to the list and return True
            if num_blocks > 0:
                audio_stats = AudioStats(self.audio)
                self.analysis_list.append(audio_stats)
                self._propagate_noise_floor()
                self.print()
                return True

            # No blocks were read
            return False

        except Exception as e:
            print(f"Error reading device: {e}")
            raise

        return bool(num_blocks > 0)

    def print(
        self,
        tone_sort_attr: str | None = "thd",
        time_sort_attr: str | None = "start",
        components: int | None = 10,
        audio_stats: bool = True,
    ) -> None:
        """
        Print the analysis results sorted on the specified statistic attribute.

        Parameters
        ----------
        tone_sort_attr : str | None
            Name of the tone statistic attribute to sort by (default is 'thd').
        time_sort_attr : str | None
            Name of the time statistic attribute to sort by (default is 'start').
        components : int | None
            Maximum number of components to print (default 10), or None to skip.
        audio_stats : bool
            Whether to print audio statistics for each audio source.
        """
        self._tone_sort_attr = tone_sort_attr
        print("\nAudio Analysis Results")
        print("======================\n")

        # Print the audio statistics table if specified
        if audio_stats:
            audio_stats_list = self.analysis_list.filtered(__class__=AudioStats)
            if audio_stats_list:
                print(audio_stats_list.summary("noise_floor"))
                print()

        # Print component details if specified
        if components is not None:
            for _, analysis in self.results.items():
                analysis.components.print(max_components=components)
                print()

        # Print time statistics table if specified
        if time_sort_attr:
            time_stats_list = self.analysis_list.filtered(__class__=TimeStats)
            if time_stats_list:
                print(time_stats_list.summary(time_sort_attr))
                print()

        # Print noise frequency response statistics table
        noise_freq_resp_list = self.analysis_list.filtered(__class__=NoiseAnalysis)
        if noise_freq_resp_list:
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

    def write_csv(
        self,
        filepath: str,
        index: bool = False,
        print_head: bool = False,
        sort_attr: str | None = None,
    ) -> None:
        """
        Write the analysis results to the specified CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        index : bool, optional
            Whether to write row indices to the CSV file (default is False).
        print_head : bool, optional
            Whether to print the CSV head (default is False).
        sort_attr : str | None, optional
            Attribute name to sort rows by (defaults to tone_sort_attr).
        """
        export_list = (
            AnalysisList(self.results.values()) if self.results else self.analysis_list
        )
        attr = sort_attr if sort_attr is not None else self._tone_sort_attr
        if isinstance(attr, str) and attr:
            export_list.sort(key=lambda item: _resolve_sort_key(item, attr))
        export_list.to_csv(filepath, index=index, print_head=print_head)
