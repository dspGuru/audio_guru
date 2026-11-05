#!/usr/bin/env python3
"""Profile the analyze.py script to identify performance bottlenecks."""

import cProfile
import pstats
from pathlib import Path
from audio_analyzer import AudioAnalyzer


def profile_analyze():
    """Run analyze with profiling."""
    # Create analyzer for a subset of files
    #
    # This analyzer is configured for quiet profiling runs; it will load a
    # representative subset of sample files for timing analysis.
    analyzer = AudioAnalyzer(verbose=False, quiet=True)

    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        # Read and analyze files
        analyzer.read(str(Path("samples") / "*1k*.wav"), quiet=True)
    finally:
        profiler.disable()

    # Print profiling results: show cumulative/top-time views to identify hotspots
    print("\n" + "=" * 100)
    print("PROFILING RESULTS - Top 50 functions by cumulative time")
    print("=" * 100)

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(50)

    print("\n" + "=" * 100)
    print("TOP 25 functions by total time (excluding subcalls)")
    print("=" * 100)

    stats.sort_stats("time")
    stats.print_stats(25)


if __name__ == "__main__":
    profile_analyze()
