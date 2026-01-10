#!/usr/bin/env python3

"""Audio Analyzer."""

import argparse

from audio_analyzer import AudioAnalyzer
from audio_player import AudioPlayer


def get_args() -> argparse.Namespace:
    """Get command-line arguments."""

    # Create and configure argument parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pattern", metavar="PATTERN", help="file name pattern", nargs="?", default=""
    )
    parser.add_argument(
        "-a",
        "--audio-stats",
        action="store_true",
        help="audio statistics",
        default=True,
    )
    parser.add_argument(
        "-c",
        "--components",
        nargs="?",
        const=10,
        default=None,
        type=int,
        help="frequency components (optional count, default 10)",
    )
    parser.add_argument(
        "-csv",
        "--write-csv",
        metavar="CSV",
        help="write statistics to a CSV file",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--device",
        action="store_true",
        help="input audio sample data from the system's default sound device",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--max-length",
        metavar="LENGTH",
        type=float,
        default=60.0,
        help="maximum test audio length in seconds",
    )
    parser.add_argument(
        "-p",
        "--play",
        metavar="PATTERN",
        help="play audio files matching the pattern",
        default=None,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="quiet output",
        default=False,
    )
    parser.add_argument(
        "-s", "--specs", action="store_true", default=False, help="unit specifications"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose output",
        default=False,
    )

    # Add mutually exclusive sorted table options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t",
        "--table-by-thd",
        action="store_const",
        const="thd",
        dest="table",
        help="statistics table sorted by THD",
    )
    group.add_argument(
        "-tf",
        "--table-by-freq",
        action="store_const",
        const="freq",
        dest="table",
        help="statistics table sorted by frequency",
    )
    group.add_argument(
        "-tf2",
        "-table-by-freq2",
        action="store_const",
        const="freq2",
        dest="table",
        help="statistics table sorted by second frequency",
    )
    group.add_argument(
        "-tn",
        "--table-by-name",
        action="store_const",
        const="name",
        dest="table",
        help="statistics table sorted by name",
    )
    group.add_argument(
        "-ts",
        "--table-by-snr",
        action="store_const",
        const="snr",
        dest="table",
        help="statistics table sorted by SNR",
    )
    group.add_argument(
        "-tt",
        "--table-by-time",
        action="store_const",
        const="start",
        dest="table",
        help="statistics table sorted by start time",
    )

    parser.set_defaults(table="thd")

    # Return parsed arguments
    return parser.parse_args()


def main():
    # Get command-line arguments
    args = get_args()

    # Create analyzer
    analyzer = AudioAnalyzer(max_segment_secs=args.max_length, verbose=args.verbose)

    # Play audio if specified
    if args.play:
        player = AudioPlayer(analyzer.get_msg_queue())
        player.play(args.play, duration=args.max_length, quiet=args.quiet)

    # Read from device if specified
    if args.device:
        if not analyzer.read_device(quiet=args.quiet):
            print("Failed to read from device")
            return
    else:
        if not analyzer.read(args.pattern, quiet=args.quiet):
            print("No files found.")
            return

    # Print the analysis results if not quiet
    if not args.quiet:
        analyzer.print(
            tone_sort_attr=args.table,
            components=args.components,
            audio_stats=args.audio_stats,
        )

    # Write the analysis results to a CSV file if specified
    if args.write_csv:
        analyzer.write_csv(args.write_csv)


if __name__ == "__main__":
    main()
