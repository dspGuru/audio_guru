"""Calculates statistics for audio tone samples"""

import argparse


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
        "-d",
        "--device",
        action="store_true",
        help="input audio sample data from the system's default sound device",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--components",
        action="store_true",
        default=False,
        help="frequency components",
    )
    parser.add_argument(
        "-csv",
        "--write-csv",
        metavar="CSV",
        help="write statistics to a CSV file",
        default=None,
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
        "-nf", "--noise-floor", action="store_true", help="noise floor", default=True
    )
    parser.add_argument(
        "-s", "--stats", action="store_true", default=False, help="statistics"
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

    parser.set_defaults(order="thd")

    # Return parsed arguments
    return parser.parse_args()
