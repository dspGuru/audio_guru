#!/usr/bin/env python3

"""Audio Player CLI."""

import argparse
from audio_player import AudioPlayer


def get_args() -> argparse.Namespace:
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pattern", metavar="PATTERN", help="file name pattern", nargs="?", default=""
    )
    parser.add_argument(
        "-l",
        "--max-length",
        metavar="LENGTH",
        type=float,
        default=15.0,
        help="maximum play time per file in seconds",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="quiet output",
        default=False,
    )

    return parser.parse_args()


def main():
    """Play audio files."""
    args = get_args()
    player = AudioPlayer()

    # Print the file pattern if not quiet
    if not args.quiet:
        print(f"Playing {args.pattern}")

    # If no file pattern is specified, print an error message and return
    if not args.pattern:
        print("Please specify a file pattern.")
        return

    # Play the audio files
    player.play(args.pattern, duration=args.max_length, quiet=args.quiet)


if __name__ == "__main__":
    main()
