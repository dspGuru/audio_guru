"""Splits audio into segments."""

import argparse
import glob

from audio import Audio


def get_args():
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
        "-nf", "--noise-floor", action="store_true", help="noise floor", default=True
    )
    parser.add_argument(
        "-s", "--stats", action="store_true", default=False, help="statistics"
    )
    parser.add_argument(
        "-w", "--write", action="store_true", default=False, help="write output files"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the noise analyzer."""

    args = get_args()
    # Get file pattern from command-line arguments
    pattern = args.pattern
    fnames = glob.glob(pattern)
    if not fnames:
        print(f"No files found matching pattern '{pattern}'")
        print("Usage: python split.py <audio_file_pattern>")
        return

    # Process each audio file
    audio = Audio()
    for fname in fnames:
        if not audio.read(fname):
            print(f"Error: Could not read audio file '{fname}'")
            continue

        base_fname = fname.rsplit(".", 1)[0]

        # Print audio file information
        print(f"Audio File: {fname}, Duration: {audio.segment.secs:0.1f} seconds")
        for i, segment in enumerate(audio):
            print(segment.desc)

            if args.write:
                out_fname = f"{base_fname}_{i+1}.wav"
                audio.write(out_fname)
                print(f"Wrote segment {i+1} to '{out_fname}'")
        print()


if __name__ == "__main__":
    main()
