#!/usr/bin/env python3

"""Splits audio into segments."""

import argparse, glob

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

    return parser.parse_args()


def split_file(pathname: str) -> list[str]:
    """
    Split a single audio file into segments.

    Parameters
    ----------
    pathname : str
        Path to the audio file to split.

    Returns
    -------
    list[str]
        List of generated file names.
    """
    audio = Audio()
    if not audio.read(pathname):
        raise FileNotFoundError(f"Error: Could not read audio file '{pathname}'")

    out_fnames: list[str] = []
    base_fname = pathname.rsplit(".", 1)[0]
    if base_fname == pathname:
        # No extension found, append suffix to original name
        base_fname = pathname

    for i, audio_segment in enumerate(audio):
        print(audio_segment.segment.desc)
        out_fname = f"{base_fname}_{i+1}.wav"
        if base_fname == pathname:
            # Ensure distinct name if original had no extension
            out_fname = f"{pathname}_{i+1}.wav"

        audio_segment.write(out_fname)
        print(f"Wrote segment {i+1} to '{out_fname}'")
        out_fnames.append(out_fname)

    return out_fnames


def main() -> None:
    """Main function to run the split an audio file into segments."""
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

        split_file(fname)
        print()


if __name__ == "__main__":
    main()
