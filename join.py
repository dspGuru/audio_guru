"""Joins audio files into a single file."""

import argparse
import glob
from pathlib import Path

from audio import Audio
from generate import silence


def get_args():
    """Get command-line arguments."""

    # Create and configure argument parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files", metavar="FILES", help="file name patterns", nargs="+", default=[]
    )
    parser.add_argument(
        "-s",
        "--silence",
        metavar="SECONDS",
        type=float,
        default=0.0,
        help="duration of the delimiting silence (seconds)",
    )
    parser.add_argument(
        "-w",
        "--write",
        metavar="OUTPUT",
        help="write output file to pathname",
        default=None,
    )
    return parser.parse_args()


def join_files(patterns: list[str | Path], silence_secs: float = 0.0) -> Audio:
    """
    Join audio files matching the patterns.

    Parameters
    ----------
    patterns : list[str]
        List of file patterns to match.
    silence_secs : float
        Seconds of silence to insert between files.

    Returns
    -------
    Audio
        The combined audio object.
    """
    if silence_secs < 0:
        raise ValueError("Silence duration must be non-negative")

    combined_audio = Audio()

    # Collect all matching files
    all_files = []
    for pattern in patterns:
        matched = sorted(glob.glob(str(pattern)))
        if not matched and Path(pattern).exists():
            matched = [pattern]
        all_files.extend(matched)

    if not all_files:
        print("No files found.")
        return combined_audio

    print(f"Joining {len(all_files)} files...")

    for i, fname in enumerate(all_files):
        # Read next file
        next_audio = Audio()
        if not next_audio.read(fname):
            print(f"Error: Could not read audio file '{fname}'")
            continue

        print(f"Adding {fname} ({next_audio.segment.secs:.2f}s)")

        # If not the first file and silence is requested, append silence first
        if i > 0 and silence_secs > 0:
            # Generate silence matching the sample rate of the FIRST file (or current combined)
            # if combined is empty (first failed?), use next_audio fs
            fs = combined_audio.fs if len(combined_audio) > 0 else next_audio.fs
            combined_audio.append(silence(secs=silence_secs, fs=fs))

        combined_audio.append(next_audio)

    return combined_audio


def main() -> None:
    """Main function to run the joiner."""

    args = get_args()

    # Process files
    audio = join_files(args.files, args.silence)

    print(f"Total Duration: {audio.segment.secs:.2f} seconds")

    if args.write:
        p = Path(args.write)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        audio.write(args.write)
        print(f"Wrote combined audio to '{args.write}'")
    else:
        print("Use -w <filename> to save the output.")


if __name__ == "__main__":
    main()
