"""Audio Analyzer."""

import glob

from analyzer_args import get_args
from audio_analyzer import AudioAnalyzer


def main():
    # Get command-line arguments
    args = get_args()

    # Create analyzer and read files
    analyzer = AudioAnalyzer()

    num_read = 0
    fnames: list[str] = glob.glob(args.pattern)
    for fname in fnames:
        if not analyzer.read(fname):
            print(f"Failed to read {fname}")
            continue
        num_read += 1

    if num_read == 0:
        print("No files found.")
        return

    # Print the analysis results
    analyzer.print(
        tone_sort_attr=args.table,
        components=args.components,
        noise_floor=args.noise_floor,
    )


if __name__ == "__main__":
    main()
