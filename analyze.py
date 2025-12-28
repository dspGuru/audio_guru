"""Audio Analyzer."""

from analyzer_args import get_args
from audio_analyzer import AudioAnalyzer


def main():
    # Get command-line arguments
    args = get_args()

    # Create analyzer
    analyzer = AudioAnalyzer(max_segment_secs=args.max_length)

    # Read from device if specified
    if args.device:
        if not analyzer.read_device():
            print("Failed to read from device")
            return
    else:
        if not analyzer.read(args.pattern):
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
