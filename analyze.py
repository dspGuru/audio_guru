"""Audio Analyzer."""

from analyzer_args import get_args
from audio_analyzer import AudioAnalyzer


def main():
    # Get command-line arguments
    args = get_args()

    # Create analyzer and read files
    analyzer = AudioAnalyzer()
    if not analyzer.read(args.pattern):
        print("No files found.")
        return

    analyzer.print(
        sort_stat_attr=args.table,
        components=args.components,
        noise_floor=args.noise_floor,
    )


if __name__ == "__main__":
    main()
