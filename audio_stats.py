"""Calculates statistics for audio tone samples
"""

import argparse

import pandas as pd

from audio_test import AudioTest
from stats_list import StatsList

def get_args():
    """Get command-line arguments."""

    # Create and configure argument parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pattern", metavar='PATTERN',
        help="file name pattern", nargs='?', default=''
    )
    parser.add_argument(
        '-c', '--components', action='store_true', default=False,
        help="frequency components"
    ) 
    parser.add_argument(
        '-csv', '--write-csv', metavar='CSV',
        help='write statistics to a CSV file', default=None
    )
    parser.add_argument(
        '-g', '--generate', action='store_true',
        help="generate test audio", default=None
    )
    parser.add_argument(
        '-l', '--max-length', metavar='LENGTH', type=float, default=0.0,
        help="maximum test audio length in seconds"
    )
    parser.add_argument(
        '-nf', '--noise-floor', action='store_true',
        help="noise floor", default=True
    )
    parser.add_argument(
        '-s', '--stats', action='store_true', default=False,
        help="statistics"
    )
    parser.add_argument(
        '-w', '--write-audio', metavar='AUDIO',
        help="write analyzed (possibly truncated/processed) audio to file", default=None
    )

    # Add mutually exclusive sorted table options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--table-by-thd', action='store_const', const='thd', dest='table',
                       help='statistics table sorted by THD')    
    group.add_argument('-tf', '--table-by-freq', action='store_const', const='freq', dest='table',
                       help='statistics table sorted by frequency')
    group.add_argument('-tf2', '-table-by-freq2', action='store_const', const='freq2', dest='table',
                       help='statistics table sorted by second frequency')
    group.add_argument('-tn', '--table-by-name', action='store_const', const='name', dest='table',
                       help='statistics table sorted by name')
    group.add_argument('-ts', '--table-by-snr', action='store_const', const='snr', dest='table',
                       help='statistics table sorted by SNR')
    group.add_argument('-tt', '--table-by-time', action='store_const', const='start', dest='table',
                       help='statistics table sorted by start time')
    
    parser.set_defaults(order='thd')

    # Return parsed arguments
    return parser.parse_args()


def main():
    # Get command-line arguments
    args = get_args()

    # Create empty audio statistics list
    stats = StatsList()

    # Generate if specified
    if args.generate:
        print("Generating audio")
        sample = AudioTest()
        for thd in (0.0, 0.01, 0.0001, 1e-5):
            sample.generate(1000.0, thd, 10.0)
            stat = sample.get_stats(None, args.max_length)
            stats.append(stat)
    
    # Load and analyze input files if specified 
    if args.pattern:
        stats.load(args)
    elif not args.generate:
        print('No files found.')
        
    # Return if no audio data is present
    if not stats:
        print('No audio data present.')
        return

    # Print stats table if specified
    if args.table:
        stats.print(args.table)
        print()

    # Write the stats dataframe if specified
    if args.write_csv:
        # Get the stats data frame
        df = stats.get_dataframe()

        # Set the global data frame output format and describe the data frame
        pd.options.display.float_format = '{:.1f}'.format
        print(df.describe())

        if df.empty:
            print(f"No statistics to write to '{args.write_csv}'")
        else:
            try:
                df.to_csv(args.write_csv, index=False, float_format='%.1f')
                print(f"Wrote statistics to '{args.write_csv}'")
            except Exception as e:
                print(f"Failed to write CSV '{args.write_csv}': {e}")


if __name__ == "__main__":
    main()
