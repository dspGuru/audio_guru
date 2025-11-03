"""Calculates statistics for audio tone samples
"""

import argparse
from util import *
from stats_list import *

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
        '-g', '--generate', action='store_true',
        help="generate test audio", default=None
    )
    parser.add_argument(
        '-s', '--stats', action='store_true', default=False,
        help="statistics"
    )

    # Add an argument to write the statistics to a CSV file
    parser.add_argument(
        '-w', '--write-csv', metavar='CSV',
        help='write statistics to a CSV file', default=None
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
    group.add_argument('-td', '--table-by-sfdr', action='store_const', const='sfdr', dest='table',
                       help='statistics table sorted by SFDR')
    
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
        stats.generate((0.01, 0.0001, 1e-6))
    
    # Load and analyze input files if specified 
    if args.pattern:
        stats.load(args)
    elif not args.generate:
        print('No files found.')

    # Return if no audio data is present
    if not stats:
        print('No audio data present.')
        return

    # Get the corresponding data frame
    df = stats.get_dataframe()

    # Print stats table if specified
    if(args.table):
        stats.print(args.table)
        print()

    # Set the global data frame output format and describe the data frame
    pd.options.display.float_format = '{:.1f}'.format
    print(df.describe())

    # Write the dataframe if specified
    if args.write_csv:
        try:
            if df.empty:
                print(f"No statistics to write to '{args.write_csv}'")
            else:
                df.to_csv(args.write_csv, index=False, float_format='%.1f')
                print(f"Wrote statistics to '{args.write_csv}'")
        except Exception as e:
            print(f"Failed to write CSV '{args.write_csv}': {e}")


if __name__ == "__main__":
    main()
