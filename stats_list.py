"""Audio statistics list class"""

import glob
from operator import attrgetter
from pathlib import Path

import pandas as pd

from audio_test import AudioTest
from segment import Segment
from stats import Stats


class StatsList(list[Stats]):
    """List of Audio Statistics"""

    def generate(self, thds: tuple[float]) -> None:
        """Generate samples with the specified THD values and append their
        statistics to the list."""
        samp = AudioTest()        
        for thd in thds:
            samp.generate(1000.0, thd)
            self.append(samp.get_stats())


    def load(self, args) -> None:
        """Load the specified samples and append their statistics to the list."""
        samp = AudioTest()
        fnames: list[str] = glob.glob(args.pattern)
        for fname in fnames:
            if not samp.read(fname):
                continue
            segments = samp.get_segments()
            for segment in segments:
                self.append(samp.get_stats(segment, args.max_length))
                if args.stats:
                    print(str(self[-1]))
                    print()
            samp.print(args)


    def print(self, sort_stat_attr: str='thd') -> None:
        """Sort the list on the specified statistic attribute and print it."""
        Stats.print_header()
        self.sort(key=attrgetter(sort_stat_attr))
        for stat in self:
            stat.print_summary()        


    def get_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame representing the list."""
        rows = []
        for s in self:
            rows.append(s.md.__dict__ | s.get_stats())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Place common columns first
        cols = ['Secs', 'F1', 'F2'] + [c for c in df.columns if c not in ('Secs', 'F1', 'F2')]
        return df[cols]
    

    def to_csv(self, filepath: str, index: bool = False) -> None:
        """Write the object's DataFrame to the specified CSV file."""
        df = self.get_dataframe()
        p = Path(filepath)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=index)
    