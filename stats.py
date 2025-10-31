"""Audio statistics classes
"""

import math
from operator import attrgetter
import numpy as np
import pandas as pd
from db import *
from pathlib import Path

class Stats:
    """Audio Statistics."""

    def __init__(self, name:str, audio:np.typing.ArrayLike, fs:float, freq:float, freq2:float):
        # File name
        self.name = name

        # Normalize audio input to a numpy array so sizing and min/max are well-defined
        self.audio = np.asarray(audio)

        # Time stats
        self.seconds = int(self.audio.shape[0]) / fs
        self.min = float(np.min(self.audio)) if self.seconds > 0 else 0.0
        self.max = float(np.max(self.audio)) if self.seconds > 0 else 0.0
        self.fs = fs
        self.freq = freq
        self.freq2 = freq2

        # Frequency stats
        self.dc = min_db
        self.snr = min_db
        self.sfdr = min_db
        self.sinad = min_db
        self.thd = min_db
        self.thd_pct = 100.0

    def __str__(self):
        """Return a multi-line string representation of the stats (time + frequency)."""
        lines = []

        # Time domain
        lines.append(f'Length = {self.seconds:0.3f} s')
        lines.append(f'Min    = {self.min:6.3f}')
        lines.append(f'Max    = {self.max:6.3f}')
        lines.append(f'Fs     = {self.fs} Hz')
        lines.append(f'Freq   = {self.freq:0.1f} Hz')
        if self.freq2 > 0.0:
            lines.append(f'Freq2  = {self.freq2:0.1f} Hz')

        # Frequency domain
        lines.append(f'SFDR  = {self.sfdr:3.1f} dB')
        lines.append(f'SNR   = {self.snr:3.1f} dB')
        lines.append(f'SINAD = {self.sinad:3.1f} dB')
        lines.append(f'DC    = {self.dc:3.1f} dB')
        lines.append(f'THD   = {self.thd:3.1f} dB')
        lines.append(f'THD%  = {self.thd_pct:.4f}%')

        return '\n'.join(lines)

    def set(self, dc: float, sig_pwr: float, noise_pwr: float, sfdr_pwr: float, dist_pwr: float):
        """Set statistics"""
        self.dc = db(dc/sig_pwr)
        self.snr = db(noise_pwr/sig_pwr)
        self.sfdr = db(sfdr_pwr/sig_pwr)
        self.sinad = db((noise_pwr + dist_pwr)/sig_pwr)
        self.thd = db(dist_pwr/sig_pwr)
        self.thd_pct = round(math.sqrt(dist_pwr/sig_pwr) * 100.0, 3)

    @staticmethod
    def print_header():
        """Print statistics header"""
        print('Name                                      THD%    THD    SNR   SFDR     F1     F2')
        print('---------------------------------------------------------------------------------')

    def print_summary(self, freq_places: int=1):
        """Print statistics summary"""
        print(f'{self.name:40} {self.thd_pct:.3f} {self.thd:6.1f} {self.snr:6.1f} {self.sfdr:6.1f}', end='')
        str = '  %5i' % round(self.freq)
        if self.freq2 > 0.0:
            str += '  ' + ('%5i' % round(self.freq2))
        print(str)

    def stats(self):
        return {
                'SFDR ':self.sfdr,
                'SNR':self.snr,
                'SINAD':self.sinad,
                'DC':self.dc,
                'THD':self.thd,   
                'THD%':self.thd_pct
                }


class StatsList(list[Stats]):
    """Statistics list"""

    def print(self, attr: str='thd'):
        Stats.print_header()
        self.sort(key=attrgetter(attr))
        for stat in self:
            stat.print_summary()        

    def get_dataframe(self):
        """Return a pandas DataFrame representing the stats list."""
        rows = []
        for s in self:
            r = s.stats().copy()
            r.update({'Name':s.name, 'F1':s.freq, 'F2':s.freq2, 'Secs':s.seconds, 'Fs':s.fs})
            rows.append(r)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Place common columns first
        cols = ['Name', 'Secs', 'Fs', 'F1', 'F2'] + [c for c in df.columns if c not in ('Name', 'Secs', 'Fs', 'F1', 'F2')]
        return df[cols]
    
    def to_csv(self, filepath: str, index: bool = False):
        """Write the object's dataframe to the specified CSV file"""
        df = self.get_dataframe()
        p = Path(filepath)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=index)