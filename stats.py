"""Audio statistics class"""

import math, os
import numpy as np
from util import *

class Stats:
    """Audio Statistics."""

    def __init__(self, sample, freq=0.0, freq2=0.0):
        # Set sample and time stats
        self.set_sample(sample)

        # Set frequency stats
        self.freq = freq
        self.freq2 = freq2

        # Set power stats. All are in dB except for THD%
        self.sig = min_db
        self.dc = min_db
        self.snr = min_db
        self.sfdr = min_db
        self.sinad = min_db
        self.thd = min_db
        self.thd_pct = 100.0
    
    def __str__(self):
        """Return a multi-line string representation of the stats."""
        lines = []

        # Time stats
        lines.append(f'Secs  = {self.secs:0.3f} s')
        lines.append(f'Min   = {self.min:0.3f}')
        lines.append(f'Max   = {self.max:0.3f}')
        lines.append(f'RMS   = {self.rms:0.3f} = {db(self.rms**2.0):0.1f} dB')

        # Frequency stats
        lines.append(f'Fs    = {self.fs} Hz')
        lines.append(f'Freq  = {self.freq:0.1f} Hz')
        if self.freq2 > 0.0:
            lines.append(f'Freq2 = {self.freq2:0.1f} Hz')

        # Power stats
        lines.append(f'Sig   = {self.sig:0.1f} dB')        
        lines.append(f'SFDR  = {self.sfdr:3.1f} dB')
        lines.append(f'SNR   = {self.snr:3.1f} dB')
        lines.append(f'SINAD = {self.sinad:3.1f} dB')
        lines.append(f'DC    = {self.dc:3.1f} dB')
        lines.append(f'THD   = {self.thd:3.1f} dB')
        lines.append(f'THD%  = {self.thd_pct:.3f}%')

        return '\n'.join(lines)

    def set_sample(self, sample):
        """Set sample data and statistics"""
        self.name = sample.name
        self.fs   = sample.fs
        self.secs = len(sample) / sample.fs
        self.min = float(np.min(sample.audio)) if self.secs > 0.0 else 0.0
        self.max = float(np.max(sample.audio)) if self.secs > 0.0 else 0.0      
        self.rms = sample.get_rms()

    def set_pwr(self, dc: float, sig_pwr: float, noise_pwr: float, sfdr_pwr: float, dist_pwr: float):
        """Set power statistics"""
        self.sig = db(sig_pwr)
        self.dc = db(dc/sig_pwr)
        self.snr = db(noise_pwr/sig_pwr)
        self.sfdr = db(sfdr_pwr/sig_pwr)
        self.sinad = db((noise_pwr + dist_pwr)/sig_pwr)
        self.thd = db(dist_pwr/sig_pwr)
        self.thd_pct = round(math.sqrt(dist_pwr/sig_pwr) * 100.0, 3)

    @staticmethod
    def print_header():
        """Print statistics header"""
        print('Name                                      THD%    THD    SNR   SFDR    F1     F2')
        print('--------------------------------------------------------------------------------')

    def print_summary(self, freq_places: int=1):
        """Print statistics summary"""
        fname = os.path.split(self.name)[1]
        print(f'{fname:40} {self.thd_pct:.3f} {self.thd:6.1f} {self.snr:6.1f} {self.sfdr:6.1f}', end='')
        str = '  %5i' % round(self.freq)
        if self.freq2 > 0.0:
            str += '  ' + ('%5i' % round(self.freq2))
        print(str)

    def desc(self) -> dict[str, str]:
        """Return a dictionary of manufacturer, model, and description derived
        from the sample's file name."""
        (mfr, model, desc) = split_fname(self.name)
        return { 'Mfr':mfr, 'Model':model, 'Desc':desc }

    def stats(self) -> dict[str, float]:
        """Return a dictionary of statistics."""
        return {
                'F1':self.freq,
                'F2':self.freq2,
                'RMS':self.rms,
                'Secs':self.secs,
                'SFDR ':self.sfdr,
                'SNR':self.snr,
                'SINAD':self.sinad,
                'DC':self.dc,
                'THD':self.thd,   
                'THD%':self.thd_pct
                }
