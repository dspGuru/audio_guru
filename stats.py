"""Audio statistics class"""

from copy import copy
from dataclasses import dataclass
import math

from audio import Audio
from metadata import Metadata
from segment import Segment
from db import *

@dataclass
class Stats:
    """Audio Statistics"""

    # Sample metadata
    md: Metadata            # samplee metadata
    segment: Segment        # audio segment

    # Audio  stats
    secs: float = 0.0       # signal duration, in seconds
    min: float = 0.0        # minimum signal value
    max: float = 0.0        # maximum signal value
    dbfs: float = 0.0       # signal dB below full scale
    rms: float = 0.0        # signal RMS value

    # Frequency stats
    freq: float = 0.0       # first tone frequency, in Hz
    freq2: float = 0.0      # second tone frequency, in Hz

    # Power stats. All are in dB except for THD%.
    sig: float = MIN_DB     # signal (first tone) power
    dc: float = MIN_DB      # DC (zero Hz) power relative to signal
    snr: float = MIN_DB     # signal-to-noise ratio
    sfdr: float = MIN_DB    # spurious-free dynamic range
    sinad = MIN_DB          # signal to noise-plus-distortion ratio
    thd: float = MIN_DB     # total harmonic distortion (THD)
    thd_pct: float = 100.0  # THD percent

    def __init__(self, audio: Audio, segment: Segment, freq:float =0.0, freq2:float =0.0):
        # Set sample and time stats
        self.segment = segment        
        self.set_audio_stats(audio)

        # Set frequency stats
        self.freq = freq
        self.freq2 = freq2
    

    def __str__(self):
        """Return a multi-line string representation of the stats."""
        lines = []

        lines.append(f'Stats ({self.segment})')
        lines.append('-----------------------')
        
        # Frequency stats
        lines.append(f'Freq  = {self.freq:0.1f} Hz')
        if self.freq2 > 0.0:
            lines.append(f'Freq2 = {self.freq2:0.1f} Hz')        

        # Time stats
        lines.append(f'Fs    = {self.fs} Hz')        
        lines.append(f'Secs  = {self.secs:0.3f} s')
        lines.append(f'Max   = {self.max:0.3f} = {dbv(self.max):0.1f} dB')
        lines.append(f'Min   = {self.min:0.3f} = {dbv(self.min):0.1f} dB')
        lines.append(f'RMS   = {self.rms:0.3f} = {dbv(self.rms):0.1f} dB')
        cf = self.max / self.rms        # crest factor
        lines.append(f'Crest = {cf:0.3f} = {dbv(cf):0.1f} dB')

        # Power stats
        lines.append(f'Sig   = {db_str(self.sig, ' dB')}')
        lines.append(f'DC    = {db_str(self.dc, ' dB')}')        
        lines.append(f'SNR   = {db_str(self.snr, ' dB')}')
        lines.append(f'SFDR  = {db_str(self.sfdr, ' dB')}')
        if self.sinad > 0.0:
            lines.append(f'SINAD = {db_str(self.sinad)}')
            lines.append(f'THD   = {db_str(self.thd)}')
            lines.append(f'THD%  = {self.thd_pct:.3f}%')

        return '\n'.join(lines)


    def set_freq_stats(self, freq: float, freq2: float) -> None:
        self.freq = freq
        self.freq2 = freq2


    def set_audio_stats(self, audio: Audio) -> None:
        """Set sample data and statistics"""
        self.md = copy(audio.md)
        self.fs = audio.fs

        self.min = audio.min
        self.max = audio.max
        self.dbfs = dbv(self.max)
        self.rms = audio.rms
        self.start = self.segment.start_secs
        self.secs = self.segment.secs


    def set_pwr_stats(self, dc: float, sig: float, noise: float, spur: float,
                       dist: float):
        """Set power statistics"""
        self.sig = sig
        self.dc = dc/sig
        self.snr = noise/sig

        # Calculate SFDR
        spur = max(spur, dist)
        if spur > MIN_PWR:
            self.sfdr = sig/spur
        else:
            self.sfdr = 0.0

        # Calculate distortion-related values
        if sig > 0.0 and (noise + dist) > MIN_PWR:
            self.sinad = sig / (noise + dist)
            self.thd = dist/sig
            self.thd_pct = round(math.sqrt(dist/sig) * 100.0, 3)
        else:
            self.sinad = 0.0
            self.thd = 0.0
            self.thd_pct = 0.0


    @staticmethod
    def print_header() -> None:
        """Print statistics header"""
        print('Name                                   THD%    THD  SINAD   SFDR     F1     F2')
        print('------------------------------------------------------------------------------')


    def print_summary(self) -> None:
        """Print statistics summary."""
        s = f'{self.md.mfr:8} {self.md.model:10} {self.md.desc:16}'
        if self.thd <= 0.0:
            s = s + '                    '
        else:
            s = s + f' {self.thd_pct:6.3f} {db_str(self.thd):6} {db_str(self.sinad):6}'
        s = s + f' {db_str(self.sfdr):6}  {round(self.freq):5.0f}'
        if self.freq2 > 0.0:
            s = s + f'  {round(self.freq2):5.0f}'
        print(s)


    def get_stats(self) -> dict[str, float]:
        """Return a dictionary of statistics."""
        return {
                'F1':self.freq,
                'F2':self.freq2,
                'RMS':self.rms,
                'Crest': dbv(self.max / self.rms),     # crest factor
                'Secs':self.secs,
                'SFDR':self.sfdr,
                'SNR':self.snr,
                'SINAD':self.sinad,
                'DC':self.dc,
                'THD':self.thd,   
                'THD%':self.thd_pct
                }
