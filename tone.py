"""Audio tone classes
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from segment import Segment
from segment import Category
from db import db

@dataclass
class Tone:
    """Audio tone, represented as a frequency, power, and category."""

    # data members
    freq: float = 0.0
    pwr: float = 0.0
    cat: Category = Category.Unknown

    def generate(self, secs: float, fs:float):
        """Generate audio."""

        # Generate constant-frequency tones using signal.chirp (f0 == f1 => pure tone)
        t = np.arange(secs * fs) / fs
        audio = self.pwr * signal.chirp(t, f0=self.freq, t1=t[-1], f1=self.freq)
        return audio


class Tones(list[Tone]):
    """List of audio tones."""

    def __init__(self, segment: Segment):
        super().__init__()
        self.segment = segment

    def print(self, ref_pwr: float=1.0) -> None:
        """Print components in dB relative to reference power."""
        print(f'Tones ({self.segment})')
        print('----------------------------------')        
        for tone in self:
            freq = round(tone.freq)
            decibels = db(tone.pwr/ref_pwr)
            print(f'  {freq:5} Hz = {decibels:6.1f} dB ', tone.cat.name)

