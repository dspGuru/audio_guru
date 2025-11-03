"""Audio tone classes
"""

from util import *

class Tone:
    """Audio tone, represented as a frequency and power."""

    def __init__(self, freq: float=0.0, pwr: float=0.0):
        self.__freq: float = float(freq)
        self.__pwr: float = float(pwr)

    @property
    def freq(self) -> float:
        return self.__freq

    @property
    def pwr(self) -> float:
        return self.__pwr

class Tones(list[Tone]):
    """List of audio tones."""

    def print(self, ref_pwr: float=1.0) -> None:
        """Print components in dB, relative to signal power."""
        print('Tones')
        print('-----------------------')        
        for tone in self:
            freq = round(tone.freq)
            decibels = db(tone.pwr/ref_pwr)
            print(f'  {freq:5} Hz = {decibels:6.1f} dB')

