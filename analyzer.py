"""Analyzes audio tone samples
"""

import sample
from stats import *

class Analyzer:
    """Audio Analyzer"""

    def __init__(self):
        self.sample = sample.Sample()
        self.tone = sample.Tone()
        self.tone2 = sample.Tone()
        self.stats = None

    def load(self, fname: str) -> int:
        """Load audio sample file."""
        return self.sample.load(fname)

    def freq_in_list(self, f: float, freqs: list[float]) -> bool:
        """Return True if the specified frequency is in the list of frequencies."""
        for freq in freqs:
            if(abs(f - freq) < self.sample.min_bins):
                return True
        return False

    def analyze(self) -> Stats:
        """Analyze audio."""
        self.sample.get_bins()
        components = self.sample.get_components()
        
        # The fundamental signal frequency is the bin with the largest power,
        # which is at index 0
        self.tone = components[0]

        # Compare the second-largest power with the first to determine if
        # this is a two-tone signal
        if((components[1].pwr < 0.001*components[0].pwr)      # 0.001 = 0.1% = -30 dB
           or (abs(components[1].freq - 2.0*components[0].pwr) < self.sample.min_bins)):
            # second-largest is a harmonic: single tone signal
            start = 1
            dist = [n * self.tone.freq for n in range(20)]
            self.tone2 = sample.Tone(0.0, 0.0)
        else:
            # second-largest is a not a harmonic: two-tone signal
            self.tone2 = components[1]
            start = 2

            # swap tones if tone2 has a lower frequency
            if self.tone2.freq < self.tone.freq:
                (self.tone, self.tone2) = (self.tone2, self.tone)

            # Calculate distortion frequencies for two-tone signal
            (f1, f2) = (self.tone.freq, self.tone2.freq)
            dist = []
            for n in range(3):
                for n2 in range(3):
                    dist.append(n*f1 + n2*f2)
                    dist.append(n*f1 - n2*f2)
                    dist.append(-n*f1 + n2*f2)  

        # Extract the power of harmonics and noise
        sfdr_pwr = 0.0
        dist_pwr = 0.0
        noise_pwr = self.sample.noise
        for comp in components[start:]:
            sfdr_pwr = max(sfdr_pwr, comp.pwr)
            if self.freq_in_list(comp.freq, dist):
                dist_pwr += comp.pwr
            else:
                noise_pwr += comp.pwr

        self.stats = Stats(self.sample.name, self.sample.audio, self.sample.fs, self.tone.freq, self.tone2.freq)

        # Set frequency statistics from powers
        self.stats.set(self.sample.dc, self.tone.pwr, noise_pwr, sfdr_pwr, dist_pwr)
        return self.stats

    def print(self, args) -> None:
        """Print the specified audio statistics."""
        if self.sample.name:
            print( f'{self.sample.name}')

        if args.components:
            self.sample.tones.print(self.tone.pwr)

        if args.stats:
            print('Stats')
            print('-----------------------')            
            print(self.stats)
