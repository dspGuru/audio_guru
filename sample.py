"""Audio sample classes and utilities
"""

import numpy as np
from scipy.fft import rfft
from scipy import signal
import loader
from util import *
from stats import Stats
from tone import Tone, Tones

# Alias for audio sample data, which is a NumPy array of 32-bit floats
Audio = np.ndarray(shape=1, dtype=np.float32)

class Sample:
    """Audio sample"""
    bins_per_tone = 5    # number of frequency bins that constitute a single tone

    def __init__(self, audio:Audio=[], fs: float=44100.0, name: str=''):
        self.audio = audio
        self.fs = fs
        self.name = name
        self.bins = []
        self.tones = Tones()
        self.noise = 0.0

    def __len__(self):
        return(len(self.audio))

    def generate(self, freq: float=1000.0, thd: float=0.001) -> None:
        """Generate audio."""
        print("Generating audio")

        # Generate constant-frequency tones using signal.chirp (f0 == f1 => pure tone)
        t = np.arange(self.fs) / self.fs
        x = signal.chirp(t, f0=freq, t1=t[-1], f1=freq, method='linear')

        # Generate and add distortion tone if THD is above the minimum dB
        x2 = signal.chirp(t, f0=2*freq, t1=t[-1], f1=2*freq, method='linear') * thd
        x += x2
        
        # Store generated audio in self
        self.audio = x
        self.name = f'Generated Audio ({freq:0.1f} at {thd * 100.0:.3f}%)'

    def get_bins(self, sl: slice):
        """Get the audio's frequency bins."""
        # Window the signal
        x = self.audio[sl]
        w = signal.windows.blackmanharris(len(x))
        xw = x*w

        # Compute normalized real FFT
        yf = rfft(xw, norm='forward')
        self.bins = yf*yf.conjugate()
        return self.bins

    def bin_to_freq(self, bin) -> float:
        """Return the frequency of the bin number."""
        return self.fs * 0.5 * bin / len(self.bins)

    def freq_to_bin(self, freq) -> int:
        """Return the bin number of the frequency."""
        return round(freq / (self.fs * 0.5) * len(self.bins))

    def freq_in_list(self, f: float, freqs) -> bool:
        """Return True if the specified frequency is in the list of frequencies."""
        for freq in freqs:
            freq = alias_freq(freq, self.fs)
            if abs(f - freq) < self.bins_per_tone:
                return True
        return False

    def get_tones(self, w_tol=50) -> list:
        """Get the audio components."""
        if(len(self.bins) == 0):
            self.get_bins(slice(0, -1))
        bins = np.absolute(self.bins)

        self.dc = sum(bins[0:self.freq_to_bin(2 * self.bins_per_tone)])
        bins[0:self.freq_to_bin(10)] = 0

        last_bin = len(bins) - 1
        self.tones = Tones()
        while len(self.tones) < 20:
            center = np.where(bins == bins.max())[0][0]
            first = max(center - w_tol, 0)
            last = min(center + w_tol, last_bin)
            pwr = sum(bins[first:last])
            bins[first:last] = 0
            self.tones.append(Tone(self.bin_to_freq(center), pwr))
        self.noise = sum(bins)
        return self.tones

    def get_dbfs(self) -> float:
        """Return the dB relative to full scale of 1.0 (peak dBFS)."""
        a = np.asarray(self.audio, dtype=np.float64)
        if a.size == 0:
            return min_db
        peak = float(np.max(np.abs(a)))
        return 2.0 * db(peak)

    def get_freq(self, sl=slice(0, -1)) -> float:
        """Get the audio's fundamental frequency from its zero-crossings."""
        # Get zero-crossings
        zc = np.where(np.diff(np.sign(self.audio[sl])))[0]

        # Remove the last zero-crossing, if necessary, to span an integer
        # number of cycles
        if len(zc) % 2 == 0:
            zc = zc[:-1]

        # Calculate the signal frequency from the zero-crossings
        nzc = len(zc)
        freq = 0.5 * (nzc - 1) / float( zc[-1] - zc[0] ) * self.fs
        return freq

    def calc_stats(self, sl=slice(0, -1)) -> bool:
        """Calculate audio statistics in self.stats. Return True on success."""

        # Return False if audio length is too small
        if(len(self.audio) < 256):
            return False
        
        self.get_bins(sl)
        tones = self.get_tones().copy()
        
        # The fundamental signal frequency is the bin with the largest power,
        # which is at index 0
        self.tone = tones[0]

        # Compare the second-largest power with the first to determine if
        # this is a two-tone signal
        if ((tones[1].pwr < 0.001*tones[0].pwr)      # 0.001 = 0.1% = -30 dB
            or (abs(tones[1].freq - 2.0*tones[0].pwr) < self.bins_per_tone)):
            # second-largest is a harmonic: single tone signal
            start = 1
            dist_freqs = [alias_freq(n * self.tone.freq, self.fs ) for n in range(10)]
            self.tone2 = Tone(0.0, 0.0)
        else:
            # Second-largest is a not a harmonic: two-tone signal
            self.tone2 = tones[1]
            start = 2

            # Swap tones if tone2 has a lower frequency
            if self.tone2.freq < self.tone.freq:
                (self.tone, self.tone2) = (self.tone2, self.tone)

            # Calculate distortion frequencies for two-tone signal
            (f1, f2) = (self.tone.freq, self.tone2.freq)
            dist_freqs = []
            for n in range(3):
                for n2 in range(3):
                    dist_freqs.append(n*f1 + n2*f2)
                    dist_freqs.append(n*f1 - n2*f2)
                    dist_freqs.append(-n*f1 + n2*f2)

        # Extract the power of distortion and noise
        sfdr_pwr = 0.0
        dist_pwr = 0.0
        noise_pwr = self.noise
        for comp in tones[start:]:
            sfdr_pwr = max(sfdr_pwr, comp.pwr)
            if self.freq_in_list(comp.freq, dist_freqs):
                dist_pwr += comp.pwr
            else:
                noise_pwr += comp.pwr      

        # Set frequency statistics from powers
        self.stats = Stats(self, self.tone.freq, self.tone2.freq)
        self.stats.set_pwr(self.dc, self.tone.pwr, noise_pwr, sfdr_pwr, dist_pwr)        
        return True

    def get_slices(self, ratio: float=0.001, seconds: float=0.1) -> list[slice]:
        """Return a list of index slices where the signal's absolute amplitude is
        below (peak * ratio) for at least the specified time.
        """

        nsamples = self.fs * seconds
        a = np.asarray(self.audio)
        n = a.size
        if n == 0 or nsamples <= 0:
            return []

        peak = float(np.max(np.abs(a)))
        if peak == 0.0:
            # Entire signal is zero -> single segment spanning whole array if nsamples satisfied
            return [slice(0, -1), ] if n >= nsamples else []

        thresh = peak * float(ratio)

        # boolean array where condition is true (below threshold)
        below = np.abs(a) <= thresh

        # find run starts and ends
        # diff will be 1 where False->True (start), -1 where True->False (end)
        d = np.diff(below.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1

        # handle case where run starts at index 0
        if below[0]:
            starts = np.concatenate(([0], starts))
        # handle case where run continues till end
        if below[-1]:
            ends = np.concatenate((ends, [n]))

        # Add slices for each non-zero segment
        slices = []
        start = 0
        for s, e in zip(starts, ends):
            s, e = int(s), int(e)
            if (e - s) >= nsamples:
                if (s - start) >= nsamples:
                    slices.append(slice(start, s))
                start = e

        # if no slices were found add a slice for the entire sample
        if not slices:
            try:
                start = int(ends[0])
            except IndexError:
                start = 0
            slices.append(slice(start, -1))

        print(self.name, slices)
        return slices

    def get_segment(self, sl: slice):
        """Return a sample copied from the specified indices (start inclusive, end exclusive)."""
        n = int(self.audio.size) if hasattr(self.audio, "size") else 0
        start = max(0, int(start_idx))
        end = min(n, int(end_idx))

        # If invalid range or empty source, return an empty Sample with same fs
        if start >= end or n == 0:
            return Sample(audio=np.array([], dtype=np.float32), fs=self.fs, name=f'{self.name} [{start}:{end}]')

        # Return a copy with the specified audio slice
        segment = Sample(audio=self.audio[sl].copy(), fs=self.fs, name=self.name)
        return segment

    def split(self) -> list[slice]:
        """Return a list of audio segment slices copied from the non-silent
        regions of the sample."""
        segments = self.get_slices()
        slices = []
        for start, end in segments:
            sub = self.get_segment(start, end)
            # skip empty sub-samples
            if getattr(sub.audio, "size", 0) > 0:
                slices.append(sub)
        return slices

    def load(self, fname: str) -> bool:
        """Load audio data from file."""
        self.name = fname
        self.bins = []
        self.tones = []
        self.audio, self.fs = loader.load(fname)

        return len(self) > 0

    def print(self, args) -> None:
        """Print the specified audio statistics."""
        if self.name:
            print( f'{self.name}')

        if args.components:
            self.tones.print(self.tone.pwr)

        if args.stats:
            print('Stats')
            print('-----------------------')            
            print(self.stats)

    def stats(self) -> dict:
        """Return a dictionary of statistics"""
        freq = self.get_freq()
        dbfs = self.get_dbfs()
        segments = self.get_slices()
        return {
                'Samples':len(self.audio),
                'Min':float(min(self.audio)),
                'Max':float(max(self.audio)),
                'Fs':self.fs,
                'Freq':freq,
                'dBFS':dbfs,
                'segments':segments
               }

