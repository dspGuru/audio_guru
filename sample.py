"""Audio sample classes
"""

import numpy as np
import scipy
from scipy.fft import rfft
from scipy import signal
from db import *
from tone import *

# Alias for audio sample data, which is a NumPy array of 32-bit floats
Audio = np.ndarray(shape=1, dtype=np.float32)

class Sample:
    """Audio sample"""

    min_bins = 5

    def __init__(self, audio:Audio=[], fs: float=44100.0, name: str=''):
        self.audio = audio
        self.fs = fs
        self.name = name
        self.bins = []
        self.tones = Tones()
        self.noise = 0.0

    def generate(self, freq: float=1000.0, thd: float=min_db) -> None:
        """Generate audio."""
        print("Generating audio")

        # Generate constant-frequency tones using signal.chirp (f0 == f1 => pure tone)
        t = np.arange(self.fs) / self.fs
        x = signal.chirp(t, f0=freq, t1=t[-1], f1=freq, method='linear')

        # Generate and add distortion tone if THD is above the minimum dB
        if(thd > min_db):
            x2 = signal.chirp(t, f0=2*freq, t1=t[-1], f1=2*freq, method='linear') * thd
            x += x2
        
        # Store generated audio in self
        self.audio = x
        self.name = f'Generated Audio ({freq:0.1f} at {thd * 100.0:.3f}%)'

    def get_bins(self):
        """Get the audio's frequency bins."""
        # Window the signal
        x = self.audio
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

    def get_components(self, w_tol=50) -> list:
        """Get the audio components."""
        if(len(self.bins) == 0):
            self.get_bins()
        bins = np.absolute(self.bins)

        self.dc = sum(bins[0:self.freq_to_bin(2 * self.min_bins)])
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
        """ Return the dB relative to full scale of 1.0 (peak dBFS)."""
        a = np.asarray(self.audio, dtype=np.float64)
        if a.size == 0:
            return min_db
        peak = float(np.max(np.abs(a)))
        return 2.0 * db(peak)

    def get_freq(self) -> float:
        """Get the audio's fundamental frequency."""
        # Get zero-crossings
        zc = np.where(np.diff(np.sign(self.audio)))[0]

        # Remove the last zero-crossing, if necessary, to span an integer
        # number of cycles
        if len(zc) % 2 == 0:
            zc = zc[:-1]

        # Calculate the signal frequency from the zero-crossings
        nzc = len(zc)
        freq = 0.5 * (nzc - 1) / float( zc[-1] - zc[0] ) * self.fs
        return freq
    
    def get_segments(self, ratio: float=0.001, seconds: float=0.1) -> list[(int, int)]:
        """Return a list of (start, stop) index pairs where the signal's
        absolute amplitude is below (peak * ratio) for at least the specified
        time.

        Returns indices in Python slice form: start inclusive, stop exclusive.
        """

        nsamples = self.fs * seconds
        a = np.asarray(self.audio)
        n = a.size
        if n == 0 or nsamples <= 0:
            return []

        peak = float(np.max(np.abs(a)))
        if peak == 0.0:
            # Entire signal is zero -> single segment spanning whole array if nsamples satisfied
            return [(0, n)] if n >= nsamples else []

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

        # pair starts and ends and filter by length
        segments = []
        for s, e in zip(starts, ends):
            if (e - s) >= nsamples:
                segments.append((int(s), int(e)))

        return segments

    def get_segment(self, start_idx:int, end_idx:int) -> 'Sample':
        """Return a sample copied from the specified indices (start inclusive, end exclusive)."""
        n = int(self.audio.size) if hasattr(self.audio, "size") else 0
        start = max(0, int(start_idx))
        end = min(n, int(end_idx))

        # If invalid range or empty source, return an empty Sample with same fs
        if start >= end or n == 0:
            return Sample(audio=np.array([], dtype=np.float32), fs=self.fs, name=f'{self.name} [{start}:{end}]')

        a = self.audio[start:end].copy()
        # Ensure float32 storage for consistency
        a = np.asarray(a, dtype=np.float32)

        ssub = Sample(audio=a, fs=self.fs, name=f'{self.name} [{start}:{end}]')
        ssub.bins = []
        ssub.tones = Tones()
        ssub.noise = 0.0
        return ssub

    def split(self) -> list:
        """Return a list of sub-samples copied from the non-silent regions of the input sample."""
        segments = self.get_segments()
        subs = []
        for start, end in segments:
            sub = self.get_segment(start, end)
            # skip empty sub-samples
            if getattr(sub.audio, "size", 0) > 0:
                subs.append(sub)
        return subs

    def load(self, fname: str) -> bool:
        """Load data from .wav file."""
        self.name = fname

        # Read WAV file using wave and convert to a mono float32 numpy array in [-1,1]
        # Read WAV file using scipy.io.wavfile.read and convert to a mono float32 numpy array in [-1,1]
        try:
            self.fs, data = scipy.io.wavfile.read(fname)
        except ValueError:
            print('Invalid audio file:', fname)
            self.fs = 1.0
            self.audio = []
            return False

        # Convert integer PCM to float32 in [-1, 1]
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
        elif data.dtype in (np.float32, np.float64):
            audio = data.astype(np.float32)
        else:
            # Fallback: cast to float32 and normalize by max absolute value
            audio = data.astype(np.float32)
            maxv = np.max(np.abs(audio))
            if maxv > 0:
                audio /= maxv

        # If multi-channel, convert to mono by averaging channels
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Save audio in self
        self.audio = audio
        self.bins = []
        self.tones = []
        return True

    def stats(self) -> dict :
        freq = self.get_freq()
        dbfs = self.get_dbfs()
        segments = self.get_segments()
        return {
                'Samples':len(self.audio),
                'Min':float(min(self.audio)),
                'Max':float(max(self.audio)),
                'Fs':self.fs,
                'Freq':freq,
                'dBFS':dbfs,
                'segments':segments
               }

