"""Audio test class"""

import numpy as np
from scipy.fft import rfft
from scipy import signal

from audio import Audio
from db import dbv
from segment import Segment
from stats import Stats
from tone import Category
from tone import Tone
from tone import Tones
from util import alias_freq
from util import is_harmonic

BINS_PER_TONE = 5
"""Number of frequency bins which constitute a single tone after BH windowing (empirical)"""

class AudioTest(Audio):
    """
    Audio test
    
    This class extends Audio to add methods for analyzing test audio signals,
    including identifying tones, distortion products, spurs, and noise, and
    calculating audio statistics.
    
    Attributes
    ----------
    fs : int
        Sampling frequency in Hz.
    name : str
        Name of the audio file.
    dc : float
        DC offset of the audio signal.
    bins : np.ndarray
        Frequency bins of the audio signal.
    tones : Tones
        Identified tones in the audio signal.
    noise : float
        Noise power in the audio signal.
    tone : Tone
        Primary tone of the audio signal.
    tone2 : Tone
        Secondary tone of the audio signal (for two-tone signals).
    md : Metadata
        Metadata associated with the audio signal.
    segment : Segment
        Segment of the audio signal selected for analysis.
    """

    """Static Constants"""

    MAX_DIST_ORDER = 3
    """Maximum harmonic distortion order to consider"""

    MAX_HARM_ORDER = 5
    """Maximum harmonic order to consider"""

    MAX_NOISE_FLOOR_DB = -30.0
    """Maximum noise floor in dBFS"""

    MAX_TONES = 20
    """Maximum number of tones to identify"""

    MIN_AUDIO_LEN = 256
    """Minimum audio length to calculate statistics"""

    MIN_SPUR_RATIO = 1e-8
    """Threshold for distinguishing spurs from noise"""

    MIN_TWO_TONE_RATIO = 1e-2
    """Minimum ratio for two-tone detection (-20 dB)"""

    def __init__(self, fs: int=44100, name: str=''):
        super().__init__(fs, name)
        self.reset()           # resets/adds bins, tones, noise, and stats
    

    def reset(self) -> None:
        """Resets or creates members which are calculated from audio"""
        self.dc = 0.0
        self.bins = []
        self.tones = Tones(self.segment)
        self.noise = 0.0


    def read(self, fname: str) -> bool:
        """Read audio data from file. Retun True if successful."""
        self.reset()
        return super().read(fname)


    def get_bins(self, segment: Segment) -> None:
        """Calculate the audio's frequency bins."""

        # Reset all calculated data
        self.reset()

        # Get audio and convert multi-channel audio to mono by averaging
        # channels
        x = self.data[segment.slice].copy()
        if x.ndim > 1:
            x = np.mean(x, axis=1)

        # Calculate DC and remove it
        self.dc = np.mean(x)
        x -= self.dc

        # Window the signal
        w = signal.windows.blackmanharris(len(x))
        xw = x*w

        # Compute normalized real FFT
        yf = rfft(xw, norm='forward')
        self.bins = yf*yf.conjugate()


    def bin_to_freq(self, bin) -> float:
        """Return the frequency of the bin number."""
        return self.fs * 0.5 * bin / len(self.bins)


    def freq_to_bin(self, freq) -> int:
        """Return the bin number of the frequency."""
        return round(freq / (self.fs * 0.5) * len(self.bins))


    def freq_in_list(self, freq: float, freqs:list[float]) -> bool:
        """
        Determine if the specified frequency is in the list of frequencies.

        Parameters
        ----------
        freq : float
            Frequency to test.
        freqs : list[float]
            List of frequencies to search.
        Returns
        -------
        bool
        True if the inputs are valid and the specified frequency is in the
        list of frequencies.
        """

        # Return False if any input is invalid
        if freq <= 0.0 or not freqs:
            return False
        
        # Check if frequency is within tolerance of any frequency in the list
        for freq in freqs:
            freq = alias_freq(freq, self.fs)
            bin_diff = self.freq_to_bin(abs(freq - freq))
            if bin_diff <= BINS_PER_TONE:
                return True         # frequency is in the list
            
        # Frequency not found in the list: return False
        return False


    def get_tones(self, segment: Segment, w_tol=BINS_PER_TONE) -> Tones:
        """
        Find the audio's tones, DC, and noise.
         
        Return the tones.
        Parameters
        ----------
        segment : Segment
            Segment of audio to analyze.
        w_tol : int
            Tolerance for tone detection, in bins.
            
        Returns
        -------
        Tones
            The tones found in the audio.
        """
        self.md.segment = segment
        self.get_bins(segment)
        bins = np.absolute(self.bins)

        # Get tones and then set their bins to zero
        last_bin = len(bins) - 1
        self.tones = Tones(segment)
        while len(self.tones) < self.MAX_TONES:
            center = np.where(bins == bins.max())[0][0]
            first = max(center - w_tol, 0)
            last = min(center + w_tol, last_bin)
            freq = self.bin_to_freq(center)
            pwr = sum(bins[first:last])
            tone = Tone(freq, pwr)            
            bins[first:last] = 0
            self.tones.append(tone)

        # Get noise as the sum of the remaining non-tone bins
        self.noise = sum(bins)
        return self.tones


    def get_stats(self, seg:Segment | None, max_secs:float =10.0) -> Stats:
        """Calculate and return audio statistics"""

        # Use the entire audio if no segment is specified
        if seg is None:
            seg = Segment(self.fs, 0, len(self.data) - 1)

        # Truncate the slice to a whole number of seconds
        seg = seg.trunc(1.0)

        if max_secs > 0.0:
            secs = min((seg.stop - seg.start + 1)*self.fs, max_secs)
            stop = round(secs * self.fs) - 1
            seg = Segment(seg.start, stop, self.fs)

        # Rase ValueError if audio length is too small to calculate meaningful
        # statistics
        if len(self) < self.MIN_AUDIO_LEN:
            raise ValueError('Audio sample is too short to analyze')
        
        tones = self.get_tones(seg).copy()
        
        # The fundamental signal frequency is the bin with the largest power,
        # which is at index zero
        tones[0].cat = Category.Tone        
        self.tone = tones[0]

        # Compare power and frequency of the second-largest power with those of
        # the first to determine if this is a two-tone signal

        harmonic = alias_freq(2.0*tones[0].freq, self.fs)
        if ((tones[1].pwr < self.MIN_TWO_TONE_RATIO*tones[0].pwr)
            or (self.freq_to_bin(abs(tones[1].freq - harmonic)) < BINS_PER_TONE)):
            # Second-largest is a harmonic: single tone signal
            start = 1
            dist_freqs = [alias_freq(n * self.tone.freq, self.fs ) for n in range(10)]
            self.tone2 = Tone(0.0, 0.0)
        else:
            # Second-largest is a not a harmonic: this is a two-tone signal
            tones[1].cat = Category.Tone            
            self.tone2 = tones[1]
            start = 2

            # Swap tones if tone2 has a lower frequency than the first
            if self.tone2.freq < self.tone.freq:
                (self.tone, self.tone2) = (self.tone2, self.tone)

            # Calculate distortion frequencies for two-tone signal
            (f1, f2) = (self.tone.freq, self.tone2.freq)
            dist_freqs = []
            for n in range(self.MAX_DIST_ORDER):
                for n2 in range(self.MAX_DIST_ORDER):
                    dist_freqs.append(n*f1 + n2*f2)
                    dist_freqs.append(n*f1 - n2*f2)
                    dist_freqs.append(-n*f1 + n2*f2)

        # Categorize remaining tones
        for tone in tones[start:]:
            if ((is_harmonic(tone.freq, self.tone.freq, self.fs, self.MAX_HARM_ORDER)) or
                (is_harmonic(tone.freq, self.tone2.freq, self.fs, self.MAX_HARM_ORDER))):
                tone.cat = Category.Harmonic
            elif self.freq_in_list(tone.freq, dist_freqs):
                tone.cat = Category.Distortion
            elif tone.pwr > self.tone.pwr * self.MIN_SPUR_RATIO:
                tone.cat = Category.Spurious
            else:
                tone.cat = Category.Noise

        # Recategorize noise tones as spurs which are harmonics of spur tones
        spur_freqs = [tone.freq for tone in tones if tone.cat == Category.Spurious]
        for idx, tone in enumerate(tones):
            if tone.cat == Category.Noise:
               for freq in spur_freqs:
                    if is_harmonic(tone.freq, freq, self.fs, self.MAX_HARM_ORDER):
                        tones[idx].cat = Category.Spurious
                           
        # Calculate SFDR, distortion, and noise from tones and based on their
        # categories and frequencies
        spur_pwr = 0.0
        dist_pwr = 0.0
        noise_pwr = 0.0
        for tone in tones:
            match tone.cat:
                case Category.Spurious:
                    spur_pwr = max(spur_pwr, tone.pwr)
                case Category.Harmonic | Category.Distortion:
                    dist_pwr += tone.pwr
                case Category.Noise:
                    noise_pwr += tone.pwr

        # Set frequency statistics from powers and return stats
        stats = Stats(self, seg, self.tone.freq, self.tone2.freq)
        stats.set_pwr_stats(self.dc, self.tone.pwr, noise_pwr, spur_pwr, dist_pwr)    
        return stats


    def print(self, args) -> None:
        """Print the specified test data."""
        if self.md.name:
            print(f'Data from {self.md.name}:')
            print()

        if args.components:
            self.tones.print(self.tone.pwr)
            print()

        if args.noise_floor:
            nf, index = self.get_noise_floor()
            secs = index / self.fs
            nf_db = dbv(nf)
            if nf_db > self.MAX_NOISE_FLOOR_DB:
                print('Noise floor not found: no silence detected.')
            else:
                print(f'Noise Floor = {nf_db:0.1f} dBFS at {secs:0.3f} secs')
            print()