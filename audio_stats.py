"""Calculates statistics for audio tone samples
"""

import math, sys, wave
import numpy as np
from scipy.fftpack import fft, rfft
from scipy import signal
import librosa as lr
import argparse
import glob

class AudioComp:

    def __init__(self, freq, pwr):
        self.freq = freq
        self.pwr = pwr

class AudioStats:

    min_db = -120.0

    def __init__(self):
        # File name
        self.name = ''

        # Time stats
        self.len = 0
        self.min = 0.0
        self.max = 0.0
        self.fs = 1.0
        self.freq = 0.0

        # Frequency stats
        self.dc = self.min_db
        self.snr = self.min_db
        self.sfdr = self.min_db
        self.sinad = self.min_db
        self.thd = self.min_db
        self.thd_pct = 100.0

    def print_time(self):
        print(f'Len   = {self.len}')
        print(f'Min   = {self.min:6.3f}')
        print(f'Max   = {self.max:6.3f}')
        print(f'Fs    = {self.fs} Hz')
        print(f'Freq  = {self.freq:.1f} Hz')
        try:
            print(f'Freq2 = {self.freq2:.1f} Hz')
        except AttributeError:
            pass

    def print_freq(self):
        print(f'SFDR  = {self.sfdr:3.1f} dB')
        print(f'SNR   = {self.snr:3.1f} dB')
        print(f'SINAD = {self.sinad:3.1f} dB')
        print(f'DC    = {self.dc:3.1f} dB')
        print(f'THD   = {self.thd:3.1f} dB')
        print(f'THD%  = {self.thd_pct:.4f}%')

    def print_header(self):
        print('Name                                THD %   THD    SNR    SFDR')
        print('---------------------------------------------------------------')

    def print(self):
        print(f'{self.name:35} {self.thd_pct:.4f} {self.thd:6.1f} {self.snr:6.1f} {self.sfdr:6.1f}')

    def stats(self):
        return {
                'DC':self.dc,
                'SNR':self.snr,
                'SFDR':self.sfdr,
                'SINAD':self.sinad,
                'THD':self.thd,
                'THD%':self.thd_pct
               }
    
    def stats_index(self):
        return [ 'DC', 'SNR', 'SFDR', 'SINAD', 'THD', 'THD%' ]
    
class AudioAnalyzer:

    def __init__(self, audio=None, fs=1.0, name=None):
        self.audio = audio
        self.fs = fs
        self.name = name

    def db(self, pwr):
        return lr.power_to_db(pwr, amin=1e-12)

    def gen_test_audio(self, thd=0.01):
        print("Generating test audio")

        # Generate a fundamental and a smaller harmonic
        fs = 44100
        freq = 1000
        x = lr.tone(freq, sr=fs, length=fs)
        x2 = lr.tone(2*freq, sr=fs, length=fs) * thd
        x += x2
        
        # Store data in self
        self.audio = x
        self.fs = fs
        self.freq = freq
    
    def load(self, fname):
        self.audio, self.fs = lr.load(fname, sr=None)
        self.name = fname

    def get_freq(self):
        # Get zero-crossings
        zc = np.where(np.diff(np.sign(self.audio)))[0]

        # Remove the last zero-crossing, if necessary, to span an integer
        # number of cycles
        if len(zc) % 2 == 0:
            zc = zc[:-1]

        # Calculate the signal frequency from the zero-crossings
        nzc = len(zc)
        self.freq = 0.5 * (nzc - 1) / ( zc[-1] - zc[0] ) * self.fs

        # Truncate the audio to span the first and last zero crossing
        return self.audio[zc[0]:zc[-1]]

    def get_bins(self):
        # Window the signal
        x = self.audio        
        w = signal.windows.blackmanharris(len(x))
        xw = x*w

        # Compute real FFT
        yf = rfft(xw)
        self.bins = yf*yf.conjugate()
        return self.bins

    def get_comps(self, w_tol=50):
        bins = self.bins

        self.dc = sum(bins[0:self.freq_to_bin(10)])
        bins[0:self.freq_to_bin(10)] = 0

        last_bin = len(bins) - 1
        self.comps = []
        while len(self.comps) < 20:
            center = np.where(bins == bins.max())[0][0]
            first = max(center - w_tol, 0)
            last = min(center + w_tol, last_bin)
            pwr = sum(bins[first:last])
            bins[first:last] = 0
            self.comps.append( AudioComp(self.bin_to_hz(center), pwr) )
        self.noise = sum(bins)

    def bin_to_hz(self, bin):
        return self.fs * 0.5 * bin / len(self.bins)

    def freq_to_bin(self, freq):
        return round(freq / (self.fs * 0.5) * len(self.bins))

    def freq_in_list(self, f, freqs):
        for freq in freqs:
            if(abs(f - freq) < 5):
                return True
        return False

    def analyze(self):
        self.get_bins()
        self.get_comps()
        
        # The fundamental signal frequency is the bin with the
        # largest power at index 0
        self.freq = self.comps[0].freq
        self.sig_pwr = self.comps[0].pwr

        # Compare the second-largest power with the first to
        # determine if this is a two-tone signal
        if((self.comps[1].pwr < 0.001*self.comps[0].pwr)      # 0.001 = 0.1% = -30 dB
           or (abs(self.comps[1].freq - 2*self.comps[0].pwr) < 5)):
            # second-largest is a harmonic: single tone signal
            start = 1
            dist = [n * self.freq for n in range(20)]
        else:
            # second-largest is a not a harmonic: two-tone signal
            self.freq2 = self.comps[1].freq
            start = 2

            # Calculate distortion frequencies for two-tone signal
            dist = []
            for n in range(3):
                for n2 in range(3):
                    dist.append(n*self.freq + n2*self.freq2)
                    dist.append(n*self.freq - n2*self.freq2)
                    dist.append(-n*self.freq + n2*self.freq2)                    

        # Extract the power of harmonics and noise:
        sfdr_pwr = 0.0
        dist_pwr = 0.0
        noise_pwr = self.noise
        for comp in self.comps[start:]:
            sfdr_pwr = max(sfdr_pwr, comp.pwr)
            if self.freq_in_list(comp.freq, dist):
                dist_pwr += comp.pwr
            else:
                noise_pwr += comp.pwr

        stats = AudioStats()

        stats.name = self.name
        stats.len = len(self.audio)
        stats.min = min(self.audio)
        stats.max = max(self.audio)
        stats.fs  = self.fs
        stats.freq  = self.freq
        try:
            stats.freq2 = self.freq2
        except AttributeError:
            pass

        # Calculate frequency statistics from powers
        stats.dc = self.db(self.dc/self.sig_pwr)
        stats.snr = self.db(noise_pwr/self.sig_pwr)
        stats.sfdr = self.db(sfdr_pwr/self.sig_pwr)
        stats.sinad = self.db((noise_pwr + dist_pwr)/self.sig_pwr)
        stats.thd = self.db(dist_pwr/self.sig_pwr)
        stats.thd_pct = math.sqrt(dist_pwr/self.sig_pwr) * 100.0

        self.stats = stats
        return stats

    def print_comps(self):
        # Print components in dB relative to signal power
        print('Components')
        print('-----------------------')        
        for comp in self.comps:
            freq = round(comp.freq)
            db = self.db(comp.pwr/self.sig_pwr)
            print(f'  {freq:5} Hz = {db:6.1f} dB')

    def print(self, args):
        if self.name:
            print( f'\n{self.name}')

        if args.components:
            self.print_comps()

        if args.stats:
            print('Statistics')
            print('-----------------------')            
            self.stats.print_time()
            self.stats.print_freq()

    def stats(self):
        return {
                'Samples':len(self.audio),
                'Min':min(self.audio),
                'Max':max(self.audio),
                'Fs':self.fs,
                'Freq':self.freq
               }
    
    def stats_index(self):
        return [ 'Samples', 'Min', 'Max', 'Fs', 'Freq' ]

def main():
    # Create and configure argument parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pattern", metavar='PATTERN',
        help="file name pattern", nargs='?', default=None
    )
    parser.add_argument(
        '-c', '--components', action='store_true', default=False,
        help="frequency components"
    ) 
    parser.add_argument(
        '-g', '--generate', action='store_true',
        help="generate test audio", default=None
    )
    parser.add_argument(
        '-s', '--stats', action='store_true', default=False,
        help="statistics"
    )
    parser.add_argument(
        '-t', '--table', action='store_true', default=False,
        help="statistics table"
    )    

    # Parse arguments
    args = parser.parse_args()

    # Create an audio analyzer object
    a = AudioAnalyzer()

    # Generate if specified
    if args.generate:
        a.gen_test_audio()
        a.analyze()
        a.print(args)
    
    # Load, analyze, and print input files if specified    
    if args.pattern:
        stats = []
        files = glob.glob(args.pattern)
        for f in files:
            a.load(f)
            stats.append(a.analyze())
            a.print(args)

        # Print stats table if specified
        if args.table:
            AudioStats().print_header()
            stats.sort(key=lambda st: st.thd)
            for stat in stats:
                stat.print()

if __name__ == "__main__":
    main()
