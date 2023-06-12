import math, sys, wave
import numpy as np
from scipy.fftpack import fft, rfft
from scipy import signal
import librosa as lr

class AudioStats:

    def __init__(self, audio=None, fs=1.0, name=None):
        self.audio = audio
        self.fs = fs
        self.name = name

    def db(self, pwr):
        return lr.power_to_db(pwr, amin=1e-12)

    def gen_test_audio(self, thd=0.01):
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
            self.comps.append( (self.bin_to_freq(center), pwr) )
        self.noise = sum(bins)

    def bin_to_freq(self, bin):
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
        self.freq = self.comps[0][0]
        self.sig_pwr = self.comps[0][1]

        # Compare the second-largest power with the first to
        # determine if this is a two-tone signal
        if(self.comps[1][1] < 0.03*self.comps[0][1]):           # 0.03 ~= -15 dB
            # second-largest is small: single tone signal
            start = 1
            dist = [n * self.freq for n in range(20)]
        else:
            # second-largest is big: two-tone signal
            self.freq2 = self.comps[1][0]
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
        for (f, pwr) in self.comps[start:]:
            sfdr_pwr = max(sfdr_pwr, pwr)
            if self.freq_in_list(f, dist):
                dist_pwr += pwr
            else:
                noise_pwr += pwr

        # Calculate statistics from powers
        self.dc = self.db(self.dc/self.sig_pwr)
        self.snr = self.db(noise_pwr/self.sig_pwr)
        self.sfdr = self.db(sfdr_pwr/self.sig_pwr)
        self.sinad = self.db((noise_pwr + dist_pwr)/self.sig_pwr)
        self.thd = self.db(dist_pwr/self.sig_pwr)
        self.thd_pct = math.sqrt(dist_pwr/self.sig_pwr) * 100.0

    def print_comps(self):
        # Print components in dB relative to signal power
        print('Audio Components')
        print('-----------------------')        
        for (freq, pwr) in self.comps:
            freq = round(freq)
            db = self.db(pwr/self.sig_pwr)
            print(f'  {freq:5} Hz = {db:6.1f} dB')

    def print_stats(self):
        print('Audio Statistics')
        print('-----------------------')
        print(f'Samples = {len(self.audio)}')
        print(f'Min     = {min(self.audio):6.3f}')
        print(f'Max     = {max(self.audio):6.3f}')
        print(f'Fs      = {self.fs} Hz')
        print(f'Freq    = {self.freq:.1f} Hz')
        try:
            print(f'Freq2   = {self.freq2:.1f} Hz')
        except AttributeError:
            pass
        print(f'SFDR    = {self.sfdr:3.1f} dB')
        print(f'SNR     = {self.snr:3.1f} dB')
        print(f'SINAD   = {self.sinad:3.1f} dB')
        print(f'DC      = {self.dc:3.1f} dB')
        print(f'THD     = {self.thd:3.1f} dB')
        print(f'THD%    = {self.thd_pct:.4f}%')

    def print(self):
        if self.name:
            print( f'Name    = {self.name}\n')
        self.print_comps()
        print()
        self.print_stats()

    def stats(self):
        return {
                'Samples':len(self.audio),
                'Min':min(self.audio),
                'Max':max(self.audio),
                'Fs':self.fs,
                'Freq':self.freq,
                'DC':self.dc,
                'SNR':self.snr,
                'SFDR':self.sfdr,
                'SINAD':self.sinad,
                'THD':self.thd,
                'THD%':self.thd_pct
               }
    
    def stats_index(self):
        return [ 'Samples', 'Min', 'Max', 'Fs', 'Freq', 'DC', 'SNR', 'SFDR', 'SINAD', 'THD', 'THD%' ]

if __name__ == "__main__":
    a = AudioStats()
    if len(sys.argv) > 1:
        a.load(sys.argv[1])
    else:
        print("Generating test audio")
        a.gen_test_audio()
    a.analyze()

    a.print()
