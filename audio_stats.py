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

    def gen_test_audio(self):
        # Generate a fundamental and a smaller harmonic
        fs = 44100
        freq = 1000
        thd = 0.01
        x = lr.tone(freq, sr=fs, length=fs)
        x2 = lr.tone(2*freq, sr=fs, length=fs) * thd
        x += x2
        
        # Store audio and fs in self
        self.audio = x
        self.fs = fs
    
    def load(self, fname):
        self.audio, self.fs = lr.load(fname, sr=None)
        self.name = fname

    def analyze(self):
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
        x = self.audio[zc[0]:zc[-1]]

        # Window the signal
        w = signal.windows.blackmanharris(len(x))
        xw = x*w
        w_tol = 7       # Window spread tolerance (+/-)

        # Compute real FFT
        yf = rfft(xw)
        pwr = yf*yf.conjugate()

        # Find the fundamental frequency, which is taken as the bin with the
        # largest power
        max_pwr = pwr.max()
        fund = int(np.nonzero(pwr == max_pwr)[0])
        self.fund = fund

        # Extract the power of DC, harmonics, and noise:
        # - DC is 'harmonic' 0, fundamental is 1, etc.
        # - Noise is the power of all other bins
        signal_pwr = 0.0
        noise_pwr = 0.0
        harm_pwr = []
        n = 0
        for i in range(len(pwr)):
            if i >= (n*fund - w_tol) and i <= (n*fund + w_tol):
                signal_pwr += pwr[i]
            else:
                noise_pwr += pwr[i]
            if i > (n+0.5)*fund:
                harm_pwr.append(signal_pwr)
                signal_pwr = 0.0
                n += 1
                
        # Normalize the noise power and all harmonics by the power
        # of the fundamental
        noise_pwr /= harm_pwr[1]
        harm_pwr /= harm_pwr[1]

        # Calculate various powers
        dist_pwr = sum(harm_pwr[2:])
        fund_pwr = harm_pwr[1]
        dc_pwr = harm_pwr[0]
        sfdr_pwr = max(harm_pwr[2:])

        # Calculate statistics from powers
        self.dc = self.db(harm_pwr[0])
        self.snr = self.db(noise_pwr / fund_pwr)
        self.sfdr = self.db(sfdr_pwr / fund_pwr)
        self.sinad = self.db((noise_pwr + dist_pwr) / fund_pwr)
        self.thd = self.db(dist_pwr/fund_pwr)
        self.thd_pct = math.sqrt(dist_pwr/fund_pwr) * 100.0

    def print(self):
        print('Audio Statistics')
        print('-----------------------')
        if self.name:
            print( f'Name    = {self.name}')
        print(f'Samples = {len(self.audio)}')
        print(f'Min     = {min(self.audio):6.3f}')
        print(f'Max     = {max(self.audio):6.3f}')
        print(f'Fs      = {self.fs} Hz')
        print(f'Freq    = {self.freq:.3f} Hz')
        print(f'DC      = {self.dc:3.1f} dB')
        print(f'SNR     = {self.snr:3.1f} dB')
        print(f'SFDR    = {self.sfdr:3.1f} dB')
        print(f'SINAD   = {self.sinad:3.1f} dB')
        print(f'THD     = {self.thd:3.1f} dB')
        print(f'THD     = {self.thd_pct:.3f}%')

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

