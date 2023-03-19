import math, sys, wave
import numpy as np
from scipy.fftpack import fft, rfft
from scipy import signal
import librosa

class AudioStats:

    def __init__(self, audio=None, fs=1.0):
        self.audio = audio
        self.fs = fs

    def gen_test_audio(self):
        # Generate
        sr = 44100
        freq = 100
        x = librosa.tone(freq, sr=sr, length=sr)
        x2 = librosa.tone(2*freq, sr=sr, length=sr) * 0.01
        x += x2
        
        # Store audio and fs in self
        self.audio = x
        self.fs = sr
    
    def load(self, fname):
        self.audio, self.fs = librosa.load(fname, sr=None)

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
        self.dc = 10*math.log10(harm_pwr[0])
        self.snr = 10*math.log10(noise_pwr / fund_pwr)
        self.sfdr = 10*math.log10(sfdr_pwr / fund_pwr)
        self.sinad = 10*math.log10((noise_pwr + dist_pwr) / fund_pwr)
        self.thd = math.sqrt(dist_pwr/fund_pwr)
        self.thd_pct = self.thd * 100.0
        self.thd_db = 20*math.log10(self.thd)

    def print_stats(self):
        print('Audio Statistics')
        print('-----------------------')
        print(f'Samples = {len(self.audio)}')
        print(f'Min     = {min(self.audio):6.3f}')
        print(f'Max     = {max(self.audio):6.3f}')
        print(f'Fs      = {self.fs} Hz')
        print(f'Freq    = {self.freq:.3f} Hz')
        print(f'DC      = {self.dc:3.1f} dB')
        print(f'SNR     = {self.snr:3.1f} dB')
        print(f'SFDR    = {self.sfdr:3.1f} dB')
        print(f'SINAD   = {self.sinad:3.1f} dB')
        print(f'THD     = {self.thd_db:3.1f} dB')
        print(f'THD     = {self.thd_pct:.3f}%')

if __name__ == "__main__":
    a = AudioStats()
    if len(sys.argv) > 1:
        a.load(sys.argv[1])
    else:
        print("Generating test audio")
        a.gen_test_audio()
    a.analyze()
    a.print_stats()
