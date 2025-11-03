import io, subprocess
import numpy as np
import scipy
from scipy.io import wavfile

def load_scipy(fname: str) -> tuple:
    """Load data from .wav file using scipy."""
 
    fs = 1.0
    audio = []

    # Read WAV file using wave and convert to a mono float32 numpy array in [-1,1]
    # Read WAV file using scipy.io.wavfile.read and convert to a mono float32 numpy array in [-1,1]
    try:
        fs, data = scipy.io.wavfile.read(fname)
    except ValueError:
        print('Invalid audio file:', fname)
        return (audio, fs)

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

    return (audio, fs)

def load_ffmpeg(fname: str) -> tuple:
    """Load data from audio file using ffmpeg."""
    try:
        # Use ffmpeg to decode any supported audio file into WAV in-memory, then read with scipy
        proc = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', fname, '-f', 'wav', '-'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        buf = io.BytesIO(proc.stdout)
        fs, data = wavfile.read(buf)

        # Normalize/convert to float32 in [-1, 1]
        if np.issubdtype(data.dtype, np.signedinteger):
            denom = 2 ** (data.dtype.itemsize * 8 - 1)
            audio = data.astype(np.float32) / denom
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
        elif np.issubdtype(data.dtype, np.floating):
            audio = data.astype(np.float32)
        else:
            audio = data.astype(np.float32)
            maxv = np.max(np.abs(audio)) if audio.size else 0.0
            if maxv > 0:
                audio /= maxv

        # Convert multi-channel to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return (audio, fs)
    except:
        print('Invalid audio file:', fname)
        return ([], 1.0)

def load_librosa(fname: str) -> tuple:
    """Load data from .wav file using scipy."""
    try:
        return librosa.load(fname, sr=None)
    except:
        print('Invalid audio file:', fname)
        return ([], 1.0)

# Try to import librosa to use its load(). If that fails, set the load
# function to load_scipy() above
try:
    import librosa
    load = load_librosa
##    import ffmpeg
##    load = load_ffmpeg
except ImportError:
    load = load_scipy