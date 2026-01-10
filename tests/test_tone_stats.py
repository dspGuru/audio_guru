import numpy as np
import pytest

from audio import Audio
from segment import Segment
from tone_stats import ToneStats


def test_tonestats_initialization():
    audio = Audio(fs=1000.0)
    audio.samples = np.zeros(1000, dtype=np.float32)
    audio.select(Segment(1000.0, 0, 999))

    # Simulate values
    freq = 100.0
    freq2 = 200.0
    sig = 1.0  # 0 dB
    noise = 0.01  # -20 dB
    spur = 0.001
    dist = 0.005

    # SNR = sig/noise per code logic
    ts = ToneStats(
        audio=audio, freq=freq, freq2=freq2, sig=sig, noise=noise, spur=spur, dist=dist
    )

    assert ts.freq == freq
    assert ts.freq2 == freq2
    assert ts.sig == sig
    assert ts.snr == pytest.approx(sig / noise)

    # SFDR: spur = max(spur, dist)
    expected_spur = max(spur, dist)
    assert ts.sfdr == pytest.approx(sig / expected_spur)

    # THD = dist / sig
    assert ts.thd == pytest.approx(dist / sig)

    # THDN = (dist + noise) / sig
    assert ts.thdn == pytest.approx((dist + noise) / sig)


def test_tonestats_str():
    audio = Audio(fs=1000.0)
    audio.samples = np.zeros(1000, dtype=np.float32)
    audio.select(Segment(1000.0, 0, 999))
    ts = ToneStats(audio, 100, 0, 1.0, 0.01, 0.0, 0.0)

    s = str(ts)
    assert "Freq  = 100.0 Hz" in s
    assert "SNR   =" in s


def test_tonestats_to_dict():
    audio = Audio(fs=1000.0)
    audio.samples = np.zeros(1000, dtype=np.float32)
    audio.select(Segment(1000.0, 0, 999))
    ts = ToneStats(audio, 100, 0, 1.0, 0.01, 0.0, 0.0)

    d = ts.to_dict()
    assert d["F1"] == 100.0
    assert "SNR" in d
    assert "THD" in d


def test_tonestats_summary():
    audio = Audio(fs=1000.0)
    audio.samples = np.zeros(1000, dtype=np.float32)
    audio.select(Segment(1000.0, 0, 999))
    ts = ToneStats(audio, 100, 0, 1.0, 0.01, 0.0, 0.0)

    s = ts.summary()
    assert "100" in s  # Freq
    # Should contain numeric values formatted


def test_tonestats_edge_cases():
    audio = Audio(fs=1000.0)
    audio.samples = np.zeros(1000, dtype=np.float32)
    audio.select(Segment(1000.0, 0, 999))

    # Low noise -> SNR uses -MIN_DB
    ts_low_noise = ToneStats(audio, 100, 0, 1.0, 1e-10, 0.0, 0.0)
    assert ts_low_noise.snr is not None

    # No signal -> THD=0
    ts_no_sig = ToneStats(audio, 100, 0, 0.0, 0.01, 0.0, 0.0)
    assert ts_no_sig.thd == 0.0
    assert ts_no_sig.thdn == 0.0

    # Freq2 > 0 with THD > 0 in __str__
    ts_dual = ToneStats(audio, 100, 200, 1.0, 0.01, 0.0, 0.01)
    s = str(ts_dual)
    assert "Freq2 = 200.0 Hz" in s
    assert "THD   =" in s

    # Summary with Freq2
    s_sum = ts_dual.summary()
    assert "200" in s_sum
