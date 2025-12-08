import pytest
from constants import DEFAULT_FS
from util import alias_freq, is_harmonic


@pytest.mark.parametrize(
    "freq,fs,expected",
    [
        (-100.0, 1000.0, 100.0),  # negative -> abs
        (1200.0, 1000.0, 200.0),  # >fs wraps down by fs
        (800.0, 1000.0, 200.0),  # >Nyquist -> aliased (fs - freq)
        (3500.0, 1000.0, 500.0),  # multiple wraps -> final 500
        (500.0, 1000.0, 500.0),  # exactly Nyquist stays
        (0.0, DEFAULT_FS, 0.0),  # zero frequency
    ],
)
def test_alias_freq_various(freq, fs, expected):
    assert alias_freq(freq, fs) == pytest.approx(expected)


def test_is_harmonic_basic_true():
    # 2nd harmonic (200 Hz) of 100 Hz at ample fs
    assert is_harmonic(200.0, 100.0, 48000.0, order=3)


def test_is_harmonic_false_nonharmonic():
    # Not a harmonic within order=2
    assert not is_harmonic(250.0, 100.0, 48000.0, order=2)


def test_is_harmonic_with_aliasing():
    # Fundamental 300 Hz, 2nd harmonic 600 Hz. With fs=800 Hz,
    # 600 aliases to 800-600 = 200 Hz, so freq=200 should be detected.
    assert is_harmonic(200.0, 300.0, 800.0, order=4)


def test_is_harmonic_tolerance():
    # 3rd harmonic is 300 Hz; freq=296 is within default tol_hz=5 -> True
    assert is_harmonic(296.0, 100.0, 48000.0, order=5)


@pytest.mark.parametrize(
    "freq,fund,fs,order",
    [
        (100.0, 0.0, DEFAULT_FS, 3),  # invalid fundamental
        (-50.0, 100.0, DEFAULT_FS, 3),  # negative freq
        (100.0, 100.0, DEFAULT_FS, 0),  # invalid order
        (100.0, 100.0, 0.0, 3),  # invalid fs
    ],
)
def test_is_harmonic_invalid_inputs_return_false(freq, fund, fs, order):
    assert not is_harmonic(freq, fund, fs, order)
