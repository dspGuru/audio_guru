import pytest
from constants import DEFAULT_FS
from util import alias_freq, is_harmonic


@pytest.mark.parametrize(
    "freq,fs,expected",
    [
        (-100.0, 1000.0, 100.0),  # Negative -> abs
        (1200.0, 1000.0, 200.0),  # >fs wraps down by fs
        (800.0, 1000.0, 200.0),  # >Nyquist -> aliased (fs - freq)
        (3500.0, 1000.0, 500.0),  # Multiple wraps -> final 500
        (500.0, 1000.0, 500.0),  # Exactly Nyquist stays
        (0.0, DEFAULT_FS, 0.0),  # Zero frequency
    ],
)
def test_alias_freq_various(freq, fs, expected):
    """Verify frequency aliasing logic for negative, wrapping, and Nyquist
    cases."""
    assert alias_freq(freq, fs) == pytest.approx(expected)


def test_is_harmonic_basic_true():
    # 2nd harmonic (200 Hz) of 100 Hz at ample fs
    assert is_harmonic(200.0, 100.0, 48000.0, order=3)


def test_is_harmonic_false_nonharmonic():
    # Not a harmonic within order=2
    assert not is_harmonic(250.0, 100.0, 48000.0, order=2)


def test_is_harmonic_with_aliasing():
    # 300 Hz fund, 2nd harm=600 Hz. fs=800 -> 600 aliases to 200 Hz
    assert is_harmonic(200.0, 300.0, 800.0, order=4)


def test_is_harmonic_tolerance():
    # 3rd harmonic=300 Hz; freq=296 within tol_hz=5 -> True
    assert is_harmonic(296.0, 100.0, 48000.0, order=5)


@pytest.mark.parametrize(
    "freq,fund,fs,order",
    [
        (100.0, 0.0, DEFAULT_FS, 3),  # Invalid fundamental
        (-50.0, 100.0, DEFAULT_FS, 3),  # Negative freq
        (100.0, 100.0, DEFAULT_FS, 0),  # Invalid order
        (100.0, 100.0, 0.0, 3),  # Invalid fs
    ],
)
def test_is_harmonic_invalid_inputs_return_false(freq, fund, fs, order):
    """Ensure invalid frequencies, sample rates, or orders return False safely."""
    assert not is_harmonic(freq, fund, fs, order)
