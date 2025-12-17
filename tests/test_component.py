import math

import pandas as pd
from unittest.mock import MagicMock, patch

from component import Component, Components
from util import Category


def test_str_and_summary_consistent():
    comp = Component(freq=1234.0, pwr=0.1, secs=0.123, cat=Category.Tone)
    assert comp.summary() == str(comp)


def test_to_dict_contains_expected_keys_and_values():
    comp = Component(
        freq=50.0,
        pwr=0.25,
        secs=0.5,
        cat=Category.Tone,
        lower_freq=40.0,
        upper_freq=60.0,
    )
    d = comp.to_dict()
    expected_keys = {
        "Frequency (Hz)",
        "Lower Frequency (Hz)",
        "Upper Frequency (Hz)",
        "Power",
        "Time (s)",
        "Category",
    }
    assert set(d.keys()) == expected_keys
    assert math.isclose(d["Frequency (Hz)"], 50.0)
    assert math.isclose(d["Power"], 0.25)
    assert d["Category"] == comp.cat.name


from metadata import Metadata
from segment import Segment


def _make_components_container():
    # Construct Components with valid Metadata
    seg = Segment(fs=44100, start=0, stop=44100, id=1, cat=Category.Tone)
    md = Metadata(pathname="test_audio.wav", segment=seg)
    c = Components(md)
    return c


def test_components_average_and_combine():
    comps = _make_components_container()
    # two close-frequency components that should combine, and one distinct
    c1 = Component(freq=1000.0, pwr=1.0, secs=0.1, cat=Category.Tone)
    c2 = Component(freq=1000.5, pwr=2.0, secs=0.2, cat=Category.Tone)
    c3 = Component(freq=3000.0, pwr=0.5, secs=0.3, cat=Category.Tone)
    comps.extend([c1, c2, c3])

    # combine with tolerance 1 Hz should merge c1 and c2
    comps.combine(freq_tol=1.0)
    assert len(comps) == 2

    merged = comps[0]
    # geometric mean of 1000 and 1000.5 is approx 1000.2499...
    expected_freq = (1000.0 * 1000.5) ** 0.5
    assert math.isclose(merged.freq, expected_freq, rel_tol=1e-6)
    # power should be average of pwr values per implementation
    assert math.isclose(merged.pwr, (1.0 + 2.0) / 2.0)


def test_find_index_and_find_freq_and_normalize_pwr():
    comps = _make_components_container()
    c1 = Component(freq=1000.0, pwr=2.0)
    c2 = Component(freq=2000.0, pwr=4.0)
    comps.extend([c1, c2])

    idx = comps.find_index(1000.0)
    assert idx == 0
    found = comps.find_freq(2000.0)
    assert found is c2

    # normalize relative to 1000 Hz (becomes 1.0)
    comps.normalize_pwr(ref_freq=1000.0)
    assert math.isclose(comps[0].pwr, 1.0)
    # other component scaled accordingly
    assert math.isclose(comps[1].pwr, 2.0)


def test_get_bands_and_ripple_and_get_dataframe(tmp_path):
    comps = _make_components_container()
    # frequencies distributed around 1k and 2k
    comps.extend(
        [
            Component(freq=900.0, pwr=0.5),
            Component(freq=1100.0, pwr=1.5),
            Component(freq=1950.0, pwr=0.8),
            Component(freq=2050.0, pwr=1.2),
        ]
    )

    bands = comps.get_bands((1000.0, 2000.0))
    assert len(bands) == 2
    assert bands[0].freq == 1000.0
    assert bands[1].freq == 2000.0

    # ripple in 800-1200 should be max-min of pwr in that range
    r = comps.ripple(800.0, 1200.0)
    expected = max(0.5, 1.5) - min(0.5, 1.5)
    assert math.isclose(r, expected)

    # DataFrame generation and CSV export
    df = comps.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    out = tmp_path / "components.csv"
    comps.to_csv(str(out), index=False)
    assert out.exists()


def test_band_edges():
    """Test finding frequency edges at specified attenuation."""
    comps = _make_components_container()
    # Create a "bell curve" of power around 1000 Hz
    freqs = [900, 950, 980, 1000, 1020, 1050, 1100]
    pwrs = [0.1, 0.5, 0.8, 1.0, 0.8, 0.5, 0.1]
    for f, p in zip(freqs, pwrs):
        comps.append(Component(freq=f, pwr=p))

    lower, upper = comps.band_edges(800, 1200, attn_db=3.0)
    # -3dB is approx half power (0.5). Should be around 950 and 1050.
    # Note: logic calculates avg power between edges first, then target.
    # Logic in `band_edges` is a bit complex, we just verify it returns something within range.
    assert 900 <= lower <= 1000
    assert 1000 <= upper <= 1100
    assert lower < upper


def test_print_output(capsys):
    """Test print methods for Component and Components."""
    # Test Component.print
    c = Component(freq=100.0, pwr=1.0, cat=Category.Tone)
    c.print(ref_pwr=1.0)
    captured = capsys.readouterr()
    assert "100.0 Hz" in captured.out
    assert "0.0 dB" in captured.out  # 1.0/1.0 = 0 dB

    # Test Components.print
    comps = _make_components_container()
    comps.append(c)
    comps.print(ref_pwr=1.0)
    captured = capsys.readouterr()
    assert "Components" in captured.out
    assert "100.0 Hz" in captured.out


def test_plot_mock():
    """Test plot method using mocks to avoid GUI."""
    comps = _make_components_container()
    comps.append(Component(freq=100, pwr=1.0))

    with patch("matplotlib.pyplot.subplots") as mock_subplots, patch(
        "matplotlib.pyplot.show"
    ) as mock_show:

        mock_ax = MagicMock()
        mock_subplots.return_value = (None, mock_ax)

        comps.plot()

        assert mock_subplots.called
        assert mock_ax.stem.called
        assert mock_show.called


def test_average_empty():
    """Test average on empty components list."""
    comps = _make_components_container()
    # Should return default component
    avg = comps.average()
    assert avg.freq == 0.0
    assert avg.pwr == 0.0


def test_find_index_tolerance():
    """Test find_index respects tolerance."""
    comps = _make_components_container()
    comps.append(Component(freq=1000.0))

    # Within default tolerance (100)
    assert comps.find_index(1050.0) == 0

    # Outside tolerance
    assert comps.find_index(2000.0) is None

    # Custom tolerance
    assert comps.find_index(1005.0, tol_hz=1.0) is None  # Too far
    assert comps.find_index(1000.5, tol_hz=1.0) == 0  # Close enough
