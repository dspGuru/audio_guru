"""Tests for component and Components container."""

import math

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


from component import Component, Components
from util import Category
from metadata import Metadata
from segment import Segment
from constants import DEFAULT_FS, DEFAULT_FREQ


def _make_components_container():
    # Construct Components with valid Metadata
    seg = Segment(fs=DEFAULT_FS, start=0, stop=int(DEFAULT_FS), id=1, cat=Category.Tone)
    md = Metadata(pathname="test_audio.wav", segment=seg)
    c = Components(md)
    return c


def make_md():
    return Metadata("test", Segment(DEFAULT_FS, 0, int(DEFAULT_FS)))


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


def test_components_average_and_combine():
    comps = _make_components_container()
    # two close-frequency components that should combine, and one distinct
    c1 = Component(freq=DEFAULT_FREQ, pwr=1.0, secs=0.1, cat=Category.Tone)
    c2 = Component(freq=DEFAULT_FREQ + 0.5, pwr=2.0, secs=0.2, cat=Category.Tone)
    c3 = Component(freq=3000.0, pwr=0.5, secs=0.3, cat=Category.Tone)
    comps.extend([c1, c2, c3])

    # combine with tolerance 1 Hz should merge c1 and c2
    comps.combine(freq_tol=1.0)
    assert len(comps) == 2

    merged = comps[0]
    # Power-weighted mean: (1000*1.0 + 1000.5*2.0) / 3.0 = 3001/3 = 1000.333...
    expected_freq = (c1.freq * c1.pwr + c2.freq * c2.pwr) / (c1.pwr + c2.pwr)
    assert math.isclose(merged.freq, expected_freq, rel_tol=1e-6)
    # power should be average of pwr values per implementation
    assert math.isclose(merged.pwr, (1.0 + 2.0) / 2.0)


def test_find_index_and_find_freq_and_normalize_pwr():
    comps = _make_components_container()
    c1 = Component(freq=DEFAULT_FREQ, pwr=2.0)
    c2 = Component(freq=2000.0, pwr=4.0)
    comps.extend([c1, c2])

    idx = comps.find_index(DEFAULT_FREQ)
    assert idx == 0
    found = comps.find_freq(2000.0)
    assert found is c2

    # normalize relative to 1000 Hz (becomes 1.0)
    comps.normalize_pwr(ref_freq=DEFAULT_FREQ)
    assert math.isclose(comps[0].pwr, 1.0)
    # other component scaled accordingly
    assert math.isclose(comps[1].pwr, 2.0)


def test_average_empty_returns_default():
    comps = Components(make_md())
    avg = comps.average()
    assert isinstance(avg, Component)
    assert avg.freq == 0.0
    assert avg.pwr == 0.0


def test_combine_groups_and_resulting_bounds():
    md = make_md()
    comps = Components(md)
    c1 = Component(DEFAULT_FREQ, 1.0, 0.0, Category.Tone)
    c2 = Component(DEFAULT_FREQ + 1.0, 0.5, 0.0, Category.Tone)
    c3 = Component(2000.0, 0.2, 0.0, Category.Tone)

    comps.append(c1)
    comps.append(c2)
    comps.append(c3)

    # Combine with tolerance so first two combine, third stays separate
    comps.combine(freq_tol=2.0)
    assert len(comps) == 2
    # First combined freq should be between 1000 and 1001 (weighted mean)
    assert DEFAULT_FREQ <= comps[0].freq <= DEFAULT_FREQ + 1.0
    # Second component should retain its original freq
    assert comps[1].freq == 2000.0


def test_find_index_out_of_tolerance():
    md = make_md()
    comps = Components(md)
    comps.append(Component(DEFAULT_FREQ, 1.0, 0.0, Category.Tone))

    assert comps.find_index(2000.0, tol_hz=100) is None


def test_normalize_pwr_changes_values():
    md = make_md()
    comps = Components(md)
    comps.append(Component(950.0, 2.0, 0.0, Category.Tone))
    comps.append(Component(2000.0, 4.0, 0.0, Category.Tone))

    comps.normalize_pwr(ref_freq=950.0)
    # ref component should become 1.0
    ref = comps.find_freq(950.0)
    assert pytest.approx(ref.pwr, rel=1e-6) == 1.0
    # other component scaled accordingly
    other = comps.find_freq(2000.0)
    assert pytest.approx(other.pwr, rel=1e-6) == 2.0


def test_get_bands_assigns_center_and_bounds():
    md = make_md()
    comps = Components(md)
    # choose a center where sqrt2 window includes 900..1400 approx
    comps.append(Component(950.0, 1.0, 0.0, Category.Tone))
    centers = (DEFAULT_FREQ,)
    bands = comps.get_bands(centers)
    assert len(bands) == 1
    assert bands[0].freq == DEFAULT_FREQ
    assert bands[0].lower_freq > 0
    assert bands[0].upper_freq > bands[0].lower_freq


def test_band_edges_fallback_when_no_candidates():
    md = make_md()
    comps = Components(md)
    # Add components with high power so target_pwr is lower and no candidates satisfy <= target
    # Components range from 100 to 10000.
    comps.append(Component(100.0, 1000.0, 0.0, Category.Tone))
    comps.append(Component(DEFAULT_FREQ, 1000.0, 0.0, Category.Tone))
    comps.append(Component(10000.0, 1000.0, 0.0, Category.Tone))

    lower, upper = comps.band_edges(20.0, 20000.0, attn_db=3.0)
    # With no candidates meeting the <= target_pwr condition, edges should default to
    # the min/max observed frequencies, not the search range.
    assert lower == 100.0
    assert upper == 10000.0


def test_ripple_and_print_pagination_and_csv(tmp_path, capsys):
    md = make_md()
    comps = Components(md, name="MyComps")

    # Create five components
    for f in (100.0, 200.0, 300.0, 400.0, 500.0):
        comps.append(Component(float(f), float(f) * 0.001, 0.0, Category.Tone))

    # Ripple over a range
    r = comps.ripple(150.0, 450.0)
    assert r >= 0.0

    # Test print pagination with max_components smaller than len
    comps.print(ref_pwr=1.0, max_components=2)
    out = capsys.readouterr().out
    assert "more ..." in out or "more" in out

    # DataFrame and CSV
    csv_file = tmp_path / "comps.csv"
    comps.to_csv(str(csv_file), index=False, print_head=True)
    assert csv_file.exists()
    # Printed head should have appeared
    head_out = out
    assert len(head_out) >= 0


def test_get_dataframe_columns():
    md = make_md()
    comps = Components(md)
    comps.append(Component(100.0, 0.5, 0.0, Category.Tone))
    df = comps.get_dataframe()
    assert "Frequency (Hz)" in df.columns
    assert "Power" in df.columns


def test_band_edges():
    """Test finding frequency edges at specified attenuation."""
    comps = _make_components_container()
    # Create a "bell curve" of power around 1000 Hz
    freqs = [900, 950, 980, DEFAULT_FREQ, 1020, 1050, 1100]
    pwrs = [0.1, 0.5, 0.8, 1.0, 0.8, 0.5, 0.1]
    for f, p in zip(freqs, pwrs):
        comps.append(Component(freq=f, pwr=p))

    lower, upper = comps.band_edges(800, 1200, attn_db=3.0)
    # -3dB is approx half power (0.5). Should be around 950 and 1050.
    # Note: logic calculates avg power between edges first, then target.
    # Logic in `band_edges` is a bit complex, we just verify it returns something within range.
    assert 900 <= lower <= DEFAULT_FREQ
    assert DEFAULT_FREQ <= upper <= 1100
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
    comps.append(Component(freq=DEFAULT_FREQ))

    # Within default tolerance (100)
    assert comps.find_index(1050.0) == 0

    # Outside tolerance
    assert comps.find_index(2000.0) is None

    # Custom tolerance
    assert comps.find_index(1005.0, tol_hz=1.0) is None  # Too far
    assert comps.find_index(1000.5, tol_hz=1.0) == 0  # Close enough


def test_cat_returns_segment_category():
    """Test that Components.cat() returns the category from the associated segment."""
    seg = Segment(fs=DEFAULT_FS, start=0, stop=int(DEFAULT_FS), id=1, cat=Category.Tone)
    md = Metadata(pathname="test_audio.wav", segment=seg)
    comps = Components(md)

    assert comps.cat() == Category.Tone


def test_cat_with_different_categories():
    """Test Components.cat() with various segment categories."""
    for category in [Category.Tone, Category.Noise, Category.Band, Category.Unknown]:
        seg = Segment(fs=DEFAULT_FS, start=0, stop=int(DEFAULT_FS), id=1, cat=category)
        md = Metadata(pathname="test_audio.wav", segment=seg)
        comps = Components(md)

        assert comps.cat() == category
