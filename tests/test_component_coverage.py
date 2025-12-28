import pytest
from unittest.mock import patch, MagicMock


from component import Component, Components
from util import Category
from metadata import Metadata
from segment import Segment


def test_component_print_edge_cases(capsys):
    # Test valid print checks internal logic for pwr > 0 usually?
    # Logic: if ref_pwr <= 0, ref_pwr = MIN_PWR.
    c = Component(freq=1000, pwr=0.5, cat=Category.Tone)

    # Test with ref_pwr <= 0
    c.print(ref_pwr=0.0)
    captured = capsys.readouterr()
    # Should print without error.
    assert "Hz" in captured.out

    # Test Category.Band print formatting
    c_band = Component(
        freq=1000, pwr=0.5, cat=Category.Band, lower_freq=900, upper_freq=1100
    )
    c_band.print()
    captured = capsys.readouterr()
    assert "Band (900-1100 Hz)" in captured.out


def test_components_lists_edge_cases():
    md = Metadata("test", Segment(1000, 0, 100))
    comps = Components(md)

    # combine empty
    comps.combine()  # Should not crash

    # normalize_pwr empty / not found
    comps.normalize_pwr(1000)  # Should not crash

    # normalize_pwr with 0 power reference
    c1 = Component(freq=100, pwr=1.0)
    c_zero = Component(freq=1000, pwr=0.0)
    comps.append(c1)
    comps.append(c_zero)
    comps.normalize_pwr(1000)  # Ref is c_zero, pwr=0 -> returns early
    assert comps[0].pwr == 1.0  # Unchanged


def test_components_band_edges_failures():
    md = Metadata("test", Segment(1000, 0, 100))
    comps = Components(md)
    # Populate with some components
    for f in range(100, 1100, 100):
        comps.append(Component(freq=f, pwr=1.0))

    # find_index returns None (out of tolerance or range)
    # band_edges asserts lower < upper
    # lower out of range
    # find_index uses tolerance 100Hz default.
    # If we ask for 5000Hz, it returns None.
    le, ue = comps.band_edges(5000, 6000)
    assert le == 5000
    assert ue == 6000

    # Test ValueError catch in generation expression (max() of empty sequence)
    # We need lower_idx and upper_idx to be found, but NO components satisfying condition.
    # Condition: lower <= freq <= center AND pwr <= target_pwr
    # To force max() failure, we need find_index to succeed (so indices valid),
    # but the generator filter to return nothing.
    # Average power will be 1.0. target_pwr for 3dB attn is ~0.5.
    # All components have pwr=1.0. So pwr <= target (0.5) will be False for all.
    # Thus generator empty.
    le, ue = comps.band_edges(100, 500, attn_db=3.0)

    # Should fall back to passed bounds
    assert le == 100
    assert ue == 500


def test_components_ripple_empty():
    md = Metadata("test", Segment(1000, 0, 100))
    comps = Components(md)
    comps.append(Component(freq=100, pwr=1.0))

    # Range with no components
    r = comps.ripple(200, 300)
    assert r == 0.0


def test_components_print_opts(capsys):
    md = Metadata("test", Segment(1000, 0, 100))
    comps = Components(md)
    for i in range(20):
        comps.append(Component(freq=i * 100, pwr=1.0))

    # Test ref_pwr <= 0
    comps.print(ref_pwr=0.0)
    # Just ensure it runs

    # Test max_components limit
    comps.print(max_components=5)
    captured = capsys.readouterr()
    assert "more ..." in captured.out


def test_to_csv_mkdir(tmp_path):
    md = Metadata("test", Segment(1000, 0, 100))
    comps = Components(md)
    comps.append(Component(freq=100, pwr=1.0))

    # Path in non-existent subdir
    out_file = tmp_path / "subdir" / "test.csv"

    # Capture stdout for print_head
    with patch("builtins.print") as mock_print:
        comps.to_csv(str(out_file), print_head=True)
        # Verify file exists
        assert out_file.exists()
        # Verify header printed (df.head())
        # The mock print is called with dataframe head
        assert mock_print.call_count >= 1


def test_components_combine_trailing_duplicates():
    md = Metadata("test", Segment(1000, 0, 100))
    comps = Components(md)
    # Add two components that should settle in the same bin
    # 1000 and 1000.5, with tol=1.0
    c1 = Component(freq=1000.0, pwr=1.0)
    c2 = Component(freq=1000.5, pwr=1.0)
    comps.append(c1)
    comps.append(c2)

    comps.combine(freq_tol=1.0)

    # Should combine into 1
    assert len(comps) == 1

    # Frequency should be geometric mean of 1000 and 1000.5
    expected_freq = (1000 * 1000.5) ** 0.5
    assert comps[0].freq == pytest.approx(expected_freq)

    # Power should be sum / count (average) == 1.0 (since both 1.0)
    # Wait, check average implementation: pwr = sum(pwr) / len.
    # Yes, (1+1)/2 = 1.0.
    assert comps[0].pwr == 1.0
