"""
Tests for analysis_list.py
"""

import pytest
from analysis_list import AnalysisList
import pandas as pd


class MockAnalysis:
    @staticmethod
    def summary_header():
        return "Header"

    def __init__(self, thd=0.0, name="test"):
        self.thd = thd
        self.name = name

    def summary(self):
        return f"{self.name} {self.thd}"

    def to_dict(self):
        return {"name": self.name, "thd": self.thd}


@pytest.fixture
def analysis_list():
    al = AnalysisList()
    al.append(MockAnalysis(thd=0.1, name="A"))
    al.append(MockAnalysis(thd=0.01, name="B"))
    al.append(MockAnalysis(thd=0.5, name="C"))
    return al


def test_print_empty(capsys):
    al = AnalysisList()
    al.print()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_unsorted(analysis_list, capsys):
    analysis_list.print()
    captured = capsys.readouterr()
    expected_output = (
        "Header\n"
        "--------------------------------------------------------------------------------\n"
        "A 0.1\n"
        "B 0.01\n"
        "C 0.5\n"
    )
    assert captured.out == expected_output


def test_print_sorted(analysis_list, capsys):
    analysis_list.print(sort_stat_attr="thd")
    captured = capsys.readouterr()
    expected_output = (
        "Header\n"
        "--------------------------------------------------------------------------------\n"
        "B 0.01\n"
        "A 0.1\n"
        "C 0.5\n"
    )
    assert captured.out == expected_output


def test_get_dataframe(analysis_list):
    df = analysis_list.get_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df["name"]) == ["A", "B", "C"]
    assert list(df["thd"]) == [0.1, 0.01, 0.5]


def test_to_csv(analysis_list, tmp_path):
    output_file = tmp_path / "subdir" / "output.csv"

    analysis_list.to_csv(str(output_file))

    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert len(df) == 3
    assert "name" in df.columns
    assert "thd" in df.columns
    # Check default float format if possible, but reading back might change exact string rep
    # Checking values is safer
    assert df.iloc[0]["thd"] == 0.1


def test_to_csv_print_head(analysis_list, tmp_path, capsys):
    output_file = tmp_path / "output.csv"
    analysis_list.to_csv(str(output_file), print_head=True)

    captured = capsys.readouterr()
    assert "name" in captured.out
    assert "thd" in captured.out
    assert output_file.exists()


def test_to_csv_with_index(analysis_list, tmp_path):
    output_file = tmp_path / "output_index.csv"
    analysis_list.to_csv(str(output_file), index=True)

    df = pd.read_csv(output_file, index_col=0)
    assert len(df) == 3
    # When index is saved, read_csv with index_col=0 should recover it
    # Default index is RangeIndex 0, 1, 2
    assert df.index.tolist() == [0, 1, 2]
