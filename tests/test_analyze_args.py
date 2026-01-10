import pytest
from unittest.mock import patch
import sys
from analyze import get_args


def run_parser(args):
    """Helper to run the parser with specified arguments."""
    with patch.object(sys, "argv", ["prog_name"] + args):
        return get_args()


def test_default_args():
    args = run_parser([])

    assert args.pattern == ""
    assert args.components is None
    assert args.write_csv is None
    assert args.write_csv is None
    assert args.device is False
    assert args.max_length == 60.0
    assert args.audio_stats is True
    assert args.specs is False
    # Default table sort
    assert args.table == "thd"


def test_flags():
    args = run_parser(["-c", "-d", "-s"])
    assert args.components == 10  # Default when -c used without value
    assert args.device is True
    assert args.specs is True


def test_components_with_value():
    args = run_parser(["-c", "5"])
    assert args.components == 5

    args = run_parser(["--components", "20"])
    assert args.components == 20


def test_values():
    args = run_parser(["test.wav", "-csv", "out.csv", "-l", "10.5"])
    assert args.pattern == "test.wav"
    assert args.write_csv == "out.csv"
    assert args.max_length == 10.5


def test_sorting_groups():
    # Test sorting options

    # Map short flags to table sort attributes:
    # -t -> thd, -tf -> freq, -tf2 -> freq2, -tn -> name, -ts -> snr, -tt -> start
    for flag, expected in [
        ("-t", "thd"),
        ("-tf", "freq"),
        ("-tf2", "freq2"),
        ("-tn", "name"),
        ("-ts", "snr"),
        ("-tt", "start"),
    ]:
        args = run_parser([flag])
        assert args.table == expected


def test_mutually_exclusive():
    # Two sorting flags should fail
    with pytest.raises(SystemExit):
        run_parser(["-t", "-tf"])
