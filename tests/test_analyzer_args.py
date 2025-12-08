import pytest
from unittest.mock import patch
import sys
from analyzer_args import get_args


def run_parser(args):
    """Helper to run the parser with specified arguments."""
    with patch.object(sys, "argv", ["prog_name"] + args):
        return get_args()


def test_default_args():
    args = run_parser([])

    assert args.pattern == ""
    assert args.components is False
    assert args.write_csv is None
    assert args.generate is None
    assert args.max_length == 0.0
    assert args.noise_floor is True
    assert args.stats is False
    # Default order set by set_defaults
    assert args.order == "thd"
    # But wait, default sorting option in group?
    # Group defaults are None if not selected, but parser.set_defaults(order="thd") is used.
    # The mutually exclusive group sets 'dest="table"'.
    # If no flag is passed, 'table' attribute might be None.
    # The 'order' attribute comes from set_defaults?
    # Inspecting analyzer_args.py:104 parser.set_defaults(order="thd")
    # But arguments store to dest="table". args.table vs args.order.
    # Let's check what attributes are present.


def test_flags():
    args = run_parser(["-c", "-g", "-s"])
    assert args.components is True
    assert args.generate is True
    assert args.stats is True


def test_values():
    args = run_parser(["test.wav", "-csv", "out.csv", "-l", "10.5"])
    assert args.pattern == "test.wav"
    assert args.write_csv == "out.csv"
    assert args.max_length == 10.5


def test_sorting_groups():
    # Test valid sorting options, identifying what 'table' is set to

    # -t -> thd
    args = run_parser(["-t"])
    assert args.table == "thd"

    # -tf -> freq
    args = run_parser(["-tf"])
    assert args.table == "freq"

    # -tm is not an option, but -tf2 is
    args = run_parser(["-tf2"])
    assert args.table == "freq2"

    # -tn -> name
    args = run_parser(["-tn"])
    assert args.table == "name"

    # -ts -> snr
    args = run_parser(["-ts"])
    assert args.table == "snr"

    # -tt -> start
    args = run_parser(["-tt"])
    assert args.table == "start"


def test_mutually_exclusive():
    # Attempting to use two sorting flags should exit/fail
    # Argparse usually exits with code 2
    with pytest.raises(SystemExit):
        run_parser(["-t", "-tf"])
