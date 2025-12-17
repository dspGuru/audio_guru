import pathlib

from constants import DEFAULT_FS
from metadata import split_pathname, Metadata
from segment import Segment
from util import Category


def test_split_pathname_three_parts():
    tup = split_pathname("Acme_C123.wav")
    assert tup == ("Acme", "C123", "")


def test_split_pathname_no_underscores_uses_basename():
    tup = split_pathname("Test Signal.wav")
    # basename should preserve spaces and drop extension
    assert tup == ("", "", "Test Signal")


def test_split_pathname_more_than_three_parts_joins_remaining():
    tup = split_pathname("Acme_C123_Test_Signal.wav")
    assert tup == ("Acme", "C123", "Test Signal")

    tup2 = split_pathname("A_B_C_D.wav")
    assert tup2 == ("A", "B", "C D")


def test_metadata_basic_fields_and_fname_and_unit_id():
    fs = DEFAULT_FS
    seg = Segment(fs=fs, start=0, stop=4095, id=1, cat=Category.Tone)
    md = Metadata("Acme_C123_Test.wav", seg)

    # fields from split_pathname
    assert md.mfr == "Acme"
    assert md.model == "C123"
    assert (
        md.desc == "Test:1"
    )  # remaining part becomes description, appended with segment id

    # fs forwarded from segment
    assert md.fs == fs

    # fname uses segment.title and default extension
    expected_fname = f"{md.mfr}_{md.model}_{seg.title}.wav"
    assert md.get_fname() == expected_fname

    # custom extension
    assert md.get_fname("mp3") == f"{md.mfr}_{md.model}_{seg.title}.mp3"

    # unit_id concatenates mfr and model
    assert md.unit_id == f"{md.mfr} {md.model}"

    # to_dict contains expected keys and values
    d = md.to_dict()
    assert d["Name"] == "Acme_C123_Test.wav"
    assert d["Mfr"] == "Acme"
    assert d["Model"] == "C123"
    assert d["Description"] == "Test:1"


def test_secs_property():
    fs = 48000.0
    seg = Segment(fs, 0, 999)
    md = Metadata("Any_Mfr_Model.wav", seg)
    assert md.secs == len(seg) / seg.fs


def test_get_fname_strips_leading_dot_in_extension():
    seg = Segment(DEFAULT_FS, 0, 101)
    md = Metadata("Acme_C123_Test.wav", seg)
    assert md.get_fname(".mp3") == f"{md.mfr}_{md.model}_{seg.title}.mp3"


def test_desc_multi_falls_back_to_segment_str():
    seg = Segment(DEFAULT_FS, 0, 10)
    md = Metadata("Acme_C123_multi.wav", seg)
    assert md._desc.lower() == "multi"
    assert md.desc == str(seg)


def test_len_and_set_and_desc_fallback():
    fs = 48000.0
    seg = Segment(fs, 0, 999)
    md = Metadata("NoUnderscore.wav", seg)

    # for a pathname without underscores, mfr/model empty and desc is basename
    assert md.mfr == ""
    assert md.model == ""
    assert md.desc == pathlib.Path("NoUnderscore.wav").stem

    # __len__ delegates to segment
    assert len(md) == len(seg)

    # set updates fields
    md.set("NewMfr", "NewModel", "NewDesc")
    assert md.mfr == "NewMfr"
    assert md.model == "NewModel"
    assert md.desc == "NewDesc"
