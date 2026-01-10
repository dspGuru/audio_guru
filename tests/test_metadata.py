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
    assert md.desc == "Test:1"  # desc appended with segment id

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

    # pathname without underscores -> mfr/model empty, desc is basename
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


def test_metadata_name_property():
    """Test Metadata name property gets and sets pathname."""
    fs = 48000.0
    seg = Segment(fs, 0, 999)
    md = Metadata("Original_Mfr_Model.wav", seg)

    # name property returns pathname
    assert md.name == "Original_Mfr_Model.wav"
    assert md.name == md.pathname

    # Setting name updates pathname and re-splits
    md.name = "NewMfr_NewModel_NewDesc.wav"
    assert md.pathname == "NewMfr_NewModel_NewDesc.wav"
    assert md.mfr == "NewMfr"
    assert md.model == "NewModel"
    assert md._desc == "NewDesc"


def test_metadata_info_is_source_info():
    """Test that Metadata.info is a SourceInfo instance."""
    from metadata import SourceInfo

    seg = Segment(48000.0, 0, 100)
    md = Metadata("test.wav", seg)

    assert isinstance(md.info, SourceInfo)
    assert isinstance(md.info, dict)


def test_source_info_inherits_from_dict():
    """Test SourceInfo is a dict subclass."""
    from metadata import SourceInfo

    info = SourceInfo({"title": "Test", "artist": "Artist"})

    assert isinstance(info, dict)
    assert info["title"] == "Test"
    assert info["artist"] == "Artist"


def test_source_info_update():
    """Test SourceInfo supports dict update."""
    from metadata import SourceInfo

    info = SourceInfo({})
    info.update({"title": "New Title", "album": "New Album"})

    assert info["title"] == "New Title"
    assert info["album"] == "New Album"


def test_source_info_set_sf_metadata_converts_types():
    """Test set_sf_metadata converts non-string values to strings."""
    import unittest.mock
    from metadata import SourceInfo

    # Create info with integer value
    info = SourceInfo({"tracknumber": 1, "title": "Test Track"})

    mock_sf = unittest.mock.Mock()
    mock_sf.info_fields = (
        SourceInfo.info_fields
    )  # Ensure fields are available if needed or iterated

    # The method iterates SourceInfo.info_fields and checks if they are in 'info'
    # Then calls setattr(mock_sf, field, str(value))
    info.set_sf_metadata(mock_sf)

    # Verify conversions
    assert getattr(mock_sf, "tracknumber") == "1"
    assert getattr(mock_sf, "title") == "Test Track"
