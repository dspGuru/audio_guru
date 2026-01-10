"""Tests for AudioList class."""

import tempfile
import pathlib
import time
from unittest.mock import patch

import numpy as np
import pytest

from audio import Audio
from audio_list import AudioList
from util import Category


class TestAudioListAppend:
    """Tests for AudioList.append method."""

    def test_append_audio_object(self):
        """Test appending a valid Audio object."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()

        audio_list.append(audio)

        assert len(audio_list) == 1
        assert audio_list[0] is audio

    def test_append_sets_tracknumber(self):
        """Test that append sets tracknumber based on list position."""
        audio_list = AudioList()

        audio1 = Audio(fs=48000, name="track1.wav")
        audio1.samples = np.ones(1000, dtype=np.float32)
        audio1.select()

        audio2 = Audio(fs=48000, name="track2.wav")
        audio2.samples = np.ones(1000, dtype=np.float32)
        audio2.select()

        audio_list.append(audio1)
        audio_list.append(audio2)

        assert audio1.md.info["tracknumber"] == 1
        assert audio2.md.info["tracknumber"] == 2

    def test_append_sets_title_from_name(self):
        """Test that append sets title from audio name."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="my_test_file.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()

        audio_list.append(audio)

        assert audio.md.info["title"] == "my_test_file.wav"

    def test_append_sets_artist_from_unit_id(self):
        """Test that append sets artist from metadata unit_id."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="Acme_Model123_Test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()

        audio_list.append(audio)

        # unit_id is "mfr model" format
        assert audio.md.info["artist"] == audio.md.unit_id

    def test_append_sets_date(self):
        """Test that append sets date to current date."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()

        expected_date = time.strftime("%Y-%m-%d")
        audio_list.append(audio)

        assert audio.md.info["date"] == expected_date

    def test_append_sets_genre_from_category(self):
        """Test that append sets genre from audio category."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()
        audio.cat = Category.Tone

        audio_list.append(audio)

        assert audio.md.info["genre"] == "Tone"

    def test_append_sets_software(self):
        """Test that append sets software to audio_guru."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()

        audio_list.append(audio)

        assert audio.md.info["software"] == "audio_guru"

    def test_append_sets_license(self):
        """Test that append sets license to CC0."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()

        audio_list.append(audio)

        assert audio.md.info["license"] == "CC0 1.0 Universal (Public Domain)"

    def test_append_non_audio_raises_typeerror(self):
        """Test that appending non-Audio object raises TypeError."""
        audio_list = AudioList()

        with pytest.raises(TypeError, match="AudioList can only contain Audio objects"):
            audio_list.append("not an audio object")

        with pytest.raises(TypeError, match="AudioList can only contain Audio objects"):
            audio_list.append({"data": [1, 2, 3]})

    def test_append_with_kwargs_updates_info(self):
        """Test that keyword arguments update audio.md.info."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()

        audio_list.append(audio, comment="Test comment", copyright="2024 Test")

        assert audio.md.info["comment"] == "Test comment"
        assert audio.md.info["copyright"] == "2024 Test"


class TestAudioListWrite:
    """Tests for AudioList.write method."""

    def test_write_creates_directory(self, tmp_path):
        """Test that write creates the output directory if it doesn't exist."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()
        audio_list.append(audio)

        new_dir = tmp_path / "subdir" / "nested"
        audio_list.write(new_dir, quiet=True)

        assert new_dir.exists()
        assert (new_dir / "test.wav").exists()

    def test_write_sets_album(self, tmp_path):
        """Test that write sets album to the path."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()
        audio_list.append(audio)

        audio_list.write(tmp_path, quiet=True)

        assert audio.md.info["album"] == str(tmp_path)

    def test_write_multiple_files(self, tmp_path):
        """Test writing multiple files."""
        audio_list = AudioList()

        for i in range(3):
            audio = Audio(fs=48000, name=f"track{i}.wav")
            audio.samples = np.ones(1000, dtype=np.float32)
            audio.select()
            audio_list.append(audio)

        audio_list.write(tmp_path, quiet=True)

        assert (tmp_path / "track0.wav").exists()
        assert (tmp_path / "track1.wav").exists()
        assert (tmp_path / "track2.wav").exists()

    def test_write_quiet_false_prints_message(self, tmp_path, capsys):
        """Test that quiet=False prints a message."""
        audio_list = AudioList()
        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(1000, dtype=np.float32)
        audio.select()
        audio_list.append(audio)

        audio_list.write(tmp_path, quiet=False)

        captured = capsys.readouterr()
        assert f"Writing audio list to '{tmp_path}'" in captured.out


class TestAudioListInheritsFromList:
    """Tests verifying AudioList inherits list behavior."""

    def test_is_instance_of_list(self):
        """Test AudioList is a list subclass."""
        audio_list = AudioList()
        assert isinstance(audio_list, list)

    def test_supports_len(self):
        """Test AudioList supports len()."""
        audio_list = AudioList()
        assert len(audio_list) == 0

        audio = Audio(fs=48000, name="test.wav")
        audio.samples = np.ones(100, dtype=np.float32)
        audio.select()
        audio_list.append(audio)

        assert len(audio_list) == 1

    def test_supports_indexing(self):
        """Test AudioList supports indexing."""
        audio_list = AudioList()

        audio1 = Audio(fs=48000, name="first.wav")
        audio1.samples = np.ones(100, dtype=np.float32)
        audio1.select()

        audio2 = Audio(fs=48000, name="second.wav")
        audio2.samples = np.ones(100, dtype=np.float32)
        audio2.select()

        audio_list.append(audio1)
        audio_list.append(audio2)

        assert audio_list[0] is audio1
        assert audio_list[1] is audio2
        assert audio_list[-1] is audio2

    def test_supports_iteration(self):
        """Test AudioList supports iteration."""
        audio_list = AudioList()

        audios = []
        for i in range(3):
            audio = Audio(fs=48000, name=f"track{i}.wav")
            audio.samples = np.ones(100, dtype=np.float32)
            audio.select()
            audio_list.append(audio)
            audios.append(audio)

        iterated = list(audio_list)
        assert iterated == audios
