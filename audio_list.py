"""Audio List Class"""

import pathlib
import time
from typing import Any

from audio import Audio


class AudioList(list):
    """
    Provides methods to operate on a list of related audio objects.
    """

    def append(self, audio: Audio, **kwargs: Any) -> None:
        """
        Append an Audio object to the list.

        Parameters
        ----------
        audio : Audio
            The Audio object to append.
        **kwargs : Any
            Optional keyword arguments. Each key-value pair will be
            set as an attribute on the Audio object before appending.

        Raises
        ------
        TypeError
            If the object being appended is not an Audio instance.
        """
        if not isinstance(audio, Audio):
            raise TypeError(
                f"AudioList can only contain Audio objects, got {type(audio).__name__}"
            )

        # Set any provided keyword arguments as attributes on the audio object
        audio.md.info.update(kwargs)

        # Set track number based on position in list
        audio.md.info["tracknumber"] = len(self) + 1

        # Set title based on segment name
        audio.md.info["title"] = audio.name

        # Set artist based on manufacturer
        audio.md.info["artist"] = audio.md.unit_id

        # Set date based on current time
        audio.md.info["date"] = time.strftime("%Y-%m-%d")

        # Set genre based on segment type
        audio.md.info["genre"] = audio.cat.name

        # Set software based on segment type
        audio.md.info["software"] = "audio_guru"

        # Set license based on segment type
        audio.md.info["license"] = "CC0 1.0 Universal (Public Domain)"

        super().append(audio)

    def write(
        self,
        path: str | pathlib.Path,
        quiet: bool = False,
        prepend_tracknumber: bool = False,
        ndigits: int = 2,
    ) -> None:
        """
        Write the audio list to a file.

        Parameters
        ----------
        path : str
            The path to the file to write.
        """

        # Set path and create directory if it doesn't exist
        path = pathlib.Path(path)
        if not quiet:
            print(f"Writing audio list to '{path}'")
        path.mkdir(parents=True, exist_ok=True)

        # Write each audio file
        for audio in self:
            # Set album to path
            audio.md.info["album"] = str(path)

            # Set name to audio name with track number prepended if specified
            name = audio.name
            if prepend_tracknumber:
                name = f"{audio.md.info['tracknumber']:0{ndigits}d}-{name}"

            # Write audio
            audio.write(path / name, quiet=quiet)
