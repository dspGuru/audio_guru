"""Audio metadata support"""

import pathlib

from segment import Segment

__all__ = ["split_pathname", "Metadata"]


def split_pathname(pathname: str) -> tuple[str]:
    """
    Split the given file pathname into three components delimited by
    underscores.

    If fewer than three components are present, the missing components are
    returned as empty strings. If no underscores are present, the entire
    filename (without path or extension) is returned as the description, with
    the manufacturer and model as empty strings.

    If more than three components are present, the first two components are
    returned as the manufacturer and model, and the remaining components are
    joined with spaces to form the description.

    Parameters
    ----------
    pathname : str
        File pathname to split.

    Returns
    -------
    tuple[str]
        A tuple of strings representing the manufacturer, model, and
        description.

    Examples
    --------
        'Acme_C123.wav' -> ('Acme', 'C123', '')
        'Test Signal.wav' -> ('', '', 'Test Signal')
        'Acme_C123_Test_Signal.wav' -> ('Acme', 'C123', 'Test Signal')
    """

    basename = pathlib.Path(pathname).stem
    parts = basename.split("_")

    if len(parts) >= 3:
        mfr, model, desc = (parts[0], parts[1], " ".join(parts[2:]))
    elif len(parts) == 2:
        mfr, model, desc = (parts[0], parts[1], "")
    elif len(parts) == 1:
        mfr = ""
        model = ""
        desc = basename  # use basename as description

    return (mfr, model, desc)


class Metadata:
    """
    Audio Metadata

    Attributes
    ----------
    segment : Segment
        Audio segment.
    pathname : str
        File pathname.
    mfr : str
        Manufacturer of unit under test.
    model : str
        Model ID of unit under test.
    _desc : str
        Description of test signal.
    """

    def __init__(self, pathname: str, segment: Segment):
        """
        Initialize metadata.

        Parameters
        ----------
        pathname : str
            File pathname.
        segment : Segment
            Audio segment associated with the metadata.
        """
        self.pathname: str = pathname
        self.segment: Segment = segment

        # Split the pathname's basename into the remaining components
        self.mfr, self.model, self._desc = split_pathname(self.pathname)

    def __len__(self):
        return len(self.segment)

    @property
    def fs(self) -> int:
        """Return the sampling frequency."""
        return self.segment.fs

    @property
    def desc(self) -> str:
        """
        Return the description. If description has not been set, return the
        string representation of the segment.
        """
        if self._desc and self._desc.lower() != "multi":
            if self.segment.id is not None:
                return f"{self._desc}:{self.segment.id}"
            return self._desc
        else:
            return str(self.segment)

    @property
    def secs(self) -> float:
        """Return the duration of the segment in seconds."""
        return len(self.segment) / self.segment.fs

    def get_fname(self, extention: str = "wav") -> str:
        """Return a file name for the metadata.

        Parameters
        ----------
        extention : str
            File extension (default: 'wav').

        Returns
        -------
        str
            File name.
        """
        return f"{self.mfr}_{self.model}_{self.segment.title}.{extention.lstrip('.')}"

    @property
    def unit_id(self) -> str:
        """Return the unit ID, which consists of the manufacturer and model."""
        return f"{self.mfr} {self.model}"

    def to_dict(self) -> dict[str, str]:
        """
        Return metadata as a dictionary.

        Returns
        -------
        dict[str, str]
            Ordered dictionary of metadata fields.
        """
        md_list = [
            ("Name", self.pathname),
            ("Mfr", self.mfr),
            ("Model", self.model),
            ("Description", self.desc),
        ]
        md = dict(md_list)
        return md

    def set(self, mfr: str, model: str = "", desc: str = "") -> None:
        """
        Set metadata.

        Parameters
        ----------
        mfr : str
            Manufacturer of the unit under test.
        model : str
            Model ID of the unit under test.
        desc : str
            Description of the test signal.
        """
        self.mfr = mfr
        self.model = model
        self._desc = desc
