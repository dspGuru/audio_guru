"""Audio metadata class"""

from dataclasses import dataclass

from segment import Segment
from util import split_fname

@dataclass
class Metadata:
    """Audio Metadata"""

    segment: Segment        # audio segment

    # File data
    name: str=''            # file pathname

    # Metadata
    mfr: str=''             # manufacturer of unit under test
    model: str=''           # model ID of unit under test
    _desc: str=''           # description of test signal


    def __init__(self, name: str, segment: Segment):
        """Return a dictionary of manufacturer, model, and description derived
        from the sample's file name."""
        self.name = name
        self.segment = segment
        (self.mfr, self.model, self._desc) = split_fname(self.name)


    def __len__(self):
        return len(self.segment)

    @property
    def fs(self) -> int:
        """Return the sampling frequency."""
        return self.segment.fs

    @property
    def desc(self) -> str:
        """Return the description of the test signal."""
        if self._desc:
            return self._desc
        else:
            return str(self.segment)


    def set(self, desc: str, mfr: str='', model: str='') -> None:
        """Set metadata"""
        self._desc = desc        
        self.mfr = mfr
        self.model = model
