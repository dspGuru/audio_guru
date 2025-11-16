"""Audio metadata support"""

from dataclasses import dataclass
import pathlib

from segment import Segment


def split_fname(pathname: str) -> tuple[str]:
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
    
    base = pathlib.Path(pathname).stem
    parts = base.split('_')

    if len(parts) >= 3:
        (mfr, model, desc) = (parts[0], parts[1], ' '.join(parts[2:]))
    elif len(parts) == 2:
        (mfr, model, desc) = (parts[0], parts[1], '')
    elif len(parts) == 1:
        mfr = ''
        model = ''
        desc = parts[0]     # entire filename as description
    
    return (mfr, model, desc)


@dataclass
class Metadata:
    """
    Audio Metadata

    Attributes
    ----------
    segment : Segment
        Audio segment.
    name : str
        File pathname.
    mfr : str
        Manufacturer of unit under test.
    model : str
        Model ID of unit under test.
    desc : str
        Description of test signal.
    """

    segment: Segment        # audio segment
    name: str=''            # file pathname    
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
        """
        Return the description. If description has not been set, return the
        string representation of the segment.
        """
        if self._desc:
            return self._desc
        else:
            return str(self.segment)


    def set(self, desc: str, mfr: str='', model: str='') -> None:
        """
        Set metadata.

        Parameters
        ----------
        desc : str
            Description of the test signal.
        mfr : str
            Manufacturer of the unit under test.
        model : str
            Model ID of the unit under test.
        """
        self._desc = desc        
        self.mfr = mfr
        self.model = model
