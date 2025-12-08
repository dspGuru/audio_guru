"""Audio frequency component class"""

from copy import copy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from analysis import Analysis
from constants import MIN_DB, MIN_PWR
from decibels import db, dbv, db_to_pwr_ratio
from metadata import Metadata
from util import Category


__all__ = ["Component", "Components"]


@dataclass
class Component(Analysis):
    """
    Audio component or band represented by a frequency, power (amplitude
    scaling), and category.

    Attributes
    ----------
    freq : float
        Center frequency of the component in Hz. A value of 0.0 represents DC.
    pwr : float
        Amplitude scaling applied to the generated waveform. Typical range depends
        on downstream use (e.g., [-1.0, 1.0] for normalized audio).
    secs : float
        Timestamp, in seconds, associated with the tone.
    cat : Category
        Semantic/category label for the tone (e.g., instrument, test tone type). May
        be used externally for classification or metadata purposes.
    lower_freq : float
        Lower frequency bound of the component in Hz.
    upper_freq : float
        Upper frequency bound of the component in Hz.
    """

    freq: float = 0.0
    pwr: float = 0.0
    secs: float = 0.0
    cat: Category = Category.Unknown
    lower_freq: float = 0.0
    upper_freq: float = 0.0

    def __str__(self) -> str:
        amp = dbv(self.pwr)
        s = f"  {self.cat.name}: {self.freq:.0f} Hz, Amplitude: {amp:.1f} dB, Time: {self.secs:.3f}s"
        return s

    # @override
    def summary(self) -> str:
        return str(self)

    def generate(self, secs: float, fs: float) -> np.ndarray:
        """
        Generate a mono audio waveform for the tone.

        Parameters
        ----------
        secs : float
            Duration of the generated audio in seconds. The number of samples
            produced is int(secs * fs). If secs * fs <= 0 an empty array is
            returned.
        fs : float
            Sampling rate in Hz (samples per second). Values <= 0 will result in
            an empty output.

        Returns
        -------
        numpy.ndarray
            1-D array of length int(secs * fs) containing the time-domain samples.
            The waveform is generated as a constant-frequency sinusoid (implemented
            via scipy.signal.chirp with f0 == f1) and scaled by the `pwr` attribute.

        Notes
        -----
        - The implementation depends on numpy (np) and scipy.signal (signal).
        - Because the generator uses a constant-frequency chirp (f0 == f1),
          the output is a pure tone at `freq` Hz.
        - This method does not perform windowing, DC offset removal, or clipping;
          those should be handled by callers if needed.

        Examples
        --------
        >>> tone = Component(freq=440.0, pwr=0.5, cat=Category.Tone)
        >>> audio = tone.generate(secs=2.0, fs=44100)
        This generates a two-second A4 tone at half amplitude sampled at 44.1
        kHz.
        """

        # Generate constant-frequency tones using signal.chirp (f0 == f1 => pure tone)
        num_samples = int(secs * fs)
        if num_samples <= 0:
            return np.array([])

        t = np.arange(num_samples) / fs
        amp = self.pwr
        audio = amp * signal.chirp(t, f0=self.freq, t1=t[-1], f1=self.freq)
        return audio

    # @override
    def to_dict(self) -> dict[str, float | str]:
        """
        Return an ordered dictionary of component attributes.

        Returns
        -------
        dict[str, float]
            Dictionary containing the component's frequency, power, timestamp,
            category (as integer), lower frequency, and upper frequency.
        """
        comp_list: list[tuple[str, float | str]] = [
            ("Frequency (Hz)", self.freq),
            ("Lower Frequency (Hz)", self.lower_freq),
            ("Upper Frequency (Hz)", self.upper_freq),
            ("Power", self.pwr),
            ("Time (s)", self.secs),
            ("Category", self.cat.name),
        ]
        return dict(comp_list)

    def print(self, ref_pwr: float = 1.0) -> None:
        """
        Prints a description of the component relative to a reference power.

        Parameters
        ----------
        ref_pwr : float
            Reference power used for dB conversion. Values <= 0 are clipped
            to a small positive minimum to avoid log errors in db().
        """
        # Avoid passing non-positive reference to db()
        if ref_pwr <= 0.0:
            ref_pwr = MIN_PWR

        db_ref = db(self.pwr / ref_pwr)
        if self.cat == Category.Band:
            lower = round(self.lower_freq)
            upper = round(self.upper_freq)
            desc = f"Band ({lower}-{upper} Hz)"
        else:
            desc = self.cat.name

        print(f"  {self.freq:7.1f} Hz = {db_ref:6.1f} dB  {desc}")


class Components(list[Component]):
    """
    Container for multiple Component objects associated with a Segment.

    Attributes
    ----------
    segment : Segment
        Owning Segment for the components.
    """

    def __init__(self, md: Metadata, name: str = "Components") -> None:
        """Initialize with the associated Metadata."""
        super().__init__()
        assert isinstance(md, Metadata)
        self.md = copy(md)
        self.name = name

    def combine(self, freq_tol: float = 1.0) -> None:
        """
        Combine consecutive duplicate frequency entries into single entries.

        This method scans through the list of Component objects and merges
        consecutive entries that have the same frequency (within a small tolerance)
        into a single entry. The power of the combined entry is the sum of the
        individual powers, and the timestamp is taken from the first occurrence.

        Parameters
        ----------
        freq_tol : float
            Frequency tolerance in Hz for considering two frequencies as duplicates.
            Default is 1.0 Hz.

        Notes
        -----
        - This method modifies the list in place.
        """
        if not self:
            return

        # Create a new Components list to hold combined entries
        combined = Components(self.md)
        group = Components(self.md)
        group.append(self[0])
        for component in self[1:]:
            if not group:
                group.append(component)
                continue

            prev = group[-1]
            if (
                component.freq > prev.freq
                and abs(component.freq - prev.freq) > freq_tol
            ):
                # New component, combine the previous group
                if len(group) == 1:
                    combined.append(group[0])
                else:
                    # Combine group into a single component
                    combined_comp = group.average(freq_tol)
                    combined_comp.lower_freq = group[0].freq
                    combined_comp.upper_freq = group[-1].freq
                    combined.append(combined_comp)

                # Start a new group
                group = Components(self.md)
            # Add to current group
            group.append(component)

        # Combine any remaining group
        if group:
            if len(group) == 1:
                combined.append(group[0])
            else:
                combined_comp = group.average()
                combined.append(combined_comp)

        # Update self to the combined list
        self.clear()
        self.extend(combined)

    def average(self, freq_tol: float = 1.0) -> Component:
        """
        Compute the average Component across all entries.

        Returns
        -------
        Component
            New Component instance with the average frequency and power of
            all components in the list. The timestamp is set to 0.0 and the
            category is set to Unknown.

        Notes
        -----
        - If the list is empty, returns a Component with all attributes set to
          their default values (0.0 frequency, 0.0 power, 0.0 timestamp,
          Unknown category).
        """
        if not self:
            return Component()

        # Calculate the average frequency as its geomentric mean
        freq = float(np.prod([comp.freq for comp in self])) ** (1.0 / len(self))

        # Calculate the average power, timestamp, and frequency bounds
        pwr = sum(comp.pwr for comp in self) / len(self)
        secs = sum(comp.secs for comp in self) / len(self)
        lower = sum(comp.lower_freq for comp in self) / len(self)
        upper = sum(comp.upper_freq for comp in self) / len(self)
        cat = Category.Tone if upper - lower < freq_tol else Category.Band
        component = Component(freq, pwr, secs, cat, lower, upper)

        return component

    def find_freq(self, freq: float) -> Component | None:
        """
        Find and return the Component whose frequency is nearest to
        the specified frequency.

        Parameters
        ----------
        freq : float
            Frequency in Hz to search for.

        Returns
        -------
        Component | None
            The Component with the frequency nearest to the specified
            frequency, or None if no nearby component is found.
        """
        idx = self.find_index(freq)
        if idx is None:
            return None

        return self[idx]

    def find_index(self, freq: float, tol_hz: float = 100.0) -> int | None:
        """
        Find and return the Component whose frequency is nearest to
        the specified frequency.

        Parameters
        ----------
        freq : float
            Frequency in Hz to search for.

        Returns
        -------
        int | None
            Index of the Component with the frequency nearest to the specified
            frequency. Returns None if the list is empty.
        """
        # Return None if the list is empty
        if not self:
            return None

        # Find the index of the nearest frequency
        freqs = np.array([comp.freq for comp in self])
        idx = int(np.argmin(np.abs(freqs - freq)))

        # Return None if outside of tolerance
        if abs(freqs[idx] - freq) > tol_hz:
            return None

        return idx

    def normalize_pwr(self, ref_freq: float = 1000.0) -> None:
        """
        Normalize component powers relative to the power at a reference frequency.

        Parameters
        ----------
        ref_freq : float
            Reference frequency in Hz. The power of the component nearest to
            this frequency will be used as the reference for normalization.

        Notes
        -----
        - This method modifies the powers of the components in place.
        - If no component is found near the reference frequency, no changes
          are made.
        """
        ref_component = self.find_freq(ref_freq)
        if ref_component is None or ref_component.pwr == 0.0:
            return

        ref_pwr = ref_component.pwr
        for component in self:
            component.pwr /= ref_pwr

    def get_bands(self, centers: tuple) -> "Components":
        """
        Reduce components to those within specified frequency bands.

        Parameters
        ----------
        centers : tuple
            List of center frequencies defining frequency bands in Hz.

        Returns
        -------
        Components
            New Components instance whose components characterize the
            frequency response within the specified bands.
        """
        sqrt2 = 2.0**0.5
        bands = Components(self.md)
        for center in centers:
            low = center / sqrt2
            high = center * sqrt2
            components = Components(self.md)
            for component in self:
                if low <= component.freq <= high:
                    components.append(component)

            band = components.average()
            band.freq = center
            band.lower_freq = low
            band.upper_freq = high
            bands.append(band)

        return bands

    def band_edges(
        self, lower: float, upper: float, attn_db: float = 3.0
    ) -> tuple[float, float]:
        """Find the lower and upper frequency edges at the specified attenuation level
        relative to the average power between the edges."""
        assert lower < upper
        lower_idx = self.find_index(lower)
        upper_idx = self.find_index(upper)

        if lower_idx is None or upper_idx is None:
            return (lower, upper)

        # Calculate average power and target power at the specified attenuation
        pwr_sum = sum(self[i].pwr for i in range(lower_idx, upper_idx + 1))
        avg_pwr = pwr_sum / (upper_idx - lower_idx + 1)
        target_pwr = avg_pwr * db_to_pwr_ratio(-attn_db)

        # Find lower edge
        center_freq = math.sqrt(lower * upper)
        candidates = (
            component.freq
            for component in self
            if (lower <= component.freq <= center_freq and component.pwr <= target_pwr)
        )
        try:
            lower_edge = max(candidates)
        except ValueError:
            lower_edge = lower

        # Find upper edge
        candidates = (
            component.freq
            for component in self
            if (center_freq <= component.freq <= upper and component.pwr <= target_pwr)
        )
        try:
            upper_edge = min(candidates)
        except ValueError:
            upper_edge = upper

        return (lower_edge, upper_edge)

    def ripple(self, f1: float, f2: float) -> float:
        """
        Return the ripple (max - min) power in the specified frequency range.

        Parameters
        ----------
        f1 : float
            Start frequency, in Hz.
        f2 : float
            Stop frequency, in Hz.

        Returns
        -------
        float
            Ripple power in the specified frequency range.
        """
        pwr_values = [comp.pwr for comp in self if f1 <= comp.freq <= f2]
        if not pwr_values:
            return 0.0

        ripple = max(pwr_values) - min(pwr_values)
        return ripple

    def plot(self) -> None:
        """Plot the components as a stem plot."""
        if not self:
            print("No components to plot")
            return

        from matplotlib import pyplot as plt

        _, ax = plt.subplots()
        freqs = [comp.freq for comp in self]
        pwrs_db = [db(comp.pwr) for comp in self]
        ax.stem(freqs, pwrs_db, bottom=MIN_DB, basefmt=" ")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title(f"Components for Metadata {self.md.desc}")
        ax.grid(True)
        plt.show()

    def print(self, ref_pwr: float = 1.0, max_components: int | None = 10) -> None:
        """
        Print tones in dB relative to a reference power.

        Parameters
        ----------
        ref_pwr : float
            Reference power used for dB conversion. Values <= 0 are clipped
            to a small positive minimum to avoid log errors in db().
        """
        # Avoid passing non-positive reference to db()
        if ref_pwr <= 0.0:
            ref_pwr = MIN_PWR

        print(f"{self.name} ({self.md.segment})")
        print("----------------------------------------")

        # Print all components if max_components is None or large enough to
        # print all
        if max_components is None or max_components >= len(self):
            for component in self:
                component.print(ref_pwr)
            return

        # Print first half of components
        half_components = max_components // 2
        for component in self[:half_components]:
            component.print(ref_pwr)

        print(f"{len(self) - max_components} more ...")

        # Print second half of components
        for component in self[-half_components:]:
            component.print(ref_pwr)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representing the list using Stats.get_stats() for
        each stat in the list.
        """
        records = [component.to_dict() for component in self]
        df = pd.DataFrame.from_records(records)
        return df

    def to_csv(
        self, filepath: str, index: bool = False, print_head: bool = False
    ) -> None:
        """
        Write the object's DataFrame to the specified CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        index : bool, optional
            Whether to write row indices to the CSV file (default is False).
        """
        df = self.get_dataframe()
        p = Path(filepath)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        # Print the data frame header if specified
        if print_head:
            print(df.head())

        # Write the DataFrame to CSV with specified options
        df.to_csv(p, header=True, index=index, float_format="%.1f")
