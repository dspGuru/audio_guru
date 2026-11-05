"""Audio tone test."""

import numpy as np

from analyzer import Analyzer
from component import Component, Components
from constants import (
    MAX_BINS_PER_TONE,
    MAX_DIST_ORDER,
    MAX_HARM_ORDER,
    MIN_SPUR_RATIO,
    MIN_PWR,
    TWO_TONE_THRESHOLD_MIN_DB,
    TWO_TONE_THRESHOLD_MAX_DB,
)
from tone_stats import ToneStats
from util import Category, alias_freq, is_harmonic

__all__ = ["ToneAnalyzer"]


class ToneAnalyzer(Analyzer):
    """
    Tone test

    Analyzes single-tone or two-tone audio signals by identifying tones,
    distortion products, spurs, and noise, and calculating the resulting
    audio statistics.

    Attributes
    ----------
        audio : Audio
            Audio data to analyze.
        tone : Component
            Primary tone component.
        tone2 : Component
            Secondary tone component (for two-tone signals).
        components : Components
            All identified components in the audio signal.
        _two_tone_threshold : float
            Adaptive two-tone detection ratio based on noise floor.
        _spur_threshold : float
            Adaptive spur detection ratio based on noise floor.
    """

    # Note: __init__() is inherited from super class

    def reset(self) -> None:
        """Resets/creates members which are calculated from audio"""
        super().reset()
        self.tone = Component()
        self.tone2 = Component()
        self._compute_adaptive_thresholds()

    def _compute_adaptive_thresholds(self) -> None:
        """Compute adaptive detection thresholds based on measured noise floor.

        Adjusts MIN_TWO_TONE_RATIO and MIN_SPUR_RATIO based on the audio's
        noise floor to improve detection sensitivity when noise is low,
        and reduce false positives when noise is high.
        """
        nf = max(self.audio.avg_noise_floor**2, MIN_PWR)

        # Two-tone ratio: higher SNR (low noise) uses lower threshold (-20 dB),
        # while lower SNR (high noise) raises threshold toward -10 dB.
        if self.tone.pwr > 0:
            snr_db = 10.0 * np.log10(self.tone.pwr / nf)
            two_tone_db = float(
                np.clip(
                    TWO_TONE_THRESHOLD_MAX_DB - (snr_db / 6.0),
                    TWO_TONE_THRESHOLD_MIN_DB,
                    TWO_TONE_THRESHOLD_MAX_DB,
                )
            )
        else:
            two_tone_db = TWO_TONE_THRESHOLD_MIN_DB
        self._two_tone_threshold = 10.0 ** (two_tone_db / 10.0)

        # Spur ratio: refine based on noise floor
        # Range: -80 dB (1e-8, default) to -70 dB (1e-7, higher noise)
        nf_above_min_db = max(0.0, 10.0 * np.log10(nf / MIN_PWR))
        self._spur_threshold = float(
            np.clip(
                MIN_SPUR_RATIO * (10.0 ** (nf_above_min_db / 60.0)),
                MIN_SPUR_RATIO,
                10.0 * MIN_SPUR_RATIO,
            )
        )

    # @override
    def get_components(self) -> Components:
        """
        Calculate and return audio tone components.

        Returns
        -------
        Components
            Calculated components.
        """

        # Get frequency components from frequency bins
        components = self.audio.bins.tones

        # Initialize distortion frequency list
        dist_freqs: list[float] = []

        # The fundamental signal frequency is the bin with the largest power,
        # which is at index zero
        components[0].cat = Category.Tone
        self.tone = components[0]
        components.ref_pwr = self.tone.pwr
        self._compute_adaptive_thresholds()
        start = 1

        # If there is more than one component, compare power and frequency of
        # the second-largest power with those of the first to determine if this
        # is a two-tone signal
        if len(components) > 1:
            harmonic = alias_freq(2.0 * components[0].freq, self.fs)

            if (components[1].pwr < self._two_tone_threshold * components[0].pwr) or (
                self.audio.bins.freq_to_bin(abs(components[1].freq - harmonic))
                < MAX_BINS_PER_TONE
            ):
                # Second-largest is a small or a harmonic: single-tone signal
                dist_freqs = [
                    alias_freq(n * self.tone.freq, self.fs)
                    for n in range(2, MAX_HARM_ORDER + 1)
                ]
                self.tone2 = Component(0.0, 0.0)  # flag as invalid
            else:
                # Second-largest is large and not a harmonic: two-tone signal
                components[1].cat = Category.Tone
                self.tone2 = components[1]
                start = 2

                # Calculate distortion frequencies for two-tone signal
                (f1, f2) = (self.tone.freq, self.tone2.freq)
                for n in range(MAX_DIST_ORDER):
                    for n2 in range(MAX_DIST_ORDER):
                        if n + n2 <= 1:
                            continue
                        dist_freqs.append(n * f1 + n2 * f2)
                        dist_freqs.append(n * f1 - n2 * f2)
                        dist_freqs.append(-n * f1 + n2 * f2)

        # Categorize remaining components
        for comp in components[start:]:
            if (is_harmonic(comp.freq, self.tone.freq, self.fs, MAX_HARM_ORDER)) or (
                is_harmonic(comp.freq, self.tone2.freq, self.fs, MAX_HARM_ORDER)
            ):
                comp.cat = Category.Harmonic
            elif self.audio.bins.freq_in_list(comp.freq, dist_freqs, MAX_BINS_PER_TONE):
                comp.cat = Category.Distortion
            elif comp.pwr > self.tone.pwr * self._spur_threshold:
                comp.cat = Category.Spurious
            elif (
                self.tone2.freq > 0.0
                and comp.pwr > self.tone2.pwr * self._spur_threshold
            ):
                comp.cat = Category.Spurious
            else:
                comp.cat = Category.Noise

        # Recategorize noise components as spurs which are harmonics of spur components
        spur_freqs = [tone.freq for tone in components if tone.cat == Category.Spurious]
        for idx, comp in enumerate(components):
            if comp.cat == Category.Noise:
                for freq in spur_freqs:
                    if is_harmonic(comp.freq, freq, self.fs, MAX_HARM_ORDER):
                        components[idx].cat = Category.Spurious

        components.name = "Tone Components"
        self.components = components
        return components

    # @override
    def analyze(self) -> ToneStats:
        """
        Calculate and return statistics from audio analysis.

        Returns
        -------
        ToneStats
            Audio tone statistics result.
        """

        # Get components via super class analyze()
        super().analyze()

        # Calculate spurious, distortion, and noise from tones, based on their
        # categories and frequencies
        spur_pwr = 0.0
        dist_pwr = 0.0
        bins_noise = getattr(self.audio.bins, "noise", 0.0)
        noise_pwr = (
            float(bins_noise)
            if isinstance(bins_noise, (int, float, np.floating))
            else 0.0
        )
        for component in self.components:
            match component.cat:
                case Category.Spurious:
                    spur_pwr = max(spur_pwr, component.pwr)
                case Category.Harmonic | Category.Distortion:
                    dist_pwr += component.pwr
                case Category.Noise:
                    noise_pwr += component.pwr
                case _:
                    pass

        # Set frequency statistics from powers and return stats
        self.analysis = ToneStats(
            self.audio,
            self.tone.freq,
            self.tone2.freq,
            self.tone.pwr + (self.tone2.pwr if self.tone2.freq > 0.0 else 0.0),
            noise_pwr,
            spur_pwr,
            dist_pwr,
        )
        return self.analysis

    def to_csv(
        self, filepath: str, index: bool = False, print_head: bool = False
    ) -> None:
        """
        Export analysis results to a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        index : bool
            Whether to include the index in the CSV output (default is False).
        print_head : bool
            Whether to print the CSV header (default is False).
        """
        tone_stats: ToneStats = self.analyze()
        tone_stats.to_csv(filepath, index, print_head)
