"""
MorphAnalyzer — abstract base interface for all morphological analyzer backends.

All backends must:
* accept a pre-tokenised list of Arabic words
* return a list-of-lists of MorphAnalysis objects  (one inner list per word)
* normalise raw backend output to the canonical MorphAnalysis schema
* honour top_k and confidence_threshold settings
* guarantee probabilities sum to ≤ 1 within each word's analysis list
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional

from sarftok import MorphAnalysis


class MorphAnalyzer(ABC):
    """Abstract morphological analyzer.

    Subclasses must implement :meth:`_raw_analyze_sentence` which returns
    a raw backend-specific structure.  This class handles normalisation,
    top-k selection, and probability calibration.

    Parameters
    ----------
    top_k:
        Maximum number of analyses to return per word.
    confidence_threshold:
        Analyses with probability below this value (after normalisation)
        are discarded.
    temperature:
        Temperature for softmax rescaling when converting raw scores to probs.
    """

    def __init__(
        self,
        top_k: int = 3,
        confidence_threshold: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        self.top_k = max(1, top_k)
        self.confidence_threshold = confidence_threshold
        self.temperature = max(temperature, 1e-6)

    # ------------------------------------------------------------------
    # Abstract method — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def _raw_analyze_word(self, word: str) -> List[MorphAnalysis]:
        """Return raw analyses for a single *word*.

        Concrete implementations should set ``prob`` to any non-negative
        score; this base class will normalise them.

        Returns
        -------
        List[MorphAnalysis]
            May be empty if the word is unanalyzable.
        """

    # ------------------------------------------------------------------
    # Normalisation pipeline
    # ------------------------------------------------------------------

    def _normalise_probs(self, analyses: List[MorphAnalysis]) -> List[MorphAnalysis]:
        """Apply temperature, softmax-normalise, filter by threshold, re-normalise."""
        if not analyses:
            return analyses

        raw_probs = [a.prob for a in analyses]
        total = sum(raw_probs)

        if total <= 0:
            # Assign uniform scores then normalise
            n = len(analyses)
            for a in analyses:
                a.prob = 1.0 / n
            return analyses

        # Temperature scaling: apply to log-odds and re-softmax
        if self.temperature != 1.0:
            scaled = [p / self.temperature for p in raw_probs]
        else:
            scaled = list(raw_probs)

        # Softmax
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        sum_exps = sum(exps)
        for a, e in zip(analyses, exps):
            a.prob = e / sum_exps

        # Filter below threshold
        analyses = [a for a in analyses if a.prob >= self.confidence_threshold]

        # Re-normalise remaining
        total2 = sum(a.prob for a in analyses)
        if total2 > 0:
            for a in analyses:
                a.prob = a.prob / total2

        return analyses

    def _select_top_k(self, analyses: List[MorphAnalysis]) -> List[MorphAnalysis]:
        """Sort descending by prob and keep top_k."""
        analyses.sort(key=lambda a: a.prob, reverse=True)
        return analyses[: self.top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_word(self, word: str) -> List[MorphAnalysis]:
        """Return top-k normalised analyses for a single *word*."""
        raw = self._raw_analyze_word(word)
        normed = self._normalise_probs(raw)
        return self._select_top_k(normed)

    def analyze_sentence(
        self,
        words: List[str],
        context: Optional[str] = None,  # noqa: ARG002 (future use for contextual analyzers)
    ) -> List[List[MorphAnalysis]]:
        """Return analyses for every word in *words*.

        Parameters
        ----------
        words:
            Pre-tokenised list of Arabic word strings.
        context:
            Full sentence string (reserved for context-sensitive backends).

        Returns
        -------
        List[List[MorphAnalysis]]
            One inner list per word.  Empty inner list = unanalyzable.
        """
        return [self.analyze_word(w) for w in words]
