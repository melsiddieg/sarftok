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
        Temperature for softmax rescaling.  Only applied when *score_type*
        is ``"logit"`` (or when it is ``"prob"`` but a non-unit temperature
        is explicitly requested).
    score_type:
        Interpretation of the ``prob`` field returned by
        :meth:`_raw_analyze_word`:

        ``"prob"``   — scores are (unnormalised) probabilities/weights and are
                       L1-normalised (divided by their sum).  This preserves a
                       backend's stated relative confidences, e.g. 0.7/0.3
                       stays 0.7/0.3 rather than being flattened by a softmax.
        ``"logit"``  — scores are logits and are converted with a temperature
                       softmax.
    """

    def __init__(
        self,
        top_k: int = 3,
        confidence_threshold: float = 0.05,
        temperature: float = 1.0,
        score_type: str = "prob",
    ) -> None:
        self.top_k = max(1, top_k)
        self.confidence_threshold = confidence_threshold
        self.temperature = max(temperature, 1e-6)
        if score_type not in ("prob", "logit"):
            raise ValueError(f"score_type must be 'prob' or 'logit', got {score_type!r}")
        self.score_type = score_type

    # ------------------------------------------------------------------
    # Abstract method — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def _raw_analyze_word(self, word: str) -> list[MorphAnalysis]:
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

    def _normalise_probs(self, analyses: list[MorphAnalysis]) -> list[MorphAnalysis]:
        """Normalise raw scores to a probability distribution.

        ``score_type="prob"`` (default): L1-normalise so a backend's stated
        confidences are preserved (0.7/0.3 stays 0.7/0.3).  A non-unit
        temperature sharpens (T<1) or flattens (T>1) the categorical via
        ``p_i^(1/T)``.

        ``score_type="logit"``: temperature softmax over the raw scores.

        Then filters by ``confidence_threshold`` and re-normalises.
        """
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

        if self.score_type == "logit":
            # Temperature softmax over logits
            scaled = [p / self.temperature for p in raw_probs]
            max_s = max(scaled)
            exps = [math.exp(s - max_s) for s in scaled]
            sum_exps = sum(exps)
            for a, e in zip(analyses, exps):
                a.prob = e / sum_exps
        else:
            # L1-normalise (preserve relative confidences)
            probs = [p / total for p in raw_probs]
            if self.temperature != 1.0:
                # Categorical temperature: sharpen/flatten then re-normalise
                powered = [p ** (1.0 / self.temperature) for p in probs]
                z = sum(powered) or 1.0
                probs = [p / z for p in powered]
            for a, p in zip(analyses, probs):
                a.prob = p

        # Filter below threshold
        analyses = [a for a in analyses if a.prob >= self.confidence_threshold]

        # Re-normalise remaining
        total2 = sum(a.prob for a in analyses)
        if total2 > 0:
            for a in analyses:
                a.prob = a.prob / total2

        return analyses

    def _select_top_k(self, analyses: list[MorphAnalysis]) -> list[MorphAnalysis]:
        """Sort descending by prob and keep top_k."""
        analyses.sort(key=lambda a: a.prob, reverse=True)
        return analyses[: self.top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_word(self, word: str) -> list[MorphAnalysis]:
        """Return top-k normalised analyses for a single *word*."""
        raw = self._raw_analyze_word(word)
        normed = self._normalise_probs(raw)
        return self._select_top_k(normed)

    def analyze_sentence(
        self,
        words: list[str],
        context: str | None = None,  # noqa: ARG002 (future use for contextual analyzers)
    ) -> list[list[MorphAnalysis]]:
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
