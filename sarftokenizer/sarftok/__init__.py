"""
SarfTok — Hybrid Probabilistic Arabic Tokenizer

Canonical data models and top-level public API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


@dataclass
class MorphAnalysis:
    """A single morphological analysis with associated probability."""

    prob: float
    """Normalised probability of this analysis (sum of top-k ≤ 1.0)."""

    proclitics: List[str] = field(default_factory=list)
    """Ordered proclitics, e.g. ['wa', 'bi', 'al']."""

    root: Optional[str] = None
    """Consonantal root in Buckwalter / transliteration form, e.g. 'ktb'."""

    pattern: Optional[str] = None
    """Morphological template, e.g. 'maCCuuC'."""

    enclitics: List[str] = field(default_factory=list)
    """Ordered enclitics, e.g. ['hu', 'hum']."""

    pos: Optional[str] = None
    """Part-of-speech tag."""

    lemma: Optional[str] = None
    """Lemma form (Arabic script)."""

    is_oov_root: bool = False
    """True when root was not found in the root vocabulary."""

    is_oov_pattern: bool = False
    """True when pattern was not found in the pattern vocabulary."""

    source: str = "analyzer"
    """Name of the backend that produced this analysis."""

    def entropy_weight(self) -> float:
        """Return this analysis's probability for use in entropy calculation."""
        return max(self.prob, 1e-9)


@dataclass
class WordMorphBundle:
    """All morphological and surface information for a single Arabic word."""

    raw_word: str
    """Original (un-normalised) word string."""

    normalized_word: str
    """Normalised word string."""

    analyses: List[MorphAnalysis] = field(default_factory=list)
    """Top-k morphological analyses, sorted descending by prob."""

    surface_pieces: List[str] = field(default_factory=list)
    """Surface subword pieces (string form) for this word."""

    word_index: int = 0
    """Position of this word in the sentence."""

    @property
    def top_analysis(self) -> Optional[MorphAnalysis]:
        return self.analyses[0] if self.analyses else None

    @property
    def is_unanalyzable(self) -> bool:
        return len(self.analyses) == 0


@dataclass
class SentenceTokenization:
    """Complete tokenization output for a single sentence."""

    raw_text: str
    """Original input text."""

    normalized_text: str
    """Normalised text (after running ArabicNormalizer)."""

    words: List[WordMorphBundle]
    """Per-word bundles, in order."""

    surface_input_ids: List[int]
    """Flat list of surface token IDs for the whole sentence."""

    word_to_surface_spans: List[Tuple[int, int]]
    """
    For each word at index j, word_to_surface_spans[j] = (start, end)
    where surface_input_ids[start:end] are the subword tokens for word j.
    """


__all__ = [
    "MorphAnalysis",
    "WordMorphBundle",
    "SentenceTokenization",
]
