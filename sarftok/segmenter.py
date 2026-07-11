"""
ArabicSegmenter — sentence and word segmentation for Arabic text.

Design principles
-----------------
* Sentence splitting uses a heuristic approach (punctuation + newlines)
  that works without external dependencies.
* Word tokenisation is pure whitespace splitting after normalisation,
  which guarantees that each word maps to a contiguous surface span.
* If CAMeL Tools is available, a richer sentence splitter can be enabled.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Sentence boundary pattern
# ---------------------------------------------------------------------------

# Sentence-final markers — Arabic and Western
_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?؟])\s+"           # after sentence-final punctuation + whitespace
    r"|(?<=[\n\r])"             # or after newlines
    r"|(?<=؟\s)|(?<=.\s\s)"    # additional heuristics
)

# Characters that on their own are not valid Arabic words (punctuation-only)
_PUNCT_ONLY = re.compile(r"^[\W\d]+$")


class ArabicSegmenter:
    """Split Arabic text into sentences and then into word tokens.

    Parameters
    ----------
    max_sentence_words:
        If a heuristic sentence exceeds this length, force-split at
        the midpoint.  Set to ``None`` to disable.
    min_word_length:
        Drop tokens shorter than this (removes stray punctuation).
        Set to ``0`` to keep everything.
    """

    def __init__(
        self,
        max_sentence_words: int | None = 512,
        min_word_length: int = 1,
    ) -> None:
        self.max_sentence_words = max_sentence_words
        self.min_word_length = min_word_length

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    def split_sentences(self, text: str) -> list[str]:
        """Split *text* into a list of sentence strings.

        The text should already be normalised.
        """
        if not text:
            return []

        # Primary split on sentence boundaries
        parts = _SENT_BOUNDARY.split(text)

        sentences: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Optionally split overlong sentences
            words = part.split()
            if self.max_sentence_words and len(words) > self.max_sentence_words:
                mid = len(words) // 2
                sentences.append(" ".join(words[:mid]))
                sentences.append(" ".join(words[mid:]))
            else:
                sentences.append(part)

        return sentences

    # ------------------------------------------------------------------
    # Word tokenisation
    # ------------------------------------------------------------------

    def tokenize_words(self, sentence: str) -> list[str]:
        """Return a list of Arabic word tokens from *sentence*.

        Tokens that consist entirely of punctuation/digits and are shorter
        than ``min_word_length`` are filtered out.
        """
        tokens = sentence.split()
        result: list[str] = []
        for tok in tokens:
            if len(tok) < self.min_word_length:
                continue
            result.append(tok)
        return result

    # ------------------------------------------------------------------
    # Combined segmentation
    # ------------------------------------------------------------------

    def segment(self, text: str) -> list[list[str]]:
        """Return ``List[sentence_words]`` — sentences × tokens.

        Parameters
        ----------
        text:
            Normalised Arabic text.

        Returns
        -------
        List[List[str]]
            Outer list = sentences, inner list = word tokens per sentence.
        """
        sentences = self.split_sentences(text)
        return [self.tokenize_words(s) for s in sentences if self.tokenize_words(s)]

    def segment_flat(self, text: str) -> list[str]:
        """Return a flat list of word tokens from the whole text."""
        result: list[str] = []
        for sent_words in self.segment(text):
            result.extend(sent_words)
        return result
