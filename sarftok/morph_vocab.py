"""
MorphVocab — vocabulary for morphological tokens.

Maintains separate sub-vocabularies for:
  * roots            → ROOT_<root>            e.g. ROOT_ktb
  * patterns         → PAT_<template>         e.g. PAT_maCCuuC
  * proclitics       → PRO<n>_<label>         e.g. PRO1_wa
  * enclitics        → ENC_<label>            e.g. ENC_hum
  * root characters  → ROOTCHAR_<char>        e.g. ROOTCHAR_k
  * pattern classes  → PATCLASS_<class>       e.g. PATCLASS_nominal

OOV strategies
--------------
Root OOV → try ROOTCHAR_ decomposition; never silently drop.
Pattern OOV → PATCLASS_* fallback, then PAT_UNKNOWN.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from sarftok import MorphAnalysis

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAT_UNKNOWN = "PAT_UNKNOWN"

# Built-in root character set (Arabic consonants + common transliteration)
_ROOTCHARS = list("بتثجحخدذرزسشصضطظعغفقكلمنهويء")
_ROOTCHAR_PREFIX = "ROOTCHAR_"

# Built-in pattern classes
_PATTERN_CLASSES = [
    "PATCLASS_verb_form_I",
    "PATCLASS_verb_form_II",
    "PATCLASS_verb_form_III",
    "PATCLASS_verb_form_IV",
    "PATCLASS_verb_form_V",
    "PATCLASS_verb_form_VI",
    "PATCLASS_verb_form_VII",
    "PATCLASS_verb_form_VIII",
    "PATCLASS_verb_form_IX",
    "PATCLASS_verb_form_X",
    "PATCLASS_nominal",
    "PATCLASS_participle",
    "PATCLASS_masdar",
    "PATCLASS_broken_plural",
    "PATCLASS_quadriliteral",
    "PATCLASS_unknown",
    PAT_UNKNOWN,
]

# Proclitic ordering labels (match heuristics.py)
_BUILTIN_PROCLITICS: List[Tuple[str, int]] = [
    ("wa", 1), ("fa", 1), ("bi", 2), ("ka", 2), ("li", 2),
    ("al", 3), ("lil", 3), ("bill", 3), ("wal", 3), ("fal", 3),
    ("na", 4), ("ya", 4),
]

_BUILTIN_ENCLITICS = [
    "hu", "hi", "ha", "hum", "hinna", "hunna", "kum", "kunna",
    "na", "ni", "ka", "ki", "ya", "y", "h", "kuma", "huma", "nn",
]


# ---------------------------------------------------------------------------
# MorphVocab
# ---------------------------------------------------------------------------


class MorphVocab:
    """Morphological token vocabulary with index-lookup tables.

    Parameters
    ----------
    root_min_freq:
        Minimum corpus count for a root to get its own composite token.
    pattern_min_freq:
        Minimum corpus count for a pattern to get its own token.
    """

    def __init__(
        self,
        root_min_freq: int = 5,
        pattern_min_freq: int = 3,
    ) -> None:
        self.root_min_freq = root_min_freq
        self.pattern_min_freq = pattern_min_freq

        # Token → integer ID maps
        self.root_vocab: Dict[str, int] = {}
        self.pattern_vocab: Dict[str, int] = {}
        self.proclitic_vocab: Dict[str, int] = {}
        self.enclitic_vocab: Dict[str, int] = {}
        self.rootchar_vocab: Dict[str, int] = {}
        self.patclass_vocab: Dict[str, int] = {}

        self._init_builtins()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_builtins(self) -> None:
        # Root chars
        for char in _ROOTCHARS:
            tok = f"{_ROOTCHAR_PREFIX}{char}"
            if tok not in self.rootchar_vocab:
                self.rootchar_vocab[tok] = len(self.rootchar_vocab)

        # Pattern classes
        for cls in _PATTERN_CLASSES:
            self.patclass_vocab[cls] = len(self.patclass_vocab)

        # Proclitics
        for label, order in _BUILTIN_PROCLITICS:
            tok = f"PRO{order}_{label}"
            if tok not in self.proclitic_vocab:
                self.proclitic_vocab[tok] = len(self.proclitic_vocab)

        # Enclitics
        for enc in _BUILTIN_ENCLITICS:
            tok = f"ENC_{enc}"
            if tok not in self.enclitic_vocab:
                self.enclitic_vocab[tok] = len(self.enclitic_vocab)

    # ------------------------------------------------------------------
    # Building from corpus
    # ------------------------------------------------------------------

    @classmethod
    def build_from_analyses(
        cls,
        analyses_iter: Iterator[List[List[MorphAnalysis]]],
        root_min_freq: int = 5,
        pattern_min_freq: int = 3,
    ) -> "MorphVocab":
        """Build vocabulary by counting root/pattern frequencies in corpus.

        Parameters
        ----------
        analyses_iter:
            Iterator over sentences; each sentence is List[List[MorphAnalysis]].
        root_min_freq, pattern_min_freq:
            Minimum frequency thresholds.

        Returns
        -------
        MorphVocab
        """
        root_counts: Counter = Counter()
        pattern_counts: Counter = Counter()

        for sentence_analyses in analyses_iter:
            for word_analyses in sentence_analyses:
                for a in word_analyses:
                    if a.root:
                        root_counts[a.root] += 1
                    if a.pattern:
                        pattern_counts[a.pattern] += 1

        vocab = cls(root_min_freq=root_min_freq, pattern_min_freq=pattern_min_freq)

        for root, count in root_counts.items():
            if count >= root_min_freq:
                tok = f"ROOT_{root}"
                if tok not in vocab.root_vocab:
                    vocab.root_vocab[tok] = len(vocab.root_vocab)

        for pattern, count in pattern_counts.items():
            if count >= pattern_min_freq:
                tok = f"PAT_{pattern}"
                if tok not in vocab.pattern_vocab:
                    vocab.pattern_vocab[tok] = len(vocab.pattern_vocab)

        return vocab

    # ------------------------------------------------------------------
    # Token ID lookup
    # ------------------------------------------------------------------

    def root_id(self, root: str) -> Tuple[Optional[int], bool]:
        """Return (id, is_composite).

        is_composite=True  → ROOT_<root> token was found.
        is_composite=False → root-char fallback should be used.
        """
        tok = f"ROOT_{root}"
        if tok in self.root_vocab:
            return self.root_vocab[tok], True
        return None, False

    def root_char_ids(self, root: str) -> List[int]:
        """Return list of ROOTCHAR_ IDs, one per letter of root.

        Unknown characters fall back to ID 0 (first ROOTCHAR token).
        """
        ids: List[int] = []
        for ch in root:
            tok = f"{_ROOTCHAR_PREFIX}{ch}"
            ids.append(self.rootchar_vocab.get(tok, 0))
        return ids

    def pattern_id(self, pattern: Optional[str]) -> Tuple[int, str]:
        """Return (id, token_used).

        Falls back to patclass, then PAT_UNKNOWN.
        """
        if pattern:
            tok = f"PAT_{pattern}"
            if tok in self.pattern_vocab:
                return self.pattern_vocab[tok], tok
        # Patclass fallback (unknown for now — subclasses can refine)
        fallback = PAT_UNKNOWN
        return self.patclass_vocab.get(fallback, 0), fallback

    def proclitic_ids(self, proclitics: List[str]) -> List[int]:
        """Return IDs for a list of proclitic labels."""
        ids: List[int] = []
        for order_or_label in proclitics:
            # Try ordered form first (e.g. PRO1_wa), then unordered
            found = False
            for order in range(1, 5):
                tok = f"PRO{order}_{order_or_label}"
                if tok in self.proclitic_vocab:
                    ids.append(self.proclitic_vocab[tok])
                    found = True
                    break
            if not found:
                # Try direct match
                tok = f"PRO1_{order_or_label}"
                ids.append(self.proclitic_vocab.get(tok, 0))
        return ids

    def enclitic_ids(self, enclitics: List[str]) -> List[int]:
        """Return IDs for a list of enclitic labels."""
        ids: List[int] = []
        for enc in enclitics:
            tok = f"ENC_{enc}"
            ids.append(self.enclitic_vocab.get(tok, 0))
        return ids

    # ------------------------------------------------------------------
    # Sizes (for nn.Embedding initialisation)
    # ------------------------------------------------------------------

    @property
    def num_roots(self) -> int:
        return max(len(self.root_vocab), 1)

    @property
    def num_patterns(self) -> int:
        return max(len(self.pattern_vocab), 1)

    @property
    def num_proclitics(self) -> int:
        return max(len(self.proclitic_vocab), 1)

    @property
    def num_enclitics(self) -> int:
        return max(len(self.enclitic_vocab), 1)

    @property
    def num_rootchars(self) -> int:
        return max(len(self.rootchar_vocab), 1)

    @property
    def num_patclasses(self) -> int:
        return max(len(self.patclass_vocab), 1)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "root_min_freq": self.root_min_freq,
            "pattern_min_freq": self.pattern_min_freq,
            "root_vocab": self.root_vocab,
            "pattern_vocab": self.pattern_vocab,
            "proclitic_vocab": self.proclitic_vocab,
            "enclitic_vocab": self.enclitic_vocab,
            "rootchar_vocab": self.rootchar_vocab,
            "patclass_vocab": self.patclass_vocab,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "MorphVocab":
        d = json.loads(Path(path).read_text())
        vocab = cls(
            root_min_freq=d.get("root_min_freq", 5),
            pattern_min_freq=d.get("pattern_min_freq", 3),
        )
        # Overwrite with saved vocabularies
        vocab.root_vocab = d.get("root_vocab", {})
        vocab.pattern_vocab = d.get("pattern_vocab", {})
        vocab.proclitic_vocab = d.get("proclitic_vocab", vocab.proclitic_vocab)
        vocab.enclitic_vocab = d.get("enclitic_vocab", vocab.enclitic_vocab)
        vocab.rootchar_vocab = d.get("rootchar_vocab", vocab.rootchar_vocab)
        vocab.patclass_vocab = d.get("patclass_vocab", vocab.patclass_vocab)
        return vocab
