"""
MorphVocab — vocabulary for morphological tokens.

Maintains separate sub-vocabularies for:
  * roots            → ROOT_<root>            e.g. ROOT_ktb
  * patterns         → PAT_<template>         e.g. PAT_maCCuuC
  * proclitics       → PRO<n>_<label>         e.g. PRO1_wa
  * enclitics        → ENC_<label>            e.g. ENC_hum
  * root characters  → ROOTCHAR_<char>        e.g. ROOTCHAR_k
  * pattern classes  → PATCLASS_<class>       e.g. PATCLASS_nominal
  * morphosyntactic factors (POS, case, mood, voice, person, number,
    gender, aspect, state, and definiteness)

OOV strategies
--------------
Root OOV → try ROOTCHAR_ decomposition; never silently drop.
Pattern OOV → PATCLASS_* fallback, then PAT_UNKNOWN.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

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
_BUILTIN_PROCLITICS: list[tuple[str, int]] = [
    ("wa", 1), ("fa", 1), ("bi", 2), ("ka", 2), ("li", 2),
    ("al", 3), ("lil", 3), ("bill", 3), ("wal", 3), ("fal", 3),
    ("na", 4), ("ya", 4),
]

_BUILTIN_ENCLITICS = [
    "hu", "hi", "ha", "hum", "hinna", "hunna", "kum", "kunna",
    "na", "ni", "ka", "ki", "ya", "y", "h", "kuma", "huma", "nn",
]

FEATURE_NAMES = (
    "pos",
    "aspect",
    "voice",
    "mood",
    "person",
    "gender",
    "number",
    "case",
    "state",
    "definiteness",
)

_BUILTIN_FEATURE_VALUES: dict[str, tuple[str, ...]] = {
    "pos": ("noun", "verb", "adj", "adv", "pron", "prep", "conj", "part", "num"),
    "aspect": ("p", "i", "c", "perfect", "imperfect", "imperative"),
    "voice": ("a", "p", "active", "passive"),
    "mood": ("i", "s", "j", "u", "indicative", "subjunctive", "jussive"),
    "person": ("1", "2", "3"),
    "gender": ("m", "f", "masculine", "feminine"),
    "number": ("s", "d", "p", "singular", "dual", "plural"),
    "case": ("n", "a", "g", "u", "nominative", "accusative", "genitive"),
    "state": ("d", "i", "c", "definite", "indefinite", "construct"),
    "definiteness": ("def", "indef", "definite", "indefinite"),
}


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
        self.root_vocab: dict[str, int] = {}
        self.pattern_vocab: dict[str, int] = {}
        self.proclitic_vocab: dict[str, int] = {}
        self.enclitic_vocab: dict[str, int] = {}
        self.rootchar_vocab: dict[str, int] = {}
        self.patclass_vocab: dict[str, int] = {}
        self.feature_vocabs: dict[str, dict[str, int]] = {
            name: {UNK_TOKEN: 0} for name in FEATURE_NAMES
        }

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

        for name, values in _BUILTIN_FEATURE_VALUES.items():
            for value in values:
                if value not in self.feature_vocabs[name]:
                    self.feature_vocabs[name][value] = len(self.feature_vocabs[name])

    # ------------------------------------------------------------------
    # Building from corpus
    # ------------------------------------------------------------------

    @classmethod
    def build_from_analyses(
        cls,
        analyses_iter: Iterator[list[list[MorphAnalysis]]],
        root_min_freq: int = 5,
        pattern_min_freq: int = 3,
    ) -> MorphVocab:
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
        observed_features: dict[str, set[str]] = {name: set() for name in FEATURE_NAMES}

        for sentence_analyses in analyses_iter:
            for word_analyses in sentence_analyses:
                for a in word_analyses:
                    if a.root:
                        root_counts[a.root] += 1
                    if a.pattern:
                        pattern_counts[a.pattern] += 1
                    for name in FEATURE_NAMES:
                        value = getattr(a, name)
                        if value:
                            observed_features[name].add(value)

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

        for name, values in observed_features.items():
            for value in sorted(values):
                if value not in vocab.feature_vocabs[name]:
                    vocab.feature_vocabs[name][value] = len(vocab.feature_vocabs[name])

        return vocab

    # ------------------------------------------------------------------
    # Token ID lookup
    # ------------------------------------------------------------------

    def root_id(self, root: str) -> tuple[int | None, bool]:
        """Return (id, is_composite).

        is_composite=True  → ROOT_<root> token was found.
        is_composite=False → root-char fallback should be used.
        """
        tok = f"ROOT_{root}"
        if tok in self.root_vocab:
            return self.root_vocab[tok], True
        return None, False

    def root_char_ids(self, root: str) -> list[int]:
        """Return list of ROOTCHAR_ IDs, one per letter of root.

        Unknown characters fall back to ID 0 (first ROOTCHAR token).
        """
        ids: list[int] = []
        for ch in root:
            tok = f"{_ROOTCHAR_PREFIX}{ch}"
            ids.append(self.rootchar_vocab.get(tok, 0))
        return ids

    def pattern_id(self, pattern: str | None) -> tuple[int, str]:
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

    def proclitic_ids(self, proclitics: list[str]) -> list[int]:
        """Return IDs for a list of proclitic labels."""
        ids: list[int] = []
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

    def enclitic_ids(self, enclitics: list[str]) -> list[int]:
        """Return IDs for a list of enclitic labels."""
        ids: list[int] = []
        for enc in enclitics:
            tok = f"ENC_{enc}"
            ids.append(self.enclitic_vocab.get(tok, 0))
        return ids

    def feature_id(self, name: str, value: str | None) -> int:
        """Return an ID for a categorical morphosyntactic feature."""
        if name not in self.feature_vocabs:
            raise KeyError(f"Unknown morphology feature: {name!r}")
        return self.feature_vocabs[name].get(value or UNK_TOKEN, 0)

    def num_feature_values(self, name: str) -> int:
        return max(len(self.feature_vocabs[name]), 1)

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
            "feature_vocabs": self.feature_vocabs,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> MorphVocab:
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
        saved_features = d.get("feature_vocabs", {})
        for name in FEATURE_NAMES:
            if name in saved_features:
                vocab.feature_vocabs[name] = saved_features[name]
        return vocab
