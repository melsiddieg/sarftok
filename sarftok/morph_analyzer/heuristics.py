"""
HeuristicMorphAnalyzer — rule-based Arabic morphology backend.

This is the default backend when CAMeL Tools is not available.
It uses:
* prefix/suffix tables for common proclitics and enclitics
* consonantal skeleton extraction as a root proxy
* coverage of the most productive Classical Arabic patterns

Design philosophy
-----------------
* Never claim high confidence — always returns at most 0.8 probability.
* Provides at least one analysis (fallback: all-null) so the pipeline
  always has *something* to work with.
* No external dependencies beyond the Python standard library.
"""
from __future__ import annotations

import re

from sarftok import MorphAnalysis
from sarftok.morph_analyzer.interface import MorphAnalyzer

# ---------------------------------------------------------------------------
# Arabic character sets
# ---------------------------------------------------------------------------

_ARABIC_CONSONANTS = set("بتثجحخدذرزسشصضطظعغفقكلمنهوي")
_LONG_VOWELS = set("اوي")

# Diacritics range
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

# ---------------------------------------------------------------------------
# Proclitic table
# (prefix string → canonical label)
# ---------------------------------------------------------------------------

_PROCLITICS: list[tuple[str, str]] = [
    ("وَبِالْ", "wa+bi+al"),
    ("وَبِ", "wa+bi"),
    ("وَلِلْ", "wa+li+al"),
    ("وَلِ", "wa+li"),
    ("وَالْ", "wa+al"),
    ("وَ", "wa"),
    ("فَبِالْ", "fa+bi+al"),
    ("فَبِ", "fa+bi"),
    ("فَلِلْ", "fa+li+al"),
    ("فَلِ", "fa+li"),
    ("فَالْ", "fa+al"),
    ("فَ", "fa"),
    ("بِالْ", "bi+al"),
    ("بِ", "bi"),
    ("كَالْ", "ka+al"),
    ("كَ", "ka"),
    ("لِلْ", "li+al"),
    ("لِ", "li"),
    ("الْ", "al"),
    ("ال", "al"),
]

_PROCLITIC_LABELS: dict[str, list[str]] = {
    v: v.split("+") for v in set(v for _, v in _PROCLITICS)
}

# ---------------------------------------------------------------------------
# Enclitic table
# (suffix string → canonical label)
# ---------------------------------------------------------------------------

_ENCLITICS: list[tuple[str, str]] = [
    ("كُمَا", "kuma"),
    ("هُمَا", "huma"),
    ("كُمْ", "kum"),
    ("هُمْ", "hum"),
    ("هِنَّ", "hinna"),
    ("كُنَّ", "kunna"),
    ("هُنَّ", "hunna"),
    ("نَا", "na"),
    ("نِي", "ni"),
    ("نَّ", "nn"),
    ("هَا", "ha"),
    ("هِ", "hi"),
    ("هُ", "hu"),
    ("كَ", "ka"),
    ("كِ", "ki"),
    ("يَ", "ya"),
    ("ي", "y"),
    ("ه", "h"),
]

# ---------------------------------------------------------------------------
# Common classical Arabic patterns (Buckwalter-like template notation)
# ---------------------------------------------------------------------------

_PATTERNS = [
    "CaCaCa",       # فَعَلَ
    "CaCiCa",       # فَعِلَ
    "CaCuCa",       # فَعُلَ
    "CaaCiC",       # فَاعِل
    "maCCaCa",      # مَفْعَلَة
    "maCCuuC",      # مَفْعُول
    "faCCaala",     # فَعَّالة
    "faCaaCiC",     # فَعَاعِل  (broken plural)
    "CuCuuC",       # فُعُول
    "CiCaaCa",      # فِعَالة
    "CaCaC",        # فَعَل (masdar)
    "taCaCCuC",     # تَفَعُّل (V form masdar)
    "inCiCaaC",     # اِنْفِعَال (VII masdar)
    "iCtiCaaC",     # اِفْتِعَال (VIII masdar)
    "iCCiCaaC",     # اِفْعِلَال (IX masdar)
    "istaCCaC",     # اِسْتَفْعَل (X imperf)
    "CaaCiCa",      # فَاعِلَة  (fem)
    "CiiC",         # فِيل
    "CawaaCiC",     # فَوَاعِل (broken plural)
    "aCCaaC",       # أَفْعَال (broken plural)
]

# Score patterns by commonality (higher = more likely)
_PATTERN_SCORES: dict[str, float] = {
    p: 1.0 - 0.04 * i for i, p in enumerate(_PATTERNS)
}


# ---------------------------------------------------------------------------
# Consonantal skeleton extraction
# ---------------------------------------------------------------------------


def _strip_diacritics(text: str) -> str:
    return _DIACRITICS.sub("", text)


def _consonantal_skeleton(word: str) -> str:
    """Extract the consonant skeleton from an Arabic word."""
    bare = _strip_diacritics(word)
    skel = "".join(c for c in bare if c in _ARABIC_CONSONANTS)
    return skel


def _skeleton_to_root(skel: str) -> str | None:
    """Guess a root from a consonantal skeleton.

    * 3-letter skeleton → direct root
    * 4-letter skeleton → quadriliteral root
    * others → None (OOV)
    """
    if len(skel) in (3, 4):
        return skel
    if len(skel) > 4:
        # Attempt to reduce by removing likely weak/geminate consonants
        # (simple heuristic: drop last repeated consonant or weak letters)
        compact = []
        for c in skel:
            if not compact or compact[-1] != c:
                compact.append(c)
        if len(compact) in (3, 4):
            return "".join(compact)
    return None


# ---------------------------------------------------------------------------
# HeuristicMorphAnalyzer
# ---------------------------------------------------------------------------


class HeuristicMorphAnalyzer(MorphAnalyzer):
    """Rule-based morphological analyzer for Arabic.

    This backend works without any external tools.  It is intended as a
    reliable fallback and for unit testing.  Accuracy is intentionally
    limited; the CAMeL or distilled backend should be used for production.

    Parameters
    ----------
    top_k, confidence_threshold, temperature:
        Inherited from :class:`~sarftok.morph_analyzer.interface.MorphAnalyzer`.
    """

    def __init__(
        self,
        top_k: int = 3,
        confidence_threshold: float = 0.05,
        temperature: float = 1.0,
        score_type: str = "prob",
    ) -> None:
        super().__init__(
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            temperature=temperature,
            score_type=score_type,
        )

    # ------------------------------------------------------------------
    # Proclitic analysis
    # ------------------------------------------------------------------

    def _strip_proclitics(self, word: str) -> tuple[list[str], str]:
        """Return (proclitic_labels, stem) by stripping known prefixes.

        Both diacritised and undiacritised matching require at least 3
        undiacritised characters remaining after stripping, so that a
        triliteral root can still be extracted from the stem.
        """
        # Try diacritised matching first
        for prefix, label in _PROCLITICS:
            if word.startswith(prefix):
                remaining = word[len(prefix):]
                remaining_nd = _strip_diacritics(remaining)
                if len(remaining_nd) >= 3:
                    return label.split("+"), remaining
        # Try undiacritised matching.  Slice by the *undiacritised* prefix
        # length: when the input carries no diacritics, using the diacritised
        # prefix length would over-slice and drop a root consonant
        # (e.g. الكتاب → تاب instead of كتاب).
        bare_nd = _strip_diacritics(word)
        for prefix, label in _PROCLITICS:
            p_nd = _strip_diacritics(prefix)
            if bare_nd.startswith(p_nd):
                remaining_len = len(bare_nd) - len(p_nd)
                if remaining_len >= 3:
                    return label.split("+"), word[len(p_nd):]
        return [], word

    def _strip_enclitics(self, stem: str) -> tuple[str, list[str]]:
        """Return (core, enclitic_labels) by stripping known suffixes."""
        for suffix, label in _ENCLITICS:
            if stem.endswith(suffix) and len(stem) > len(suffix) + 1:
                return stem[: -len(suffix)], [label]
        # Try without diacritics
        stem_nd = _strip_diacritics(stem)
        for suffix, label in _ENCLITICS:
            s_nd = _strip_diacritics(suffix)
            if stem_nd.endswith(s_nd) and len(stem_nd) > len(s_nd) + 1:
                return stem[: -len(suffix)], [label]
        return stem, []

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _guess_pattern(self, core: str, skel: str) -> str | None:
        """Pick a plausible pattern based on skeleton length and word shape."""
        n = len(skel)
        bare = _strip_diacritics(core)

        # Form I triliteral — CaCaCa / CaCiCa
        if n == 3:
            if re.match(r"^.\u064E.\u064E.$", core):
                return "CaCaCa"
            if re.match(r"^.\u064E.\u0650.$", core):  # \u0650 = kasra (the -i- vowel)
                return "CaCiCa"
            if bare.startswith("\u0645"):
                return "maCCaCa"
            return "CaCaCa"  # default triliteral

        # Quadriliteral
        if n == 4:
            return "faCaaCiC"

        return None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _raw_analyze_word(self, word: str) -> list[MorphAnalysis]:
        if not word:
            return []

        proclitics, stem_with_enc = self._strip_proclitics(word)
        core, enclitics = self._strip_enclitics(stem_with_enc)

        skel = _consonantal_skeleton(core)
        root = _skeleton_to_root(skel)
        pattern = self._guess_pattern(core, skel)

        is_oov_root = root is None
        is_oov_pattern = pattern is None

        # Build primary analysis
        base_prob = 0.7 if root else 0.3

        primary = MorphAnalysis(
            prob=base_prob,
            proclitics=proclitics,
            root=root,
            pattern=pattern,
            enclitics=enclitics,
            is_oov_root=is_oov_root,
            is_oov_pattern=is_oov_pattern,
            source="heuristic",
        )

        analyses: list[MorphAnalysis] = [primary]

        # Add a lower-confidence alternative if we have a root
        # (e.g. passive / verbal noun interpretation)
        if root and not is_oov_root:
            alt_pattern = "maCCuuC" if pattern != "maCCuuC" else "CaaCiC"
            alt = MorphAnalysis(
                prob=0.3,
                proclitics=proclitics,
                root=root,
                pattern=alt_pattern,
                enclitics=enclitics,
                is_oov_root=False,
                is_oov_pattern=False,
                source="heuristic",
            )
            analyses.append(alt)

        return analyses
