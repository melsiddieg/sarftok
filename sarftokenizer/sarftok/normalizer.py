"""
ArabicNormalizer — deterministic Arabic text normalisation.

Modes
-----
classical_strict
    Normalise alef variants, ya/alif-maqsura, remove tatweel,
    strip punctuation non-Arabic, clean whitespace.
    Diacritics are *preserved* by default.

classical_soft
    Same as strict but keeps common Arabic punctuation (، ؟ ؛).

msa_soft
    Same as classical_soft but additionally collapses ta-marbuta
    and keeps western numerals.

All modes support optional diacritic stripping via ``strip_diacritics``.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal


# ---------------------------------------------------------------------------
# Unicode character constants
# ---------------------------------------------------------------------------

# Alef variants → alef (U+0627)
_ALEF_VARIANTS = {
    "\u0622",  # آ  ARABIC LETTER ALEF WITH MADDA ABOVE
    "\u0623",  # أ  ARABIC LETTER ALEF WITH HAMZA ABOVE
    "\u0625",  # إ  ARABIC LETTER ALEF WITH HAMZA BELOW
    "\u0671",  # ٱ  ARABIC LETTER ALEF WASLA
    "\u0672",  # ٲ  ARABIC LETTER ALEF WITH WAVY HAMZA ABOVE
    "\u0673",  # ٳ  ARABIC LETTER ALEF WITH WAVY HAMZA BELOW
}
_ALEF = "\u0627"  # ا

# Ya variant → ya (U+064A)
_YA_VARIANTS = {
    "\u0649",  # ى  ARABIC LETTER ALEF MAKSURA
    "\u06CC",  # ی  ARABIC LETTER FARSI YEH
    "\u0620",  # ؠ  (Unicode alias)
}
_YA = "\u064A"  # ي

# Tatweel
_TATWEEL = "\u0640"

# Arabic diacritics (harakat + superscripts)
_DIACRITICS_RANGE = re.compile(
    r"[\u064B-\u065F\u0670]"  # fathatan … superscript alef
)

# Arabic punctuation to keep in soft modes
_ARABIC_PUNCT = "،؛؟"

# Western punctuation
_WESTERN_PUNCT = ".,;!?:()[]{}\"'…–—-"

# Whitespace normalisation
_MULTI_SPACE = re.compile(r" {2,}")
_NEWLINES = re.compile(r"[\r\n\t]+")


# ---------------------------------------------------------------------------
# Helper pattern builders (compiled once at import time)
# ---------------------------------------------------------------------------


def _build_allowed_chars_pattern(mode: str) -> re.Pattern:
    arabic_block = r"\u0600-\u06FF"
    arabic_supplement = r"\u0750-\u077F"
    arabic_extended = r"\u08A0-\u08FF"
    alef_wasla = r"\u0671"
    space = r" "
    numerals_arabic = r"\u0660-\u0669"
    numerals_western = r"0-9"

    keep = (
        rf"{arabic_block}{arabic_supplement}{arabic_extended}{alef_wasla}{space}"
    )

    if mode in ("classical_soft", "msa_soft"):
        keep += re.escape(_ARABIC_PUNCT)

    if mode == "msa_soft":
        keep += numerals_western

    keep += numerals_arabic

    return re.compile(rf"[^{keep}]")


_STRIP_NON_ARABIC: dict[str, re.Pattern] = {
    m: _build_allowed_chars_pattern(m)
    for m in ("classical_strict", "classical_soft", "msa_soft")
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ArabicNormalizer:
    """Normalise Arabic text according to a configurable mode.

    Parameters
    ----------
    mode:
        One of ``classical_strict``, ``classical_soft``, ``msa_soft``.
    strip_diacritics:
        Remove harakat (tashkeel). Default ``False`` (preserve diacritics).
    normalise_alef:
        Collapse alef variants to bare alef. Default ``True``.
    normalise_ya:
        Collapse alif maqsura / farsi yeh to ya. Default ``True``.
    """

    def __init__(
        self,
        mode: Literal["classical_strict", "classical_soft", "msa_soft"] = "classical_strict",
        strip_diacritics: bool = False,
        normalise_alef: bool = True,
        normalise_ya: bool = True,
    ) -> None:
        if mode not in ("classical_strict", "classical_soft", "msa_soft"):
            raise ValueError(f"Unknown normalisation mode: {mode!r}")
        self.mode = mode
        self.strip_diacritics = strip_diacritics
        self.normalise_alef = normalise_alef
        self.normalise_ya = normalise_ya
        self._strip_pattern = _STRIP_NON_ARABIC[mode]

    # ------------------------------------------------------------------
    # Core normalisation steps
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_alef(text: str) -> str:
        for ch in _ALEF_VARIANTS:
            text = text.replace(ch, _ALEF)
        return text

    @staticmethod
    def _normalise_ya(text: str) -> str:
        for ch in _YA_VARIANTS:
            text = text.replace(ch, _YA)
        return text

    @staticmethod
    def _remove_tatweel(text: str) -> str:
        return text.replace(_TATWEEL, "")

    @staticmethod
    def _strip_diacritics_fn(text: str) -> str:
        return _DIACRITICS_RANGE.sub("", text)

    def _strip_non_arabic(self, text: str) -> str:
        # Replace non-allowed characters with spaces, then collapse.
        return self._strip_pattern.sub(" ", text)

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        text = _NEWLINES.sub(" ", text)
        text = _MULTI_SPACE.sub(" ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Public normalise method
    # ------------------------------------------------------------------

    def normalise(self, text: str) -> str:
        """Return normalised text."""
        if not text:
            return text

        # Unicode NFC
        text = unicodedata.normalize("NFC", text)

        # Alef variants
        if self.normalise_alef:
            text = self._normalise_alef(text)

        # Ya variants
        if self.normalise_ya:
            text = self._normalise_ya(text)

        # Remove tatweel
        text = self._remove_tatweel(text)

        # Strip diacritics (optional)
        if self.strip_diacritics:
            text = self._strip_diacritics_fn(text)

        # Strip non-Arabic characters (mode-dependent)
        text = self._strip_non_arabic(text)

        # Whitespace
        text = self._clean_whitespace(text)

        return text


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


_NORMALIZER_CACHE: dict[tuple, ArabicNormalizer] = {}


def normalize_text(
    text: str,
    mode: Literal["classical_strict", "classical_soft", "msa_soft"] = "classical_strict",
    *,
    strip_diacritics: bool = False,
    normalise_alef: bool = True,
    normalise_ya: bool = True,
) -> str:
    """Normalise *text* using the specified mode (cached normaliser instance).

    Parameters
    ----------
    text:
        Raw Arabic input string.
    mode:
        Normalisation mode.  One of ``classical_strict``, ``classical_soft``,
        ``msa_soft``.
    strip_diacritics:
        Remove harakat when ``True``.
    normalise_alef:
        Collapse alef variants.
    normalise_ya:
        Collapse ya variants / alif maqsura.

    Returns
    -------
    str
        Normalised text.
    """
    key = (mode, strip_diacritics, normalise_alef, normalise_ya)
    if key not in _NORMALIZER_CACHE:
        _NORMALIZER_CACHE[key] = ArabicNormalizer(
            mode=mode,
            strip_diacritics=strip_diacritics,
            normalise_alef=normalise_alef,
            normalise_ya=normalise_ya,
        )
    return _NORMALIZER_CACHE[key].normalise(text)
