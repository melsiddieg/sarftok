"""
CamelMorphAnalyzer — CAMeL Tools morphological analyzer backend.

This is an optional backend that requires:
    pip install camel-tools

It wraps the CAMeL Tools Analyzer and maps its output fields to the
canonical SarfTok MorphAnalysis schema.

If camel-tools is not installed, importing this module raises ImportError
with a helpful message.
"""
from __future__ import annotations

from typing import List, Optional

from sarftok import MorphAnalysis
from sarftok.morph_analyzer.interface import MorphAnalyzer

try:
    from camel_tools.morphology.analyzer import Analyzer as CamelAnalyzer  # type: ignore
    from camel_tools.morphology.database import MorphologyDB  # type: ignore

    _CAMEL_AVAILABLE = True
except ImportError:
    _CAMEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

# CAMeL Tools proclitic feature keys (in order: wa, fa, bi, ka, li, al, ...)
_PROCLITIC_KEYS = ["prc3", "prc2", "prc1", "prc0"]
_ENCLITIC_KEYS = ["enc0"]


def _extract_clitics(analysis: dict, keys: List[str]) -> List[str]:
    """Extract non-null clitic values from a CAMeL analysis dict."""
    result: List[str] = []
    for k in keys:
        v = analysis.get(k, "")
        if v and v not in ("", "0", "na"):
            result.append(v)
    return result


def _get_root(analysis: dict) -> Optional[str]:
    root = analysis.get("root", "")
    if not root or root in ("-", "NOAN", "na"):
        return None
    return root


def _get_pattern(analysis: dict) -> Optional[str]:
    pattern = analysis.get("pattern", "")
    if not pattern or pattern in ("-", "NOAN", "na"):
        return None
    return pattern


def _get_pos(analysis: dict) -> Optional[str]:
    return analysis.get("pos") or None


def _get_lemma(analysis: dict) -> Optional[str]:
    return analysis.get("lex") or analysis.get("lemma") or None


def _get_prob(analysis: dict, index: int, n_analyses: int) -> float:
    """Assign a proxy score when CAMeL does not report confidence."""
    # CAMeL sometimes attaches 'count' or 'freq'; fall back to rank-based score.
    count = analysis.get("count", None)
    if count is not None:
        try:
            return float(count)
        except (ValueError, TypeError):
            pass
    # Rank-based: higher rank → higher score
    return float(n_analyses - index)


# ---------------------------------------------------------------------------
# CamelMorphAnalyzer
# ---------------------------------------------------------------------------


class CamelMorphAnalyzer(MorphAnalyzer):
    """Morphological analyzer backed by CAMeL Tools.

    Parameters
    ----------
    db_name:
        Name of the CAMeL Tools morphology database to load.
        Default: ``"calima-msa-r13"`` (MSA database).
        Use ``"calima-clx-r13"`` for classical Arabic.
    top_k, confidence_threshold, temperature:
        Inherited from :class:`~sarftok.morph_analyzer.interface.MorphAnalyzer`.
    """

    def __init__(
        self,
        db_name: str = "calima-msa-r13",
        top_k: int = 3,
        confidence_threshold: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        if not _CAMEL_AVAILABLE:
            raise ImportError(
                "camel-tools is required for CamelMorphAnalyzer.\n"
                "Install it with:  pip install camel-tools\n"
                "Then download the data:  camel_data -i morphology-db-msa-r13"
            )
        super().__init__(
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            temperature=temperature,
        )
        db = MorphologyDB.builtin_db(db_name)
        self._analyzer = CamelAnalyzer(db)

    def _raw_analyze_word(self, word: str) -> List[MorphAnalysis]:
        try:
            raw_analyses = self._analyzer.analyze(word)
        except Exception:  # noqa: BLE001
            return []

        result: List[MorphAnalysis] = []
        n = len(raw_analyses)
        for idx, raw in enumerate(raw_analyses):
            d = dict(raw)
            result.append(
                MorphAnalysis(
                    prob=_get_prob(d, idx, n),
                    proclitics=_extract_clitics(d, _PROCLITIC_KEYS),
                    root=_get_root(d),
                    pattern=_get_pattern(d),
                    enclitics=_extract_clitics(d, _ENCLITIC_KEYS),
                    pos=_get_pos(d),
                    lemma=_get_lemma(d),
                    source="camel",
                )
            )
        return result
