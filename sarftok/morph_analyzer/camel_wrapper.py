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


def _extract_clitics(analysis: dict, keys: list[str]) -> list[str]:
    """Extract non-null clitic values from a CAMeL analysis dict."""
    result: list[str] = []
    for k in keys:
        v = analysis.get(k, "")
        if v and v not in ("", "0", "na"):
            result.append(v)
    return result


def _get_root(analysis: dict) -> str | None:
    root = analysis.get("root", "")
    if not root or root in ("-", "NOAN", "na"):
        return None
    return root


def _get_pattern(analysis: dict) -> str | None:
    pattern = analysis.get("pattern", "")
    if not pattern or pattern in ("-", "NOAN", "na"):
        return None
    return pattern


def _get_pos(analysis: dict) -> str | None:
    return analysis.get("pos") or None


def _get_lemma(analysis: dict) -> str | None:
    return analysis.get("lex") or analysis.get("lemma") or None


def _get_prob(analysis: dict) -> tuple[float, str]:
    """Return a defensible non-contextual weight and its provenance.

    CAMeL's analyzer output ordering is not a calibrated posterior. When no
    corpus frequency exists we therefore use equal weights, producing maximum
    entropy instead of manufacturing confidence from rank.
    """
    count = analysis.get("count", analysis.get("freq"))
    if count is not None:
        try:
            return float(count), "frequency"
        except (ValueError, TypeError):
            pass
    return 1.0, "uniform"


_FEATURE_KEYS = {
    "aspect": "asp",
    "voice": "vox",
    "mood": "mod",
    "person": "per",
    "gender": "gen",
    "number": "num",
    "case": "cas",
    "state": "stt",
    "definiteness": "det",
}


def _feature(analysis: dict, name: str) -> str | None:
    value = analysis.get(_FEATURE_KEYS[name]) or analysis.get(name)
    return None if value in (None, "", "na", "0", "-") else str(value)


def _map_analysis(
    raw: dict,
    *,
    score: float | None = None,
    contextual: bool = False,
    score_source: str | None = None,
) -> MorphAnalysis:
    if score is None:
        score, inferred_source = _get_prob(raw)
        score_source = score_source or inferred_source
    return MorphAnalysis(
        prob=max(float(score), 0.0),
        proclitics=_extract_clitics(raw, _PROCLITIC_KEYS),
        root=_get_root(raw),
        pattern=_get_pattern(raw),
        enclitics=_extract_clitics(raw, _ENCLITIC_KEYS),
        pos=_get_pos(raw),
        lemma=_get_lemma(raw),
        aspect=_feature(raw, "aspect"),
        voice=_feature(raw, "voice"),
        mood=_feature(raw, "mood"),
        person=_feature(raw, "person"),
        gender=_feature(raw, "gender"),
        number=_feature(raw, "number"),
        case=_feature(raw, "case"),
        state=_feature(raw, "state"),
        definiteness=_feature(raw, "definiteness"),
        is_contextual=contextual,
        score_source=score_source or "unknown",
        source="camel",
    )


# ---------------------------------------------------------------------------
# CamelMorphAnalyzer
# ---------------------------------------------------------------------------


class CamelMorphAnalyzer(MorphAnalyzer):
    """Morphological analyzer backed by CAMeL Tools.

    Parameters
    ----------
    db_name:
        Name of the CAMeL Tools morphology database to load.
        Default: ``"calima-clx-r13"`` (Classical Arabic database).
        Use ``"calima-msa-r13"`` explicitly for MSA.
    top_k, confidence_threshold, temperature:
        Inherited from :class:`~sarftok.morph_analyzer.interface.MorphAnalyzer`.
    """

    def __init__(
        self,
        db_name: str = "calima-clx-r13",
        top_k: int = 3,
        confidence_threshold: float = 0.05,
        temperature: float = 1.0,
        score_type: str = "prob",
        contextual_disambiguator=None,
    ) -> None:
        if not _CAMEL_AVAILABLE:
            raise ImportError(
                "camel-tools is required for CamelMorphAnalyzer.\n"
                "Install it with:  pip install camel-tools\n"
                "Then install the morphology data required by db_name with camel_data."
            )
        super().__init__(
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            temperature=temperature,
            score_type=score_type,
        )
        db = MorphologyDB.builtin_db(db_name)
        self._analyzer = CamelAnalyzer(db)
        self.db_name = db_name
        self.contextual_disambiguator = contextual_disambiguator

    def _raw_analyze_word(self, word: str) -> list[MorphAnalysis]:
        try:
            raw_analyses = self._analyzer.analyze(word)
        except Exception:  # noqa: BLE001
            return []

        return [_map_analysis(dict(raw)) for raw in raw_analyses]

    def _raw_analyze_sentence(
        self,
        words: list[str],
        context: str | None = None,
    ) -> list[list[MorphAnalysis]]:
        """Use an injected CAMeL-compatible disambiguator when available.

        A disambiguator may expose ``disambiguate(words)`` (the CAMeL Tools
        API) or be a callable accepting ``(words, context)``. Its per-word
        results may be CAMeL ``DisambiguatedWord`` objects or plain lists of
        scored analyses. Without one, the base class safely falls back to
        word-independent, uniformly weighted analyses.
        """
        if self.contextual_disambiguator is None:
            return super()._raw_analyze_sentence(words, context=context)

        disambiguator = self.contextual_disambiguator
        if hasattr(disambiguator, "disambiguate"):
            outputs = disambiguator.disambiguate(words)
        elif callable(disambiguator):
            outputs = disambiguator(words, context)
        else:
            raise TypeError(
                "contextual_disambiguator must be callable or expose disambiguate(words)"
            )

        sentence: list[list[MorphAnalysis]] = []
        for output in outputs:
            candidates = getattr(output, "analyses", output)
            word_results: list[MorphAnalysis] = []
            for candidate in candidates:
                if isinstance(candidate, MorphAnalysis):
                    candidate.is_contextual = True
                    candidate.score_source = "contextual_posterior"
                    word_results.append(candidate)
                    continue
                raw = getattr(candidate, "analysis", candidate)
                score = getattr(candidate, "score", None)
                if not isinstance(raw, dict):
                    raw = dict(raw)
                word_results.append(
                    _map_analysis(
                        raw,
                        score=score,
                        contextual=True,
                        score_source="contextual_posterior",
                    )
                )
            sentence.append(word_results)
        return sentence
