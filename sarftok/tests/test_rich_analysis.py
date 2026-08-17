"""Tests for rich morphology mapping and serialization."""

import pytest

from sarftok import MorphAnalysis
from sarftok.config import SarfTokConfig
from sarftok.morph_analyzer.camel_wrapper import _map_analysis
from sarftok.serialization import analysis_to_dict, dict_to_analysis


def test_rich_analysis_serialization_roundtrip():
    analysis = MorphAnalysis(
        prob=0.8,
        root="كتب",
        pattern="فعل",
        pos="verb",
        aspect="p",
        voice="a",
        mood="i",
        person="3",
        gender="m",
        number="s",
        case="n",
        state="d",
        definiteness="def",
        is_contextual=True,
        score_source="posterior",
    )
    assert dict_to_analysis(analysis_to_dict(analysis)) == analysis


def test_camel_mapping_extracts_full_feature_set_and_uses_uniform_fallback():
    analysis = _map_analysis(
        {
            "root": "كتب",
            "pattern": "فعل",
            "pos": "verb",
            "asp": "p",
            "vox": "a",
            "mod": "i",
            "per": "3",
            "gen": "m",
            "num": "s",
            "cas": "n",
            "stt": "d",
            "det": "def",
        }
    )
    assert analysis.prob == 1.0
    assert analysis.score_source == "uniform"
    assert (analysis.aspect, analysis.voice, analysis.case) == ("p", "a", "n")


def test_analyzer_factory_passes_classical_database(monkeypatch):
    import sarftok.morph_analyzer.camel_wrapper as camel_wrapper
    from sarftok.tokenizer_api import _build_analyzer

    captured = {}

    class FakeCamel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(camel_wrapper, "CamelMorphAnalyzer", FakeCamel)
    _build_analyzer(SarfTokConfig(analyzer_backend="camel"))
    assert captured["db_name"] == "calima-clx-r13"


def test_invalid_contextual_result_length_is_rejected():
    from sarftok.morph_analyzer.interface import MorphAnalyzer

    class Broken(MorphAnalyzer):
        def _raw_analyze_word(self, word):
            return []

        def _raw_analyze_sentence(self, words, context=None):
            return []

    with pytest.raises(ValueError, match="different number"):
        Broken().analyze_sentence(["كتب"])
