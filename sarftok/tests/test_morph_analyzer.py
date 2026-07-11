"""
test_morph_analyzer.py — tests for heuristic analyzer and interface.
"""
import pytest

from sarftok import MorphAnalysis
from sarftok.morph_analyzer.heuristics import HeuristicMorphAnalyzer
from sarftok.morph_analyzer.interface import MorphAnalyzer


@pytest.fixture
def analyzer():
    return HeuristicMorphAnalyzer(top_k=3, confidence_threshold=0.05)


class TestHeuristicOutputFormat:
    def test_returns_list_of_morph_analysis(self, analyzer):
        results = analyzer.analyze_word("كتب")
        assert isinstance(results, list)
        assert all(isinstance(a, MorphAnalysis) for a in results)

    def test_at_least_one_analysis(self, analyzer):
        results = analyzer.analyze_word("كتب")
        assert len(results) >= 1

    def test_probs_sum_to_one(self, analyzer):
        results = analyzer.analyze_word("كتب")
        total = sum(a.prob for a in results)
        assert abs(total - 1.0) < 0.01

    def test_sorted_descending_by_prob(self, analyzer):
        results = analyzer.analyze_word("كتب")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].prob >= results[i + 1].prob

    def test_top_k_respected(self, analyzer):
        for word in ["والطالبين", "بمكتوبهم", "فاستخرجوها"]:
            results = analyzer.analyze_word(word)
            assert len(results) <= 3


class TestHeuristicSentenceAnalysis:
    def test_analyze_sentence(self, analyzer):
        words = ["كتب", "الطالب", "الدرس"]
        results = analyzer.analyze_sentence(words)
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_analyze_sentence_with_context(self, analyzer):
        words = ["كتب"]
        results = analyzer.analyze_sentence(words, context="كتب الطالب")
        assert len(results) == 1


class TestHeuristicProclitics:
    def test_proclitic_detection(self, analyzer):
        results = analyzer.analyze_word("وكتب")
        if results and results[0].proclitics:
            assert "wa" in results[0].proclitics

    def test_no_proclitics_for_short_word(self, analyzer):
        # Very short words should not have their only content stripped
        results = analyzer.analyze_word("من")
        assert isinstance(results, list)


class TestHeuristicRoot:
    def test_root_extraction(self, analyzer):
        results = analyzer.analyze_word("كتب")
        if results:
            # Should detect a root
            has_root = any(a.root is not None for a in results)
            assert has_root

    def test_source_is_heuristic(self, analyzer):
        results = analyzer.analyze_word("كتب")
        for a in results:
            assert a.source == "heuristic"


class TestProbNormalisation:
    def test_temperature_scaling(self):
        analyzer = HeuristicMorphAnalyzer(top_k=3, temperature=0.5)
        results = analyzer.analyze_word("كتب")
        total = sum(a.prob for a in results)
        assert abs(total - 1.0) < 0.01

    def test_confidence_threshold_filtering(self):
        analyzer = HeuristicMorphAnalyzer(top_k=5, confidence_threshold=0.4)
        results = analyzer.analyze_word("كتب")
        for a in results:
            assert a.prob >= 0.4


class TestDistilledStub:
    def test_distilled_raises(self):
        from sarftok.morph_analyzer.distilled_tagger import DistilledMorphTagger
        with pytest.raises(NotImplementedError):
            DistilledMorphTagger()


class TestProclitcSliceRegression:
    """F2 — undiacritised proclitic stripping must not drop a root consonant."""

    def test_definite_article_undiacritised(self, analyzer):
        # الكتاب → strip 'al' → stem كتاب → root كتب (previously wrongly تاب)
        results = analyzer.analyze_word("الكتاب")
        assert "al" in results[0].proclitics
        assert results[0].root == "كتب"

    def test_wa_proclitic_undiacritised(self, analyzer):
        # وكتب → strip 'wa' → stem كتب → root كتب
        results = analyzer.analyze_word("وكتب")
        assert "wa" in results[0].proclitics
        assert results[0].root == "كتب"


class TestPatternRegression:
    """F3 — CaCiCa must match the kasra vowel, not the ya letter."""

    def test_cacica_pattern_uses_kasra(self, analyzer):
        # كَتِب — fatha then kasra → CaCiCa
        results = analyzer.analyze_word("كَتِب")
        assert results[0].pattern == "CaCiCa"


class TestProbabilitySemantics:
    """F4 — 'prob' score_type L1-normalises and preserves relative confidences."""

    def test_l1_preserves_distribution(self, analyzer):
        # Heuristic emits primary 0.7 / alt 0.3 for a known root; these must be
        # preserved rather than flattened by a softmax (which gives ~0.6/0.4).
        results = analyzer.analyze_word("كتب")
        assert len(results) == 2
        assert abs(results[0].prob - 0.7) < 0.02
        assert abs(results[1].prob - 0.3) < 0.02

    def test_logit_score_type_uses_softmax(self):
        from sarftok import MorphAnalysis

        class _Logit(MorphAnalyzer):
            def _raw_analyze_word(self, word):
                return [
                    MorphAnalysis(prob=1.0, root="ktb", source="t"),
                    MorphAnalysis(prob=0.0, root="ktb", source="t"),
                ]

        probs = [a.prob for a in _Logit(score_type="logit").analyze_word("x")]
        # softmax([1,0]) ≈ [0.731, 0.269] — not L1's [1.0, 0.0]
        assert abs(probs[0] - 0.731) < 0.02

    def test_invalid_score_type_raises(self):
        from sarftok.morph_analyzer.heuristics import HeuristicMorphAnalyzer
        with pytest.raises(ValueError, match="score_type"):
            HeuristicMorphAnalyzer(score_type="bogus")
