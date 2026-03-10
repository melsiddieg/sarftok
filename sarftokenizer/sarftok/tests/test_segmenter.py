"""
test_segmenter.py — unit tests for ArabicSegmenter.
"""
import pytest

from sarftok.segmenter import ArabicSegmenter


@pytest.fixture
def seg():
    return ArabicSegmenter()


class TestSentenceSplitting:
    def test_single_sentence(self, seg):
        result = seg.split_sentences("كتب الطالب الدرس")
        assert len(result) >= 1

    def test_multiple_sentences_by_period(self, seg):
        result = seg.split_sentences("كتب الطالب. قرأ المعلم.")
        assert len(result) >= 2

    def test_empty_text(self, seg):
        assert seg.split_sentences("") == []


class TestWordTokenisation:
    def test_whitespace_splitting(self, seg):
        words = seg.tokenize_words("كتب الطالب الدرس")
        assert len(words) == 3
        assert words[0] == "كتب"
        assert words[1] == "الطالب"
        assert words[2] == "الدرس"

    def test_empty_string(self, seg):
        assert seg.tokenize_words("") == []


class TestSegment:
    def test_segment_returns_list_of_lists(self, seg):
        result = seg.segment("كتب الطالب الدرس")
        assert isinstance(result, list)
        assert all(isinstance(s, list) for s in result)

    def test_segment_flat(self, seg):
        result = seg.segment_flat("كتب الطالب الدرس")
        assert isinstance(result, list)
        assert all(isinstance(w, str) for w in result)
        assert len(result) >= 3
