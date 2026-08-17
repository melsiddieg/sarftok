"""
test_morph_vocab.py — tests for MorphVocab build, save/load, and OOV fallback.
"""
import json
import tempfile
from pathlib import Path

import pytest

from sarftok import MorphAnalysis
from sarftok.morph_vocab import MorphVocab


@pytest.fixture
def vocab():
    return MorphVocab(root_min_freq=2, pattern_min_freq=2)


class TestBuiltins:
    def test_rootchar_vocab_populated(self, vocab):
        assert vocab.num_rootchars > 0
        assert any("ROOTCHAR_" in k for k in vocab.rootchar_vocab)

    def test_patclass_vocab_populated(self, vocab):
        assert vocab.num_patclasses > 0
        assert "PAT_UNKNOWN" in vocab.patclass_vocab

    def test_proclitic_vocab_populated(self, vocab):
        assert vocab.num_proclitics > 0

    def test_enclitic_vocab_populated(self, vocab):
        assert vocab.num_enclitics > 0

    def test_morphosyntactic_factor_vocabs_populated(self, vocab):
        for name in ("pos", "case", "mood", "voice", "person", "number", "gender"):
            assert vocab.num_feature_values(name) > 1


class TestRootLookup:
    def test_unknown_root_returns_not_composite(self, vocab):
        rid, is_composite = vocab.root_id("xyz_unknown")
        assert not is_composite

    def test_root_char_ids(self, vocab):
        ids = vocab.root_char_ids("كتب")
        assert len(ids) == 3  # one per root character
        assert all(isinstance(i, int) for i in ids)


class TestPatternLookup:
    def test_unknown_pattern_falls_back_to_pat_unknown(self, vocab):
        pid, tok = vocab.pattern_id("some_unknown_pattern_xyz")
        assert tok == "PAT_UNKNOWN"


class TestBuildFromAnalyses:
    def test_build_creates_frequent_roots(self):
        def make_iter():
            a = MorphAnalysis(prob=1.0, root="ktb", pattern="CaCaCa")
            for _ in range(5):
                yield [[a]]

        vocab = MorphVocab.build_from_analyses(make_iter(), root_min_freq=3)
        rid, is_composite = vocab.root_id("ktb")
        assert is_composite
        assert rid is not None

    def test_build_skips_rare_roots(self):
        def make_iter():
            a = MorphAnalysis(prob=1.0, root="rareroot")
            yield [[a]]  # only 1 occurrence

        vocab = MorphVocab.build_from_analyses(make_iter(), root_min_freq=5)
        rid, is_composite = vocab.root_id("rareroot")
        assert not is_composite

    def test_build_records_observed_feature_values(self):
        analysis = MorphAnalysis(
            prob=1.0,
            root="كتب",
            case="custom_case",
            mood="custom_mood",
        )
        vocab = MorphVocab.build_from_analyses(iter([[[analysis]]]))
        assert vocab.feature_id("case", "custom_case") != 0
        assert vocab.feature_id("mood", "custom_mood") != 0


class TestSaveLoad:
    def test_roundtrip(self, vocab):
        # Add a root manually
        vocab.root_vocab["ROOT_ktb"] = len(vocab.root_vocab)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        vocab.save(path)
        loaded = MorphVocab.load(path)

        assert loaded.root_vocab == vocab.root_vocab
        assert loaded.pattern_vocab == vocab.pattern_vocab
        assert loaded.root_min_freq == vocab.root_min_freq
        assert loaded.feature_vocabs == vocab.feature_vocabs

    def test_json_valid(self, vocab):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        vocab.save(path)
        data = json.loads(Path(path).read_text())
        assert "root_vocab" in data
        assert "pattern_vocab" in data
