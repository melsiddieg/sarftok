"""
test_morph_encoder.py — tests for MorphEncoder nn.Module.
"""
import pytest
import torch

from sarftok import MorphAnalysis
from sarftok.morph_encoder import MorphEncoder
from sarftok.morph_vocab import MorphVocab


@pytest.fixture
def vocab():
    v = MorphVocab(root_min_freq=1, pattern_min_freq=1)
    # Add a known root and pattern
    v.root_vocab["ROOT_ktb"] = len(v.root_vocab)
    v.pattern_vocab["PAT_CaCaCa"] = len(v.pattern_vocab)
    return v


@pytest.fixture
def encoder(vocab):
    return MorphEncoder(vocab=vocab, hidden_dim=64, use_composite_root=True)


class TestForwardWordShape:
    def test_output_shape(self, encoder):
        analyses = [
            MorphAnalysis(prob=0.7, root="ktb", pattern="CaCaCa", source="test"),
            MorphAnalysis(prob=0.3, root="ktb", pattern="maCCuuC", is_oov_pattern=True, source="test"),
        ]
        out = encoder.forward_word(analyses)
        assert out.shape == (64,)

    def test_empty_analyses_returns_zeros(self, encoder):
        out = encoder.forward_word([])
        assert out.shape == (64,)
        assert torch.all(out == 0)


class TestProbabilisticMixture:
    def test_mixture_is_weighted(self, encoder):
        a1 = MorphAnalysis(prob=1.0, root="ktb", pattern="CaCaCa", source="test")
        a2 = MorphAnalysis(prob=0.0, root="ktb", pattern="CaCaCa", is_oov_pattern=True, source="test")
        out_single = encoder.forward_word([a1])

        out_mix = encoder.forward_word([a1, a2])
        # With prob=0 for a2, the outputs should be identical to just a1
        assert torch.allclose(out_single, out_mix, atol=1e-6)


class TestBatchForward:
    def test_batch_output_shape(self, encoder):
        a = MorphAnalysis(prob=1.0, root="ktb", pattern="CaCaCa", source="test")
        batch = [
            [[a], [a, a]],  # sentence 1: 2 words
            [[a]],          # sentence 2: 1 word
        ]
        out = encoder(batch)
        assert out.shape == (2, 2, 64)

    def test_empty_batch(self, encoder):
        out = encoder([])
        assert out.shape[0] == 0


class TestOOVFallback:
    def test_oov_root_uses_rootchar(self, encoder):
        a = MorphAnalysis(
            prob=1.0, root="xyz", is_oov_root=True,
            pattern="CaCaCa", source="test"
        )
        out = encoder.forward_word([a])
        assert out.shape == (64,)
        # Should not be all zeros (rootchar embeddings should contribute)

    def test_oov_pattern_uses_fallback(self, encoder):
        a = MorphAnalysis(
            prob=1.0, root="ktb", pattern="unknown_xyz",
            is_oov_pattern=True, source="test"
        )
        out = encoder.forward_word([a])
        assert out.shape == (64,)
