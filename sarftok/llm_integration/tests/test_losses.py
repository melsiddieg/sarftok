"""
test_losses.py — tests for training losses.
"""
import pytest
import torch

from sarftok import MorphAnalysis
from sarftok.llm_integration.losses import lm_loss, orthogonality_loss
from sarftok.morph_encoder import MorphEncoder
from sarftok.morph_vocab import MorphVocab


@pytest.fixture
def vocab():
    v = MorphVocab(root_min_freq=1, pattern_min_freq=1)
    v.root_vocab["ROOT_ktb"] = len(v.root_vocab)
    v.root_vocab["ROOT_qra"] = len(v.root_vocab)
    v.pattern_vocab["PAT_CaCaCa"] = len(v.pattern_vocab)
    v.pattern_vocab["PAT_maCCuuC"] = len(v.pattern_vocab)
    return v


@pytest.fixture
def encoder(vocab):
    return MorphEncoder(vocab=vocab, hidden_dim=32)


class TestLMLoss:
    def test_output_is_scalar(self):
        logits = torch.randn(2, 5, 100)
        labels = torch.randint(0, 100, (2, 5))
        loss = lm_loss(logits, labels)
        assert loss.dim() == 0

    def test_ignores_padding(self):
        logits = torch.randn(1, 3, 10)
        labels = torch.tensor([[1, -100, -100]])
        loss = lm_loss(logits, labels, ignore_index=-100)
        assert loss.isfinite()


class TestOrthogonalityLoss:
    def test_finite_output(self, encoder):
        analyses = [
            [[MorphAnalysis(prob=0.9, root="ktb", pattern="CaCaCa", source="test")]],
        ]
        loss = orthogonality_loss(encoder, analyses, min_confidence=0.5, lambda_=0.01)
        assert loss.isfinite()
        assert loss.item() >= 0

    def test_zero_when_no_qualifying(self, encoder):
        analyses = [
            [[MorphAnalysis(prob=0.1, root="ktb", pattern="CaCaCa", source="test")]],
        ]
        loss = orthogonality_loss(encoder, analyses, min_confidence=0.5, lambda_=0.01)
        assert loss.item() == 0.0

    def test_decreases_in_toy_optimization(self, encoder):
        """Verify ortho loss decreases when root/pattern are optimized to be orthogonal."""
        a = MorphAnalysis(prob=1.0, root="ktb", pattern="CaCaCa", source="test")
        analyses = [
            [[a]],
        ]
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = orthogonality_loss(encoder, analyses, min_confidence=0.5, lambda_=1.0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (or stay at 0)
        assert losses[-1] <= losses[0] + 1e-5

    def test_empty_analyses(self, encoder):
        loss = orthogonality_loss(encoder, [], min_confidence=0.5)
        assert loss.item() == 0.0

    def test_oov_root_excluded(self, encoder):
        analyses = [
            [[MorphAnalysis(prob=0.9, root="unknown_xyz", pattern="CaCaCa",
                            is_oov_root=True, source="test")]],
        ]
        loss = orthogonality_loss(encoder, analyses, min_confidence=0.5)
        assert loss.item() == 0.0
