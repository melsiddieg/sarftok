"""
test_embedding_fusion.py — tests for ProbabilisticEmbedder fusion modes.
"""
import math

import pytest
import torch

from sarftok import MorphAnalysis
from sarftok.probabilistic_embedder import ProbabilisticEmbedder, _morphological_entropy


@pytest.fixture
def embedder_broadcast():
    return ProbabilisticEmbedder(
        hidden_dim=32,
        fusion_mode="broadcast",
        alpha=0.5,
        entropy_gating=False,
    )


@pytest.fixture
def embedder_fp():
    return ProbabilisticEmbedder(
        hidden_dim=32,
        fusion_mode="first_piece",
        alpha=0.5,
        entropy_gating=False,
    )


@pytest.fixture
def embedder_gated():
    return ProbabilisticEmbedder(
        hidden_dim=32,
        fusion_mode="broadcast",
        alpha=1.0,
        entropy_gating=True,
        beta=1.0,
    )


class TestBroadcastMode:
    def test_all_surface_positions_modified(self, embedder_broadcast):
        surface = torch.zeros(5, 32)
        morph = torch.ones(2, 32)
        spans = [(0, 3), (3, 5)]

        out = embedder_broadcast.forward_sentence(surface, morph, spans)
        assert out.shape == (5, 32)
        # All positions should be non-zero (0 + 0.5 * 1 = 0.5)
        assert torch.all(out != 0)

    def test_correct_span_assignment(self, embedder_broadcast):
        surface = torch.zeros(4, 32)
        morph = torch.stack([torch.ones(32) * 1.0, torch.ones(32) * 2.0])
        spans = [(0, 2), (2, 4)]

        out = embedder_broadcast.forward_sentence(surface, morph, spans)
        # First word span should have value 0.5 (alpha * 1.0)
        assert torch.allclose(out[0], torch.ones(32) * 0.5)
        assert torch.allclose(out[1], torch.ones(32) * 0.5)
        # Second word span should have value 1.0 (alpha * 2.0)
        assert torch.allclose(out[2], torch.ones(32) * 1.0)


class TestFirstPieceMode:
    def test_only_first_piece_modified(self, embedder_fp):
        surface = torch.zeros(5, 32)
        morph = torch.ones(2, 32)
        spans = [(0, 3), (3, 5)]

        out = embedder_fp.forward_sentence(surface, morph, spans)
        assert out.shape == (5, 32)
        # Only positions 0 and 3 should be modified
        assert torch.all(out[0] != 0)
        assert torch.all(out[1] == 0)  # not modified
        assert torch.all(out[2] == 0)
        assert torch.all(out[3] != 0)
        assert torch.all(out[4] == 0)


class TestEntropyGating:
    def test_entropy_computation(self):
        analyses = [
            MorphAnalysis(prob=0.5, source="test"),
            MorphAnalysis(prob=0.5, source="test"),
        ]
        H = _morphological_entropy(analyses)
        # H = -2 * 0.5 * ln(0.5) = ln(2) ≈ 0.693
        assert abs(H - math.log(2)) < 0.01

    def test_high_entropy_reduces_alpha(self, embedder_gated):
        # Uniform distribution → max entropy → small alpha
        analyses = [
            MorphAnalysis(prob=0.33, source="test"),
            MorphAnalysis(prob=0.33, source="test"),
            MorphAnalysis(prob=0.34, source="test"),
        ]
        eff = embedder_gated._effective_alpha(analyses)
        assert eff < 1.0  # should be reduced from base alpha=1.0

    def test_low_entropy_preserves_alpha(self, embedder_gated):
        analyses = [MorphAnalysis(prob=1.0, source="test")]
        eff = embedder_gated._effective_alpha(analyses)
        # H ≈ 0, so alpha ≈ 1.0 * exp(0) = 1.0
        assert abs(eff - 1.0) < 0.01

    def test_normalized_entropy_in_unit_range(self):
        # A uniform distribution should give H_norm ≈ 1.0 regardless of k.
        for k in (2, 3, 5):
            analyses = [MorphAnalysis(prob=1.0 / k, source="test") for _ in range(k)]
            h_norm = _morphological_entropy(analyses, normalize=True)
            assert abs(h_norm - 1.0) < 1e-6

    def test_normalization_stabilizes_gate_across_k(self):
        # With normalisation on, a uniform word yields the same effective alpha
        # (α₀·exp(-β)) whether it has 2 or 5 analyses — β stays scale-stable.
        emb = ProbabilisticEmbedder(
            hidden_dim=8, alpha=1.0, entropy_gating=True, beta=1.0,
            entropy_normalize=True,
        )
        eff_k2 = emb._effective_alpha([MorphAnalysis(prob=0.5, source="t") for _ in range(2)])
        eff_k5 = emb._effective_alpha([MorphAnalysis(prob=0.2, source="t") for _ in range(5)])
        assert abs(eff_k2 - eff_k5) < 1e-6
        # Without normalisation the two would differ (raw H grows with k).
        emb_raw = ProbabilisticEmbedder(
            hidden_dim=8, alpha=1.0, entropy_gating=True, beta=1.0,
            entropy_normalize=False,
        )
        raw_k2 = emb_raw._effective_alpha([MorphAnalysis(prob=0.5, source="t") for _ in range(2)])
        raw_k5 = emb_raw._effective_alpha([MorphAnalysis(prob=0.2, source="t") for _ in range(5)])
        assert raw_k5 < raw_k2


class TestBatchForward:
    def test_batch_shapes(self, embedder_broadcast):
        surface = torch.zeros(2, 5, 32)
        morph = torch.ones(2, 2, 32)
        spans = [[(0, 3), (3, 5)], [(0, 2), (2, 5)]]

        out = embedder_broadcast(surface, morph, spans)
        assert out.shape == (2, 5, 32)


class TestEmptySpan:
    def test_empty_span_skipped(self, embedder_broadcast):
        surface = torch.zeros(3, 32)
        morph = torch.ones(1, 32)
        spans = [(1, 1)]  # empty span

        out = embedder_broadcast.forward_sentence(surface, morph, spans)
        # Nothing should be modified since span is empty
        assert torch.all(out == 0)
