"""
test_hf_integration.py — tests for HF model wrapper and data collator.

Uses a tiny nn.Module to simulate a CausalLM (no HF transformers required).
"""
import pytest
import torch
import torch.nn as nn

from sarftok import MorphAnalysis
from sarftok.config import SarfTokConfig
from sarftok.llm_integration.hf_data_collator import SarfTokDataCollator
from sarftok.llm_integration.hf_modeling_embeddings import (
    HybridSarfTokEmbedding,
)
from sarftok.morph_encoder import MorphEncoder
from sarftok.morph_vocab import MorphVocab
from sarftok.probabilistic_embedder import ProbabilisticEmbedder
from sarftok.serialization import analysis_to_dict

# ---------------------------------------------------------------------------
# Toy model fixtures
# ---------------------------------------------------------------------------


class ToyEmbedding(nn.Embedding):
    pass


class ToyOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class ToyCausalLM(nn.Module):
    """Minimal fake CausalLM for testing."""

    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                labels=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        logits = self.head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return ToyOutput(loss=loss, logits=logits)

    def save_pretrained(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))


@pytest.fixture
def vocab():
    v = MorphVocab(root_min_freq=1, pattern_min_freq=1)
    v.root_vocab["ROOT_ktb"] = len(v.root_vocab)
    v.pattern_vocab["PAT_CaCaCa"] = len(v.pattern_vocab)
    return v


@pytest.fixture
def config():
    return SarfTokConfig(
        hidden_dim=32,
        alpha=0.5,
        entropy_gating=False,
        ortho_lambda=0.01,
        ortho_min_confidence=0.3,
    )


# ---------------------------------------------------------------------------
# Data collator tests
# ---------------------------------------------------------------------------


class TestSarfTokDataCollator:
    def test_padding(self):
        collator = SarfTokDataCollator(pad_token_id=0)
        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "word_to_surface_spans": [(0, 2), (2, 3)],
                "morph_analyses": [[{"prob": 1.0, "root": "ktb"}], [{"prob": 1.0}]],
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "word_to_surface_spans": [(0, 2)],
                "morph_analyses": [[{"prob": 1.0, "root": "qra"}]],
            },
        ]
        batch = collator(features)
        assert batch["input_ids"].shape == (2, 3)  # padded to longest
        assert batch["attention_mask"].shape == (2, 3)
        assert batch["labels"].shape == (2, 3)
        # Padded position should have label -100
        assert batch["labels"][1, 2].item() == -100

    def test_morph_analyses_preserved(self):
        collator = SarfTokDataCollator()
        features = [
            {
                "input_ids": [1],
                "attention_mask": [1],
                "word_to_surface_spans": [(0, 1)],
                "morph_analyses": [[{"prob": 1.0}]],
            },
        ]
        batch = collator(features)
        assert len(batch["morph_analyses"]) == 1
        assert isinstance(batch["morph_analyses"][0], list)


# ---------------------------------------------------------------------------
# Embedding + model tests
# ---------------------------------------------------------------------------


class TestHybridSarfTokEmbedding:
    def test_forward_with_morph(self, vocab):
        base_emb = nn.Embedding(100, 32)
        enc = MorphEncoder(vocab=vocab, hidden_dim=32)
        pe = ProbabilisticEmbedder(hidden_dim=32, alpha=0.5, entropy_gating=False)
        hybrid = HybridSarfTokEmbedding(base_emb, enc, pe)

        ids = torch.tensor([[1, 2, 3, 4, 5]])
        spans = [[(0, 2), (2, 5)]]
        morph = [
            [
                [analysis_to_dict(MorphAnalysis(prob=1.0, root="ktb", pattern="CaCaCa", source="test"))],
                [analysis_to_dict(MorphAnalysis(prob=1.0, root="qra", pattern="CaCaCa", source="test"))],
            ]
        ]
        out = hybrid(ids, spans, morph)
        assert out.shape == (1, 5, 32)

    def test_forward_without_morph(self, vocab):
        base_emb = nn.Embedding(100, 32)
        enc = MorphEncoder(vocab=vocab, hidden_dim=32)
        pe = ProbabilisticEmbedder(hidden_dim=32, alpha=0.5, entropy_gating=False)
        hybrid = HybridSarfTokEmbedding(base_emb, enc, pe)

        ids = torch.tensor([[1, 2, 3]])
        out = hybrid(ids, [], None)
        assert out.shape == (1, 3, 32)


class TestHybridSarfTokCausalLM:
    def test_forward_backward(self, vocab, config):
        from sarftok.llm_integration.hf_modeling_embeddings import HybridSarfTokCausalLM

        base = ToyCausalLM(vocab_size=100, hidden_dim=32)
        model = HybridSarfTokCausalLM(base, config, vocab)

        ids = torch.tensor([[1, 2, 3, 4, 5]])
        mask = torch.ones_like(ids)
        labels = torch.tensor([[2, 3, 4, 5, 6]])
        spans = [[(0, 2), (2, 5)]]
        morph = [
            [
                [analysis_to_dict(MorphAnalysis(prob=0.8, root="ktb", pattern="CaCaCa", source="test"))],
                [analysis_to_dict(MorphAnalysis(prob=0.9, root="qra", pattern="CaCaCa", source="test"))],
            ]
        ]

        out = model(ids, mask, labels, spans, morph)
        assert "loss" in out
        assert "logits" in out
        # Can backprop
        out["loss"].backward()

    def test_forward_without_morph(self, vocab, config):
        from sarftok.llm_integration.hf_modeling_embeddings import HybridSarfTokCausalLM

        base = ToyCausalLM(vocab_size=100, hidden_dim=32)
        model = HybridSarfTokCausalLM(base, config, vocab)

        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        labels = torch.tensor([[2, 3, 4]])
        out = model(ids, mask, labels)
        assert hasattr(out, "loss") or "loss" in out
