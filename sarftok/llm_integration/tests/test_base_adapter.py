"""
test_base_adapter.py — tests for SarfTokBaseAdapter (align morphology to a base tokenizer).

The unit tests use a deterministic mock fast tokenizer so they run without network access.
A single opt-in integration test exercises a real HuggingFace fast tokenizer and is skipped
when the tokenizer cannot be downloaded (offline CI).
"""
import pytest
import torch.nn as nn

from sarftok import MorphAnalysis
from sarftok.alignment import validate_spans
from sarftok.config import SarfTokConfig
from sarftok.llm_integration.base_tokenizer_adapter import (
    SarfTokBaseAdapter,
    _spans_from_word_ids,
)

# ---------------------------------------------------------------------------
# Mock fast tokenizer
# ---------------------------------------------------------------------------


class _MockEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_ids):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0):
        return self._word_ids


class MockFastTokenizer:
    """Splits each word into ceil(len/2) fake subtokens; adds BOS(1)/EOS(2) specials."""

    is_fast = True

    def __call__(
        self,
        words,
        is_split_into_words=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        input_ids: list[int] = []
        word_ids: list[int | None] = []
        if add_special_tokens:
            input_ids.append(1)
            word_ids.append(None)
        if is_split_into_words:
            for wi, w in enumerate(words):
                n = max(1, (len(w) + 1) // 2)
                for _ in range(n):
                    input_ids.append(100 + wi)
                    word_ids.append(wi)
        else:
            for _ in str(words).split():
                input_ids.append(100)
                word_ids.append(None)
        if add_special_tokens:
            input_ids.append(2)
            word_ids.append(None)
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
            word_ids = word_ids[:max_length]
        return _MockEncoding(input_ids, [1] * len(input_ids), word_ids)


@pytest.fixture
def config():
    return SarfTokConfig(analyzer_backend="heuristic", top_k=3)


@pytest.fixture
def adapter(config):
    return SarfTokBaseAdapter(MockFastTokenizer(), config)


# ---------------------------------------------------------------------------
# _spans_from_word_ids
# ---------------------------------------------------------------------------


class TestSpansFromWordIds:
    def test_contiguous_with_specials(self):
        # BOS, w0 x2, w1 x1, EOS
        word_ids = [None, 0, 0, 1, None]
        spans = _spans_from_word_ids(word_ids, num_words=2)
        assert spans == [(1, 3), (3, 4)]

    def test_dropped_word_gets_empty_span(self):
        word_ids = [None, 0, 0]  # word 1 truncated away
        spans = _spans_from_word_ids(word_ids, num_words=2)
        assert spans[0] == (1, 3)
        # Empty span anchored at the cursor (not (0,0)) so starts stay monotonic.
        assert spans[1] == (3, 3)
        validate_spans(spans, total_ids=3)


# ---------------------------------------------------------------------------
# Adapter encoding contract
# ---------------------------------------------------------------------------


class TestAdapterEncoding:
    def test_requires_fast_tokenizer(self, config):
        class Slow:
            is_fast = False

        with pytest.raises(ValueError, match="fast tokenizer"):
            SarfTokBaseAdapter(Slow(), config)

    def test_encode_sentence_shapes(self, adapter):
        out = adapter.encode_sentence("كتب الطالب الدرس")
        n_words = out["num_words"]
        assert n_words == 3
        assert len(out["word_to_surface_spans"]) == n_words
        assert len(out["morph_analyses"]) == n_words
        assert len(out["input_ids"]) == len(out["attention_mask"])

    def test_spans_are_valid_and_contiguous(self, adapter):
        out = adapter.encode_sentence("كتب الطالب الدرس")
        spans = out["word_to_surface_spans"]
        # Must pass the pipeline's own span validator (allows BOS/EOS gaps).
        validate_spans(spans, total_ids=len(out["input_ids"]))
        # Each word maps to a non-empty contiguous span here (no truncation).
        for s, e in spans:
            assert 0 <= s < e <= len(out["input_ids"])

    def test_morph_analyses_are_dicts(self, adapter):
        out = adapter.encode_sentence("الكتاب")
        wa = out["morph_analyses"][0]
        assert isinstance(wa, list) and wa
        assert "prob" in wa[0] and "root" in wa[0]
        # F2 regression flows through the adapter too.
        assert wa[0]["root"] == "كتب"

    def test_truncation_clamps_spans(self, adapter):
        out = adapter.encode_sentence("كتب الطالب الدرس والمعلم", max_length=4)
        assert len(out["input_ids"]) == 4
        validate_spans(out["word_to_surface_spans"], total_ids=4)

    def test_batch_call(self, adapter):
        batch = adapter(["كتب الطالب", "قرأ المعلم الكتاب"])
        assert len(batch["input_ids"]) == 2
        assert batch["num_words"] == [2, 3]

    def test_empty_text(self, adapter):
        out = adapter.encode_sentence("")
        assert out["num_words"] == 0
        assert out["word_to_surface_spans"] == []

    def test_contextual_analyzer_receives_lossless_classical_text(self):
        class Recorder:
            def analyze_sentence(self, words, context=None):
                self.words = words
                self.context = context
                return [[MorphAnalysis(prob=1.0, root="علو", is_contextual=True)]]

        recorder = Recorder()
        adapter = SarfTokBaseAdapter(
            MockFastTokenizer(),
            SarfTokConfig(),
            analyzer=recorder,
        )
        out = adapter.encode_sentence("أعلى")
        assert recorder.words == ["أعلى"]
        assert recorder.context == "أعلى"
        assert out["morph_analyses"][0][0]["is_contextual"] is True


# ---------------------------------------------------------------------------
# End-to-end fusion with the mock tokenizer + a real embedding table
# ---------------------------------------------------------------------------


class TestFusionEndToEnd:
    def test_adapter_feeds_hybrid_embedding(self, config):
        from sarftok.llm_integration.hf_data_collator import SarfTokDataCollator
        from sarftok.llm_integration.hf_modeling_embeddings import HybridSarfTokEmbedding
        from sarftok.morph_encoder import MorphEncoder
        from sarftok.morph_vocab import MorphVocab
        from sarftok.probabilistic_embedder import ProbabilisticEmbedder

        adapter = SarfTokBaseAdapter(MockFastTokenizer(), config)
        collator = SarfTokDataCollator(pad_token_id=0)
        feats = [adapter.encode_sentence("كتب الطالب الدرس"),
                 adapter.encode_sentence("قرأ المعلم")]
        batch = collator(feats)

        vocab = MorphVocab(root_min_freq=1, pattern_min_freq=1)
        base_emb = nn.Embedding(1000, 16)  # sized to the base model's vocab
        hybrid = HybridSarfTokEmbedding(
            base_emb,
            MorphEncoder(vocab=vocab, hidden_dim=16),
            ProbabilisticEmbedder(hidden_dim=16, alpha=0.5, entropy_gating=True),
        )
        fused = hybrid(
            batch["input_ids"],
            batch["word_to_surface_spans"],
            batch["morph_analyses"],
        )
        assert fused.shape == (2, batch["input_ids"].size(1), 16)
        fused.sum().backward()
        # With an empty root vocab the composite root path is unused; morphology
        # flows through the root-char fallback table, which must receive gradient.
        assert hybrid.morph_encoder.rootchar_emb.weight.grad is not None


# ---------------------------------------------------------------------------
# Opt-in: real HuggingFace fast tokenizer (skipped offline)
# ---------------------------------------------------------------------------


def _load_real_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("gpt2")
    except Exception:  # noqa: BLE001 — offline / not installed
        return None


@pytest.mark.skipif(_load_real_tokenizer() is None, reason="gpt2 tokenizer unavailable offline")
def test_real_tokenizer_alignment():
    config = SarfTokConfig(analyzer_backend="heuristic", top_k=3)
    tok = _load_real_tokenizer()
    adapter = SarfTokBaseAdapter(tok, config)
    out = adapter.encode_sentence("كتب الطالب الدرس")
    validate_spans(out["word_to_surface_spans"], total_ids=len(out["input_ids"]))
    assert out["num_words"] == 3
    assert len(out["morph_analyses"]) == 3
