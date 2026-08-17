"""
base_tokenizer_adapter.py — align SarfTok morphology to an *existing* model's tokenizer.

Motivation
----------
:class:`~sarftok.llm_integration.hf_modeling_embeddings.HybridSarfTokCausalLM` fuses a
per-word morphology embedding onto surface token embeddings.  The surface IDs and the base
model's embedding table must therefore share the same vocabulary.  For pretraining from
scratch that vocabulary is SarfTok's own SentencePiece surface model.  To *fine-tune an
existing* model (e.g. Qwen), the surface IDs must instead be the base model's own token IDs.

This adapter produces exactly the collator-compatible encoding contract used elsewhere
(``input_ids`` / ``attention_mask`` / ``word_to_surface_spans`` / ``morph_analyses``), but
built against a base HuggingFace *fast* tokenizer:

1. Whitespace pre-tokenise the text into words (the units morphology is defined over).
2. Encode with ``is_split_into_words=True`` and read ``word_ids()`` to map every base token
   back to its word — giving a **contiguous** ``(start, end)`` span per word, the invariant
   :class:`~sarftok.probabilistic_embedder.ProbabilisticEmbedder` relies on.
3. Run the configured SarfTok analyzer through its sentence-level API so a
   contextual backend can condition candidate probabilities on the full input.

The base model's real embedding table then handles ``input_ids`` and morphology is added on
top — IDs and embeddings stay consistent.

Usage::

    from transformers import AutoTokenizer
    base_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")   # fast tokenizer
    adapter = SarfTokBaseAdapter(base_tok, SarfTokConfig())
    batch = adapter(["كتب الطالب الدرس", "قرأ المعلم الكتاب"])
    # feed through SarfTokDataCollator → HybridSarfTokCausalLM
"""
from __future__ import annotations

from typing import Any

from sarftok.config import SarfTokConfig
from sarftok.normalizer import ArabicNormalizer
from sarftok.serialization import analysis_to_dict
from sarftok.tokenizer_api import _build_analyzer


def _spans_from_word_ids(
    word_ids: list[int | None],
    num_words: int,
) -> list[tuple[int, int]]:
    """Build one contiguous ``(start, end)`` span per word from HF ``word_ids()``.

    Tokens whose ``word_ids`` entry is ``None`` (special tokens) are skipped, producing a
    permitted gap.  Words that produced no tokens (e.g. dropped by truncation) get an empty
    span anchored at the running cursor, which downstream fusion skips.  Anchoring at the
    cursor (rather than ``(0, 0)``) keeps span starts monotonically non-decreasing so the
    result still passes :func:`sarftok.alignment.validate_spans`.
    """
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for tok_pos, w in enumerate(word_ids):
        if w is None:
            continue
        if w not in first:
            first[w] = tok_pos
        last[w] = tok_pos

    spans: list[tuple[int, int]] = []
    cursor = 0
    for w in range(num_words):
        if w in first:
            end = last[w] + 1
            spans.append((first[w], end))
            cursor = end
        else:
            spans.append((cursor, cursor))
    return spans


class SarfTokBaseAdapter:
    """Encode text against a base model's tokenizer, with SarfTok morphology attached.

    Parameters
    ----------
    base_tokenizer:
        A HuggingFace *fast* tokenizer (``PreTrainedTokenizerFast``); ``word_ids()`` requires
        the fast implementation.
    config:
        :class:`~sarftok.config.SarfTokConfig` (selects analyzer backend, normalisation, etc.).
    normalize_for_analysis:
        If True (default), each word is normalised with :class:`ArabicNormalizer` before
        morphological analysis.  The base tokenizer always sees the *original* words so the
        language-model text is never altered.
    word_cache_size:
        Max cached word→analyses entries (LRU-ish; 0 disables).
    analyzer:
        Optional analyzer instance. Inject a context-capable analyzer or CAMeL
        disambiguator wrapper here; otherwise the configured backend is built.
    """

    def __init__(
        self,
        base_tokenizer: Any,
        config: SarfTokConfig,
        *,
        normalize_for_analysis: bool = True,
        word_cache_size: int = 50_000,
        analyzer=None,
    ) -> None:
        if not getattr(base_tokenizer, "is_fast", False):
            raise ValueError(
                "SarfTokBaseAdapter requires a fast tokenizer (PreTrainedTokenizerFast); "
                "word_ids() is unavailable on the slow implementation."
            )
        self.base_tok = base_tokenizer
        self.config = config
        self.normalizer = ArabicNormalizer(
            mode=config.norm_mode,
            strip_diacritics=config.strip_diacritics,
            normalise_alef=config.normalise_alef,
            normalise_ya=config.normalise_ya,
        )
        self.analyzer = analyzer if analyzer is not None else _build_analyzer(config)
        self.normalize_for_analysis = normalize_for_analysis
        self._cache: dict = {}
        self._cache_size = word_cache_size

    # ------------------------------------------------------------------
    # Per-word analysis (cached)
    # ------------------------------------------------------------------

    def _analyze(self, word: str) -> list:
        key = word
        if self._cache_size > 0 and key in self._cache:
            return self._cache[key]
        target = self.normalizer.normalise(word) if self.normalize_for_analysis else word
        result = self.analyzer.analyze_word(target) if target else []
        if self._cache_size > 0:
            if len(self._cache) >= self._cache_size:
                for k in list(self._cache.keys())[: self._cache_size // 2]:
                    del self._cache[k]
            self._cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Core encode
    # ------------------------------------------------------------------

    def encode_sentence(
        self,
        text: str,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        """Encode a single text into the collator-compatible dict."""
        words = text.split()

        if not words:
            enc = self.base_tok(
                text,
                truncation=max_length is not None,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            )
            ids = enc["input_ids"]
            return {
                "input_ids": ids,
                "attention_mask": enc.get("attention_mask", [1] * len(ids)),
                "word_to_surface_spans": [],
                "morph_analyses": [],
                "num_words": 0,
            }

        enc = self.base_tok(
            words,
            is_split_into_words=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", [1] * len(input_ids))
        spans = _spans_from_word_ids(enc.word_ids(), len(words))

        if self.config.contextual_analysis:
            analysis_words = [
                self.normalizer.normalise(w) if self.normalize_for_analysis else w
                for w in words
            ]
            analyses = self.analyzer.analyze_sentence(
                analysis_words,
                context=" ".join(analysis_words),
            )
        else:
            analyses = [self._analyze(w) for w in words]
        morph_analyses = [
            [analysis_to_dict(a) for a in word_analyses]
            for word_analyses in analyses
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_to_surface_spans": spans,
            "morph_analyses": morph_analyses,
            "num_words": len(words),
        }

    def __call__(
        self,
        texts: str | list[str],
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[Any]]:
        """Encode one or more texts; returns a batch dict (one list per field)."""
        if isinstance(texts, str):
            texts = [texts]
        encodings = [
            self.encode_sentence(t, max_length=max_length, add_special_tokens=add_special_tokens)
            for t in texts
        ]
        return {
            "input_ids": [e["input_ids"] for e in encodings],
            "attention_mask": [e["attention_mask"] for e in encodings],
            "word_to_surface_spans": [e["word_to_surface_spans"] for e in encodings],
            "morph_analyses": [e["morph_analyses"] for e in encodings],
            "num_words": [e["num_words"] for e in encodings],
        }
