"""
hf_tokenizer_wrapper.py — HuggingFace-compatible SarfTok tokenizer.

This wrapper presents a standard HF tokenizer interface but additionally
returns morph metadata alongside the standard input_ids / attention_mask.
It is NOT a PreTrainedTokenizer subclass because that would force us to
adopt BPE only; instead it wraps SarfTokTokenizer and matches the dict
output contract expected by SarfTokDataCollator.

Usage::

    tok = SarfTokHFTokenizer(config)
    batch = tok(["كتب الطالب الدرس", "قرأ المعلم الكتاب"])
    # batch["input_ids"], batch["attention_mask"],
    # batch["word_to_surface_spans"], batch["morph_analyses"]
"""
from __future__ import annotations

from typing import Any

from sarftok import SentenceTokenization
from sarftok.config import SarfTokConfig
from sarftok.serialization import analysis_to_dict
from sarftok.tokenizer_api import SarfTokTokenizer


class SarfTokHFTokenizer:
    """HF-style tokenizer returning surface IDs plus morphology metadata.

    Parameters
    ----------
    config:
        :class:`~sarftok.config.SarfTokConfig`
    """

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    def __init__(self, config: SarfTokConfig) -> None:
        self.config = config
        self._tokenizer = SarfTokTokenizer(config)

    # ------------------------------------------------------------------
    # Core encode
    # ------------------------------------------------------------------

    def encode_sentence(self, text: str) -> dict[str, Any]:
        """Return a single-sentence encoding dict."""
        sent: SentenceTokenization = self._tokenizer.tokenize_sentence(text)
        return {
            "input_ids": sent.surface_input_ids,
            "attention_mask": [1] * len(sent.surface_input_ids),
            "word_to_surface_spans": list(sent.word_to_surface_spans),
            "morph_analyses": [
                [analysis_to_dict(a) for a in w.analyses] for w in sent.words
            ],
            "num_words": len(sent.words),
        }

    def __call__(
        self,
        texts: str | list[str],
        **kwargs,
    ) -> dict[str, list[Any]]:
        """Encode one or more texts. Returns a batch dict (list per field)."""
        if isinstance(texts, str):
            texts = [texts]

        encodings = [self.encode_sentence(t) for t in texts]
        return {
            "input_ids": [e["input_ids"] for e in encodings],
            "attention_mask": [e["attention_mask"] for e in encodings],
            "word_to_surface_spans": [e["word_to_surface_spans"] for e in encodings],
            "morph_analyses": [e["morph_analyses"] for e in encodings],
            "num_words": [e["num_words"] for e in encodings],
        }

    # ------------------------------------------------------------------
    # HF compatibility helpers
    # ------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        if self._tokenizer.surface_tokenizer is not None:
            return self._tokenizer.surface_tokenizer.vocab_size
        return 0

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        if self._tokenizer.surface_tokenizer is not None:
            return [self._tokenizer.surface_tokenizer.id_to_piece(i) for i in ids]
        return [str(i) for i in ids]

    def save_pretrained(self, save_dir: str) -> None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.config.to_json(os.path.join(save_dir, "sarftok_config.json"))

    @classmethod
    def from_pretrained(cls, save_dir: str) -> SarfTokHFTokenizer:
        import os
        return cls(SarfTokConfig.from_json(os.path.join(save_dir, "sarftok_config.json")))
