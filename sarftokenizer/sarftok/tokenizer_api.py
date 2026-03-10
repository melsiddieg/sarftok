"""
SarfTokTokenizer — high-level pipeline combining all components.

Pipeline:
  text → normalize → segment → surface-tokenize + alignment → analyze → bundle
"""
from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from sarftok import MorphAnalysis, SentenceTokenization, WordMorphBundle
from sarftok.config import SarfTokConfig
from sarftok.normalizer import ArabicNormalizer
from sarftok.segmenter import ArabicSegmenter


def _build_analyzer(config: SarfTokConfig):
    """Factory function — returns the configured MorphAnalyzer backend."""
    if config.analyzer_backend == "heuristic":
        from sarftok.morph_analyzer.heuristics import HeuristicMorphAnalyzer
        return HeuristicMorphAnalyzer(
            top_k=config.top_k,
            confidence_threshold=config.confidence_threshold,
            temperature=config.analyzer_temperature,
        )
    elif config.analyzer_backend == "camel":
        from sarftok.morph_analyzer.camel_wrapper import CamelMorphAnalyzer
        return CamelMorphAnalyzer(
            top_k=config.top_k,
            confidence_threshold=config.confidence_threshold,
            temperature=config.analyzer_temperature,
        )
    elif config.analyzer_backend == "distilled":
        from sarftok.morph_analyzer.distilled_tagger import DistilledMorphTagger
        return DistilledMorphTagger(top_k=config.top_k)
    else:
        raise ValueError(f"Unknown analyzer backend: {config.analyzer_backend!r}")


def _build_surface_tokenizer(config: SarfTokConfig):
    """Return a SurfaceTokenizer if model path is provided."""
    if config.surface_model_path and Path(config.surface_model_path).exists():
        from sarftok.surface_tokenizer import SurfaceTokenizer
        return SurfaceTokenizer(config.surface_model_path)
    return None


class SarfTokTokenizer:
    """Main tokenizer — wraps the full SarfTok pipeline.

    Parameters
    ----------
    config:
        :class:`~sarftok.config.SarfTokConfig` with all settings.
    word_cache_size:
        Maximum number of word→analyses pairs to cache (LRU).  Set to 0
        to disable caching.
    """

    def __init__(
        self,
        config: SarfTokConfig,
        word_cache_size: int = 50_000,
    ) -> None:
        self.config = config
        self.normalizer = ArabicNormalizer(
            mode=config.norm_mode,
            strip_diacritics=config.strip_diacritics,
        )
        self.segmenter = ArabicSegmenter()
        self.surface_tokenizer = _build_surface_tokenizer(config)
        self.analyzer = _build_analyzer(config)
        self._word_cache: dict = {}
        self._cache_size = word_cache_size

    # ------------------------------------------------------------------
    # Analysis caching
    # ------------------------------------------------------------------

    def _analyze_word_cached(self, word: str) -> List[MorphAnalysis]:
        if self._cache_size > 0:
            if word in self._word_cache:
                return self._word_cache[word]
        result = self.analyzer.analyze_word(word)
        if self._cache_size > 0:
            if len(self._word_cache) >= self._cache_size:
                # Simple eviction: clear oldest half
                keys = list(self._word_cache.keys())
                for k in keys[: self._cache_size // 2]:
                    del self._word_cache[k]
            self._word_cache[word] = result
        return result

    # ------------------------------------------------------------------
    # Core tokenization
    # ------------------------------------------------------------------

    def tokenize_sentence(
        self,
        text: str,
        raw_text: Optional[str] = None,
    ) -> SentenceTokenization:
        """Tokenize a single sentence or short text.

        Parameters
        ----------
        text:
            Input Arabic text (will be normalised internally).
        raw_text:
            Optional override for the ``raw_text`` field in the output.

        Returns
        -------
        SentenceTokenization
        """
        raw = raw_text if raw_text is not None else text
        normalised = self.normalizer.normalise(text)
        words_list = self.segmenter.tokenize_words(normalised)

        # Surface tokenize with alignment
        if self.surface_tokenizer is not None:
            surface_ids, spans = self.surface_tokenizer.encode_with_alignment(words_list)
        else:
            # Fallback: treat each word as a single fake token ID
            surface_ids = list(range(len(words_list)))
            spans = [(i, i + 1) for i in range(len(words_list))]

        # Morphological analysis (cached per-word)
        analyses_per_word = [self._analyze_word_cached(w) for w in words_list]

        # Build surface piece strings
        if self.surface_tokenizer is not None:
            pieces_per_word = [
                self.surface_tokenizer.encode_pieces(w) for w in words_list
            ]
        else:
            pieces_per_word = [[w] for w in words_list]

        # Assemble WordMorphBundle list
        word_bundles = []
        for i, (raw_w, norm_w, analyses, pieces) in enumerate(
            zip(words_list, words_list, analyses_per_word, pieces_per_word)
        ):
            word_bundles.append(
                WordMorphBundle(
                    raw_word=raw_w,
                    normalized_word=norm_w,
                    analyses=analyses,
                    surface_pieces=pieces,
                    word_index=i,
                )
            )

        return SentenceTokenization(
            raw_text=raw,
            normalized_text=normalised,
            words=word_bundles,
            surface_input_ids=surface_ids,
            word_to_surface_spans=spans,
        )

    def tokenize_batch(
        self,
        texts: List[str],
        num_workers: int = 1,
    ) -> List[SentenceTokenization]:
        """Tokenize a list of texts, optionally in parallel.

        Warning: multiprocessing is safe only when using the heuristic
        backend.  CAMeL Tools is not fork-safe — use num_workers=1 with it.
        """
        if num_workers <= 1 or len(texts) < 4:
            return [self.tokenize_sentence(t) for t in texts]

        # For multiprocessing, use a simple map (no shared state with functools)
        results = [None] * len(texts)
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(self.tokenize_sentence, t): i for i, t in enumerate(texts)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save_config(self, path: str) -> None:
        self.config.to_json(path)

    @classmethod
    def from_config(cls, path: str) -> "SarfTokTokenizer":
        return cls(SarfTokConfig.from_json(path))
