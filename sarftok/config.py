"""
SarfTokConfig — master configuration dataclass.

Every tunable knob in the SarfTok pipeline lives here so that experiments
can be reproduced by serialising a single config object.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class SarfTokConfig:
    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    norm_mode: Literal["classical_strict", "classical_soft", "msa_soft"] = (
        "classical_strict"
    )
    """Arabic text normalisation mode."""

    strip_diacritics: bool = False
    """Strip harakat when True (for ablations only)."""

    # ------------------------------------------------------------------
    # Surface tokenizer
    # ------------------------------------------------------------------
    surface_vocab_size: int = 32_000
    """Target vocabulary size for the SentencePiece surface tokenizer."""

    surface_model_type: Literal["bpe", "unigram"] = "bpe"
    """SentencePiece model type."""

    surface_model_path: Optional[str] = None
    """Path to a pre-trained SentencePiece .model file."""

    # ------------------------------------------------------------------
    # Morphological analyzer
    # ------------------------------------------------------------------
    analyzer_backend: Literal["heuristic", "camel", "distilled"] = "heuristic"
    """Which morphological analyzer backend to use."""

    top_k: int = 3
    """Maximum analyses to keep per word."""

    confidence_threshold: float = 0.05
    """Discard analyses with probability below this value."""

    analyzer_temperature: float = 1.0
    """Temperature for softmax re-normalisation of analyzer scores."""

    # ------------------------------------------------------------------
    # Morphological vocabulary
    # ------------------------------------------------------------------
    morph_vocab_path: Optional[str] = None
    """Path to a pre-built morph vocabulary JSON file."""

    root_min_freq: int = 5
    """Minimum corpus frequency to give a root its own composite token."""

    pattern_min_freq: int = 3
    """Minimum corpus frequency to give a pattern its own composite token."""

    # ------------------------------------------------------------------
    # Embedding fusion
    # ------------------------------------------------------------------
    hidden_dim: int = 2048
    """Embedding / hidden dimension of the transformer."""

    fusion_mode: Literal["broadcast", "first_piece"] = "broadcast"
    """
    broadcast   — add morphology embedding to ALL surface pieces of a word.
    first_piece — add only to the FIRST surface piece.
    """

    alpha: float = 0.5
    """Base scalar weight for morphology channel (α₀)."""

    learnable_alpha: bool = False
    """If True, α is a learned scalar parameter."""

    entropy_gating: bool = True
    """If True, scale α by exp(-β·H(p)) per word."""

    beta: float = 1.0
    """Entropy decay rate β in α_word = α₀ · exp(-β·H(p))."""

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------
    ortho_lambda: float = 0.01
    """Coefficient for the orthogonality regularisation loss."""

    ortho_min_confidence: float = 0.5
    """Only apply ortho loss for analyses with prob ≥ this threshold."""

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    num_workers: int = 4
    """Parallel workers for offline corpus preprocessing."""

    shard_size: int = 10_000
    """Sentences per output shard."""

    output_format: Literal["jsonl", "parquet"] = "jsonl"
    """Serialisation format for corpus shards."""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def from_dict(cls, d: dict) -> "SarfTokConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str | Path) -> "SarfTokConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # Experiment presets
    # ------------------------------------------------------------------
    @classmethod
    def baseline_a(cls, **kwargs) -> "SarfTokConfig":
        """Surface tokenizer only — no morphology."""
        return cls(alpha=0.0, entropy_gating=False, top_k=0, **kwargs)

    @classmethod
    def baseline_b(cls, **kwargs) -> "SarfTokConfig":
        """Surface + deterministic top-1 morphology."""
        return cls(top_k=1, entropy_gating=False, **kwargs)

    @classmethod
    def main_c(cls, **kwargs) -> "SarfTokConfig":
        """Surface + probabilistic top-3 morphology."""
        return cls(top_k=3, entropy_gating=True, **kwargs)

    @classmethod
    def main_d(cls, **kwargs) -> "SarfTokConfig":
        """Surface + probabilistic top-3 + orthogonality loss."""
        return cls(top_k=3, entropy_gating=True, ortho_lambda=0.01, **kwargs)
