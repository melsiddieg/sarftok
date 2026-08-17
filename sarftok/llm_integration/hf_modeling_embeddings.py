"""
hf_modeling_embeddings.py — HybridSarfTokEmbedding and HybridSarfTokCausalLM.

Architecture
------------
1. Standard token embed lookup → surface_emb  (batch, seq_len, d_model)
2. MorphEncoder encodes top-k analyses per word → morph_emb  (batch, max_words, d_model)
3. ProbabilisticEmbedder fuses morph_emb into surface_emb
4. Pass fused inputs_embeds to base transformer (no internals changed)

Supported base models
---------------------
* Any CausalLM with `.get_input_embeddings()` (Llama, GPTNeoX, GPT-2, etc.)

The wrapper never modifies the base model's weights by default.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from sarftok import MorphAnalysis
from sarftok.config import SarfTokConfig
from sarftok.morph_encoder import MorphEncoder
from sarftok.morph_vocab import MorphVocab
from sarftok.probabilistic_embedder import ProbabilisticEmbedder
from sarftok.serialization import dict_to_analysis

# ---------------------------------------------------------------------------
# Convert collator's morph_analyses (List[List[dict]]) → List[List[MorphAnalysis]]
# ---------------------------------------------------------------------------

def _dicts_to_analyses(
    morph_analyses_batch: list[list[list[dict]]],
) -> list[list[list[MorphAnalysis]]]:
    """Convert raw dict analyses (from collator) to MorphAnalysis objects."""
    return [
        [
            [dict_to_analysis(a) for a in word_analyses]
            for word_analyses in sentence_analyses
        ]
        for sentence_analyses in morph_analyses_batch
    ]


# ---------------------------------------------------------------------------
# HybridSarfTokEmbedding
# ---------------------------------------------------------------------------


class HybridSarfTokEmbedding(nn.Module):
    """Drop-in embedding layer that fuses surface and morphology embeddings.

    Parameters
    ----------
    base_embedding:
        The base model's ``nn.Embedding`` (token ID → d_model).
    morph_encoder:
        Pre-built :class:`~sarftok.morph_encoder.MorphEncoder`.
    probabilistic_embedder:
        Pre-built :class:`~sarftok.probabilistic_embedder.ProbabilisticEmbedder`.
    """

    def __init__(
        self,
        base_embedding: nn.Embedding,
        morph_encoder: MorphEncoder,
        probabilistic_embedder: ProbabilisticEmbedder,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.base_embedding = base_embedding
        self.morph_encoder = morph_encoder
        self.probabilistic_embedder = probabilistic_embedder
        self.layer_norm = nn.LayerNorm(base_embedding.embedding_dim) if layer_norm else None

    def forward(
        self,
        input_ids: torch.Tensor,
        word_to_surface_spans: list[list[tuple[int, int]]],
        morph_analyses: list[list[list[Any]]] | None = None,
    ) -> torch.Tensor:
        """Return fused embeddings for the full batch.

        Parameters
        ----------
        input_ids:
            Shape ``(batch, seq_len)``.
        word_to_surface_spans:
            Per-example list of (start, end) spans.
        morph_analyses:
            Nested list of analyses dicts (batch × words × analyses).
            If None, morphology channel is skipped.

        Returns
        -------
        torch.Tensor  shape: (batch, seq_len, d_model)
        """
        # Step 1: base surface embeddings
        surface_emb = self.base_embedding(input_ids)  # (B, T, D)

        if morph_analyses is None or self.probabilistic_embedder.alpha == 0:
            return surface_emb

        # Step 2: convert dicts → MorphAnalysis objects
        analyses_obj = _dicts_to_analyses(morph_analyses)

        # Step 3: morphology embeddings
        morph_emb = self.morph_encoder(analyses_obj)  # (B, W, D)

        # Step 4: fuse
        fused = self.probabilistic_embedder(
            surface_emb=surface_emb,
            morph_emb=morph_emb,
            word_to_surface_spans=word_to_surface_spans,
            analyses_batch=analyses_obj,
        )
        return self.layer_norm(fused) if self.layer_norm is not None else fused


# ---------------------------------------------------------------------------
# HybridSarfTokCausalLM
# ---------------------------------------------------------------------------


class HybridSarfTokCausalLM(nn.Module):
    """Wraps any HF CausalLM and injects hybrid embeddings.

    This wrapper:
    1. Replaces the base model's default embedding with HybridSarfTokEmbedding.
    2. Calls the base model with ``inputs_embeds`` instead of ``input_ids``.
    3. Adds an optional orthogonality loss contribution.

    Parameters
    ----------
    base_model:
        A HuggingFace CausalLM (e.g. LlamaForCausalLM, GPTNeoXForCausalLM).
    config:
        :class:`~sarftok.config.SarfTokConfig`.
    vocab:
        :class:`~sarftok.morph_vocab.MorphVocab`.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: SarfTokConfig,
        vocab: MorphVocab,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.sarftok_config = config

        base_emb: nn.Embedding = base_model.get_input_embeddings()
        hidden_dim = base_emb.embedding_dim

        morph_encoder = MorphEncoder(
            vocab=vocab,
            hidden_dim=hidden_dim,
            use_composite_root=True,
            use_root_pattern_interaction=config.root_pattern_interaction,
        )
        prob_embedder = ProbabilisticEmbedder(
            hidden_dim=hidden_dim,
            fusion_mode=config.fusion_mode,
            alpha=config.alpha,
            learnable_alpha=config.learnable_alpha,
            entropy_gating=config.entropy_gating,
            beta=config.beta,
            entropy_normalize=config.entropy_normalize,
            broadcast_piece_scaling=config.broadcast_piece_scaling,
        )
        self.hybrid_embedding = HybridSarfTokEmbedding(
            base_embedding=base_emb,
            morph_encoder=morph_encoder,
            probabilistic_embedder=prob_embedder,
            layer_norm=config.fusion_layer_norm,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        word_to_surface_spans: list[list[tuple[int, int]]] | None = None,
        morph_analyses: list[list[list[Any]]] | None = None,
        template_pairs: list[tuple[Any, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Forward pass through the hybrid embedding + base transformer.

        Returns
        -------
        dict with keys: loss, logits, (+ auxiliary losses if configured)
        """
        # Compute fused embeddings
        inputs_embeds = self.hybrid_embedding(
            input_ids=input_ids,
            word_to_surface_spans=word_to_surface_spans or [],
            morph_analyses=morph_analyses,
        )

        # Forward through base model (no input_ids — use inputs_embeds)
        outputs = self.base_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        auxiliary: dict[str, torch.Tensor] = {}
        if labels is not None and morph_analyses is not None:
            analyses_obj = _dicts_to_analyses(morph_analyses)
            if self.sarftok_config.ortho_lambda > 0.0:
                from sarftok.llm_integration.losses import orthogonality_loss

                auxiliary["ortho_loss"] = orthogonality_loss(
                    encoder=self.hybrid_embedding.morph_encoder,
                    analyses_batch=analyses_obj,
                    min_confidence=self.sarftok_config.ortho_min_confidence,
                    lambda_=self.sarftok_config.ortho_lambda,
                )

        if labels is not None and template_pairs and self.sarftok_config.template_lambda > 0:
            from sarftok.llm_integration.losses import template_parallelism_loss

            object_pairs = [
                (
                    dict_to_analysis(before) if isinstance(before, dict) else before,
                    dict_to_analysis(after) if isinstance(after, dict) else after,
                )
                for before, after in template_pairs
            ]
            auxiliary["template_loss"] = template_parallelism_loss(
                self.hybrid_embedding.morph_encoder,
                object_pairs,
                lambda_=self.sarftok_config.template_lambda,
            )

        if auxiliary:
            lm = outputs.loss if hasattr(outputs, "loss") else None
            total_loss = sum(auxiliary.values(), lm if lm is not None else 0.0)
            return {
                "loss": total_loss,
                "logits": outputs.logits,
                "lm_loss": lm,
                **auxiliary,
            }

        return outputs

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> nn.Embedding:
        return self.hybrid_embedding.base_embedding

    def get_output_embeddings(self) -> nn.Module | None:
        return self.base_model.get_output_embeddings()

    def save_pretrained(self, save_dir: str) -> None:
        """Save base model + hybrid embedding weights."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.base_model.save_pretrained(save_dir)
        torch.save(
            {
                "morph_encoder": self.hybrid_embedding.morph_encoder.state_dict(),
                "probabilistic_embedder": self.hybrid_embedding.probabilistic_embedder.state_dict(),
                "layer_norm": (
                    self.hybrid_embedding.layer_norm.state_dict()
                    if self.hybrid_embedding.layer_norm is not None
                    else None
                ),
            },
            os.path.join(save_dir, "sarftok_embeddings.pt"),
        )
        self.sarftok_config.to_json(os.path.join(save_dir, "sarftok_config.json"))

    def load_sarftok_weights(self, save_dir: str) -> None:
        """Load previously saved SarfTok embedding weights."""
        import os
        state = torch.load(
            os.path.join(save_dir, "sarftok_embeddings.pt"),
            map_location="cpu",
        )
        self.hybrid_embedding.morph_encoder.load_state_dict(state["morph_encoder"])
        self.hybrid_embedding.probabilistic_embedder.load_state_dict(
            state["probabilistic_embedder"]
        )
        if self.hybrid_embedding.layer_norm is not None and state.get("layer_norm") is not None:
            self.hybrid_embedding.layer_norm.load_state_dict(state["layer_norm"])
