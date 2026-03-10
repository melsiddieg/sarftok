"""
ProbabilisticEmbedder — inject morphology into surface embeddings.

Two fusion modes
----------------
broadcast   : add morphology embedding to ALL surface pieces of each word.
first_piece : add morphology embedding only to the FIRST surface piece.

Entropy-aware gating
--------------------
When entropy_gating=True:
    α_word = α₀ · exp(-β · H(p))
where H(p) = -Σ p_i · log(p_i) is the morphological entropy.
High entropy (uncertain morphology) → smaller α → surface dominates.
"""
from __future__ import annotations

import math
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from sarftok import MorphAnalysis


def _morphological_entropy(analyses: List[MorphAnalysis]) -> float:
    """Compute Shannon entropy of a word's analysis distribution."""
    entropy = 0.0
    for a in analyses:
        p = max(a.prob, 1e-9)
        entropy -= p * math.log(p)
    return entropy


class ProbabilisticEmbedder(nn.Module):
    """Fuse morphology embeddings into the surface embedding sequence.

    Parameters
    ----------
    hidden_dim:
        Embedding / hidden dimension.
    fusion_mode:
        ``"broadcast"`` or ``"first_piece"``.
    alpha:
        Base morphology weight α₀.
    learnable_alpha:
        If True, alpha is an ``nn.Parameter`` initialised from *alpha*.
    entropy_gating:
        If True, scale αword = α₀ · exp(-β · H(p)).
    beta:
        Entropy decay rate.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        fusion_mode: Literal["broadcast", "first_piece"] = "broadcast",
        alpha: float = 0.5,
        learnable_alpha: bool = False,
        entropy_gating: bool = True,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.entropy_gating = entropy_gating
        self.beta = beta

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def _effective_alpha(self, analyses: Optional[List[MorphAnalysis]]) -> float:
        """Return the effective alpha for a word given its analyses."""
        base = float(self.alpha)
        if not self.entropy_gating or not analyses:
            return base
        H = _morphological_entropy(analyses)
        return base * math.exp(-self.beta * H)

    # ------------------------------------------------------------------
    # Forward (single-sentence, for clarity and debugging)
    # ------------------------------------------------------------------

    def forward_sentence(
        self,
        surface_emb: torch.Tensor,
        morph_emb: torch.Tensor,
        word_to_surface_spans: List[Tuple[int, int]],
        analyses: Optional[List[List[MorphAnalysis]]] = None,
    ) -> torch.Tensor:
        """Fuse morphology into surface embeddings for one sentence.

        Parameters
        ----------
        surface_emb:
            Shape ``(seq_len, hidden_dim)``.
        morph_emb:
            Shape ``(num_words, hidden_dim)`` — output of MorphEncoder.
        word_to_surface_spans:
            ``spans[j] = (start, end)`` for word j.
        analyses:
            Optional list-of-analyses per word (for entropy gating).

        Returns
        -------
        torch.Tensor  shape: (seq_len, hidden_dim)
        """
        fused = surface_emb.clone()
        num_words = len(word_to_surface_spans)

        for j in range(num_words):
            s, e = word_to_surface_spans[j]
            if s >= e:
                continue  # empty span — skip

            word_analyses = analyses[j] if analyses and j < len(analyses) else None
            eff_alpha = self._effective_alpha(word_analyses)

            if eff_alpha == 0.0:
                continue

            m_emb = morph_emb[j]  # (hidden_dim,)

            if self.fusion_mode == "broadcast":
                fused[s:e] = fused[s:e] + eff_alpha * m_emb.unsqueeze(0)
            else:  # first_piece
                fused[s] = fused[s] + eff_alpha * m_emb

        return fused

    # ------------------------------------------------------------------
    # Batched forward
    # ------------------------------------------------------------------

    def forward(
        self,
        surface_emb: torch.Tensor,
        morph_emb: torch.Tensor,
        word_to_surface_spans: List[List[Tuple[int, int]]],
        analyses_batch: Optional[List[List[List[MorphAnalysis]]]] = None,
    ) -> torch.Tensor:
        """Fuse morphology for a batch.

        Parameters
        ----------
        surface_emb:
            Shape ``(batch, seq_len, hidden_dim)``.
        morph_emb:
            Shape ``(batch, max_words, hidden_dim)`` from :class:`MorphEncoder`.
        word_to_surface_spans:
            Outer list = batch; inner list = spans per word.
        analyses_batch:
            Optional batch of per-word analyses for entropy gating.

        Returns
        -------
        torch.Tensor  shape: (batch, seq_len, hidden_dim)
        """
        batch_size = surface_emb.size(0)
        outputs = []
        for b in range(batch_size):
            ans = analyses_batch[b] if analyses_batch else None
            out = self.forward_sentence(
                surface_emb=surface_emb[b],
                morph_emb=morph_emb[b],
                word_to_surface_spans=word_to_surface_spans[b],
                analyses=ans,
            )
            outputs.append(out)
        return torch.stack(outputs, dim=0)
