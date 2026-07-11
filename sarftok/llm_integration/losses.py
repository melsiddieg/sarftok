"""
losses.py — training objectives for SarfTok.

Implemented losses
------------------
1. lm_loss        — standard causal next-token cross-entropy (delegate to HF)
2. orthogonality_loss — penalise cosine similarity between root and pattern vecs
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sarftok import MorphAnalysis
from sarftok.morph_encoder import MorphEncoder


def lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard causal LM cross-entropy loss.

    Parameters
    ----------
    logits:
        Shape ``(batch, seq_len, vocab_size)``.
    labels:
        Shape ``(batch, seq_len)``.  Positions to ignore have value *ignore_index*.

    Returns
    -------
    torch.Tensor  scalar
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        labels.view(B * T),
        ignore_index=ignore_index,
    )


def orthogonality_loss(
    encoder: MorphEncoder,
    analyses_batch: list[list[list[MorphAnalysis]]],
    min_confidence: float = 0.5,
    lambda_: float = 0.01,
) -> torch.Tensor:
    """Penalise cosine similarity between root and pattern embeddings.

    For each high-confidence analysis:
        L_ortho = (cosine(E_root, E_pat))²

    Batch loss = λ · mean over all qualifying (root, pattern) pairs.

    Parameters
    ----------
    encoder:
        :class:`~sarftok.morph_encoder.MorphEncoder`.
    analyses_batch:
        Batch of per-word analyses.
    min_confidence:
        Only penalise analyses with prob ≥ this value.
    lambda_:
        Loss coefficient.

    Returns
    -------
    torch.Tensor  scalar
    """
    device = encoder.root_emb.weight.device
    cosines_sq: list[torch.Tensor] = []

    for sent in analyses_batch:
        for word_analyses in sent:
            for a in word_analyses:
                if a.prob < min_confidence:
                    continue
                if a.root is None or a.is_oov_root:
                    continue
                if a.is_oov_pattern:
                    continue

                e_root = encoder._root_embedding(a)         # (D,)
                e_pat = encoder._pattern_embedding(a)       # (D,)

                cos = F.cosine_similarity(
                    e_root.unsqueeze(0), e_pat.unsqueeze(0)
                )
                cosines_sq.append(cos ** 2)

    if not cosines_sq:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return lambda_ * torch.stack(cosines_sq).mean()


def template_parallelism_loss(
    encoder: MorphEncoder,
    pairs: list[tuple],
    lambda_: float = 0.01,
) -> torch.Tensor:
    """Optional Phase-3 loss encouraging template-parallel vector differences.

    Not implemented for MVP — returns 0 with a warning.

    Parameters
    ----------
    pairs:
        List of (analysis_a, analysis_b) pairs differing only by pattern.
    lambda_:
        Loss coefficient.
    """
    import warnings
    warnings.warn(
        "template_parallelism_loss is a Phase-3 feature and not yet implemented. "
        "Returning 0.",
        stacklevel=2,
    )
    device = encoder.root_emb.weight.device
    return torch.tensor(0.0, device=device)
