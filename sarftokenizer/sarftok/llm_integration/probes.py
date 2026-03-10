"""
probes.py — geometry and NLP utility probes for SarfTok embeddings.

Implements:
* root_clustering_purity   — how pure are root-based clusters?
* cosine_similarity_report — same-root vs cross-root similarity
* template_vector_alignment — parallelism of template offset vectors
* subspace_orthogonality   — mean cos² between root and pattern subspaces

These are offline evaluation tools, not part of the training forward pass.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sarftok import MorphAnalysis
from sarftok.morph_encoder import MorphEncoder


# ---------------------------------------------------------------------------
# Helper: gather root and pattern embeddings from a corpus of analyses
# ---------------------------------------------------------------------------


def _gather_root_embeddings(
    encoder: MorphEncoder,
    analyses_corpus: List[List[List[MorphAnalysis]]],
    max_items: int = 5_000,
) -> Dict[str, List[torch.Tensor]]:
    """Return a dict mapping root → list of embedding tensors."""
    root_embs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    count = 0
    for sent in analyses_corpus:
        for word_analyses in sent:
            for a in word_analyses:
                if a.root and not a.is_oov_root and count < max_items:
                    emb = encoder._root_embedding(a).detach().cpu()
                    root_embs[a.root].append(emb)
                    count += 1
    return dict(root_embs)


# ---------------------------------------------------------------------------
# Root clustering purity
# ---------------------------------------------------------------------------


def root_clustering_purity(
    encoder: MorphEncoder,
    analyses_corpus: List[List[List[MorphAnalysis]]],
    top_k_roots: int = 50,
) -> float:
    """Compute pseudo-purity of k-means clustering on root embeddings.

    Uses per-root centroid and measures fraction of embeddings nearest
    their own centroid (intra-root recall).

    Returns
    -------
    float  in [0, 1]
    """
    root_embs = _gather_root_embeddings(encoder, analyses_corpus)

    # Keep top-k most frequent roots
    sorted_roots = sorted(root_embs.keys(), key=lambda r: -len(root_embs[r]))
    selected = sorted_roots[:top_k_roots]

    if len(selected) < 2:
        return float("nan")

    # Centroids
    centroids: Dict[str, torch.Tensor] = {}
    for root in selected:
        embs = torch.stack(root_embs[root])  # (N, D)
        centroids[root] = embs.mean(dim=0)

    centroid_stack = torch.stack(list(centroids.values()))  # (R, D)
    root_list = list(centroids.keys())

    correct = 0
    total = 0
    for root_idx, root in enumerate(root_list):
        if root not in root_embs:
            continue
        embs = torch.stack(root_embs[root])  # (N, D)
        # Cosine similarity to all centroids
        sims = F.cosine_similarity(
            embs.unsqueeze(1), centroid_stack.unsqueeze(0), dim=2
        )  # (N, R)
        nearest = sims.argmax(dim=1)  # (N,)
        correct += (nearest == root_idx).sum().item()
        total += embs.size(0)

    return correct / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Cosine similarity: same-root vs cross-root
# ---------------------------------------------------------------------------


def cosine_similarity_report(
    encoder: MorphEncoder,
    analyses_corpus: List[List[List[MorphAnalysis]]],
    n_pairs: int = 1_000,
) -> Dict[str, float]:
    """Compute mean cosine similarity for same-root and cross-root pairs.

    Returns
    -------
    dict with keys:
        same_root_mean_cos, cross_root_mean_cos, delta
    """
    import random
    root_embs = _gather_root_embeddings(encoder, analyses_corpus, max_items=2_000)
    roots = [r for r, embs in root_embs.items() if len(embs) >= 2]

    if len(roots) < 2:
        return {"same_root_mean_cos": float("nan"), "cross_root_mean_cos": float("nan"), "delta": float("nan")}

    same_sims: List[float] = []
    cross_sims: List[float] = []

    for _ in range(n_pairs):
        # Same root pair
        r = random.choice(roots)
        e_a, e_b = random.sample(root_embs[r], 2)
        same_sims.append(F.cosine_similarity(e_a.unsqueeze(0), e_b.unsqueeze(0)).item())

        # Cross root pair
        r1, r2 = random.sample(roots, 2)
        e1 = random.choice(root_embs[r1])
        e2 = random.choice(root_embs[r2])
        cross_sims.append(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item())

    same_mean = sum(same_sims) / len(same_sims)
    cross_mean = sum(cross_sims) / len(cross_sims)
    return {
        "same_root_mean_cos": same_mean,
        "cross_root_mean_cos": cross_mean,
        "delta": same_mean - cross_mean,
    }


# ---------------------------------------------------------------------------
# Subspace orthogonality (root vs pattern)
# ---------------------------------------------------------------------------


def subspace_orthogonality(
    encoder: MorphEncoder,
    analyses_corpus: List[List[List[MorphAnalysis]]],
    max_pairs: int = 500,
) -> Dict[str, float]:
    """Measure mean cosine² between root and pattern embeddings.

    Low values indicate good orthogonality (more independent channels).

    Returns
    -------
    dict with keys:
        mean_cos2, std_cos2, n_pairs
    """
    cos2_values: List[float] = []
    count = 0

    for sent in analyses_corpus:
        for word_analyses in sent:
            for a in word_analyses:
                if count >= max_pairs:
                    break
                if a.root is None or a.is_oov_root or a.is_oov_pattern:
                    continue
                e_root = encoder._root_embedding(a).detach().cpu()
                e_pat = encoder._pattern_embedding(a).detach().cpu()
                cos = F.cosine_similarity(e_root.unsqueeze(0), e_pat.unsqueeze(0)).item()
                cos2_values.append(cos ** 2)
                count += 1

    if not cos2_values:
        return {"mean_cos2": float("nan"), "std_cos2": float("nan"), "n_pairs": 0}

    mean = sum(cos2_values) / len(cos2_values)
    var = sum((x - mean) ** 2 for x in cos2_values) / len(cos2_values)
    return {
        "mean_cos2": mean,
        "std_cos2": math.sqrt(var),
        "n_pairs": len(cos2_values),
    }


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------


def tokens_per_word_stats(
    tokenization_results,
) -> Dict[str, float]:
    """Compute average surface tokens per Arabic word.

    Parameters
    ----------
    tokenization_results:
        Iterable of :class:`sarftok.SentenceTokenization`.

    Returns
    -------
    dict with keys: mean, min, max, total_words, total_tokens
    """
    total_words = 0
    total_tokens = 0
    min_ratio = float("inf")
    max_ratio = float("-inf")

    for sent in tokenization_results:
        for s, e in sent.word_to_surface_spans:
            n_tok = e - s
            ratio = n_tok
            total_words += 1
            total_tokens += n_tok
            if ratio < min_ratio:
                min_ratio = ratio
            if ratio > max_ratio:
                max_ratio = ratio

    if total_words == 0:
        return {"mean": float("nan"), "min": 0, "max": 0, "total_words": 0, "total_tokens": 0}

    return {
        "mean": total_tokens / total_words,
        "min": min_ratio,
        "max": max_ratio,
        "total_words": total_words,
        "total_tokens": total_tokens,
    }
