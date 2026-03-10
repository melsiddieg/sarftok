"""
MorphEncoder — nn.Module that converts morphological analyses into embeddings.

Architecture
------------
For each analysis i of a word:
    E_i = E_root_i + E_pat_i + Σ_j E_pro_j + Σ_k E_enc_k

Probabilistic mixture:
    E_morph = Σ_i p_i * E_i

Root fallback: if root token is OOV, pool ROOTCHAR embeddings.
Pattern fallback: if pattern is OOV, use PATCLASS or PAT_UNKNOWN embedding.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from sarftok import MorphAnalysis
from sarftok.morph_vocab import MorphVocab


class MorphEncoder(nn.Module):
    """Compute a single morphology embedding per word from top-k analyses.

    Parameters
    ----------
    vocab:
        Pre-built :class:`~sarftok.morph_vocab.MorphVocab`.
    hidden_dim:
        Embedding dimension (must match the main model hidden size).
    use_composite_root:
        If True, prefer ROOT_ composite tokens; fall back to ROOTCHAR_ pooling.
        If False, always use ROOTCHAR_ pooling.
    dropout:
        Dropout probability applied to the output morphology embedding.
    """

    def __init__(
        self,
        vocab: MorphVocab,
        hidden_dim: int = 768,
        use_composite_root: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.use_composite_root = use_composite_root

        # Embedding tables
        self.root_emb = nn.Embedding(vocab.num_roots, hidden_dim, padding_idx=None)
        self.pattern_emb = nn.Embedding(
            vocab.num_patterns, hidden_dim, padding_idx=None
        )
        self.proclitic_emb = nn.Embedding(
            vocab.num_proclitics, hidden_dim, padding_idx=None
        )
        self.enclitic_emb = nn.Embedding(
            vocab.num_enclitics, hidden_dim, padding_idx=None
        )
        self.rootchar_emb = nn.Embedding(
            vocab.num_rootchars, hidden_dim, padding_idx=None
        )
        self.patclass_emb = nn.Embedding(
            vocab.num_patclasses, hidden_dim, padding_idx=None
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (
            self.root_emb,
            self.pattern_emb,
            self.proclitic_emb,
            self.enclitic_emb,
            self.rootchar_emb,
            self.patclass_emb,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Per-analysis embedding
    # ------------------------------------------------------------------

    def _root_embedding(self, analysis: MorphAnalysis) -> torch.Tensor:
        """Return root embedding, falling back to root-char pooling."""
        device = self.root_emb.weight.device
        if analysis.root is None:
            # No root info — return zeros
            return torch.zeros(self.hidden_dim, device=device)

        if self.use_composite_root and not analysis.is_oov_root:
            root_id, is_composite = self.vocab.root_id(analysis.root)
            if is_composite and root_id is not None:
                idx = torch.tensor([root_id], device=device)
                return self.root_emb(idx).squeeze(0)

        # Root-char fallback: mean-pool character embeddings
        char_ids = self.vocab.root_char_ids(analysis.root)
        if not char_ids:
            return torch.zeros(self.hidden_dim, device=device)
        idx = torch.tensor(char_ids, device=device)
        return self.rootchar_emb(idx).mean(dim=0)

    def _pattern_embedding(self, analysis: MorphAnalysis) -> torch.Tensor:
        """Return pattern embedding, falling back to pattern class / PAT_UNKNOWN."""
        device = self.pattern_emb.weight.device

        if not analysis.is_oov_pattern and analysis.pattern is not None:
            pat_id, tok = self.vocab.pattern_id(analysis.pattern)
            if not tok == "PAT_UNKNOWN":
                idx = torch.tensor([pat_id], device=device)
                return self.pattern_emb(idx).squeeze(0)

        # Fallback: pattern class embedding
        fallback_id = self.vocab.patclass_vocab.get("PAT_UNKNOWN", 0)
        idx = torch.tensor([fallback_id], device=device)
        return self.patclass_emb(idx).squeeze(0)

    def _clitic_sum(
        self, ids: List[int], emb_table: nn.Embedding
    ) -> torch.Tensor:
        """Sum embedding vectors for a list of token IDs."""
        device = emb_table.weight.device
        if not ids:
            return torch.zeros(self.hidden_dim, device=device)
        idx = torch.tensor(ids, device=device)
        return emb_table(idx).sum(dim=0)

    def _analysis_embedding(self, analysis: MorphAnalysis) -> torch.Tensor:
        """Compute E_i = E_root + E_pat + Σ E_pro + Σ E_enc."""
        e_root = self._root_embedding(analysis)
        e_pat = self._pattern_embedding(analysis)
        e_pro = self._clitic_sum(
            self.vocab.proclitic_ids(analysis.proclitics), self.proclitic_emb
        )
        e_enc = self._clitic_sum(
            self.vocab.enclitic_ids(analysis.enclitics), self.enclitic_emb
        )
        return e_root + e_pat + e_pro + e_enc

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_word(self, analyses: List[MorphAnalysis]) -> torch.Tensor:
        """Encode one word's top-k analyses into a single embedding.

        Parameters
        ----------
        analyses:
            List of :class:`~sarftok.MorphAnalysis` (top-k, sorted by prob).

        Returns
        -------
        torch.Tensor  shape: (hidden_dim,)
        """
        device = self.root_emb.weight.device

        if not analyses:
            # Unanalyzable word — return zero embedding
            return torch.zeros(self.hidden_dim, device=device)

        # Compute E_i for each analysis and weight by probability
        emb_sum = torch.zeros(self.hidden_dim, device=device)
        total_prob = sum(a.prob for a in analyses)
        if total_prob <= 0:
            total_prob = 1.0

        for analysis in analyses:
            e_i = self._analysis_embedding(analysis)
            emb_sum = emb_sum + (analysis.prob / total_prob) * e_i

        return self.dropout(emb_sum)

    def forward(
        self, analyses_batch: List[List[List[MorphAnalysis]]]
    ) -> torch.Tensor:
        """Encode a batch of sentences.

        Parameters
        ----------
        analyses_batch:
            Outer list = batch; middle list = words per sentence;
            inner list = analyses per word.

        Returns
        -------
        torch.Tensor  shape: (batch_size, max_words, hidden_dim)
            Padded with zeros for shorter sentences.
        """
        device = self.root_emb.weight.device
        batch_size = len(analyses_batch)
        max_words = max((len(sent) for sent in analyses_batch), default=0)

        if max_words == 0:
            return torch.zeros(batch_size, 1, self.hidden_dim, device=device)

        out = torch.zeros(batch_size, max_words, self.hidden_dim, device=device)
        for b, sentence_analyses in enumerate(analyses_batch):
            for w, word_analyses in enumerate(sentence_analyses):
                out[b, w] = self.forward_word(word_analyses)

        return out
