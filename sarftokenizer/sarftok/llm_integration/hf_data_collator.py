"""
hf_data_collator.py — custom data collator for SarfTok batches.

Handles:
* padding of surface token IDs to the longest sequence in the batch
* creating attention masks
* preserving (and optionally padding) word_to_surface_spans
* collecting morph_analyses as nested Python lists (no tensor conversion —
  morph analyses stay as dicts and are converted to tensors inside the
  model's forward pass via MorphEncoder)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


class SarfTokDataCollator:
    """Collate a list of SarfTok encoding dicts into a padded batch.

    Parameters
    ----------
    pad_token_id:
        Token ID used for padding surface sequences.
    max_length:
        Optional truncation / padding target length.
        If None, pad to the longest sequence in the batch.
    label_pad_token_id:
        Padding value for labels (typically -100 for cross-entropy ignore).
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        label_pad_token_id: int = -100,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collate a list of single-example encoding dicts.

        Input keys per example (from SarfTokHFTokenizer):
            input_ids, attention_mask, word_to_surface_spans, morph_analyses

        Output keys:
            input_ids          : LongTensor (batch, seq_len)
            attention_mask     : LongTensor (batch, seq_len)
            word_to_surface_spans : List[List[Tuple[int,int]]]  (not tensorised)
            morph_analyses     : List[List[List[dict]]]         (not tensorised)
            labels             : LongTensor (batch, seq_len)  — copy of input_ids
                                 with pad positions set to label_pad_token_id
        """
        batch_size = len(features)

        # --------------- surface tokens ---------------
        all_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        all_masks = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]

        target_len = self.max_length or max(t.size(0) for t in all_ids)

        def _pad_1d(t: torch.Tensor, length: int, pad_val: int) -> torch.Tensor:
            curr = t.size(0)
            if curr >= length:
                return t[:length]
            return torch.cat([t, torch.full((length - curr,), pad_val, dtype=t.dtype)])

        padded_ids = torch.stack(
            [_pad_1d(t, target_len, self.pad_token_id) for t in all_ids]
        )
        padded_masks = torch.stack(
            [_pad_1d(t, target_len, 0) for t in all_masks]
        )

        # Labels: same as input_ids but padded positions → label_pad_token_id
        labels = padded_ids.clone()
        labels[padded_masks == 0] = self.label_pad_token_id

        # --------------- morph metadata (kept as Python objects) ---------------
        word_to_surface_spans: List[List[Tuple[int, int]]] = [
            [tuple(sp) for sp in f.get("word_to_surface_spans", [])]
            for f in features
        ]
        morph_analyses: List[List[List[dict]]] = [
            f.get("morph_analyses", []) for f in features
        ]

        return {
            "input_ids": padded_ids,
            "attention_mask": padded_masks,
            "labels": labels,
            "word_to_surface_spans": word_to_surface_spans,
            "morph_analyses": morph_analyses,
        }
