"""
SurfaceTokenizer — SentencePiece wrapper with word-aligned subword spans.

This module wraps ``sentencepiece`` to provide:
* training with configurable vocab size and model type (BPE / Unigram)
* encode/decode with word→span alignment guarantees
* whitespace pretokenisation enforcement so word boundaries are always honoured

The alignment guarantee is critical: each word must map to a *contiguous*
range of surface tokens so the morphology embedding can be broadcast correctly.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import sentencepiece as spm  # type: ignore
except ImportError:  # pragma: no cover
    spm = None  # type: ignore


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def train_surface_tokenizer(
    corpus_path: str | Path,
    vocab_size: int = 32_000,
    model_type: str = "bpe",
    output_dir: str | Path = "models/surface",
    model_prefix: str = "spm",
    character_coverage: float = 0.9999,
    pad_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    unk_id: int = 3,
    user_defined_symbols: Optional[List[str]] = None,
) -> str:
    """Train a SentencePiece model on *corpus_path* and save to *output_dir*.

    Parameters
    ----------
    corpus_path:
        Plain-text corpus file, one sentence per line.
    vocab_size:
        Target vocabulary size (e.g. 16000, 24000, 32000).
    model_type:
        ``"bpe"`` or ``"unigram"``.
    output_dir:
        Directory where ``spm.model`` and ``spm.vocab`` are written.
    model_prefix:
        Prefix for output files.
    character_coverage:
        SentencePiece character coverage parameter.
    pad_id, bos_id, eos_id, unk_id:
        Special token IDs.
    user_defined_symbols:
        Extra symbols to add to the vocabulary verbatim (e.g. morph tokens).

    Returns
    -------
    str
        Path to the trained ``.model`` file.
    """
    if spm is None:
        raise ImportError("sentencepiece is required: pip install sentencepiece")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_prefix}.model"

    uds = ",".join(user_defined_symbols) if user_defined_symbols else ""

    training_args = (
        f"--input={corpus_path}"
        f" --model_prefix={output_dir / model_prefix}"
        f" --vocab_size={vocab_size}"
        f" --model_type={model_type}"
        f" --character_coverage={character_coverage}"
        f" --pad_id={pad_id}"
        f" --bos_id={bos_id}"
        f" --eos_id={eos_id}"
        f" --unk_id={unk_id}"
        f" --pad_piece=<pad>"
        f" --bos_piece=<s>"
        f" --eos_piece=</s>"
        f" --unk_piece=<unk>"
        " --byte_fallback=true"
        " --add_dummy_prefix=false"  # we handle whitespace explicitly
    )
    if uds:
        training_args += f" --user_defined_symbols={uds}"

    spm.SentencePieceTrainer.train(training_args)
    return str(model_path)


# ---------------------------------------------------------------------------
# SurfaceTokenizer class
# ---------------------------------------------------------------------------


class SurfaceTokenizer:
    """Wraps a SentencePiece model with word-alignment support.

    The tokenizer always encodes text word-by-word (whitespace-pretokenised)
    so that each word maps to a *contiguous* sub-span of the full token ID
    sequence.

    Parameters
    ----------
    model_path:
        Path to a trained SentencePiece ``.model`` file.
    add_bos:
        Prepend BOS token to every encoded sequence.
    add_eos:
        Append EOS token to every encoded sequence.
    """

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    UNK_ID: int = 3

    def __init__(
        self,
        model_path: str | Path,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> None:
        if spm is None:
            raise ImportError("sentencepiece is required: pip install sentencepiece")
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_path))
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size: int = self._sp.get_piece_size()

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    def piece_to_id(self, piece: str) -> int:
        return self._sp.piece_to_id(piece)

    def id_to_piece(self, idx: int) -> str:
        return self._sp.id_to_piece(idx)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode *text* to a list of integer token IDs."""
        ids = self._sp.encode(text, out_type=int)
        if self.add_bos:
            ids = [self.BOS_ID, *ids]
        if self.add_eos:
            ids = [*ids, self.EOS_ID]
        return ids

    def encode_pieces(self, text: str) -> List[str]:
        """Encode *text* to a list of string pieces."""
        return self._sp.encode(text, out_type=str)

    def encode_with_alignment(
        self, words: List[str]
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Encode a pre-tokenised word list and return (ids, spans).

        Each word is encoded independently so that word boundaries are
        *guaranteed* to align with token span boundaries.

        Parameters
        ----------
        words:
            List of (normalised) Arabic word strings.

        Returns
        -------
        ids:
            Flat list of surface token IDs.
        spans:
            ``spans[j] = (start, end)`` — half-open index into *ids* for
            word ``j`` (i.e. ``ids[start:end]``).
        """
        ids: List[int] = []
        spans: List[Tuple[int, int]] = []
        offset = 0

        if self.add_bos:
            ids.append(self.BOS_ID)
            offset = 1

        for word in words:
            word_ids = self._sp.encode(word, out_type=int)
            if not word_ids:
                # Fallback: treat as single UNK token
                word_ids = [self.UNK_ID]
            start = offset
            ids.extend(word_ids)
            offset += len(word_ids)
            spans.append((start, offset))

        if self.add_eos:
            ids.append(self.EOS_ID)

        return ids, spans

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a string."""
        # Filter special tokens before decoding
        filtered = [
            i for i in ids if i not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)
        ]
        return self._sp.decode(filtered)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save_model_bytes(self) -> bytes:
        """Return the raw bytes of the SentencePiece model."""
        return self._sp.serialized_model_proto()

    @classmethod
    def from_model_bytes(cls, data: bytes, **kwargs) -> "SurfaceTokenizer":
        """Load tokenizer from raw model bytes (e.g. from an HF asset)."""
        tmp = _write_tmp_model(data)
        return cls(tmp, **kwargs)


def _write_tmp_model(data: bytes) -> str:
    import tempfile

    tf = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
    tf.write(data)
    tf.close()
    return tf.name
