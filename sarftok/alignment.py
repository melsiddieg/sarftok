"""
alignment.py — word-to-surface-span alignment utilities.

After encoding a list of words with SurfaceTokenizer.encode_with_alignment(),
these helpers validate and manipulate the resulting span map.
"""
from __future__ import annotations


def validate_spans(
    spans: list[tuple[int, int]],
    total_ids: int,
) -> None:
    """Assert that spans are contiguous, non-overlapping, and in bounds.

    Parameters
    ----------
    spans:
        List of (start, end) half-open ranges.
    total_ids:
        Total length of the surface ID sequence.

    Raises
    ------
    ValueError
        If any span is invalid.
    """
    expected_start = 0
    for i, (s, e) in enumerate(spans):
        if s < 0 or e > total_ids:
            raise ValueError(
                f"Span [{i}] = ({s}, {e}) is out of bounds for sequence length {total_ids}."
            )
        if s > e:
            raise ValueError(f"Span [{i}] = ({s}, {e}) has start > end.")
        if s < expected_start:
            raise ValueError(
                f"Span [{i}] = ({s}, {e}) overlaps previous span (expected start ≥ {expected_start})."
            )
        if s > expected_start:
            # Gap between spans — permitted when BOS/EOS are inserted
            pass
        expected_start = e


def span_lengths(spans: list[tuple[int, int]]) -> list[int]:
    """Return the number of surface tokens per word span."""
    return [e - s for s, e in spans]


def build_word_mask(
    span: tuple[int, int],
    seq_len: int,
) -> list[bool]:
    """Return a boolean mask of length *seq_len* that is True within *span*."""
    mask = [False] * seq_len
    for i in range(span[0], span[1]):
        mask[i] = True
    return mask


def broadcast_to_surface(
    word_values: list[float],
    spans: list[tuple[int, int]],
    seq_len: int,
    default: float = 0.0,
) -> list[float]:
    """Broadcast per-word scalar values to surface token positions.

    Parameters
    ----------
    word_values:
        One scalar per word.
    spans:
        Word-to-surface span mapping.
    seq_len:
        Total number of surface tokens.
    default:
        Fill value for positions not covered by any word span.

    Returns
    -------
    List[float]
        Length *seq_len*, where position ``t`` carries the value of the word
        whose span covers it.
    """
    if len(word_values) != len(spans):
        raise ValueError(
            f"len(word_values)={len(word_values)} != len(spans)={len(spans)}"
        )
    result = [default] * seq_len
    for val, (s, e) in zip(word_values, spans):
        for t in range(s, e):
            result[t] = val
    return result


def first_piece_indices(spans: list[tuple[int, int]]) -> list[int | None]:
    """Return the index of the first surface piece for each word span.

    Returns ``None`` for empty spans.
    """
    indices: list[int | None] = []
    for s, e in spans:
        indices.append(s if s < e else None)
    return indices
