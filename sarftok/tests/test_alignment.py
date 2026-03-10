"""
test_alignment.py — tests for word-to-surface span alignment utilities.
"""
import pytest

from sarftok.alignment import (
    broadcast_to_surface,
    build_word_mask,
    first_piece_indices,
    span_lengths,
    validate_spans,
)


class TestValidateSpans:
    def test_valid_contiguous_spans(self):
        validate_spans([(0, 2), (2, 5), (5, 7)], total_ids=7)

    def test_gap_between_spans_allowed(self):
        # BOS token at position 0, words start at 1
        validate_spans([(1, 3), (3, 6)], total_ids=7)

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            validate_spans([(0, 10)], total_ids=5)

    def test_start_gt_end_raises(self):
        with pytest.raises(ValueError, match="start > end"):
            validate_spans([(5, 3)], total_ids=10)

    def test_overlapping_spans_raises(self):
        with pytest.raises(ValueError, match="overlaps"):
            validate_spans([(0, 5), (3, 7)], total_ids=10)


class TestSpanLengths:
    def test_lengths(self):
        assert span_lengths([(0, 2), (2, 5), (5, 7)]) == [2, 3, 2]

    def test_empty(self):
        assert span_lengths([]) == []


class TestBuildWordMask:
    def test_mask(self):
        mask = build_word_mask((2, 5), seq_len=7)
        assert mask == [False, False, True, True, True, False, False]


class TestBroadcast:
    def test_broadcast_values(self):
        result = broadcast_to_surface(
            word_values=[1.0, 2.0],
            spans=[(0, 3), (3, 5)],
            seq_len=5,
        )
        assert result == [1.0, 1.0, 1.0, 2.0, 2.0]

    def test_broadcast_with_gap(self):
        result = broadcast_to_surface(
            word_values=[1.0],
            spans=[(1, 3)],
            seq_len=4,
            default=0.0,
        )
        assert result == [0.0, 1.0, 1.0, 0.0]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="len"):
            broadcast_to_surface([1.0], [(0, 1), (1, 2)], 2)


class TestFirstPieceIndices:
    def test_normal(self):
        assert first_piece_indices([(0, 2), (2, 5)]) == [0, 2]

    def test_empty_span(self):
        assert first_piece_indices([(3, 3)]) == [None]
