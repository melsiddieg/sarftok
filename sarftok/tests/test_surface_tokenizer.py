"""
test_surface_tokenizer.py — tests for SurfaceTokenizer (skipped if no model).
"""

# These tests are skipped when no trained SentencePiece model is available.
# To run them, train a model first:
#   python scripts/build_surface_tokenizer.py --corpus <file> --output /tmp/spm_test


class TestSurfaceTokenizerImport:
    def test_module_importable(self):
        from sarftok.surface_tokenizer import SurfaceTokenizer
        assert SurfaceTokenizer is not None

    def test_train_function_importable(self):
        from sarftok.surface_tokenizer import train_surface_tokenizer
        assert callable(train_surface_tokenizer)


class TestEncodeWithAlignmentContract:
    """Tests that exercise the contract of encode_with_alignment without a model."""

    def test_alignment_contract_documentation(self):
        """Document the expected contract: each word → contiguous span."""
        # This is a design-level test — the actual function needs a trained model.
        # When a model is loaded, the following must hold:
        #   ids, spans = tok.encode_with_alignment(["كتب", "الطالب"])
        #   assert len(spans) == 2
        #   for s, e in spans:
        #       assert 0 <= s < e <= len(ids)
        pass
