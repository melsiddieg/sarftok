"""
build_surface_tokenizer.py — CLI for training the SentencePiece surface tokenizer.

Usage
-----
sarftok-build-surface \\
    --corpus data/corpus.txt \\
    --vocab-size 32000 \\
    --model-type bpe \\
    --output models/surface/

Or run as a script:
    python scripts/build_surface_tokenizer.py --corpus ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece surface tokenizer for Arabic."
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to plain-text corpus file (one sentence per line).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_000,
        dest="vocab_size",
        help="Target vocabulary size (default: 32000).",
    )
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram"],
        default="bpe",
        dest="model_type",
        help="SentencePiece model type (default: bpe).",
    )
    parser.add_argument(
        "--output",
        default="models/surface",
        help="Output directory for the trained model.",
    )
    parser.add_argument(
        "--prefix", default="spm", help="Model file prefix (default: spm)."
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9999,
        dest="character_coverage",
        help="Character coverage (default: 0.9999).",
    )
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"Error: corpus file not found: {corpus}", file=sys.stderr)
        sys.exit(1)

    print(f"Training {args.model_type.upper()} tokenizer")
    print(f"  corpus:     {corpus}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  output:     {args.output}")

    from sarftok.surface_tokenizer import train_surface_tokenizer

    model_path = train_surface_tokenizer(
        corpus_path=corpus,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        output_dir=args.output,
        model_prefix=args.prefix,
        character_coverage=args.character_coverage,
    )
    print(f"Done. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
