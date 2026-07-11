"""
preprocess_corpus.py — offline corpus preprocessing pipeline.

Pipeline per sentence:
  text → normalize → segment → surface tokenize → morph analyze → serialize

Usage
-----
sarftok-preprocess \\
    --input data/corpus.txt \\
    --output data/shards/ \\
    --surface-model models/surface/spm.model \\
    --analyzer heuristic \\
    --top-k 3 \\
    --workers 4 \\
    --shard-size 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocess Arabic corpus into SarfTok shards."
    )
    parser.add_argument("--input", required=True, help="Input corpus file path.")
    parser.add_argument("--output", required=True, help="Output shard directory.")
    parser.add_argument(
        "--surface-model",
        default=None,
        dest="surface_model",
        help="Path to SentencePiece .model file (optional).",
    )
    parser.add_argument(
        "--analyzer",
        choices=["heuristic", "camel"],
        default="heuristic",
        help="Morphological analyzer backend.",
    )
    parser.add_argument("--top-k", type=int, default=3, dest="top_k")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (use 1 for CAMeL backend).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        dest="shard_size",
        help="Sentences per output shard.",
    )
    parser.add_argument(
        "--norm-mode",
        choices=["classical_strict", "classical_soft", "msa_soft"],
        default="classical_strict",
        dest="norm_mode",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    from sarftok.config import SarfTokConfig
    from sarftok.serialization import JsonlShardWriter
    from sarftok.tokenizer_api import SarfTokTokenizer

    cfg = SarfTokConfig(
        norm_mode=args.norm_mode,
        analyzer_backend=args.analyzer,
        top_k=args.top_k,
        surface_model_path=args.surface_model,
    )
    tokenizer = SarfTokTokenizer(cfg)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing: {input_path} → {out_dir}")
    total = 0
    with JsonlShardWriter(out_dir, shard_size=args.shard_size) as writer:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sent = tokenizer.tokenize_sentence(line)
                    if sent.words:
                        writer.write(sent)
                        total += 1
                except Exception as exc:
                    print(f"Warning: skipping line due to error: {exc}", file=sys.stderr)
                if total % 1_000 == 0 and total > 0:
                    print(f"  Processed {total:,} sentences...")

    print(f"Done. Wrote {total:,} sentences to {out_dir}")


if __name__ == "__main__":
    main()
