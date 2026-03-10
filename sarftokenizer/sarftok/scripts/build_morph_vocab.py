"""
build_morph_vocab.py — CLI for building the morphological vocabulary.

Reads preprocessed JSONL shards, counts root/pattern frequencies,
and saves a MorphVocab JSON file.

Usage
-----
sarftok-build-vocab \\
    --shards data/shards/ \\
    --output models/morph_vocab.json \\
    --root-min-freq 5 \\
    --pattern-min-freq 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build morphological vocabulary from preprocessed corpus shards."
    )
    parser.add_argument(
        "--shards",
        required=True,
        help="Directory containing JSONL corpus shards.",
    )
    parser.add_argument(
        "--output",
        default="models/morph_vocab.json",
        help="Output path for the morph vocab JSON.",
    )
    parser.add_argument(
        "--root-min-freq",
        type=int,
        default=5,
        dest="root_min_freq",
        help="Minimum root frequency for composite token (default: 5).",
    )
    parser.add_argument(
        "--pattern-min-freq",
        type=int,
        default=3,
        dest="pattern_min_freq",
        help="Minimum pattern frequency for composite token (default: 3).",
    )
    args = parser.parse_args(argv)

    shard_dir = Path(args.shards)
    if not shard_dir.exists():
        print(f"Error: shard directory not found: {shard_dir}", file=sys.stderr)
        sys.exit(1)

    from sarftok.morph_vocab import MorphVocab
    from sarftok.serialization import read_jsonl_shards

    def analyses_iter():
        for sent in read_jsonl_shards(shard_dir):
            yield [[a for a in w.analyses] for w in sent.words]

    print("Building morph vocab...")
    vocab = MorphVocab.build_from_analyses(
        analyses_iter(),
        root_min_freq=args.root_min_freq,
        pattern_min_freq=args.pattern_min_freq,
    )

    out = Path(args.output)
    vocab.save(out)
    print(f"Vocab saved to: {out}")
    print(f"  roots:    {vocab.num_roots}")
    print(f"  patterns: {vocab.num_patterns}")
    print(f"  proclitics: {vocab.num_proclitics}")
    print(f"  enclitics:  {vocab.num_enclitics}")


if __name__ == "__main__":
    main()
