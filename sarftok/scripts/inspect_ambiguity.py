"""
inspect_ambiguity.py — CLI tool for inspecting morphological ambiguity.

Reads preprocessed JSONL shards and reports ambiguity statistics:
* per-word analysis count distribution
* most ambiguous words
* entropy distribution

Usage
-----
sarftok-inspect --shards data/shards/ --top 20
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect morphological ambiguity in preprocessed SarfTok shards."
    )
    parser.add_argument("--shards", required=True, help="Directory of JSONL shards.")
    parser.add_argument("--top", type=int, default=20, help="Top-N most ambiguous words to show.")
    args = parser.parse_args(argv)

    shard_dir = Path(args.shards)
    if not shard_dir.exists():
        print(f"Error: shard directory not found: {shard_dir}", file=sys.stderr)
        sys.exit(1)

    from sarftok.serialization import read_jsonl_shards

    analysis_count_dist: Counter = Counter()
    word_entropy: defaultdict[str, list] = defaultdict(list)
    total_words = 0

    for sent in read_jsonl_shards(shard_dir):
        for w in sent.words:
            n = len(w.analyses)
            analysis_count_dist[n] += 1
            total_words += 1
            if w.analyses:
                probs = [a.prob for a in w.analyses]
                word_entropy[w.normalized_word].append(_entropy(probs))

    print(f"Total words: {total_words:,}")
    print()

    print("Analysis count distribution:")
    for k in sorted(analysis_count_dist.keys()):
        pct = 100.0 * analysis_count_dist[k] / total_words
        print(f"  {k} analyses: {analysis_count_dist[k]:>8,} ({pct:.1f}%)")
    print()

    # Most ambiguous by mean entropy
    mean_entropy = {
        w: sum(es) / len(es)
        for w, es in word_entropy.items()
        if len(es) >= 2
    }
    top_ambiguous = sorted(mean_entropy.items(), key=lambda x: -x[1])[: args.top]

    print(f"Top {args.top} most ambiguous words (by mean entropy):")
    for word, ent in top_ambiguous:
        print(f"  {word:>20s}  H={ent:.3f}  (seen {len(word_entropy[word])}x)")


if __name__ == "__main__":
    main()
