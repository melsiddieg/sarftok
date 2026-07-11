"""
train_distilled_guesser.py — Phase-4 stub for training the distilled morphology guesser.

Not implemented in v1. This script serves as a placeholder with a clear
development roadmap in the docstring.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a distilled morphology tagger (Phase 4 — not yet implemented)."
    )
    parser.add_argument("--data", help="Training data directory.")
    parser.add_argument("--output", help="Output model directory.")
    parser.parse_args(argv)

    print(
        "ERROR: train_distilled_guesser is a Phase-4 feature and has not been "
        "implemented yet.\n\n"
        "Development roadmap:\n"
        "  1. Collect (word, context) → top-k MorphAnalysis pairs from CAMeL backend\n"
        "  2. Train a small seq-labelling model (e.g. char-CNN + BiLSTM) to predict\n"
        "     root/pattern/clitics/probs from the word surface + context window\n"
        "  3. Calibrate probabilities via temperature scaling\n"
        "  4. Integrate as analyzer_backend='distilled'\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
