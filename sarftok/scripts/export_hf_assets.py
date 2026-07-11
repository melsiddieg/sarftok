"""
export_hf_assets.py — export surface tokenizer + morph vocab as HF-compatible assets.

Copies the SentencePiece model, morph vocab JSON, and SarfTok config into a
single directory that can be loaded with SarfTokHFTokenizer.from_pretrained().

Usage
-----
sarftok-export-hf \\
    --surface-model models/surface/spm.model \\
    --morph-vocab models/morph_vocab.json \\
    --config sarftok_config.json \\
    --output hf_assets/sarftok/
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export SarfTok assets for HuggingFace integration."
    )
    parser.add_argument("--surface-model", default=None, dest="surface_model",
                        help="Path to SentencePiece .model file.")
    parser.add_argument("--morph-vocab", default=None, dest="morph_vocab",
                        help="Path to morph vocab JSON.")
    parser.add_argument("--config", default=None,
                        help="Path to sarftok_config.json.")
    parser.add_argument("--output", required=True,
                        help="Output directory for HF assets.")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    copied = []
    if args.surface_model:
        shutil.copy2(args.surface_model, out / "spm.model")
        copied.append("spm.model")

    if args.morph_vocab:
        shutil.copy2(args.morph_vocab, out / "morph_vocab.json")
        copied.append("morph_vocab.json")

    if args.config:
        shutil.copy2(args.config, out / "sarftok_config.json")
        copied.append("sarftok_config.json")

    print(f"Exported to {out}: {', '.join(copied) or 'nothing'}")


if __name__ == "__main__":
    main()
