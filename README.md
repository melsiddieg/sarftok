# SarfTok — Hybrid Probabilistic Arabic Tokenizer

**SarfTok** combines a surface subword channel with a probabilistic morphology channel to produce factorized additive word embeddings for Arabic LLM pretraining.

## Architecture

```
Arabic text
    │
    ▼
ArabicNormalizer (classical_strict / classical_soft / msa_soft)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
SurfaceTokenizer                    MorphAnalyzer
(SentencePiece BPE/Unigram)        (CAMeL / Heuristic / Distilled)
    │                                      │
    │  word → surface span mapping         │  top-k MorphAnalysis[]
    │                                      ▼
    │                              MorphEncoder (nn.Module)
    │                              E_i = E_root + E_pat + E_pro + E_enc
    │                              E_morph = Σ p_i · E_i
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
         ProbabilisticEmbedder
         α_word = α₀ · exp(-β·H(p))   ← entropy-aware gate
         surface_emb[s:e] += α · E_morph[j]
                   │
                   ▼
         HybridSarfTokCausalLM
         (passes inputs_embeds into frozen transformer)
```

## Quickstart

```bash
# Install
pip install -e ".[hf,dev]"

# Optional CAMeL Tools morphology backend
pip install -e ".[camel]"

# Tokenize a sentence
python -c "
from sarftok import SarfTokTokenizer, SarfTokConfig
cfg = SarfTokConfig(analyzer_backend='heuristic')
tok = SarfTokTokenizer(cfg)
result = tok.tokenize_sentence('كَتَبَ الطَّالِبُ الدَّرْسَ')
for w in result.words:
    print(w.raw_word, '→', w.surface_pieces, '|', [(a.root, a.prob) for a in w.analyses])
"

# Build surface tokenizer
sarftok-build-surface --corpus data/corpus.txt --vocab-size 32000 --output models/surface/

# Build morphological vocabulary
sarftok-build-vocab --corpus data/shards/ --output models/morph_vocab.json

# Preprocess corpus
sarftok-preprocess --input data/corpus.txt --output data/shards/ --analyzer heuristic --workers 8
```

## Configuration

```python
from sarftok import SarfTokConfig

cfg = SarfTokConfig(
    # normalization
    norm_mode="classical_strict",
    strip_diacritics=False,
    # surface tokenizer
    surface_vocab_size=32000,
    surface_model_type="bpe",
    surface_model_path="models/surface/spm.model",
    # morphology
    analyzer_backend="heuristic",   # "heuristic" | "camel" | "distilled"
    top_k=3,
    confidence_threshold=0.05,
    morph_vocab_path="models/morph_vocab.json",
    # embedding fusion
    hidden_dim=2048,
    fusion_mode="broadcast",        # "broadcast" | "first_piece"
    alpha=0.5,
    beta=1.0,
    learnable_alpha=False,
    entropy_gating=True,
    # training losses
    ortho_lambda=0.01,
    ortho_min_confidence=0.5,
)
```

## Package Layout

```
sarftok/
  config.py               SarfTokConfig
  normalizer.py           ArabicNormalizer
  segmenter.py            ArabicSegmenter
  surface_tokenizer.py    SurfaceTokenizer
  morph_analyzer/
    interface.py          MorphAnalyzer ABC
    camel_wrapper.py      CAMeL Tools backend (optional)
    heuristics.py         Rule-based fallback backend
    distilled_tagger.py   Phase-4 stub
  morph_vocab.py          MorphVocab
  morph_encoder.py        MorphEncoder (nn.Module)
  alignment.py            word-to-surface alignment
  probabilistic_embedder.py  ProbabilisticEmbedder (nn.Module)
  tokenizer_api.py        SarfTokTokenizer (high-level)
  serialization.py        Shard writers/readers
  tests/

llm_integration/
  hf_tokenizer_wrapper.py
  hf_data_collator.py
  hf_modeling_embeddings.py
  hf_trainer_patch.py
  losses.py
  probes.py
  tests/

scripts/
  build_surface_tokenizer.py
  build_morph_vocab.py
  preprocess_corpus.py
  inspect_ambiguity.py
  export_hf_assets.py
  train_distilled_guesser.py
```

## Experiment Configs

| Run | Surface | Morphology | Mode | Loss |
|-----|---------|-----------|------|------|
| Baseline A | ✓ | — | — | LM |
| Baseline B | ✓ | top-1 det. | broadcast | LM |
| Main C | ✓ | top-3 prob. | broadcast | LM |
| Main D | ✓ | top-3 prob. | broadcast | LM + Ortho |
| Optional E | — | top-3 prob. | — | LM |
| Optional F | ✓ | char-root fallback | broadcast | LM |
| Optional G | ✓ | top-3 prob. | first_piece | LM |

## License

MIT
