# SarfTok — Hybrid Probabilistic Arabic Tokenizer

**SarfTok** combines a surface subword channel with a probabilistic morphology channel to produce factorized word embeddings for Arabic LLM pretraining and continued pretraining of existing models.

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
    │                              E_rp = MLP(E_root, E_pat, E_root ⊙ E_pat)
    │                              E_i = E_rp + E_POS + E_case + E_mood + …
    │                                    + E_pro + E_enc
    │                              E_morph = Σ p_i · E_i
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
         ProbabilisticEmbedder
         α_word = α₀ · exp(-β·H(p))   ← entropy-aware gate
         surface_emb[s:e] += α / √n_pieces · E_morph[j]
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
from sarftok.config import SarfTokConfig
from sarftok.tokenizer_api import SarfTokTokenizer
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
from sarftok.config import SarfTokConfig

cfg = SarfTokConfig(
    # normalization
    norm_mode="classical_strict",
    strip_diacritics=False,
    normalise_alef=False,           # preserve Classical orthography
    normalise_ya=False,
    # surface tokenizer
    surface_vocab_size=32000,
    surface_model_type="bpe",
    surface_model_path="models/surface/spm.model",
    # morphology
    analyzer_backend="heuristic",   # "heuristic" | "camel" | "distilled"
    camel_db="calima-clx-r13",      # Classical Arabic, not the MSA database
    contextual_analysis=True,       # use sentence-level analyzer API
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
    entropy_normalize=True,         # normalise H by ln(k) so β is stable across top_k
    broadcast_piece_scaling="sqrt", # "none" | "sqrt" | "linear"
    root_pattern_interaction=True,
    fusion_layer_norm=False,        # opt-in for checkpoint compatibility
    # training losses
    ortho_lambda=0.0,                # optional ablation, disabled by default
    ortho_min_confidence=0.5,
    template_lambda=0.0,            # enable when supplying template_pairs
)
```

### Analyzer probability semantics

Analyzer backends declare how their per-analysis scores should be interpreted:

- `score_type="prob"` (default) — scores are (unnormalised) probabilities and are
  **L1-normalised**, preserving a backend's stated confidences (e.g. `0.7 / 0.3`).
- `score_type="logit"` — scores are logits and are converted with a temperature softmax.

CAMeL analyzer order is not treated as a posterior. If an analysis has no count/frequency,
its candidates receive uniform weights so entropy gating correctly weakens the morphology
channel. For contextual probabilities, inject a CAMeL-compatible disambiguator into
`CamelMorphAnalyzer(contextual_disambiguator=...)` and pass that analyzer to
`SarfTokTokenizer(..., analyzer=analyzer)` or `SarfTokBaseAdapter(..., analyzer=analyzer)`.
The disambiguator is called once per sentence and its candidate scores are preserved.

## Fine-tuning an existing LLM

Beyond from-scratch pretraining on the surface vocabulary, SarfTok can morph-fuse an
**existing** checkpoint. `SarfTokBaseAdapter` aligns morphology to the base model's own
tokenizer (via `word_ids()`), so the surface IDs and the base embedding table stay
consistent while the morphology embedding is added on top — no transformer weights change.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from sarftok.config import SarfTokConfig
from sarftok.llm_integration.base_tokenizer_adapter import SarfTokBaseAdapter
from sarftok.llm_integration.hf_data_collator import SarfTokDataCollator
from sarftok.llm_integration.hf_modeling_embeddings import HybridSarfTokCausalLM
from sarftok.morph_vocab import MorphVocab

cfg = SarfTokConfig(analyzer_backend="heuristic", alpha=0.5, ortho_lambda=0.01)
tok = AutoTokenizer.from_pretrained("gpt2")          # any HF *fast* tokenizer
base = AutoModelForCausalLM.from_pretrained("gpt2")

adapter  = SarfTokBaseAdapter(tok, cfg)
collator = SarfTokDataCollator(pad_token_id=tok.pad_token_id or 0)
model    = HybridSarfTokCausalLM(base, cfg, MorphVocab(root_min_freq=1, pattern_min_freq=1))

batch = collator([adapter.encode_sentence("كتب الطالب الدرس")])
out = model(**batch)   # out["loss"] backprops into the morphology tables only
```

A runnable end-to-end version (with a tiny model, no download) is in
[`examples/finetune_existing_llm.py`](examples/finetune_existing_llm.py). For a full run,
wrap the model in a `Trainer`/`SFTTrainer` with `remove_unused_columns=False` and this
collator.

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
