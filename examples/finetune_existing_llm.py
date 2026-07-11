"""
finetune_existing_llm.py — morph-fuse an *existing* HuggingFace causal LM.

Demonstrates the fine-tuning integration path:

    raw Arabic text
        → SarfTokBaseAdapter (aligns morphology to the base model's own tokenizer)
        → SarfTokDataCollator (pads, keeps morph metadata as Python objects)
        → HybridSarfTokCausalLM (adds the morphology embedding onto surface embeddings)
        → one optimiser step

Run with a real model::

    python examples/finetune_existing_llm.py --model Qwen/Qwen3.5-0.8B

With no ``--model`` (default) it builds a tiny randomly-initialised GPT-2 so the whole
pipeline runs on CPU in seconds and requires no download — this is what the smoke test and
CI exercise.  The point is to prove morphology fuses into an existing checkpoint's embeddings
and that a training step completes with finite loss and gradients on the morph tables.
"""
from __future__ import annotations

import argparse

import torch

from sarftok.config import SarfTokConfig
from sarftok.llm_integration.base_tokenizer_adapter import SarfTokBaseAdapter
from sarftok.llm_integration.hf_data_collator import SarfTokDataCollator
from sarftok.llm_integration.hf_modeling_embeddings import HybridSarfTokCausalLM
from sarftok.morph_vocab import MorphVocab

SAMPLES = [
    "كتب الطالب الدرس في الفصل",
    "قرأ المعلم الكتاب على الطلاب",
    "ذهبت الفتاة إلى المدرسة صباحا",
    "يدرس الأطفال اللغة العربية بجد",
]


def _build_tiny_model():
    """A small randomly-initialised GPT-2 + its fast tokenizer (no network)."""
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    try:
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Could not load the gpt2 fast tokenizer (offline?). "
            "Re-run with --model pointing at a locally available model.\n"
            f"Underlying error: {exc}"
        ) from exc
    cfg = GPT2Config(vocab_size=tok.vocab_size, n_embd=64, n_layer=2, n_head=2, n_positions=128)
    model = GPT2LMHeadModel(cfg)
    return model, tok


def _load_named_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if not tok.is_fast:
        raise SystemExit(f"{name} does not provide a fast tokenizer; word alignment needs one.")
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF model id (default: tiny random GPT-2).")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args(argv)

    base_model, base_tok = (
        _load_named_model(args.model) if args.model else _build_tiny_model()
    )
    if base_tok.pad_token_id is None:
        base_tok.pad_token = base_tok.eos_token

    config = SarfTokConfig(
        analyzer_backend="heuristic",
        top_k=3,
        alpha=0.5,
        entropy_gating=True,
        ortho_lambda=0.01,
    )
    vocab = MorphVocab(root_min_freq=1, pattern_min_freq=1)

    adapter = SarfTokBaseAdapter(base_tok, config)
    collator = SarfTokDataCollator(pad_token_id=base_tok.pad_token_id or 0)
    model = HybridSarfTokCausalLM(base_model, config, vocab)

    features = [adapter.encode_sentence(t, max_length=64) for t in SAMPLES]
    batch = collator(features)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    print(f"Model: {args.model or 'tiny-random-gpt2'} | "
          f"batch input_ids {tuple(batch['input_ids'].shape)}")
    for step in range(args.steps):
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            word_to_surface_spans=batch["word_to_surface_spans"],
            morph_analyses=batch["morph_analyses"],
        )
        loss = out["loss"] if isinstance(out, dict) else out.loss
        loss.backward()
        optimizer.step()
        extra = ""
        if isinstance(out, dict) and out.get("ortho_loss") is not None:
            extra = f" | ortho={out['ortho_loss'].item():.4f}"
        print(f"step {step}: loss={loss.item():.4f}{extra}")
        assert torch.isfinite(loss), "loss became non-finite"

    morph_grad = model.hybrid_embedding.morph_encoder.rootchar_emb.weight.grad
    assert morph_grad is not None, "morphology tables received no gradient — fusion inactive"
    print("OK — morphology fused into an existing LLM; morph tables received gradient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
