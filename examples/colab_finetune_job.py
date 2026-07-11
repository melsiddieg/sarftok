"""
colab_finetune_job.py — a SarfTok morphology-fusion finetune, sized for free Colab.

What it does
------------
Freezes an existing HuggingFace causal LM and trains *only* the SarfTok morphology tables
(root/pattern/clitic embeddings + the fusion gate) so the model learns to use the injected
Arabic morphology — a cheap adapter-style finetune that leaves every transformer weight
untouched.

* Model: ``$SARFTOK_MODEL`` if set, else ``Qwen/Qwen3-0.6B`` on GPU / ``gpt2`` on CPU (both
  ship fast tokenizers; alignment needs ``word_ids()``).
* Data: streamed from ``$SARFTOK_DATASET`` (default ``FreedomIntelligence/sharegpt-arabic``),
  ``$SARFTOK_SAMPLES`` examples; falls back to a few inline sentences if the dataset can't be
  reached. Trained with real minibatches over ``$SARFTOK_EPOCHS`` epochs.
* Reports: per-epoch mean loss, throughput, that only morph params are trainable, and that
  the morph tables actually moved.

Run on Colab via the CLI::

    colab exec -s <session> -f examples/colab_finetune_job.py

Knobs (env): SARFTOK_MODEL, SARFTOK_DATASET, SARFTOK_SAMPLES, SARFTOK_BATCH,
SARFTOK_EPOCHS, SARFTOK_MAXLEN, SARFTOK_LR.
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sarftok.config import SarfTokConfig
from sarftok.llm_integration.base_tokenizer_adapter import SarfTokBaseAdapter
from sarftok.llm_integration.hf_data_collator import SarfTokDataCollator
from sarftok.llm_integration.hf_modeling_embeddings import HybridSarfTokCausalLM
from sarftok.morph_vocab import MorphVocab

INLINE_SAMPLES = [
    "كتب الطالب الدرس في الفصل صباحا",
    "قرأ المعلم الكتاب على الطلاب بصوت واضح",
    "ذهبت الفتاة إلى المدرسة مبكرا",
    "يدرس الأطفال اللغة العربية بجد ونشاط",
    "شرح الأستاذ القاعدة النحوية للتلاميذ",
    "تكتب المهندسة التقرير الفني بدقة",
    "لعب الأولاد في الحديقة بعد المطر",
    "حفظ التلميذ القصيدة كاملة عن ظهر قلب",
]


def _extract_text(ex: dict) -> str:
    """Pull a training string out of a ShareGPT-style record."""
    for key in ("conversations", "messages"):
        conv = ex.get(key)
        if conv:
            parts = [
                (t.get("value") or t.get("content") or "").strip()
                for t in conv
                if isinstance(t, dict)
            ]
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined
    return (ex.get("text") or "").strip()


def _load_texts(dataset_id: str, n: int) -> tuple[list[str], str]:
    """Stream up to *n* non-empty texts from *dataset_id*; fall back to inline."""
    if dataset_id.lower() == "inline":
        return [INLINE_SAMPLES[i % len(INLINE_SAMPLES)] for i in range(n)], "inline"
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train", streaming=True)
        texts: list[str] = []
        for ex in ds:
            t = _extract_text(ex)
            if t:
                texts.append(t)
            if len(texts) >= n:
                break
        if texts:
            return texts, dataset_id
    except Exception as exc:  # noqa: BLE001
        print(f"[data] dataset '{dataset_id}' unavailable ({type(exc).__name__}); "
              f"using inline samples", flush=True)
    return [INLINE_SAMPLES[i % len(INLINE_SAMPLES)] for i in range(n)], "inline"


def main() -> int:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_model = "Qwen/Qwen3-0.6B" if device == "cuda" else "gpt2"
    model_id = os.environ.get("SARFTOK_MODEL", default_model)
    dataset_id = os.environ.get("SARFTOK_DATASET", "FreedomIntelligence/sharegpt-arabic")
    n_samples = int(os.environ.get("SARFTOK_SAMPLES", "512"))
    batch_size = int(os.environ.get("SARFTOK_BATCH", "8"))
    epochs = int(os.environ.get("SARFTOK_EPOCHS", "3"))
    max_len = int(os.environ.get("SARFTOK_MAXLEN", "64"))
    lr = float(os.environ.get("SARFTOK_LR", "1e-3"))

    print(f"[env] torch={torch.__version__} device={device} model={model_id}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    if not tok.is_fast:
        raise SystemExit(f"{model_id} lacks a fast tokenizer (word_ids() required).")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    base.to(device)

    cfg = SarfTokConfig(
        analyzer_backend="heuristic",
        norm_mode="classical_soft",
        top_k=3,
        alpha=0.5,
        entropy_gating=True,
        ortho_lambda=0.01,
    )
    vocab = MorphVocab(root_min_freq=1, pattern_min_freq=1)

    adapter = SarfTokBaseAdapter(tok, cfg)
    collator = SarfTokDataCollator(pad_token_id=tok.pad_token_id or 0)
    model = HybridSarfTokCausalLM(base, cfg, vocab).to(device)

    # Freeze the base model; train only the SarfTok morphology tables + gate.
    for p in model.base_model.parameters():
        p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable={n_train:,} / total={n_total:,} "
          f"({100 * n_train / n_total:.3f}% — morphology only)", flush=True)

    # Load + pre-encode the corpus (analyzer runs once per sample here).
    t_enc = time.time()
    texts, source = _load_texts(dataset_id, n_samples)
    # Cap to ~max_len words so the analyzer isn't run on words that truncation drops.
    features = [
        adapter.encode_sentence(" ".join(t.split()[:max_len]), max_length=max_len)
        for t in texts
    ]
    features = [f for f in features if f["num_words"] > 0]
    n_steps = (len(features) + batch_size - 1) // batch_size * epochs
    print(f"[data] source={source} | {len(features)} samples encoded in "
          f"{time.time() - t_enc:.1f}s | batch={batch_size} epochs={epochs} "
          f"({n_steps} steps)", flush=True)

    # Snapshot a morph table to prove it moves.
    before = model.hybrid_embedding.morph_encoder.rootchar_emb.weight.detach().clone()

    optim = torch.optim.AdamW(trainable, lr=lr)
    model.train()

    first_loss = last_loss = None
    t0 = time.time()
    tokens = 0
    for epoch in range(epochs):
        perm = torch.randperm(len(features)).tolist()
        ep_loss, ep_batches = 0.0, 0
        for i in range(0, len(perm), batch_size):
            chunk = [features[j] for j in perm[i : i + batch_size]]
            batch = collator(chunk)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            optim.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                word_to_surface_spans=batch["word_to_surface_spans"],
                morph_analyses=batch["morph_analyses"],
            )
            loss = out["loss"] if isinstance(out, dict) else out.loss
            loss.backward()
            optim.step()

            tokens += int(batch["attention_mask"].sum().item())
            last_loss = loss.item()
            if first_loss is None:
                first_loss = last_loss
            ep_loss += last_loss
            ep_batches += 1

        print(f"[train] epoch {epoch + 1}/{epochs}  mean_loss={ep_loss / ep_batches:.4f}",
              flush=True)

    dt = time.time() - t0
    moved = (model.hybrid_embedding.morph_encoder.rootchar_emb.weight.detach() - before)
    moved_norm = float(moved.norm())

    print(f"[done] {n_steps} steps in {dt:.1f}s "
          f"({tokens / dt:.0f} tok/s) | loss {first_loss:.4f} -> {last_loss:.4f}", flush=True)
    print(f"[check] morph table moved: ||Δ||={moved_norm:.4f} "
          f"({'trained ✓' if moved_norm > 0 else 'NOT trained ✗'})", flush=True)

    # Save just the SarfTok weights (small) to prove artifacts are produced.
    out_dir = "/content/sarftok_ft"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "morph_encoder": model.hybrid_embedding.morph_encoder.state_dict(),
            "probabilistic_embedder": model.hybrid_embedding.probabilistic_embedder.state_dict(),
        },
        os.path.join(out_dir, "sarftok_embeddings.pt"),
    )
    cfg.to_json(os.path.join(out_dir, "sarftok_config.json"))
    print(f"[save] wrote SarfTok adapter weights to {out_dir}", flush=True)

    ok = last_loss < first_loss and moved_norm > 0
    print("RESULT:", "PASS ✓" if ok else "FAIL ✗", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
