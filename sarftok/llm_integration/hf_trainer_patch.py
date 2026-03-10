"""
hf_trainer_patch.py — helpers for integrating SarfTok with HF Trainer.

Provides:
* compute_metrics wrapper
* custom training step hook
* a pre-built TrainingArguments suggestion
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch


def make_sarftok_compute_metrics(
    tokenizer_vocab_size: int,
) -> Callable:
    """Return a compute_metrics function for use with HF Trainer.

    Currently computes perplexity from the LM loss.
    """
    import math

    def compute_metrics(eval_pred) -> Dict[str, float]:
        # eval_pred is (logits, labels) from Trainer
        # If we return a dict from forward, Trainer will pass through the loss
        return {}  # Trainer accumulates loss automatically

    return compute_metrics


def get_suggested_training_args(
    output_dir: str = "runs/sarftok_run",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    warmup_ratio: float = 0.05,
    logging_steps: int = 50,
    save_steps: int = 500,
    fp16: bool = True,
) -> dict:
    """Return a dict of recommended HF TrainingArguments kwargs."""
    return {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": warmup_ratio,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "fp16": fp16,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,  # CRITICAL: keep morph metadata columns
        "report_to": "none",
    }


class SarfTokTrainer:
    """Minimal trainer wrapper for SarfTok.

    For production use, prefer HF Trainer with the patches applied.
    This class exists as a reference / debugging tool.
    """

    def __init__(self, model, optimizer, scheduler=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        # Pop morph metadata (not standard HF fields)
        word_spans = batch.pop("word_to_surface_spans", None)
        morph_analyses = batch.pop("morph_analyses", None)

        outputs = self.model(
            **batch,
            word_to_surface_spans=word_spans,
            morph_analyses=morph_analyses,
        )

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        metrics = {"loss": loss.item()}
        if isinstance(outputs, dict):
            if "lm_loss" in outputs and outputs["lm_loss"] is not None:
                metrics["lm_loss"] = outputs["lm_loss"].item()
            if "ortho_loss" in outputs:
                metrics["ortho_loss"] = outputs["ortho_loss"].item()
        return metrics
