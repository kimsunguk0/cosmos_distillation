"""Trainer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.training.losses import (
    DistillationLossWeights,
    auxiliary_action_loss,
    masked_token_accuracy,
    self_consistency_penalty,
    weighted_causal_ce,
)


@dataclass(slots=True)
class TrainerConfig:
    stage_name: str
    max_length: int = 4096
    bf16: bool = True
    learning_rate: float = 2e-5
    batch_size: int = 1
    max_steps: int | None = None


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a nested batch dict onto the target device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


def run_train_step(
    model,
    batch: dict[str, Any],
    weights: DistillationLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one train step and return total loss plus scalar logs."""
    hard_outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    hard_ce, _ = weighted_causal_ce(
        hard_outputs["logits"],
        batch["labels"],
        batch["hard_ce_weights"],
    )
    aux = auxiliary_action_loss(
        hard_outputs["meta_action_logits"],
        batch["action_class_labels"],
        batch["hard_ce_weights"],
    )
    self_cons = self_consistency_penalty(
        hard_outputs["meta_action_logits"],
        batch["action_class_labels"],
        batch["consistency_scores"],
    )

    seq_kd = torch.tensor(0.0, device=batch["input_ids"].device)
    teacher_batch = batch.get("teacher_batch")
    if teacher_batch is not None:
        teacher_outputs = model(
            input_ids=teacher_batch["input_ids"],
            attention_mask=teacher_batch["attention_mask"],
        )
        seq_kd, _ = weighted_causal_ce(
            teacher_outputs["logits"],
            teacher_batch["labels"],
            batch["seq_kd_weights"][teacher_batch["present_mask"]],
        )

    total = (
        weights.hard_ce * hard_ce
        + weights.seq_kd * seq_kd
        + weights.aux * aux
        + weights.self_cons * self_cons
    )
    metrics = {
        "hard_ce": float(hard_ce.detach().cpu()),
        "seq_kd": float(seq_kd.detach().cpu()),
        "aux": float(aux.detach().cpu()),
        "self_cons": float(self_cons.detach().cpu()),
        "token_acc": float(masked_token_accuracy(hard_outputs["logits"], batch["labels"]).detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    return total, metrics
