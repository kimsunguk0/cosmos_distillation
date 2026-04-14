"""Trainer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.training.losses import (
    DistillationLossWeights,
    auxiliary_action_loss,
    feature_alignment_loss,
    logit_kd_loss,
    masked_token_accuracy,
    ranking_consistency_loss,
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
        elif isinstance(value, list):
            converted = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    converted.append(item.to(device))
                elif isinstance(item, dict):
                    converted.append(move_batch_to_device(item, device))
                else:
                    converted.append(item)
            moved[key] = converted
        else:
            moved[key] = value
    return moved


def run_train_step(
    model,
    batch: dict[str, Any],
    weights: DistillationLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one train step and return total loss plus scalar logs."""
    hard_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
    )
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
    logit_kd = torch.tensor(0.0, device=batch["input_ids"].device)
    feat_align = torch.tensor(0.0, device=batch["input_ids"].device)
    teacher_target_batches = batch.get("teacher_target_batches") or []
    if teacher_target_batches:
        seq_losses = []
        logit_losses = []
        feat_losses = []
        for teacher_batch in teacher_target_batches:
            teacher_outputs = model(
                input_ids=teacher_batch["input_ids"],
                attention_mask=teacher_batch["attention_mask"],
                pixel_values=teacher_batch.get("pixel_values"),
                image_grid_thw=teacher_batch.get("image_grid_thw"),
            )
            target_weights = teacher_batch["weights"].to(batch["input_ids"].device)
            seq_loss, _ = weighted_causal_ce(
                teacher_outputs["logits"],
                teacher_batch["labels"],
                target_weights,
            )
            seq_losses.append(seq_loss)
            logit_losses.append(
                logit_kd_loss(
                    teacher_outputs["logits"],
                    teacher_batch.get("teacher_topk_indices"),
                    teacher_batch.get("teacher_topk_logits"),
                    teacher_batch.get("teacher_topk_mask"),
                    teacher_batch["labels"],
                    target_weights,
                )
            )
            feat_losses.append(
                feature_alignment_loss(
                    teacher_outputs["hidden_states"],
                    teacher_batch.get("teacher_pooled_hidden"),
                    teacher_batch["attention_mask"],
                    target_weights,
                )
            )
        seq_kd = torch.stack(seq_losses).mean()
        logit_kd = torch.stack(logit_losses).mean()
        feat_align = torch.stack(feat_losses).mean()

    rank = ranking_consistency_loss(
        hard_outputs["meta_action_logits"],
        batch["teacher_action_class_labels"],
        batch["teacher_action_present_mask"],
        batch["teacher_selection_scores"],
        batch["rank_weights"],
    )

    total = (
        weights.hard_ce * hard_ce
        + weights.seq_kd * seq_kd
        + weights.logit_kd * logit_kd
        + weights.feat * feat_align
        + weights.aux * aux
        + weights.self_cons * self_cons
        + weights.rank * rank
    )
    metrics = {
        "hard_ce": float(hard_ce.detach().cpu()),
        "seq_kd": float(seq_kd.detach().cpu()),
        "logit_kd": float(logit_kd.detach().cpu()),
        "feat_align": float(feat_align.detach().cpu()),
        "aux": float(aux.detach().cpu()),
        "self_cons": float(self_cons.detach().cpu()),
        "rank": float(rank.detach().cpu()),
        "token_acc": float(masked_token_accuracy(hard_outputs["logits"], batch["labels"]).detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    return total, metrics
