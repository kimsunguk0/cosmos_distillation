"""Loss configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class DistillationLossWeights:
    hard_ce: float
    seq_kd: float
    logit_kd: float
    feat: float
    aux: float
    self_cons: float
    rank: float


STAGE_DEFAULTS = {
    "stage_a": DistillationLossWeights(1.0, 0.3, 0.2, 0.05, 0.4, 0.2, 0.0),
    "stage_b": DistillationLossWeights(1.0, 0.6, 0.5, 0.15, 0.3, 0.15, 0.1),
    "stage_c": DistillationLossWeights(1.1, 0.3, 0.3, 0.1, 0.35, 0.25, 0.2),
}


def get_stage_weights(stage_name: str) -> DistillationLossWeights:
    """Look up a predefined stage weight schedule."""
    try:
        return STAGE_DEFAULTS[stage_name]
    except KeyError as exc:
        raise ValueError(f"Unknown stage name: {stage_name}") from exc


def _token_losses(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid_mask = (shift_labels != -100).float()
    token_count = valid_mask.sum(dim=1).clamp(min=1.0)
    loss_per_sample = (flat_loss * valid_mask).sum(dim=1) / token_count
    return loss_per_sample, token_count


def weighted_causal_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted CE loss and per-sample token counts."""
    loss_per_sample, token_count = _token_losses(logits, labels)
    if sample_weights is None:
        return loss_per_sample.mean(), token_count
    weight_sum = sample_weights.sum().clamp(min=1e-6)
    return (loss_per_sample * sample_weights).sum() / weight_sum, token_count


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute token accuracy over non-ignored labels."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    preds = shift_logits.argmax(dim=-1)
    valid_mask = shift_labels != -100
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)
    correct = ((preds == shift_labels) & valid_mask).float().sum()
    total = valid_mask.float().sum().clamp(min=1.0)
    return correct / total


def auxiliary_action_loss(
    meta_action_logits: torch.Tensor,
    action_labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy loss over the meta-action head."""
    per_sample = F.cross_entropy(meta_action_logits, action_labels, reduction="none")
    if sample_weights is None:
        return per_sample.mean()
    denom = sample_weights.sum().clamp(min=1e-6)
    return (per_sample * sample_weights).sum() / denom


def self_consistency_penalty(
    meta_action_logits: torch.Tensor,
    action_labels: torch.Tensor,
    consistency_scores: torch.Tensor,
) -> torch.Tensor:
    """Encourage agreement on samples that already have higher consistency confidence."""
    target_loss = F.cross_entropy(meta_action_logits, action_labels, reduction="none")
    weights = consistency_scores.clamp(min=0.0, max=1.0)
    denom = weights.sum().clamp(min=1e-6)
    return (target_loss * weights).sum() / denom
