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


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL distillation over token logits when teacher logits are available."""
    if teacher_logits is None:
        return torch.tensor(0.0, device=student_logits.device)

    shift_student = student_logits[:, :-1, :].contiguous()
    shift_teacher = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != -100
    if not valid_mask.any():
        return torch.tensor(0.0, device=student_logits.device)

    student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    token_kl = token_kl * valid_mask.float()
    per_sample = token_kl.sum(dim=1) / valid_mask.float().sum(dim=1).clamp(min=1.0)
    if sample_weights is None:
        return per_sample.mean()
    denom = sample_weights.sum().clamp(min=1e-6)
    return (per_sample * sample_weights).sum() / denom


def feature_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    attention_mask: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align pooled hidden states when teacher features are available."""
    if teacher_hidden is None:
        return torch.tensor(0.0, device=student_hidden.device)

    if attention_mask is None:
        student_pooled = student_hidden.mean(dim=1)
    else:
        mask = attention_mask.unsqueeze(-1).to(student_hidden.dtype)
        student_pooled = (student_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    teacher_pooled = teacher_hidden.mean(dim=1)
    per_sample = F.mse_loss(student_pooled, teacher_pooled, reduction="none").mean(dim=-1)
    if sample_weights is None:
        return per_sample.mean()
    denom = sample_weights.sum().clamp(min=1e-6)
    return (per_sample * sample_weights).sum() / denom


def ranking_consistency_loss(
    meta_action_logits: torch.Tensor,
    teacher_action_labels: torch.Tensor,
    teacher_action_present_mask: torch.Tensor,
    selection_scores: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encourage the teacher-selected action class to outrank alternatives."""
    valid = teacher_action_present_mask.bool()
    if not valid.any():
        return torch.tensor(0.0, device=meta_action_logits.device)

    chosen_logits = meta_action_logits[valid, teacher_action_labels[valid]]
    all_logits = meta_action_logits[valid]
    competitor_logits = all_logits.masked_fill(
        F.one_hot(teacher_action_labels[valid], num_classes=all_logits.shape[-1]).bool(),
        float("-inf"),
    ).max(dim=-1).values
    margin = 0.2
    per_sample = F.relu(margin - (chosen_logits - competitor_logits))
    weights = selection_scores[valid].clamp(min=0.0, max=1.0)
    if sample_weights is not None:
        weights = weights * sample_weights[valid]
    denom = weights.sum().clamp(min=1e-6)
    return (per_sample * weights).sum() / denom
