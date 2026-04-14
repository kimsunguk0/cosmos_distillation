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


def _apply_sample_weights(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    """Scale losses by sample weights while preserving magnitude for batch_size=1."""
    if sample_weights is None:
        return values.mean()
    weights = sample_weights.to(values.device, dtype=values.dtype)
    return (values * weights).mean()


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
    return _apply_sample_weights(loss_per_sample, sample_weights), token_count


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
    return _apply_sample_weights(per_sample, sample_weights)


def self_consistency_penalty(
    meta_action_logits: torch.Tensor,
    action_labels: torch.Tensor,
    consistency_scores: torch.Tensor,
) -> torch.Tensor:
    """Encourage agreement on samples that already have higher consistency confidence."""
    target_loss = F.cross_entropy(meta_action_logits, action_labels, reduction="none")
    weights = consistency_scores.clamp(min=0.0, max=1.0)
    return _apply_sample_weights(target_loss, weights)


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor | None,
    teacher_topk_logits: torch.Tensor | None,
    teacher_topk_mask: torch.Tensor | None,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL distillation over sparse top-k teacher logits when available."""
    if teacher_topk_indices is None or teacher_topk_logits is None or teacher_topk_mask is None:
        return torch.tensor(0.0, device=student_logits.device)

    shift_student = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != -100
    if not valid_mask.any():
        return torch.tensor(0.0, device=student_logits.device)

    gather_indices = teacher_topk_indices.to(student_logits.device)
    gather_values = teacher_topk_logits.to(student_logits.device)
    sparse_mask = teacher_topk_mask.to(student_logits.device)

    vocab_size = shift_student.shape[-1]
    per_sample_losses = []
    per_sample_weights = []
    for sample_index in range(shift_student.shape[0]):
        student_target_logits = shift_student[sample_index][valid_mask[sample_index]]
        teacher_target_indices = gather_indices[sample_index][sparse_mask[sample_index]]
        teacher_target_values = gather_values[sample_index][sparse_mask[sample_index]]
        aligned_tokens = min(student_target_logits.shape[0], teacher_target_indices.shape[0])
        if aligned_tokens <= 0:
            continue
        student_target_logits = student_target_logits[:aligned_tokens]
        teacher_target_indices = teacher_target_indices[:aligned_tokens]
        teacher_target_values = teacher_target_values[:aligned_tokens]
        token_losses = []
        for token_index in range(aligned_tokens):
            token_teacher_indices = teacher_target_indices[token_index]
            token_teacher_values = teacher_target_values[token_index]
            keep = (token_teacher_indices >= 0) & (token_teacher_indices < vocab_size)
            if not keep.any():
                continue
            token_teacher_indices = token_teacher_indices[keep]
            token_teacher_values = token_teacher_values[keep]
            gathered_student = torch.gather(
                student_target_logits[token_index],
                dim=-1,
                index=token_teacher_indices,
            )
            student_log_probs = F.log_softmax(gathered_student / temperature, dim=-1)
            teacher_probs = F.softmax(token_teacher_values / temperature, dim=-1)
            token_losses.append(F.kl_div(student_log_probs, teacher_probs, reduction="sum"))
        if not token_losses:
            continue
        per_sample_losses.append(torch.stack(token_losses).mean())
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=student_logits.device))
        else:
            per_sample_weights.append(sample_weights[sample_index])

    if not per_sample_losses:
        return torch.tensor(0.0, device=student_logits.device)

    losses = torch.stack(per_sample_losses)
    weights_tensor = torch.stack(per_sample_weights).to(student_logits.device)
    return _apply_sample_weights(losses, None if sample_weights is None else weights_tensor)


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

    teacher_pooled = teacher_hidden.mean(dim=1) if teacher_hidden.ndim == 3 else teacher_hidden
    if teacher_pooled.shape[-1] != student_pooled.shape[-1]:
        return torch.tensor(0.0, device=student_hidden.device)
    per_sample = F.mse_loss(student_pooled, teacher_pooled, reduction="none").mean(dim=-1)
    return _apply_sample_weights(per_sample, sample_weights)


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
    return _apply_sample_weights(per_sample, weights)
