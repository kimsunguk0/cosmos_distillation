"""Loss configuration helpers for v3.2 distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


@dataclass(slots=True)
class DistillationLossWeights:
    hard_cot_ce: float
    teacher_seq_ce: float
    teacher_logit_kd: float
    traj_ce: float
    format_ce: float
    action_aux: float
    feat_align: float
    traj_aux_reg: float = 0.0
    teacher_traj_ce: float | None = None
    teacher_traj_topk_kd: float = 0.0
    teacher_traj_hidden_align: float = 0.0
    traj_xyz_reg: float = 0.0
    traj_delta_reg: float = 0.0
    traj_final_reg: float = 0.0
    traj_control_reg: float = 0.0
    traj_control_delta_reg: float = 0.0
    traj_aux_xyz_reg: float = 0.0
    traj_aux_final_reg: float = 0.0
    traj_aux_guided_kd: float = 0.0
    traj_aux_pseudo_ce: float = 0.0


@dataclass(slots=True)
class TrajectoryDecodeConfig:
    traj_token_start_idx: int
    num_bins: int
    dims_min: tuple[float, float]
    dims_max: tuple[float, float]
    accel_mean: float
    accel_std: float
    curvature_mean: float
    curvature_std: float
    dt: float
    n_waypoints: int
    short_horizon_steps: int = 24
    short_horizon_weight: float = 1.0


@dataclass(slots=True)
class TrajectoryAuxInterfaceConfig:
    num_buckets: int = 1
    normalize_targets: bool = False
    tanh_bound: float = 3.0
    huber_delta: float = 1.0
    decode_tanh_scale: float = 1.0
    decode_edge_margin: int = 0
    decode_prior_sigma: float = 96.0
    target_means: tuple[tuple[float, float], ...] | None = None
    target_stds: tuple[tuple[float, float], ...] | None = None


STAGE_DEFAULTS = {
    "stage_a": DistillationLossWeights(1.0, 0.12, 0.18, 0.25, 0.5, 0.0, 0.0),
    "stage_b": DistillationLossWeights(1.0, 0.25, 0.4, 0.2, 0.25, 0.05, 0.0),
    "stage_c": DistillationLossWeights(1.0, 0.2, 0.35, 0.15, 0.15, 0.08, 0.0),
}


LOSS_WEIGHT_ALIASES: dict[str, tuple[str, ...]] = {
    "hard_cot_ce": ("gt_cot_loss",),
    "teacher_seq_ce": ("teacher_cot_loss",),
    "teacher_logit_kd": ("teacher_topk_kd_loss",),
    "traj_ce": ("traj_loss",),
    "traj_aux_reg": ("traj_aux_loss", "traj_interface_loss"),
    "teacher_traj_ce": ("teacher_traj_loss",),
    "teacher_traj_topk_kd": ("teacher_traj_topk_kd_loss",),
    "teacher_traj_hidden_align": ("teacher_traj_hidden_align_loss",),
    "format_ce": ("output_format_loss", "structure_loss"),
    "action_aux": ("meta_action_loss", "action_aux_loss"),
    "feat_align": ("feature_align_loss",),
    "traj_xyz_reg": ("traj_xyz_loss",),
    "traj_delta_reg": ("traj_delta_loss",),
    "traj_final_reg": ("traj_final_loss",),
    "traj_control_reg": ("traj_control_loss",),
    "traj_control_delta_reg": ("traj_control_delta_loss",),
    "traj_aux_xyz_reg": ("traj_aux_xyz_loss", "traj_aux_gt_xyz_loss"),
    "traj_aux_final_reg": ("traj_aux_final_loss", "traj_aux_gt_final_loss"),
    "traj_aux_guided_kd": ("traj_aux_guided_kd_loss", "traj_aux_prior_kd_loss"),
    "traj_aux_pseudo_ce": ("traj_aux_pseudo_ce_loss", "traj_aux_token_ce_loss"),
}


LOSS_WEIGHT_EXPORT_NAMES: dict[str, str] = {
    "hard_cot_ce": "gt_cot_loss",
    "teacher_seq_ce": "teacher_cot_loss",
    "teacher_logit_kd": "teacher_topk_kd_loss",
    "traj_ce": "traj_loss",
    "traj_aux_reg": "traj_aux_loss",
    "teacher_traj_ce": "teacher_traj_loss",
    "teacher_traj_topk_kd": "teacher_traj_topk_kd_loss",
    "teacher_traj_hidden_align": "teacher_traj_hidden_align_loss",
    "format_ce": "output_format_loss",
    "action_aux": "meta_action_loss",
    "feat_align": "feature_align_loss",
    "traj_xyz_reg": "traj_xyz_loss",
    "traj_delta_reg": "traj_delta_loss",
    "traj_final_reg": "traj_final_loss",
    "traj_control_reg": "traj_control_loss",
    "traj_control_delta_reg": "traj_control_delta_loss",
    "traj_aux_xyz_reg": "traj_aux_xyz_loss",
    "traj_aux_final_reg": "traj_aux_final_loss",
    "traj_aux_guided_kd": "traj_aux_guided_kd_loss",
    "traj_aux_pseudo_ce": "traj_aux_pseudo_ce_loss",
}


METRIC_EXPORT_NAMES: dict[str, str] = {
    "hard_cot_ce": "gt_cot_loss",
    "teacher_seq_ce": "teacher_cot_loss",
    "teacher_logit_kd": "teacher_topk_kd_loss",
    "hard_traj_ce": "gt_traj_loss",
    "traj_aux_reg": "traj_aux_loss",
    "teacher_traj_ce": "teacher_traj_loss",
    "teacher_traj_topk_kd": "teacher_traj_topk_kd_loss",
    "teacher_traj_hidden_align": "teacher_traj_hidden_align_loss",
    "traj_ce": "traj_loss",
    "format_ce": "output_format_loss",
    "action_aux": "meta_action_loss",
    "feat_align": "feature_align_loss",
    "traj_xyz_reg": "traj_xyz_loss",
    "traj_delta_reg": "traj_delta_loss",
    "traj_final_reg": "traj_final_loss",
    "traj_control_reg": "traj_control_loss",
    "traj_control_delta_reg": "traj_control_delta_loss",
    "traj_aux_xyz_reg": "traj_aux_xyz_loss",
    "traj_aux_final_reg": "traj_aux_final_loss",
    "traj_aux_guided_kd": "traj_aux_guided_kd_loss",
    "traj_aux_pseudo_ce": "traj_aux_pseudo_ce_loss",
    "hard_token_acc": "response_token_acc",
    "hard_cot_acc": "gt_cot_token_acc",
    "hard_traj_acc": "traj_token_acc",
    "teacher_cot_acc": "teacher_cot_token_acc",
}


def resolve_loss_weight_value(
    weights: Mapping[str, object] | None,
    canonical_name: str,
    default: float,
) -> float:
    """Read a loss weight from either the legacy or user-facing key."""
    if not weights:
        return float(default)
    for key in (canonical_name, *LOSS_WEIGHT_ALIASES.get(canonical_name, ())):
        if key in weights:
            return float(weights[key])
    return float(default)


def resolve_optional_loss_weight_value(
    weights: Mapping[str, object] | None,
    canonical_name: str,
) -> float | None:
    """Read an optional loss weight and return None when no key is present."""
    if not weights:
        return None
    for key in (canonical_name, *LOSS_WEIGHT_ALIASES.get(canonical_name, ())):
        if key in weights:
            return float(weights[key])
    return None


def export_loss_weights(weights: DistillationLossWeights) -> dict[str, float]:
    """Return user-facing loss weight names for logs and summaries."""
    values = {
        "hard_cot_ce": weights.hard_cot_ce,
        "teacher_seq_ce": weights.teacher_seq_ce,
        "teacher_logit_kd": weights.teacher_logit_kd,
        "traj_ce": weights.traj_ce,
        "traj_aux_reg": weights.traj_aux_reg,
        "format_ce": weights.format_ce,
        "action_aux": weights.action_aux,
        "feat_align": weights.feat_align,
        "teacher_traj_topk_kd": weights.teacher_traj_topk_kd,
        "teacher_traj_hidden_align": weights.teacher_traj_hidden_align,
        "traj_xyz_reg": weights.traj_xyz_reg,
        "traj_delta_reg": weights.traj_delta_reg,
        "traj_final_reg": weights.traj_final_reg,
        "traj_control_reg": weights.traj_control_reg,
        "traj_control_delta_reg": weights.traj_control_delta_reg,
        "traj_aux_xyz_reg": weights.traj_aux_xyz_reg,
        "traj_aux_final_reg": weights.traj_aux_final_reg,
        "traj_aux_guided_kd": weights.traj_aux_guided_kd,
        "traj_aux_pseudo_ce": weights.traj_aux_pseudo_ce,
    }
    exported = {LOSS_WEIGHT_EXPORT_NAMES[key]: float(value) for key, value in values.items()}
    if weights.teacher_traj_ce is not None:
        exported[LOSS_WEIGHT_EXPORT_NAMES["teacher_traj_ce"]] = float(weights.teacher_traj_ce)
    return exported


def export_metric_logs(metrics: Mapping[str, float]) -> dict[str, float]:
    """Return user-facing metric names while keeping total_loss unchanged."""
    exported: dict[str, float] = {}
    for key, value in metrics.items():
        exported[METRIC_EXPORT_NAMES.get(key, key)] = float(value)
    return exported


def _zero(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def _apply_sample_weights(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    """Scale per-sample losses while preserving batch_size=1 behavior."""
    if values.numel() == 0:
        return _zero(values.device)
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


def _masked_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    label_token_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_label_token_weights = None
    if label_token_weights is not None:
        shift_label_token_weights = label_token_weights[:, 1:].contiguous().to(device=shift_logits.device)
    valid_mask = shift_labels != IGNORE_INDEX
    if token_mask is not None:
        valid_mask = valid_mask & token_mask[:, 1:].to(dtype=torch.bool, device=shift_labels.device)
    token_count = valid_mask.float().sum(dim=1)
    loss_per_sample = torch.zeros((shift_logits.shape[0],), dtype=shift_logits.dtype, device=shift_logits.device)
    for sample_index in range(shift_logits.shape[0]):
        sample_mask = valid_mask[sample_index]
        if not sample_mask.any():
            continue
        selected_logits = shift_logits[sample_index][sample_mask]
        selected_labels = shift_labels[sample_index][sample_mask]
        if shift_label_token_weights is None:
            loss_per_sample[sample_index] = F.cross_entropy(
                selected_logits,
                selected_labels,
                reduction="mean",
            )
            continue
        selected_token_weights = shift_label_token_weights[sample_index][sample_mask].to(dtype=shift_logits.dtype)
        per_token_loss = F.cross_entropy(
            selected_logits,
            selected_labels,
            reduction="none",
        )
        loss_per_sample[sample_index] = (per_token_loss * selected_token_weights).sum() / selected_token_weights.sum().clamp(min=1e-6)
    return loss_per_sample, token_count


def weighted_causal_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    label_token_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted CE loss and per-sample token counts."""
    loss_per_sample, token_count = _masked_token_losses(logits, labels, token_mask, label_token_weights)
    return _apply_sample_weights(loss_per_sample, sample_weights), token_count


def masked_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute token accuracy over the selected non-ignored labels."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != IGNORE_INDEX
    if token_mask is not None:
        valid_mask = valid_mask & token_mask[:, 1:].to(dtype=torch.bool, device=shift_labels.device)
    if not valid_mask.any():
        return _zero(logits.device)
    preds = shift_logits.argmax(dim=-1)
    correct = ((preds == shift_labels) & valid_mask).float().sum()
    total = valid_mask.float().sum().clamp(min=1.0)
    return correct / total


def auxiliary_action_loss(
    meta_action_logits: torch.Tensor,
    action_labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy loss over the action auxiliary head."""
    per_sample = F.cross_entropy(meta_action_logits, action_labels, reduction="none")
    return _apply_sample_weights(per_sample, sample_weights)


def teacher_logit_kd_loss(
    student_logits: torch.Tensor,
    cot_content_mask: torch.Tensor | None,
    teacher_topk_indices: torch.Tensor | None,
    teacher_topk_logprobs: torch.Tensor | None,
    teacher_topk_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL distillation over sparse teacher top-k distributions on CoT token positions."""
    if (
        cot_content_mask is None
        or teacher_topk_indices is None
        or teacher_topk_logprobs is None
        or teacher_topk_mask is None
    ):
        return _zero(student_logits.device)

    shift_student = student_logits[:, :-1, :].contiguous()
    span_mask = cot_content_mask[:, 1:].to(dtype=torch.bool, device=student_logits.device)
    gather_indices = teacher_topk_indices.to(student_logits.device)
    gather_values = teacher_topk_logprobs.to(student_logits.device)
    sparse_mask = teacher_topk_mask.to(student_logits.device)
    vocab_size = shift_student.shape[-1]

    per_sample_losses = []
    per_sample_weights = []
    for sample_index in range(shift_student.shape[0]):
        student_target_logits = shift_student[sample_index][span_mask[sample_index]]
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
            token_losses.append(F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (temperature**2))
        if not token_losses:
            continue
        per_sample_losses.append(torch.stack(token_losses).mean())
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=student_logits.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(student_logits.device))

    if not per_sample_losses:
        return _zero(student_logits.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def feature_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    attention_mask: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align pooled hidden states when teacher features are available."""
    if teacher_hidden is None:
        return _zero(student_hidden.device)

    if attention_mask is None:
        student_pooled = student_hidden.mean(dim=1)
    else:
        mask = attention_mask.unsqueeze(-1).to(student_hidden.dtype)
        student_pooled = (student_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    teacher_pooled = teacher_hidden.mean(dim=1) if teacher_hidden.ndim == 3 else teacher_hidden
    if teacher_pooled.shape[-1] != student_pooled.shape[-1]:
        return _zero(student_hidden.device)
    per_sample = F.mse_loss(student_pooled, teacher_pooled, reduction="none").mean(dim=-1)
    return _apply_sample_weights(per_sample, sample_weights)


def token_hidden_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    token_mask: torch.Tensor | None,
    teacher_token_mask: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align per-token hidden states over a selected masked span."""
    if teacher_hidden is None or token_mask is None:
        return _zero(student_hidden.device)

    shift_student = student_hidden[:, :-1, :].contiguous()
    span_mask = token_mask[:, 1:].to(dtype=torch.bool, device=student_hidden.device)
    teacher_hidden = teacher_hidden.to(device=student_hidden.device, dtype=shift_student.dtype)

    per_sample_losses = []
    per_sample_weights = []
    for sample_index in range(shift_student.shape[0]):
        student_target_hidden = shift_student[sample_index][span_mask[sample_index]]
        teacher_target_hidden = teacher_hidden[sample_index]
        if teacher_token_mask is not None:
            teacher_target_hidden = teacher_target_hidden[
                teacher_token_mask[sample_index].to(dtype=torch.bool, device=student_hidden.device)
            ]
        aligned_tokens = min(student_target_hidden.shape[0], teacher_target_hidden.shape[0])
        if aligned_tokens <= 0:
            continue
        student_target_hidden = student_target_hidden[:aligned_tokens]
        teacher_target_hidden = teacher_target_hidden[:aligned_tokens]
        if student_target_hidden.shape[-1] != teacher_target_hidden.shape[-1]:
            continue
        per_sample_losses.append(
            F.mse_loss(student_target_hidden, teacher_target_hidden, reduction="none").mean(dim=-1).mean()
        )
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=student_hidden.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(student_hidden.device))

    if not per_sample_losses:
        return _zero(student_hidden.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def token_hidden_alignment_bridge_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    token_mask: torch.Tensor | None,
    teacher_token_mask: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    *,
    cosine_weight: float = 0.8,
    mse_weight: float = 0.2,
) -> torch.Tensor:
    """Align projected token states in a shared bridge space using cosine-heavy loss."""
    if teacher_hidden is None or token_mask is None:
        return _zero(student_hidden.device)

    shift_student = student_hidden[:, :-1, :].contiguous()
    span_mask = token_mask[:, 1:].to(dtype=torch.bool, device=student_hidden.device)
    teacher_hidden = teacher_hidden.to(device=student_hidden.device, dtype=shift_student.dtype)

    per_sample_losses = []
    per_sample_weights = []
    for sample_index in range(shift_student.shape[0]):
        student_target_hidden = shift_student[sample_index][span_mask[sample_index]]
        teacher_target_hidden = teacher_hidden[sample_index]
        if teacher_token_mask is not None:
            teacher_target_hidden = teacher_target_hidden[
                teacher_token_mask[sample_index].to(dtype=torch.bool, device=student_hidden.device)
            ]
        aligned_tokens = min(student_target_hidden.shape[0], teacher_target_hidden.shape[0])
        if aligned_tokens <= 0:
            continue
        student_target_hidden = student_target_hidden[:aligned_tokens]
        teacher_target_hidden = teacher_target_hidden[:aligned_tokens]
        if student_target_hidden.shape[-1] != teacher_target_hidden.shape[-1]:
            continue
        cosine_loss = 1.0 - F.cosine_similarity(student_target_hidden, teacher_target_hidden, dim=-1)
        mse_loss = F.mse_loss(student_target_hidden, teacher_target_hidden, reduction="none").mean(dim=-1)
        per_sample_losses.append((cosine_weight * cosine_loss + mse_weight * mse_loss).mean())
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=student_hidden.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(student_hidden.device))

    if not per_sample_losses:
        return _zero(student_hidden.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def token_hidden_relation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor | None,
    token_mask: torch.Tensor | None,
    teacher_token_mask: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match token-to-token cosine geometry in a shared bridge space."""
    if teacher_hidden is None or token_mask is None:
        return _zero(student_hidden.device)

    shift_student = student_hidden[:, :-1, :].contiguous()
    span_mask = token_mask[:, 1:].to(dtype=torch.bool, device=student_hidden.device)
    teacher_hidden = teacher_hidden.to(device=student_hidden.device, dtype=shift_student.dtype)

    per_sample_losses = []
    per_sample_weights = []
    for sample_index in range(shift_student.shape[0]):
        student_target_hidden = shift_student[sample_index][span_mask[sample_index]]
        teacher_target_hidden = teacher_hidden[sample_index]
        if teacher_token_mask is not None:
            teacher_target_hidden = teacher_target_hidden[
                teacher_token_mask[sample_index].to(dtype=torch.bool, device=student_hidden.device)
            ]
        aligned_tokens = min(student_target_hidden.shape[0], teacher_target_hidden.shape[0])
        if aligned_tokens <= 1:
            continue
        student_target_hidden = F.normalize(student_target_hidden[:aligned_tokens], dim=-1)
        teacher_target_hidden = F.normalize(teacher_target_hidden[:aligned_tokens], dim=-1)
        student_rel = student_target_hidden @ student_target_hidden.transpose(0, 1)
        teacher_rel = teacher_target_hidden @ teacher_target_hidden.transpose(0, 1)
        triu = torch.triu_indices(aligned_tokens, aligned_tokens, offset=1, device=student_hidden.device)
        if triu.shape[1] <= 0:
            continue
        per_sample_losses.append(
            F.mse_loss(
                student_rel[triu[0], triu[1]],
                teacher_rel[triu[0], triu[1]],
                reduction="mean",
            )
        )
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=student_hidden.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(student_hidden.device))

    if not per_sample_losses:
        return _zero(student_hidden.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def token_hidden_variance_floor_loss(
    student_hidden: torch.Tensor,
    token_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None = None,
    *,
    target_std: float = 0.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Discourage token states from collapsing into a tiny subspace."""
    if token_mask is None:
        return _zero(student_hidden.device)

    shift_student = student_hidden[:, :-1, :].contiguous()
    span_mask = token_mask[:, 1:].to(dtype=torch.bool, device=student_hidden.device)

    per_sample_losses = []
    per_sample_weights = []
    for sample_index in range(shift_student.shape[0]):
        student_target_hidden = shift_student[sample_index][span_mask[sample_index]]
        if student_target_hidden.shape[0] <= 1:
            continue
        centered = student_target_hidden - student_target_hidden.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
        penalty = F.relu(float(target_std) - std).mean()
        per_sample_losses.append(penalty)
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=student_hidden.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(student_hidden.device))

    if not per_sample_losses:
        return _zero(student_hidden.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def token_hidden_covariance_loss(
    student_hidden: torch.Tensor,
    token_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce feature-axis covariance so bridge features do not collapse into one direction."""
    if token_mask is None:
        return _zero(student_hidden.device)

    shift_student = student_hidden[:, :-1, :].contiguous()
    span_mask = token_mask[:, 1:].to(dtype=torch.bool, device=student_hidden.device)

    per_sample_losses = []
    per_sample_weights = []
    for sample_index in range(shift_student.shape[0]):
        student_target_hidden = shift_student[sample_index][span_mask[sample_index]]
        if student_target_hidden.shape[0] <= 1:
            continue
        centered = student_target_hidden - student_target_hidden.mean(dim=0, keepdim=True)
        cov = centered.transpose(0, 1) @ centered / max(int(centered.shape[0] - 1), 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        per_sample_losses.append((off_diag**2).mean())
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=student_hidden.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(student_hidden.device))

    if not per_sample_losses:
        return _zero(student_hidden.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def _gather_expected_traj_controls(
    logits: torch.Tensor,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor | None,
    config: TrajectoryDecodeConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if traj_token_mask is None:
        return [], []
    valid_mask = (shift_labels != IGNORE_INDEX) & traj_token_mask[:, 1:].to(dtype=torch.bool, device=shift_labels.device)
    traj_slice = slice(config.traj_token_start_idx, config.traj_token_start_idx + config.num_bins)
    bin_indices = torch.arange(config.num_bins, device=shift_logits.device, dtype=shift_logits.dtype)
    dims_min = torch.tensor(config.dims_min, device=shift_logits.device, dtype=shift_logits.dtype)
    dims_max = torch.tensor(config.dims_max, device=shift_logits.device, dtype=shift_logits.dtype)

    predicted_controls: list[torch.Tensor] = []
    target_controls: list[torch.Tensor] = []
    for sample_index in range(shift_logits.shape[0]):
        sample_mask = valid_mask[sample_index]
        if not sample_mask.any():
            predicted_controls.append(torch.empty((0, 2), device=shift_logits.device, dtype=shift_logits.dtype))
            target_controls.append(torch.empty((0, 2), device=shift_logits.device, dtype=shift_logits.dtype))
            continue

        selected_logits = shift_logits[sample_index][sample_mask][:, traj_slice]
        selected_labels = shift_labels[sample_index][sample_mask] - int(config.traj_token_start_idx)
        usable_count = min(selected_logits.shape[0], selected_labels.shape[0], config.n_waypoints * 2)
        usable_count -= usable_count % 2
        if usable_count <= 0:
            predicted_controls.append(torch.empty((0, 2), device=shift_logits.device, dtype=shift_logits.dtype))
            target_controls.append(torch.empty((0, 2), device=shift_logits.device, dtype=shift_logits.dtype))
            continue

        selected_logits = selected_logits[:usable_count]
        selected_labels = selected_labels[:usable_count].clamp(min=0, max=config.num_bins - 1).to(dtype=shift_logits.dtype)
        probs = F.softmax(selected_logits, dim=-1)
        expected_bins = (probs * bin_indices.unsqueeze(0)).sum(dim=-1)

        dim_ids = torch.arange(usable_count, device=shift_logits.device) % 2
        dim_min = dims_min[dim_ids]
        dim_max = dims_max[dim_ids]
        expected_controls_flat = expected_bins / float(config.num_bins - 1) * (dim_max - dim_min) + dim_min
        target_controls_flat = selected_labels / float(config.num_bins - 1) * (dim_max - dim_min) + dim_min
        predicted_controls.append(expected_controls_flat.reshape(-1, 2))
        target_controls.append(target_controls_flat.reshape(-1, 2))
    return predicted_controls, target_controls


def _gather_aux_predicted_controls(
    traj_aux_values: torch.Tensor,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor | None,
    config: TrajectoryDecodeConfig,
    aux_config: TrajectoryAuxInterfaceConfig | None = None,
) -> list[torch.Tensor]:
    """Gather bucketed auxiliary-head control predictions over traj body positions."""
    if traj_token_mask is None:
        return []
    shift_aux = traj_aux_values[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = (shift_labels != IGNORE_INDEX) & traj_token_mask[:, 1:].to(dtype=torch.bool, device=shift_labels.device)
    num_buckets = max(int(shift_aux.shape[-1] // 2), 1)
    dims_min = torch.tensor(config.dims_min, device=shift_aux.device, dtype=shift_aux.dtype)
    dims_max = torch.tensor(config.dims_max, device=shift_aux.device, dtype=shift_aux.dtype)
    stats_mean, stats_std = _resolve_aux_target_stats(aux_config, shift_aux.device, shift_aux.dtype, num_buckets)

    predicted_controls: list[torch.Tensor] = []
    for sample_index in range(shift_aux.shape[0]):
        sample_mask = valid_mask[sample_index]
        if not sample_mask.any():
            predicted_controls.append(torch.empty((0, 2), device=shift_aux.device, dtype=shift_aux.dtype))
            continue
        selected_aux = shift_aux[sample_index][sample_mask]
        usable_count = min(selected_aux.shape[0], config.n_waypoints * 2)
        usable_count -= usable_count % 2
        if usable_count <= 0:
            predicted_controls.append(torch.empty((0, 2), device=shift_aux.device, dtype=shift_aux.dtype))
            continue

        selected_aux = selected_aux[:usable_count]
        token_indices = torch.arange(usable_count, device=shift_aux.device)
        waypoint_ids = token_indices // 2
        parity = token_indices % 2
        bucket_ids = torch.clamp((waypoint_ids * num_buckets) // max(int(config.n_waypoints), 1), max=num_buckets - 1)
        channel_ids = bucket_ids * 2 + parity
        flat_controls = selected_aux[token_indices, channel_ids]
        if stats_mean is not None and stats_std is not None:
            tanh_bound = max(float(aux_config.tanh_bound if aux_config is not None else 3.0), 1e-6)
            flat_controls = tanh_bound * torch.tanh(flat_controls)
            flat_controls = stats_mean[bucket_ids, parity] + stats_std[bucket_ids, parity] * flat_controls
            flat_controls = torch.clamp(flat_controls, dims_min[parity], dims_max[parity])
        predicted_controls.append(flat_controls.reshape(-1, 2))
    return predicted_controls


def trajectory_aux_guided_kd_loss(
    logits: torch.Tensor,
    traj_aux_values: torch.Tensor | None,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
    config: TrajectoryDecodeConfig | None,
    aux_config: TrajectoryAuxInterfaceConfig | None = None,
) -> torch.Tensor:
    """Use the bounded auxiliary trajectory head as a soft prior for LM traj-token logits."""
    if traj_aux_values is None or config is None or traj_token_mask is None:
        device = labels.device if isinstance(labels, torch.Tensor) else torch.device("cpu")
        return _zero(device)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_aux = traj_aux_values[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = (shift_labels != IGNORE_INDEX) & traj_token_mask[:, 1:].to(dtype=torch.bool, device=shift_labels.device)
    num_buckets = max(int(shift_aux.shape[-1] // 2), 1)
    dims_min = torch.tensor(config.dims_min, device=shift_aux.device, dtype=shift_aux.dtype)
    dims_max = torch.tensor(config.dims_max, device=shift_aux.device, dtype=shift_aux.dtype)
    stats_mean, stats_std = _resolve_aux_target_stats(aux_config, shift_aux.device, shift_aux.dtype, num_buckets)
    traj_slice = slice(config.traj_token_start_idx, config.traj_token_start_idx + config.num_bins)
    bin_indices = torch.arange(config.num_bins, device=shift_aux.device, dtype=shift_aux.dtype)
    sigma = max(float(aux_config.decode_prior_sigma if aux_config is not None else 96.0), 1.0)
    tanh_scale = max(float(aux_config.decode_tanh_scale if aux_config is not None else 1.0), 0.0)
    edge_margin = max(int(aux_config.decode_edge_margin if aux_config is not None else 0), 0)

    per_sample_losses: list[torch.Tensor] = []
    per_sample_weights: list[torch.Tensor] = []
    for sample_index in range(shift_aux.shape[0]):
        sample_mask = valid_mask[sample_index]
        if not sample_mask.any():
            continue
        selected_logits = shift_logits[sample_index][sample_mask][:, traj_slice]
        selected_aux = shift_aux[sample_index][sample_mask]
        usable_count = min(selected_logits.shape[0], selected_aux.shape[0], config.n_waypoints * 2)
        if usable_count <= 0:
            continue
        selected_logits = selected_logits[:usable_count]
        selected_aux = selected_aux[:usable_count]

        token_indices = torch.arange(usable_count, device=shift_aux.device)
        waypoint_ids = token_indices // 2
        parity = token_indices % 2
        bucket_ids = torch.clamp((waypoint_ids * num_buckets) // max(int(config.n_waypoints), 1), max=num_buckets - 1)
        channel_ids = bucket_ids * 2 + parity
        pred_scalar = selected_aux[token_indices, channel_ids]

        dim_min = dims_min[parity]
        dim_max = dims_max[parity]
        if stats_mean is not None and stats_std is not None:
            tanh_bound = max(float(aux_config.tanh_bound if aux_config is not None else 3.0), 1e-6) * tanh_scale
            pred_scalar = tanh_bound * torch.tanh(pred_scalar)
            pred_scalar = stats_mean[bucket_ids, parity] + stats_std[bucket_ids, parity] * pred_scalar
        pred_scalar = torch.clamp(pred_scalar, dim_min, dim_max)

        ratio = torch.where(
            (dim_max - dim_min) > 1e-6,
            (pred_scalar - dim_min) / (dim_max - dim_min),
            torch.zeros_like(pred_scalar),
        )
        center_bins = torch.round(ratio * float(config.num_bins - 1)).clamp(min=0, max=config.num_bins - 1)
        if edge_margin > 0 and config.num_bins > edge_margin * 2:
            center_bins = center_bins.clamp(min=edge_margin, max=config.num_bins - 1 - edge_margin)

        prior_logits = -torch.square((bin_indices.unsqueeze(0) - center_bins.unsqueeze(1)) / sigma)
        target_probs = F.softmax(prior_logits, dim=-1).to(dtype=selected_logits.dtype)
        student_log_probs = F.log_softmax(selected_logits, dim=-1)
        per_token_loss = -(target_probs * student_log_probs).sum(dim=-1)

        token_weights = torch.ones((usable_count,), device=shift_aux.device, dtype=shift_aux.dtype)
        short_steps = max(int(config.short_horizon_steps), 1)
        short_weight = max(float(config.short_horizon_weight), 1.0)
        token_weights[waypoint_ids < short_steps] = short_weight
        per_sample_losses.append((per_token_loss * token_weights).sum() / token_weights.sum().clamp(min=1e-6))
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=shift_aux.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(shift_aux.device))

    if not per_sample_losses:
        return _zero(shift_aux.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def trajectory_aux_pseudo_ce_loss(
    logits: torch.Tensor,
    traj_aux_values: torch.Tensor | None,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
    config: TrajectoryDecodeConfig | None,
    aux_config: TrajectoryAuxInterfaceConfig | None = None,
) -> torch.Tensor:
    """Train LM traj logits toward the bounded aux-derived token ids used by hybrid decode."""
    if traj_aux_values is None or config is None or traj_token_mask is None:
        device = labels.device if isinstance(labels, torch.Tensor) else torch.device("cpu")
        return _zero(device)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_aux = traj_aux_values[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = (shift_labels != IGNORE_INDEX) & traj_token_mask[:, 1:].to(dtype=torch.bool, device=shift_labels.device)
    num_buckets = max(int(shift_aux.shape[-1] // 2), 1)
    dims_min = torch.tensor(config.dims_min, device=shift_aux.device, dtype=shift_aux.dtype)
    dims_max = torch.tensor(config.dims_max, device=shift_aux.device, dtype=shift_aux.dtype)
    stats_mean, stats_std = _resolve_aux_target_stats(aux_config, shift_aux.device, shift_aux.dtype, num_buckets)
    traj_slice = slice(config.traj_token_start_idx, config.traj_token_start_idx + config.num_bins)
    tanh_scale = max(float(aux_config.decode_tanh_scale if aux_config is not None else 1.0), 0.0)
    edge_margin = max(int(aux_config.decode_edge_margin if aux_config is not None else 0), 0)

    per_sample_losses: list[torch.Tensor] = []
    per_sample_weights: list[torch.Tensor] = []
    for sample_index in range(shift_aux.shape[0]):
        sample_mask = valid_mask[sample_index]
        if not sample_mask.any():
            continue
        selected_logits = shift_logits[sample_index][sample_mask][:, traj_slice]
        selected_aux = shift_aux[sample_index][sample_mask]
        usable_count = min(selected_logits.shape[0], selected_aux.shape[0], config.n_waypoints * 2)
        if usable_count <= 0:
            continue
        selected_logits = selected_logits[:usable_count]
        selected_aux = selected_aux[:usable_count]

        token_indices = torch.arange(usable_count, device=shift_aux.device)
        waypoint_ids = token_indices // 2
        parity = token_indices % 2
        bucket_ids = torch.clamp((waypoint_ids * num_buckets) // max(int(config.n_waypoints), 1), max=num_buckets - 1)
        channel_ids = bucket_ids * 2 + parity
        pred_scalar = selected_aux[token_indices, channel_ids]

        dim_min = dims_min[parity]
        dim_max = dims_max[parity]
        if stats_mean is not None and stats_std is not None:
            tanh_bound = max(float(aux_config.tanh_bound if aux_config is not None else 3.0), 1e-6) * tanh_scale
            pred_scalar = tanh_bound * torch.tanh(pred_scalar)
            pred_scalar = stats_mean[bucket_ids, parity] + stats_std[bucket_ids, parity] * pred_scalar
        pred_scalar = torch.clamp(pred_scalar, dim_min, dim_max)

        ratio = torch.where(
            (dim_max - dim_min) > 1e-6,
            (pred_scalar - dim_min) / (dim_max - dim_min),
            torch.zeros_like(pred_scalar),
        )
        center_bins = torch.round(ratio * float(config.num_bins - 1)).clamp(min=0, max=config.num_bins - 1).to(dtype=torch.long)
        if edge_margin > 0 and config.num_bins > edge_margin * 2:
            center_bins = center_bins.clamp(min=edge_margin, max=config.num_bins - 1 - edge_margin)

        per_token_loss = F.cross_entropy(selected_logits, center_bins, reduction="none")
        token_weights = torch.ones((usable_count,), device=shift_aux.device, dtype=shift_aux.dtype)
        short_steps = max(int(config.short_horizon_steps), 1)
        short_weight = max(float(config.short_horizon_weight), 1.0)
        token_weights[waypoint_ids < short_steps] = short_weight
        per_sample_losses.append((per_token_loss * token_weights).sum() / token_weights.sum().clamp(min=1e-6))
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=shift_aux.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(shift_aux.device))

    if not per_sample_losses:
        return _zero(shift_aux.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def _resolve_aux_target_stats(
    aux_config: TrajectoryAuxInterfaceConfig | None,
    device: torch.device,
    dtype: torch.dtype,
    num_buckets: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if (
        aux_config is None
        or not aux_config.normalize_targets
        or aux_config.target_means is None
        or aux_config.target_stds is None
    ):
        return None, None
    means = torch.tensor(aux_config.target_means, device=device, dtype=dtype)
    stds = torch.tensor(aux_config.target_stds, device=device, dtype=dtype)
    if means.shape != (num_buckets, 2) or stds.shape != (num_buckets, 2):
        return None, None
    stds = stds.clamp(min=1e-3)
    return means, stds


def trajectory_aux_regression_loss(
    traj_aux_values: torch.Tensor | None,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
    config: TrajectoryDecodeConfig | None,
    aux_config: TrajectoryAuxInterfaceConfig | None = None,
) -> torch.Tensor:
    """Supervise a training-only trajectory interface head in continuous control space."""
    if traj_aux_values is None or config is None or traj_token_mask is None:
        device = labels.device if isinstance(labels, torch.Tensor) else torch.device("cpu")
        return _zero(device)

    shift_aux = traj_aux_values[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = (shift_labels != IGNORE_INDEX) & traj_token_mask[:, 1:].to(dtype=torch.bool, device=shift_labels.device)
    dims_min = torch.tensor(config.dims_min, device=shift_aux.device, dtype=shift_aux.dtype)
    dims_max = torch.tensor(config.dims_max, device=shift_aux.device, dtype=shift_aux.dtype)
    num_buckets = max(int(shift_aux.shape[-1] // 2), 1)
    stats_mean, stats_std = _resolve_aux_target_stats(aux_config, shift_aux.device, shift_aux.dtype, num_buckets)

    per_sample_losses: list[torch.Tensor] = []
    per_sample_weights: list[torch.Tensor] = []
    for sample_index in range(shift_aux.shape[0]):
        sample_mask = valid_mask[sample_index]
        if not sample_mask.any():
            continue
        selected_aux = shift_aux[sample_index][sample_mask]
        selected_labels = shift_labels[sample_index][sample_mask] - int(config.traj_token_start_idx)
        usable_count = min(selected_aux.shape[0], selected_labels.shape[0], config.n_waypoints * 2)
        if usable_count <= 0:
            continue
        selected_aux = selected_aux[:usable_count]
        selected_labels = selected_labels[:usable_count].clamp(min=0, max=config.num_bins - 1).to(dtype=shift_aux.dtype)

        parity = torch.arange(usable_count, device=shift_aux.device) % 2
        token_indices = torch.arange(usable_count, device=shift_aux.device)
        waypoint_ids = token_indices // 2
        bucket_ids = torch.clamp((waypoint_ids * num_buckets) // max(int(config.n_waypoints), 1), max=num_buckets - 1)
        channel_ids = bucket_ids * 2 + parity
        pred_scalar = selected_aux[token_indices, channel_ids]

        dim_min = dims_min[parity]
        dim_max = dims_max[parity]
        target_scalar = selected_labels / float(config.num_bins - 1) * (dim_max - dim_min) + dim_min

        if stats_mean is not None and stats_std is not None:
            tanh_bound = max(float(aux_config.tanh_bound if aux_config is not None else 3.0), 1e-6)
            huber_delta = max(float(aux_config.huber_delta if aux_config is not None else 1.0), 1e-6)
            target_mean = stats_mean[bucket_ids, parity]
            target_std = stats_std[bucket_ids, parity]
            pred_scalar = tanh_bound * torch.tanh(pred_scalar)
            target_scalar = torch.clamp((target_scalar - target_mean) / target_std, -tanh_bound, tanh_bound)
            per_token_loss = F.huber_loss(pred_scalar, target_scalar, reduction="none", delta=huber_delta)
        else:
            per_token_loss = F.smooth_l1_loss(pred_scalar, target_scalar, reduction="none")
        token_weights = torch.ones((usable_count,), device=shift_aux.device, dtype=shift_aux.dtype)
        short_steps = max(int(config.short_horizon_steps), 1)
        short_weight = max(float(config.short_horizon_weight), 1.0)
        token_weights[waypoint_ids < short_steps] = short_weight
        per_sample_losses.append((per_token_loss * token_weights).sum() / token_weights.sum().clamp(min=1e-6))
        if sample_weights is None:
            per_sample_weights.append(torch.tensor(1.0, device=shift_aux.device))
        else:
            per_sample_weights.append(sample_weights[sample_index].to(shift_aux.device))

    if not per_sample_losses:
        return _zero(shift_aux.device)

    losses = torch.stack(per_sample_losses)
    weights = None if sample_weights is None else torch.stack(per_sample_weights)
    return _apply_sample_weights(losses, weights)


def decoded_traj_aux_anchor_losses(
    traj_aux_values: torch.Tensor | None,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor | None,
    ego_history_xyz: torch.Tensor | None,
    ego_history_mask: torch.Tensor | None,
    ego_future_xyz: torch.Tensor | None,
    ego_future_mask: torch.Tensor | None,
    config: TrajectoryDecodeConfig | None,
    aux_config: TrajectoryAuxInterfaceConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode auxiliary-head controls into xyz and compare with GT prefix/final anchors."""
    if (
        traj_aux_values is None
        or config is None
        or traj_token_mask is None
        or ego_history_xyz is None
        or ego_future_xyz is None
    ):
        device = labels.device if isinstance(labels, torch.Tensor) else torch.device("cpu")
        return _zero(device), _zero(device)

    predicted_controls = _gather_aux_predicted_controls(
        traj_aux_values,
        labels,
        traj_token_mask,
        config,
        aux_config=aux_config,
    )
    xyz_losses: list[torch.Tensor] = []
    final_losses: list[torch.Tensor] = []

    for sample_index, controls in enumerate(predicted_controls):
        if controls.numel() == 0:
            continue
        pred_xyz = _decode_expected_future_xyz(
            controls,
            ego_history_xyz[sample_index],
            None if ego_history_mask is None else ego_history_mask[sample_index],
            config,
        )
        gt_mask = None if ego_future_mask is None else ego_future_mask[sample_index]
        if gt_mask is None:
            gt_steps = ego_future_xyz[sample_index].shape[0]
        else:
            gt_steps = int(gt_mask.to(dtype=torch.long).sum().item())
        common_steps = min(pred_xyz.shape[0], gt_steps, config.n_waypoints)
        if common_steps <= 0:
            continue
        pred_xy = pred_xyz[:common_steps, :2]
        gt_xy = ego_future_xyz[sample_index, :common_steps, :2].to(device=pred_xy.device, dtype=pred_xy.dtype)

        prefix_steps = min(common_steps, max(int(config.short_horizon_steps), 1))
        if prefix_steps > 0:
            xyz_losses.append(
                F.smooth_l1_loss(
                    pred_xy[:prefix_steps],
                    gt_xy[:prefix_steps],
                    reduction="none",
                ).mean(dim=-1).mean()
            )
        final_losses.append(F.smooth_l1_loss(pred_xy[common_steps - 1], gt_xy[common_steps - 1], reduction="mean"))

    device = traj_aux_values.device
    xyz_loss = torch.stack(xyz_losses).mean() if xyz_losses else _zero(device)
    final_loss = torch.stack(final_losses).mean() if final_losses else _zero(device)
    return xyz_loss, final_loss


def _decode_expected_future_xyz(
    controls: torch.Tensor,
    history_xyz: torch.Tensor,
    history_mask: torch.Tensor | None,
    config: TrajectoryDecodeConfig,
) -> torch.Tensor:
    if controls.numel() == 0:
        return torch.empty((0, 3), device=controls.device, dtype=controls.dtype)

    if history_mask is None:
        valid_length = history_xyz.shape[0]
    else:
        valid_length = int(history_mask.to(dtype=torch.long).sum().item())
    valid_length = max(valid_length, 1)
    last_index = valid_length - 1
    prev_index = max(last_index - 1, 0)

    last_xy = history_xyz[last_index, :2]
    prev_xy = history_xyz[prev_index, :2]
    last_z = history_xyz[last_index, 2] if history_xyz.shape[-1] > 2 else torch.tensor(0.0, device=controls.device)
    v0 = torch.linalg.norm(last_xy - prev_xy) / max(config.dt, 1e-6)

    accel = controls[:, 0] * float(config.accel_std) + float(config.accel_mean)
    kappa = controls[:, 1] * float(config.curvature_std) + float(config.curvature_mean)
    dt = float(config.dt)

    velocity = torch.cat(
        [
            v0.view(1),
            v0.view(1) + torch.cumsum(accel * dt, dim=0),
        ],
        dim=0,
    )
    theta = torch.cat(
        [
            torch.zeros((1,), device=controls.device, dtype=controls.dtype),
            torch.cumsum(kappa * velocity[:-1] * dt + kappa * accel * (0.5 * dt * dt), dim=0),
        ],
        dim=0,
    )
    half_dt = 0.5 * dt
    x = torch.cumsum(velocity[:-1] * torch.cos(theta[:-1]) * half_dt + velocity[1:] * torch.cos(theta[1:]) * half_dt, dim=0)
    y = torch.cumsum(velocity[:-1] * torch.sin(theta[:-1]) * half_dt + velocity[1:] * torch.sin(theta[1:]) * half_dt, dim=0)
    z = torch.full_like(x, fill_value=last_z)
    return torch.stack([x, y, z], dim=-1)


def decoded_traj_geometry_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor | None,
    ego_history_xyz: torch.Tensor | None,
    ego_history_mask: torch.Tensor | None,
    ego_future_xyz: torch.Tensor | None,
    ego_future_mask: torch.Tensor | None,
    config: TrajectoryDecodeConfig | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode expected trajectory controls into xyz and compare with GT geometry."""
    if (
        config is None
        or traj_token_mask is None
        or ego_history_xyz is None
        or ego_future_xyz is None
    ):
        device = logits.device
        return _zero(device), _zero(device), _zero(device)

    predicted_controls, _ = _gather_expected_traj_controls(logits, labels, traj_token_mask, config)
    xyz_losses: list[torch.Tensor] = []
    delta_losses: list[torch.Tensor] = []
    final_losses: list[torch.Tensor] = []

    for sample_index, controls in enumerate(predicted_controls):
        if controls.numel() == 0:
            continue
        pred_xyz = _decode_expected_future_xyz(
            controls,
            ego_history_xyz[sample_index],
            None if ego_history_mask is None else ego_history_mask[sample_index],
            config,
        )
        gt_mask = None if ego_future_mask is None else ego_future_mask[sample_index]
        if gt_mask is None:
            gt_steps = ego_future_xyz[sample_index].shape[0]
        else:
            gt_steps = int(gt_mask.to(dtype=torch.long).sum().item())
        common_steps = min(pred_xyz.shape[0], gt_steps, config.n_waypoints)
        if common_steps <= 0:
            continue
        pred_xy = pred_xyz[:common_steps, :2]
        gt_xy = ego_future_xyz[sample_index, :common_steps, :2].to(device=pred_xy.device, dtype=pred_xy.dtype)

        short_steps = min(common_steps, max(int(config.short_horizon_steps), 1))
        short_weight = max(float(config.short_horizon_weight), 1.0)

        per_step_xy = F.smooth_l1_loss(pred_xy, gt_xy, reduction="none").mean(dim=-1)
        xy_weights = torch.ones((common_steps,), device=pred_xy.device, dtype=pred_xy.dtype)
        xy_weights[:short_steps] = short_weight
        xyz_losses.append((per_step_xy * xy_weights).sum() / xy_weights.sum().clamp(min=1e-6))

        if common_steps >= 2:
            pred_delta = pred_xy[1:] - pred_xy[:-1]
            gt_delta = gt_xy[1:] - gt_xy[:-1]
            per_step_delta = F.smooth_l1_loss(pred_delta, gt_delta, reduction="none").mean(dim=-1)
            delta_weights = torch.ones((common_steps - 1,), device=pred_xy.device, dtype=pred_xy.dtype)
            delta_weights[: max(short_steps - 1, 1)] = short_weight
            delta_losses.append((per_step_delta * delta_weights).sum() / delta_weights.sum().clamp(min=1e-6))
        final_losses.append(F.smooth_l1_loss(pred_xy[common_steps - 1], gt_xy[common_steps - 1], reduction="mean"))

    device = logits.device
    xyz_loss = torch.stack(xyz_losses).mean() if xyz_losses else _zero(device)
    delta_loss = torch.stack(delta_losses).mean() if delta_losses else _zero(device)
    final_loss = torch.stack(final_losses).mean() if final_losses else _zero(device)
    return xyz_loss, delta_loss, final_loss


def trajectory_control_regression_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor | None,
    sample_weights: torch.Tensor | None,
    config: TrajectoryDecodeConfig | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare LM expected traj controls directly against the target traj control sequence."""
    if config is None or traj_token_mask is None:
        device = labels.device if isinstance(labels, torch.Tensor) else torch.device("cpu")
        return _zero(device), _zero(device)

    predicted_controls, target_controls = _gather_expected_traj_controls(logits, labels, traj_token_mask, config)
    control_losses: list[torch.Tensor] = []
    delta_losses: list[torch.Tensor] = []
    sample_weight_values: list[torch.Tensor] = []

    for sample_index, (pred_controls, tgt_controls) in enumerate(zip(predicted_controls, target_controls)):
        common_steps = min(pred_controls.shape[0], tgt_controls.shape[0], config.n_waypoints)
        if common_steps <= 0:
            continue
        pred_controls = pred_controls[:common_steps]
        tgt_controls = tgt_controls[:common_steps]
        control_loss = F.smooth_l1_loss(pred_controls, tgt_controls, reduction="none").mean(dim=-1)
        step_weights = torch.ones((common_steps,), device=pred_controls.device, dtype=pred_controls.dtype)
        short_steps = min(common_steps, max(int(config.short_horizon_steps), 1))
        if short_steps > 0:
            step_weights[:short_steps] = max(float(config.short_horizon_weight), 1.0)
        control_losses.append((control_loss * step_weights).sum() / step_weights.sum().clamp(min=1e-6))

        if common_steps >= 2:
            pred_delta = pred_controls[1:] - pred_controls[:-1]
            tgt_delta = tgt_controls[1:] - tgt_controls[:-1]
            delta_loss = F.smooth_l1_loss(pred_delta, tgt_delta, reduction="none").mean(dim=-1)
            delta_step_weights = step_weights[1:]
            delta_losses.append(
                (delta_loss * delta_step_weights).sum() / delta_step_weights.sum().clamp(min=1e-6)
            )
        else:
            delta_losses.append(_zero(pred_controls.device))

        if sample_weights is None:
            sample_weight_values.append(torch.tensor(1.0, device=pred_controls.device))
        else:
            sample_weight_values.append(sample_weights[sample_index].to(pred_controls.device))

    if not control_losses:
        return _zero(logits.device), _zero(logits.device)

    weights = None if sample_weights is None else torch.stack(sample_weight_values)
    control = _apply_sample_weights(torch.stack(control_losses), weights)
    delta = _apply_sample_weights(torch.stack(delta_losses), weights)
    return control, delta
