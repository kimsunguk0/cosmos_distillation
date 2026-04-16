"""Trainer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.training.losses import (
    DistillationLossWeights,
    TrajectoryDecodeConfig,
    auxiliary_action_loss,
    decoded_traj_geometry_losses,
    export_metric_logs,
    feature_alignment_loss,
    masked_token_accuracy,
    teacher_logit_kd_loss,
    token_hidden_alignment_loss,
    weighted_causal_ce,
)


@dataclass(slots=True)
class TrainerConfig:
    stage_name: str
    epochs: float = 1.0
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


def _zero(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def _teacher_view_has_active_supervision(
    teacher_view: dict[str, Any] | None,
    weights: DistillationLossWeights,
) -> bool:
    """Return whether the teacher branch should run for this batch."""
    if teacher_view is None:
        return False

    quality = teacher_view.get("teacher_quality_multiplier")
    if quality is None:
        quality = torch.ones_like(teacher_view["teacher_view_weight"], dtype=torch.float32)

    def _active(name: str) -> bool:
        tensor = teacher_view.get(name)
        if tensor is None:
            return False
        return bool(torch.any((tensor * quality) > 0).item())

    if weights.teacher_seq_ce > 0 and _active("teacher_view_weight"):
        return True
    if weights.teacher_logit_kd > 0 and _active("teacher_logit_kd_weight"):
        return True
    if weights.feat_align > 0 and _active("teacher_view_weight"):
        return True
    if weights.teacher_traj_ce is not None and weights.teacher_traj_ce > 0 and _active("traj_weights"):
        return True
    return False


def run_train_step(
    model,
    batch: dict[str, Any],
    weights: DistillationLossWeights,
    traj_decode_config: TrajectoryDecodeConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one train step and return total loss plus scalar logs."""
    device = batch["input_ids"].device
    hard_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
    )

    hard_cot_ce, _ = weighted_causal_ce(
        hard_outputs["logits"],
        batch["labels"],
        batch["hard_cot_weights"],
        batch["cot_span_mask"],
    )
    hard_traj_ce, _ = weighted_causal_ce(
        hard_outputs["logits"],
        batch["labels"],
        batch["traj_weights"],
        batch["traj_token_mask"],
        batch.get("traj_token_label_weights"),
    )
    format_ce, _ = weighted_causal_ce(
        hard_outputs["logits"],
        batch["labels"],
        batch["hard_cot_weights"],
        batch["format_token_mask"],
    )
    action_aux = auxiliary_action_loss(
        hard_outputs["meta_action_logits"],
        batch["action_class_labels"],
        batch["action_aux_weight"],
    )
    traj_xyz_reg, traj_delta_reg, traj_final_reg = decoded_traj_geometry_losses(
        hard_outputs["logits"],
        batch["labels"],
        batch["traj_token_mask"],
        batch.get("ego_history_xyz"),
        batch.get("ego_history_mask"),
        batch.get("ego_future_xyz"),
        batch.get("ego_future_mask"),
        traj_decode_config,
    )
    hard_token_mask = batch["cot_span_mask"] | batch["traj_span_mask"]
    hard_token_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], hard_token_mask)
    hard_cot_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], batch["cot_span_mask"])
    hard_traj_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], batch["traj_token_mask"])

    teacher_seq_ce = _zero(device)
    teacher_logit_kd = _zero(device)
    teacher_traj_ce = _zero(device)
    teacher_traj_topk_kd = _zero(device)
    teacher_traj_hidden_align = _zero(device)
    feat_align = _zero(device)

    teacher_traj_sample_weights = None
    if batch.get("teacher_traj_available") is not None:
        teacher_traj_sample_weights = batch["teacher_traj_available"].float()
        if batch.get("teacher_traj_quality_multiplier") is not None:
            teacher_traj_sample_weights = (
                teacher_traj_sample_weights * batch["teacher_traj_quality_multiplier"].float()
            )
    if weights.teacher_traj_ce is not None and batch.get("teacher_traj_labels") is not None:
        teacher_traj_ce, _ = weighted_causal_ce(
            hard_outputs["logits"],
            batch["teacher_traj_labels"],
            teacher_traj_sample_weights,
            batch["traj_token_mask"],
            batch.get("traj_token_label_weights"),
        )
    if weights.teacher_traj_topk_kd > 0:
        teacher_traj_topk_kd = teacher_logit_kd_loss(
            hard_outputs["logits"],
            batch.get("traj_token_mask"),
            batch.get("teacher_traj_topk_indices"),
            batch.get("teacher_traj_topk_logprobs"),
            batch.get("teacher_traj_topk_mask"),
            teacher_traj_sample_weights,
        )
    if weights.teacher_traj_hidden_align > 0:
        teacher_traj_hidden_align = token_hidden_alignment_loss(
            hard_outputs["hidden_states"],
            batch.get("teacher_traj_hidden"),
            batch.get("traj_token_mask"),
            teacher_traj_sample_weights,
        )
    del hard_outputs

    teacher_view = batch.get("teacher_view")
    teacher_cot_acc = _zero(device)
    if _teacher_view_has_active_supervision(teacher_view, weights):
        teacher_outputs = model(
            input_ids=teacher_view["input_ids"],
            attention_mask=teacher_view["attention_mask"],
            pixel_values=teacher_view.get("pixel_values"),
            image_grid_thw=teacher_view.get("image_grid_thw"),
        )
        seq_weights = teacher_view["teacher_view_weight"] * teacher_view["teacher_quality_multiplier"]
        teacher_seq_ce, _ = weighted_causal_ce(
            teacher_outputs["logits"],
            teacher_view["labels"],
            seq_weights,
            teacher_view["cot_span_mask"],
        )
        teacher_logit_weights = teacher_view["teacher_logit_kd_weight"] * teacher_view["teacher_quality_multiplier"]
        teacher_logit_kd = teacher_logit_kd_loss(
            teacher_outputs["logits"],
            teacher_view.get("cot_content_mask"),
            teacher_view.get("teacher_topk_indices"),
            teacher_view.get("teacher_topk_logprobs"),
            teacher_view.get("teacher_topk_mask"),
            teacher_logit_weights,
        )
        teacher_traj_ce, _ = weighted_causal_ce(
            teacher_outputs["logits"],
            teacher_view["labels"],
            teacher_view["traj_weights"],
            teacher_view["traj_token_mask"],
            teacher_view.get("traj_token_label_weights"),
        )
        if weights.feat_align > 0 and teacher_view.get("teacher_pooled_hidden") is not None:
            feat_weights = seq_weights
            hidden_mask = teacher_view.get("teacher_pooled_hidden_mask")
            if hidden_mask is not None:
                feat_weights = feat_weights * hidden_mask.float()
            feat_align = feature_alignment_loss(
                teacher_outputs["hidden_states"],
                teacher_view.get("teacher_pooled_hidden"),
                teacher_view["attention_mask"],
                feat_weights,
            )
        teacher_cot_acc = masked_token_accuracy(
            teacher_outputs["logits"],
            teacher_view["labels"],
            teacher_view["cot_span_mask"],
        )
        del teacher_outputs

    traj_ce = hard_traj_ce
    traj_total = weights.traj_ce * hard_traj_ce
    if weights.teacher_traj_ce is not None:
        traj_total = weights.traj_ce * hard_traj_ce
        if _teacher_view_has_active_supervision(teacher_view, weights):
            traj_total = traj_total + weights.teacher_traj_ce * teacher_traj_ce

    total = (
        weights.hard_cot_ce * hard_cot_ce
        + weights.teacher_seq_ce * teacher_seq_ce
        + weights.teacher_logit_kd * teacher_logit_kd
        + traj_total
        + weights.format_ce * format_ce
        + weights.action_aux * action_aux
        + weights.feat_align * feat_align
        + weights.teacher_traj_topk_kd * teacher_traj_topk_kd
        + weights.teacher_traj_hidden_align * teacher_traj_hidden_align
        + weights.traj_xyz_reg * traj_xyz_reg
        + weights.traj_delta_reg * traj_delta_reg
        + weights.traj_final_reg * traj_final_reg
    )

    metrics = export_metric_logs(
        {
        "hard_cot_ce": float(hard_cot_ce.detach().cpu()),
        "teacher_seq_ce": float(teacher_seq_ce.detach().cpu()),
        "teacher_logit_kd": float(teacher_logit_kd.detach().cpu()),
        "hard_traj_ce": float(hard_traj_ce.detach().cpu()),
        "teacher_traj_ce": float(teacher_traj_ce.detach().cpu()),
        "teacher_traj_topk_kd": float(teacher_traj_topk_kd.detach().cpu()),
        "teacher_traj_hidden_align": float(teacher_traj_hidden_align.detach().cpu()),
        "traj_ce": float(traj_ce.detach().cpu()),
        "format_ce": float(format_ce.detach().cpu()),
        "action_aux": float(action_aux.detach().cpu()),
        "feat_align": float(feat_align.detach().cpu()),
        "traj_xyz_reg": float(traj_xyz_reg.detach().cpu()),
        "traj_delta_reg": float(traj_delta_reg.detach().cpu()),
        "traj_final_reg": float(traj_final_reg.detach().cpu()),
        "hard_token_acc": float(hard_token_acc.detach().cpu()),
        "hard_cot_acc": float(hard_cot_acc.detach().cpu()),
        "hard_traj_acc": float(hard_traj_acc.detach().cpu()),
        "teacher_cot_acc": float(teacher_cot_acc.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        }
    )
    return total, metrics
