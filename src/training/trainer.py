"""Trainer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.training.losses import (
    DistillationLossWeights,
    TrajectoryAuxInterfaceConfig,
    TrajectoryDecodeConfig,
    auxiliary_action_loss,
    decoded_traj_geometry_losses,
    decoded_traj_aux_anchor_losses,
    export_metric_logs,
    feature_alignment_loss,
    masked_token_accuracy,
    teacher_logit_kd_loss,
    token_hidden_covariance_loss,
    trajectory_aux_regression_loss,
    trajectory_aux_guided_kd_loss,
    trajectory_aux_pseudo_ce_loss,
    token_hidden_alignment_bridge_loss,
    token_hidden_alignment_loss,
    token_hidden_contrastive_loss,
    token_hidden_residual_diagonal_alignment_loss,
    token_hidden_relation_loss,
    token_hidden_soft_relation_kl_loss,
    token_hidden_spectrum_loss,
    token_hidden_temporal_delta_loss,
    token_hidden_variance_floor_loss,
    trajectory_control_regression_losses,
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


@dataclass(slots=True)
class ScheduledSamplingConfig:
    enabled: bool = False
    probability: float = 0.0
    traj_token_start_id: int = 0
    replacement_vocab_size: int = 3000


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


def _restrict_traj_token_mask_to_prefix(
    traj_token_mask: torch.Tensor,
    max_body_tokens: int | None,
) -> torch.Tensor:
    """Optionally keep only the first N trajectory body tokens active."""
    if max_body_tokens is None or int(max_body_tokens) <= 0:
        return traj_token_mask
    body_order = torch.cumsum(traj_token_mask.to(dtype=torch.int64), dim=1) - 1
    return traj_token_mask & (body_order < int(max_body_tokens))


def _apply_scheduled_sampling(
    model,
    batch: dict[str, Any],
    active_traj_token_mask: torch.Tensor,
    config: ScheduledSamplingConfig | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replace some trajectory body input tokens with current-model predictions."""
    if config is None or not config.enabled or float(config.probability) <= 0:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": 0.0,
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
        }

    probability = min(max(float(config.probability), 0.0), 1.0)
    vocab_start = int(config.traj_token_start_id)
    vocab_size = int(config.replacement_vocab_size)
    if vocab_start < 0 or vocab_size <= 0:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": 0.0,
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
        }

    # Causal alignment: logits[:, j-1] predicts token position j. We replace
    # input_ids[:, j] so later trajectory positions see model-sampled context.
    candidate_mask = active_traj_token_mask[:, 1:].to(dtype=torch.bool)
    candidate_count = int(candidate_mask.sum().detach().cpu())
    if candidate_count <= 0:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": 0.0,
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
        }

    with torch.no_grad():
        teacher_forced_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
        )
        shifted_logits = teacher_forced_outputs["logits"][:, :-1, vocab_start : vocab_start + vocab_size]
        predicted_tokens = shifted_logits.argmax(dim=-1).to(dtype=batch["input_ids"].dtype) + vocab_start
        del teacher_forced_outputs

    replacement_draw = torch.rand(candidate_mask.shape, device=candidate_mask.device) < probability
    replacement_mask = candidate_mask & replacement_draw
    replaced_count = int(replacement_mask.sum().detach().cpu())
    if replaced_count <= 0:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": float(candidate_count),
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
        }

    sampled_input_ids = batch["input_ids"].clone()
    sampled_tail = sampled_input_ids[:, 1:]
    sampled_tail[replacement_mask] = predicted_tokens[replacement_mask]
    return sampled_input_ids, {
        "scheduled_sampling_candidates": float(candidate_count),
        "scheduled_sampling_replaced": float(replaced_count),
        "scheduled_sampling_rate": float(replaced_count / max(candidate_count, 1)),
    }


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
    traj_aux_interface_config: TrajectoryAuxInterfaceConfig | None = None,
    traj_body_prefix_tokens: int | None = None,
    traj_hidden_bridge_config: dict[str, float] | None = None,
    scheduled_sampling_config: ScheduledSamplingConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one train step and return total loss plus scalar logs."""
    device = batch["input_ids"].device
    unwrapped_model = getattr(model, "module", model)
    active_traj_token_mask = _restrict_traj_token_mask_to_prefix(
        batch["traj_token_mask"],
        traj_body_prefix_tokens,
    )
    input_ids, scheduled_sampling_logs = _apply_scheduled_sampling(
        model,
        batch,
        active_traj_token_mask,
        scheduled_sampling_config,
    )
    hard_outputs = model(
        input_ids=input_ids,
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
        active_traj_token_mask,
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
    traj_aux_reg = trajectory_aux_regression_loss(
        hard_outputs.get("traj_aux_values"),
        batch["labels"],
        active_traj_token_mask,
        batch["traj_weights"],
        traj_decode_config,
        aux_config=traj_aux_interface_config,
    )
    traj_aux_xyz_reg, traj_aux_final_reg = decoded_traj_aux_anchor_losses(
        hard_outputs.get("traj_aux_values"),
        batch["labels"],
        active_traj_token_mask,
        batch.get("ego_history_xyz"),
        batch.get("ego_history_mask"),
        batch.get("ego_future_xyz"),
        batch.get("ego_future_mask"),
        traj_decode_config,
        aux_config=traj_aux_interface_config,
    )
    traj_xyz_reg, traj_delta_reg, traj_final_reg = decoded_traj_geometry_losses(
        hard_outputs["logits"],
        batch["labels"],
        active_traj_token_mask,
        batch.get("ego_history_xyz"),
        batch.get("ego_history_mask"),
        batch.get("ego_future_xyz"),
        batch.get("ego_future_mask"),
        traj_decode_config,
    )
    traj_control_reg, traj_control_delta_reg = trajectory_control_regression_losses(
        hard_outputs["logits"],
        batch["labels"],
        active_traj_token_mask,
        batch["traj_weights"],
        traj_decode_config,
    )
    traj_aux_guided_kd = trajectory_aux_guided_kd_loss(
        hard_outputs["logits"],
        hard_outputs.get("traj_aux_values"),
        batch["labels"],
        active_traj_token_mask,
        batch["traj_weights"],
        traj_decode_config,
        aux_config=traj_aux_interface_config,
    )
    traj_aux_pseudo_ce = trajectory_aux_pseudo_ce_loss(
        hard_outputs["logits"],
        hard_outputs.get("traj_aux_values"),
        batch["labels"],
        active_traj_token_mask,
        batch["traj_weights"],
        traj_decode_config,
        aux_config=traj_aux_interface_config,
    )
    hard_token_mask = batch["cot_span_mask"] | batch["traj_span_mask"]
    hard_token_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], hard_token_mask)
    hard_cot_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], batch["cot_span_mask"])
    hard_traj_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], active_traj_token_mask)
    traj_aux_tensor = hard_outputs.get("traj_aux_values")
    traj_aux_abs_max = float(traj_aux_tensor.detach().abs().max().cpu()) if traj_aux_tensor is not None else 0.0

    teacher_seq_ce = _zero(device)
    teacher_logit_kd = _zero(device)
    teacher_traj_ce = _zero(device)
    teacher_traj_topk_kd = _zero(device)
    teacher_traj_hidden_align = _zero(device)
    teacher_traj_hidden_relation = _zero(device)
    teacher_traj_hidden_variance = _zero(device)
    teacher_traj_hidden_covariance = _zero(device)
    teacher_traj_hidden_raw_relation = _zero(device)
    teacher_traj_hidden_raw_relation_centered = _zero(device)
    teacher_traj_hidden_raw_spectrum = _zero(device)
    teacher_traj_hidden_latent_spectrum = _zero(device)
    teacher_traj_hidden_temporal = _zero(device)
    teacher_traj_hidden_contrastive = _zero(device)
    teacher_traj_hidden_soft_relation = _zero(device)
    teacher_traj_hidden_residual_diag = _zero(device)
    feat_align = _zero(device)

    hard_teacher_pair_weights = None
    if batch.get("teacher_pair_weight") is not None:
        hard_teacher_pair_weights = batch["teacher_pair_weight"].float()
        if batch.get("teacher_pair_quality_multiplier") is not None:
            hard_teacher_pair_weights = (
                hard_teacher_pair_weights * batch["teacher_pair_quality_multiplier"].float()
            )
    if weights.teacher_logit_kd > 0 and batch.get("teacher_topk_indices") is not None:
        teacher_logit_kd = teacher_logit_kd + teacher_logit_kd_loss(
            hard_outputs["logits"],
            batch.get("cot_content_mask"),
            batch.get("teacher_topk_indices"),
            batch.get("teacher_topk_logprobs"),
            batch.get("teacher_topk_mask"),
            hard_teacher_pair_weights,
        )
    if weights.feat_align > 0 and batch.get("teacher_pooled_hidden") is not None:
        feat_weights = hard_teacher_pair_weights
        hidden_mask = batch.get("teacher_pooled_hidden_mask")
        if feat_weights is None and hidden_mask is not None:
            feat_weights = hidden_mask.float()
        elif feat_weights is not None and hidden_mask is not None:
            feat_weights = feat_weights * hidden_mask.float()
        feat_align = feat_align + feature_alignment_loss(
            hard_outputs["hidden_states"],
            batch.get("teacher_pooled_hidden"),
            batch["attention_mask"],
            feat_weights,
        )

    teacher_traj_sample_weights = None
    if batch.get("teacher_traj_available") is not None:
        teacher_traj_sample_weights = batch["teacher_traj_available"].float()
        if batch.get("teacher_traj_quality_multiplier") is not None:
            teacher_traj_sample_weights = (
                teacher_traj_sample_weights * batch["teacher_traj_quality_multiplier"].float()
            )
    teacher_traj_label_token_weights = batch.get("teacher_traj_label_token_weights")
    if teacher_traj_label_token_weights is None:
        teacher_traj_label_token_weights = batch.get("traj_token_label_weights")
    if weights.teacher_traj_ce is not None and batch.get("teacher_traj_labels") is not None:
        teacher_traj_ce, _ = weighted_causal_ce(
            hard_outputs["logits"],
            batch["teacher_traj_labels"],
            teacher_traj_sample_weights,
            active_traj_token_mask,
            teacher_traj_label_token_weights,
        )
    teacher_traj_topk_on_teacher_view = bool(batch.get("teacher_traj_topk_on_teacher_view", False))
    if weights.teacher_traj_topk_kd > 0 and not teacher_traj_topk_on_teacher_view:
        teacher_traj_topk_kd = teacher_logit_kd_loss(
            hard_outputs["logits"],
            active_traj_token_mask,
            batch.get("teacher_traj_topk_indices"),
            batch.get("teacher_traj_topk_logprobs"),
            batch.get("teacher_traj_topk_mask"),
            teacher_traj_sample_weights,
            token_weights=batch.get("teacher_traj_token_weights"),
        )
    if weights.teacher_traj_hidden_align > 0:
        hidden_bridge_cfg = dict(traj_hidden_bridge_config or {})
        student_hidden_for_distill = None
        teacher_hidden_for_distill = None

        bridge_student_hidden = hard_outputs.get("traj_hidden_bridge_states")
        if bridge_student_hidden is not None and hasattr(unwrapped_model, "project_teacher_traj_hidden"):
            bridge_teacher_hidden = unwrapped_model.project_teacher_traj_hidden(batch.get("teacher_traj_hidden"))
            if bridge_teacher_hidden is not None:
                student_hidden_for_distill = bridge_student_hidden
                teacher_hidden_for_distill = bridge_teacher_hidden

        if student_hidden_for_distill is None and batch.get("teacher_traj_hidden") is not None:
            direct_student_hidden = hard_outputs.get("traj_hidden_states", hard_outputs["hidden_states"])
            direct_teacher_hidden = batch.get("teacher_traj_hidden")
            if (
                direct_student_hidden is not None
                and direct_teacher_hidden is not None
                and int(direct_student_hidden.shape[-1]) == int(direct_teacher_hidden.shape[-1])
            ):
                student_hidden_for_distill = direct_student_hidden
                teacher_hidden_for_distill = direct_teacher_hidden

        if student_hidden_for_distill is not None and teacher_hidden_for_distill is not None:
            teacher_traj_hidden_align = token_hidden_alignment_bridge_loss(
                student_hidden_for_distill,
                teacher_hidden_for_distill,
                active_traj_token_mask,
                batch.get("teacher_traj_hidden_mask"),
                teacher_traj_sample_weights,
                cosine_weight=float(hidden_bridge_cfg.get("cosine_weight", 0.8)),
                mse_weight=float(hidden_bridge_cfg.get("mse_weight", 0.2)),
            )
            teacher_traj_hidden_relation = token_hidden_relation_loss(
                student_hidden_for_distill,
                teacher_hidden_for_distill,
                active_traj_token_mask,
                batch.get("teacher_traj_hidden_mask"),
                teacher_traj_sample_weights,
            )
            teacher_traj_hidden_variance = token_hidden_variance_floor_loss(
                student_hidden_for_distill,
                active_traj_token_mask,
                teacher_traj_sample_weights,
                target_std=float(hidden_bridge_cfg.get("variance_target", 0.5)),
            )
            teacher_traj_hidden_covariance = token_hidden_covariance_loss(
                student_hidden_for_distill,
                active_traj_token_mask,
                teacher_traj_sample_weights,
            )
            teacher_traj_hidden_latent_spectrum = token_hidden_spectrum_loss(
                student_hidden_for_distill,
                teacher_hidden_for_distill,
                active_traj_token_mask,
                batch.get("teacher_traj_hidden_mask"),
                teacher_traj_sample_weights,
            )
            teacher_traj_hidden_temporal = token_hidden_temporal_delta_loss(
                student_hidden_for_distill,
                teacher_hidden_for_distill,
                active_traj_token_mask,
                batch.get("teacher_traj_hidden_mask"),
                teacher_traj_sample_weights,
                second_order_weight=float(hidden_bridge_cfg.get("temporal_second_weight", 1.0)),
            )
            teacher_traj_hidden_contrastive = token_hidden_contrastive_loss(
                student_hidden_for_distill,
                teacher_hidden_for_distill,
                active_traj_token_mask,
                batch.get("teacher_traj_hidden_mask"),
                teacher_traj_sample_weights,
                temperature=float(hidden_bridge_cfg.get("contrastive_temperature", 0.07)),
            )
            teacher_traj_hidden_soft_relation = token_hidden_soft_relation_kl_loss(
                student_hidden_for_distill,
                teacher_hidden_for_distill,
                active_traj_token_mask,
                batch.get("teacher_traj_hidden_mask"),
                teacher_traj_sample_weights,
                student_temperature=float(hidden_bridge_cfg.get("soft_relation_student_temperature", 0.10)),
                teacher_temperature=float(hidden_bridge_cfg.get("soft_relation_teacher_temperature", 0.10)),
                diagonal_alpha=float(hidden_bridge_cfg.get("soft_relation_diagonal_alpha", 0.0)),
            )
            teacher_traj_hidden_residual_diag = token_hidden_residual_diagonal_alignment_loss(
                student_hidden_for_distill,
                teacher_hidden_for_distill,
                active_traj_token_mask,
                batch.get("teacher_traj_hidden_mask"),
                teacher_traj_sample_weights,
            )
            raw_teacher_hidden = batch.get("teacher_traj_hidden_raw")
            raw_teacher_hidden_mask = batch.get("teacher_traj_hidden_raw_mask")
            if raw_teacher_hidden is not None:
                raw_student_hidden = hard_outputs["hidden_states"]
                teacher_traj_hidden_raw_relation = token_hidden_relation_loss(
                    raw_student_hidden,
                    raw_teacher_hidden,
                    active_traj_token_mask,
                    raw_teacher_hidden_mask,
                    teacher_traj_sample_weights,
                    center=False,
                )
                teacher_traj_hidden_raw_relation_centered = token_hidden_relation_loss(
                    raw_student_hidden,
                    raw_teacher_hidden,
                    active_traj_token_mask,
                    raw_teacher_hidden_mask,
                    teacher_traj_sample_weights,
                    center=True,
                )
                teacher_traj_hidden_raw_spectrum = token_hidden_spectrum_loss(
                    raw_student_hidden,
                    raw_teacher_hidden,
                    active_traj_token_mask,
                    raw_teacher_hidden_mask,
                    teacher_traj_sample_weights,
                )
            teacher_traj_hidden_align = (
                teacher_traj_hidden_align
                + float(hidden_bridge_cfg.get("relation_weight", 0.0)) * teacher_traj_hidden_relation
                + float(hidden_bridge_cfg.get("variance_weight", 0.0)) * teacher_traj_hidden_variance
                + float(hidden_bridge_cfg.get("covariance_weight", 0.0)) * teacher_traj_hidden_covariance
                + float(hidden_bridge_cfg.get("latent_spectrum_weight", 0.0)) * teacher_traj_hidden_latent_spectrum
                + float(hidden_bridge_cfg.get("temporal_weight", 0.0)) * teacher_traj_hidden_temporal
                + float(hidden_bridge_cfg.get("soft_relation_weight", 0.0)) * teacher_traj_hidden_soft_relation
                + float(hidden_bridge_cfg.get("residual_diag_weight", 0.0)) * teacher_traj_hidden_residual_diag
                + float(hidden_bridge_cfg.get("contrastive_weight", 0.0)) * teacher_traj_hidden_contrastive
                + float(hidden_bridge_cfg.get("raw_relation_weight", 0.0)) * teacher_traj_hidden_raw_relation
                + float(hidden_bridge_cfg.get("raw_centered_relation_weight", 0.0))
                * teacher_traj_hidden_raw_relation_centered
                + float(hidden_bridge_cfg.get("raw_spectrum_weight", 0.0)) * teacher_traj_hidden_raw_spectrum
            )
        elif batch.get("teacher_traj_hidden") is not None:
            teacher_traj_hidden_align = token_hidden_alignment_loss(
                hard_outputs.get("traj_hidden_states", hard_outputs["hidden_states"]),
                batch.get("teacher_traj_hidden"),
                active_traj_token_mask,
                batch.get("teacher_traj_hidden_mask"),
                teacher_traj_sample_weights,
            )
    del hard_outputs

    teacher_view = batch.get("teacher_view")
    teacher_cot_acc = _zero(device)
    hard_teacher_traj_active = batch.get("teacher_traj_labels") is not None
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
        teacher_logit_kd = teacher_logit_kd + teacher_logit_kd_loss(
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
        if weights.teacher_traj_topk_kd > 0 and teacher_traj_topk_on_teacher_view:
            teacher_traj_topk_kd = teacher_traj_topk_kd + teacher_logit_kd_loss(
                teacher_outputs["logits"],
                teacher_view["traj_token_mask"],
                teacher_view.get("teacher_traj_topk_indices"),
                teacher_view.get("teacher_traj_topk_logprobs"),
                teacher_view.get("teacher_traj_topk_mask"),
                teacher_view["traj_weights"] * teacher_view["teacher_quality_multiplier"],
            )
        if weights.feat_align > 0 and teacher_view.get("teacher_pooled_hidden") is not None:
            feat_weights = seq_weights
            hidden_mask = teacher_view.get("teacher_pooled_hidden_mask")
            if hidden_mask is not None:
                feat_weights = feat_weights * hidden_mask.float()
            feat_align = feat_align + feature_alignment_loss(
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
        if hard_teacher_traj_active or _teacher_view_has_active_supervision(teacher_view, weights):
            traj_total = traj_total + weights.teacher_traj_ce * teacher_traj_ce

    total = (
        weights.hard_cot_ce * hard_cot_ce
        + weights.teacher_seq_ce * teacher_seq_ce
        + weights.teacher_logit_kd * teacher_logit_kd
        + traj_total
        + weights.traj_aux_reg * traj_aux_reg
        + weights.traj_aux_xyz_reg * traj_aux_xyz_reg
        + weights.traj_aux_final_reg * traj_aux_final_reg
        + weights.format_ce * format_ce
        + weights.action_aux * action_aux
        + weights.feat_align * feat_align
        + weights.teacher_traj_topk_kd * teacher_traj_topk_kd
        + weights.teacher_traj_hidden_align * teacher_traj_hidden_align
        + weights.traj_xyz_reg * traj_xyz_reg
        + weights.traj_delta_reg * traj_delta_reg
        + weights.traj_final_reg * traj_final_reg
        + weights.traj_control_reg * traj_control_reg
        + weights.traj_control_delta_reg * traj_control_delta_reg
        + weights.traj_aux_guided_kd * traj_aux_guided_kd
        + weights.traj_aux_pseudo_ce * traj_aux_pseudo_ce
    )

    metrics = export_metric_logs(
        {
        "hard_cot_ce": float(hard_cot_ce.detach().cpu()),
        "teacher_seq_ce": float(teacher_seq_ce.detach().cpu()),
        "teacher_logit_kd": float(teacher_logit_kd.detach().cpu()),
        "hard_traj_ce": float(hard_traj_ce.detach().cpu()),
        "traj_aux_reg": float(traj_aux_reg.detach().cpu()),
        "traj_aux_xyz_reg": float(traj_aux_xyz_reg.detach().cpu()),
        "traj_aux_final_reg": float(traj_aux_final_reg.detach().cpu()),
        "teacher_traj_ce": float(teacher_traj_ce.detach().cpu()),
        "teacher_traj_topk_kd": float(teacher_traj_topk_kd.detach().cpu()),
        "teacher_traj_hidden_align": float(teacher_traj_hidden_align.detach().cpu()),
        "teacher_traj_hidden_relation": float(teacher_traj_hidden_relation.detach().cpu()),
        "teacher_traj_hidden_variance": float(teacher_traj_hidden_variance.detach().cpu()),
        "teacher_traj_hidden_covariance": float(teacher_traj_hidden_covariance.detach().cpu()),
        "teacher_traj_hidden_raw_relation": float(teacher_traj_hidden_raw_relation.detach().cpu()),
        "teacher_traj_hidden_raw_relation_centered": float(teacher_traj_hidden_raw_relation_centered.detach().cpu()),
        "teacher_traj_hidden_raw_spectrum": float(teacher_traj_hidden_raw_spectrum.detach().cpu()),
        "teacher_traj_hidden_latent_spectrum": float(teacher_traj_hidden_latent_spectrum.detach().cpu()),
        "teacher_traj_hidden_temporal": float(teacher_traj_hidden_temporal.detach().cpu()),
        "teacher_traj_hidden_contrastive": float(teacher_traj_hidden_contrastive.detach().cpu()),
        "teacher_traj_hidden_soft_relation": float(teacher_traj_hidden_soft_relation.detach().cpu()),
        "teacher_traj_hidden_residual_diag": float(teacher_traj_hidden_residual_diag.detach().cpu()),
        "traj_ce": float(traj_ce.detach().cpu()),
        "format_ce": float(format_ce.detach().cpu()),
        "action_aux": float(action_aux.detach().cpu()),
        "feat_align": float(feat_align.detach().cpu()),
        "traj_xyz_reg": float(traj_xyz_reg.detach().cpu()),
        "traj_delta_reg": float(traj_delta_reg.detach().cpu()),
        "traj_final_reg": float(traj_final_reg.detach().cpu()),
        "traj_control_reg": float(traj_control_reg.detach().cpu()),
        "traj_control_delta_reg": float(traj_control_delta_reg.detach().cpu()),
        "traj_aux_guided_kd": float(traj_aux_guided_kd.detach().cpu()),
        "traj_aux_pseudo_ce": float(traj_aux_pseudo_ce.detach().cpu()),
        "hard_token_acc": float(hard_token_acc.detach().cpu()),
        "hard_cot_acc": float(hard_cot_acc.detach().cpu()),
        "hard_traj_acc": float(hard_traj_acc.detach().cpu()),
        "teacher_cot_acc": float(teacher_cot_acc.detach().cpu()),
        "traj_aux_abs_max": traj_aux_abs_max,
        "traj_body_prefix_tokens": float(traj_body_prefix_tokens or 0),
        **scheduled_sampling_logs,
        "total_loss": float(total.detach().cpu()),
        }
    )
    return total, metrics
