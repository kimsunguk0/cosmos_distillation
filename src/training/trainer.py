"""Trainer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.training.losses import (
    DistillationLossWeights,
    TrajectoryAuxInterfaceConfig,
    TrajectoryDecodeConfig,
    auxiliary_action_loss,
    boundary_action_xyz_loss,
    decoded_traj_geometry_losses,
    decoded_traj_aux_anchor_losses,
    export_metric_logs,
    feature_alignment_loss,
    masked_token_accuracy,
    teacher_logit_kd_loss,
    token_hidden_covariance_loss,
    token_hidden_positions_alignment_bridge_loss,
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
from src.training.flex_batch import compress_batch_for_flex


@dataclass(slots=True)
class TrainerConfig:
    stage_name: str
    epochs: float = 1.0
    max_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-5
    batch_size: int = 1
    max_steps: int | None = None


@dataclass(slots=True)
class ScheduledSamplingConfig:
    enabled: bool = False
    probability: float = 0.0
    probability_start: float | None = None
    ramp_steps: int = 0
    traj_token_start_id: int = 0
    replacement_vocab_size: int = 3000
    mode: str = "token"
    prefix_tokens: int = 0
    autoregressive_prefix: bool = False
    generated_prefix_topk_kd_scale: float = 1.0


class _FutureTokenRangeLogitsProcessor(LogitsProcessor):
    """Restrict generated scheduled-sampling prefix tokens to the traj codebook."""

    def __init__(self, vocab_start: int, vocab_size: int) -> None:
        self.vocab_start = int(vocab_start)
        self.vocab_end = int(vocab_start) + int(vocab_size)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.vocab_start <= 0 and self.vocab_end >= int(scores.shape[-1]):
            return scores
        masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked[:, self.vocab_start : self.vocab_end] = scores[:, self.vocab_start : self.vocab_end]
        return masked


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a nested batch dict onto the target device."""
    non_blocking = device.type == "cuda"
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
        elif isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        elif isinstance(value, list):
            converted = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    converted.append(item.to(device, non_blocking=non_blocking))
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


def _scheduled_sampling_probability(config: ScheduledSamplingConfig, global_step: int | None) -> float:
    target_probability = min(max(float(config.probability), 0.0), 1.0)
    ramp_steps = max(int(getattr(config, "ramp_steps", 0) or 0), 0)
    if ramp_steps <= 0:
        return target_probability
    start_probability = getattr(config, "probability_start", None)
    if start_probability is None:
        start_probability = 0.0
    start_probability = min(max(float(start_probability), 0.0), 1.0)
    step = max(int(global_step or 0), 0)
    ramp_ratio = min(float(step) / float(ramp_steps), 1.0)
    return start_probability + (target_probability - start_probability) * ramp_ratio


def _infer_pad_token_id(model, batch: dict[str, Any]) -> int:
    backbone = getattr(getattr(model, "module", model), "backbone", None)
    for owner in (backbone, getattr(backbone, "generation_config", None), getattr(backbone, "config", None)):
        pad_token_id = getattr(owner, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        pad_positions = attention_mask == 0
        if bool(torch.any(pad_positions).item()):
            return int(input_ids[pad_positions][0].item())
    return 0


def _model_logits_and_past(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, dict):
        logits = output["logits"]
        backbone_outputs = output.get("backbone_outputs")
        return logits, getattr(backbone_outputs, "past_key_values", None)
    return output.logits, getattr(output, "past_key_values", None)


def _past_seq_len(past_key_values: Any) -> int | None:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    try:
        return int(past_key_values[0][0].shape[-2])
    except Exception:  # noqa: BLE001
        return None


def _flex_enabled(model) -> bool:
    unwrapped_model = getattr(model, "module", model)
    return bool(hasattr(unwrapped_model, "flex_enabled") and unwrapped_model.flex_enabled())


def _slice_multimodal_rows(
    batch: dict[str, Any],
    row_indices: list[int],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        return pixel_values, image_grid_thw

    batch_size = int(batch["input_ids"].shape[0])
    grid_rows = int(image_grid_thw.shape[0])
    if batch_size <= 0 or grid_rows % batch_size != 0:
        return pixel_values, image_grid_thw

    grids_per_sample = grid_rows // batch_size
    grid_counts = image_grid_thw.to(dtype=torch.int64).prod(dim=1)
    grid_offsets = torch.zeros(grid_rows + 1, dtype=torch.int64, device=grid_counts.device)
    grid_offsets[1:] = torch.cumsum(grid_counts, dim=0)

    selected_grids: list[torch.Tensor] = []
    selected_pixels: list[torch.Tensor] = []
    for row_index in row_indices:
        grid_start = int(row_index) * grids_per_sample
        grid_end = grid_start + grids_per_sample
        pixel_start = int(grid_offsets[grid_start].item())
        pixel_end = int(grid_offsets[grid_end].item())
        selected_grids.append(image_grid_thw[grid_start:grid_end])
        selected_pixels.append(pixel_values[pixel_start:pixel_end])

    return torch.cat(selected_pixels, dim=0), torch.cat(selected_grids, dim=0)


def _slice_row_tensor(batch: dict[str, Any], key: str, row_indices: list[int]) -> torch.Tensor | None:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    batch_size = int(batch["input_ids"].shape[0])
    if int(value.shape[0]) != batch_size:
        return value
    indices = torch.tensor(row_indices, dtype=torch.long, device=value.device)
    return value.index_select(0, indices)


def _build_left_padded_prefix_batch(
    batch: dict[str, Any],
    active_traj_token_mask: torch.Tensor,
    row_indices: list[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].to(dtype=torch.bool)
    prefixes: list[torch.Tensor] = []
    traj_positions_by_row: list[torch.Tensor] = []
    max_prefix_len = 1
    for row_index in row_indices:
        traj_positions = torch.nonzero(active_traj_token_mask[row_index].to(dtype=torch.bool), as_tuple=False).flatten()
        traj_positions_by_row.append(traj_positions)
        first_traj_position = int(traj_positions[0].item()) if traj_positions.numel() > 0 else int(attention_mask[row_index].sum().item())
        prefix_ids = input_ids[row_index, :first_traj_position]
        prefix_mask = attention_mask[row_index, :first_traj_position]
        prefix_ids = prefix_ids[prefix_mask]
        if prefix_ids.numel() == 0:
            prefix_ids = input_ids[row_index, :1]
        prefixes.append(prefix_ids)
        max_prefix_len = max(max_prefix_len, int(prefix_ids.numel()))

    prefix_input_ids = torch.full(
        (len(prefixes), max_prefix_len),
        int(pad_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    prefix_attention_mask = torch.zeros(
        (len(prefixes), max_prefix_len),
        dtype=batch["attention_mask"].dtype,
        device=input_ids.device,
    )
    for row, prefix_ids in enumerate(prefixes):
        length = int(prefix_ids.numel())
        prefix_input_ids[row, -length:] = prefix_ids
        prefix_attention_mask[row, -length:] = 1
    return prefix_input_ids, prefix_attention_mask, traj_positions_by_row


def _generate_flex_autoregressive_prefix_tokens(
    model,
    prefix_batch: dict[str, torch.Tensor],
    *,
    prefix_tokens: int,
    logits_processor: LogitsProcessorList,
) -> torch.Tensor | None:
    generated = prefix_batch["input_ids"].clone()
    attention_mask = prefix_batch["attention_mask"].clone()
    prefill_keys = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "camera_indices",
        "relative_timestamps",
        "camera_counts",
        "frames_per_camera",
    )
    prefill_kwargs = {key: prefix_batch[key] for key in prefill_keys if key in prefix_batch}
    prefill_kwargs.update(
        {
            "return_hidden_states": False,
            "compute_meta_action": False,
            "compute_traj_aux": False,
            "use_cache": True,
        }
    )
    output = model(**prefill_kwargs)
    logits, past_key_values = _model_logits_and_past(output)

    for token_index in range(max(int(prefix_tokens), 0)):
        scores = logits[:, -1, :]
        scores = logits_processor(generated, scores)
        next_token = scores.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)],
            dim=1,
        )
        if token_index + 1 >= int(prefix_tokens):
            break
        decode_kwargs: dict[str, Any] = {
            "input_ids": next_token,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "return_hidden_states": False,
            "compute_meta_action": False,
            "compute_traj_aux": False,
            "use_cache": True,
        }
        past_seq_len = _past_seq_len(past_key_values)
        if past_seq_len is not None:
            decode_kwargs["cache_position"] = torch.arange(
                past_seq_len,
                past_seq_len + int(next_token.shape[1]),
                device=next_token.device,
                dtype=torch.long,
            )
        output = model(**decode_kwargs)
        logits, past_key_values = _model_logits_and_past(output)
    return generated[:, prefix_batch["input_ids"].shape[1] :]


def _generate_autoregressive_prefix_tokens(
    model,
    batch: dict[str, Any],
    active_traj_token_mask: torch.Tensor,
    row_indices: list[int],
    *,
    prefix_tokens: int,
    vocab_start: int,
    vocab_size: int,
) -> tuple[torch.Tensor | None, list[torch.Tensor], float]:
    unwrapped_model = getattr(model, "module", model)
    backbone = getattr(unwrapped_model, "backbone", None)
    if backbone is None or not hasattr(backbone, "generate"):
        return None, [], 0.0

    pad_token_id = _infer_pad_token_id(model, batch)
    prefix_input_ids, prefix_attention_mask, traj_positions_by_row = _build_left_padded_prefix_batch(
        batch,
        active_traj_token_mask,
        row_indices,
        pad_token_id,
    )
    selected_pixel_values, selected_image_grid_thw = _slice_multimodal_rows(batch, row_indices)
    logits_processor = LogitsProcessorList([
        _FutureTokenRangeLogitsProcessor(vocab_start, vocab_size),
    ])
    if _flex_enabled(model):
        prefix_batch: dict[str, torch.Tensor] = {
            "input_ids": prefix_input_ids,
            "attention_mask": prefix_attention_mask,
        }
        if selected_pixel_values is not None:
            prefix_batch["pixel_values"] = selected_pixel_values
        if selected_image_grid_thw is not None:
            prefix_batch["image_grid_thw"] = selected_image_grid_thw
        for key in ("camera_indices", "relative_timestamps", "camera_counts", "frames_per_camera"):
            selected = _slice_row_tensor(batch, key, row_indices)
            if selected is not None:
                prefix_batch[key] = selected
        generated_tokens = _generate_flex_autoregressive_prefix_tokens(
            model,
            prefix_batch,
            prefix_tokens=prefix_tokens,
            logits_processor=logits_processor,
        )
        if generated_tokens is None:
            return None, traj_positions_by_row, 0.0
        generated_tokens = generated_tokens[:, : int(prefix_tokens)]
        valid_mask = (generated_tokens >= vocab_start) & (generated_tokens < vocab_start + vocab_size)
        valid_rate = float(valid_mask.float().mean().detach().cpu()) if generated_tokens.numel() else 0.0
        return generated_tokens.to(dtype=batch["input_ids"].dtype), traj_positions_by_row, valid_rate

    generate_kwargs: dict[str, Any] = {
        "input_ids": prefix_input_ids,
        "attention_mask": prefix_attention_mask,
        "max_new_tokens": int(prefix_tokens),
        "min_new_tokens": int(prefix_tokens),
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": int(pad_token_id),
        "eos_token_id": None,
        "forced_eos_token_id": None,
        "return_dict_in_generate": False,
        "logits_processor": logits_processor,
    }
    if selected_pixel_values is not None:
        generate_kwargs["pixel_values"] = selected_pixel_values
    if selected_image_grid_thw is not None:
        generate_kwargs["image_grid_thw"] = selected_image_grid_thw

    generated = backbone.generate(**generate_kwargs)
    if not isinstance(generated, torch.Tensor):
        generated = getattr(generated, "sequences", None)
    if generated is None:
        return None, traj_positions_by_row, 0.0
    generated_tokens = generated[:, prefix_input_ids.shape[1] : prefix_input_ids.shape[1] + int(prefix_tokens)]
    valid_mask = (generated_tokens >= vocab_start) & (generated_tokens < vocab_start + vocab_size)
    valid_rate = float(valid_mask.float().mean().detach().cpu()) if generated_tokens.numel() else 0.0
    return generated_tokens.to(dtype=batch["input_ids"].dtype), traj_positions_by_row, valid_rate


def _apply_scheduled_sampling(
    model,
    batch: dict[str, Any],
    active_traj_token_mask: torch.Tensor,
    config: ScheduledSamplingConfig | None,
    global_step: int | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
    """Replace some trajectory body input tokens with current-model predictions."""
    if config is None or not config.enabled or float(config.probability) <= 0:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": 0.0,
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
        }, None

    probability = _scheduled_sampling_probability(config, global_step)
    vocab_start = int(config.traj_token_start_id)
    vocab_size = int(config.replacement_vocab_size)
    if vocab_start < 0 or vocab_size <= 0:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": 0.0,
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
        }, None

    mode = str(getattr(config, "mode", "token") or "token").strip().lower()
    prefix_tokens = max(int(getattr(config, "prefix_tokens", 0) or 0), 0)
    prefix_mode = mode in {"generated_prefix", "prefix", "sample_prefix"} and prefix_tokens > 0
    autoregressive_prefix = bool(getattr(config, "autoregressive_prefix", False)) and prefix_mode

    # Causal alignment: logits[:, j-1] predicts token position j. The legacy
    # one-forward replacement starts at token 1. True AR prefix generation
    # generates body token 0 from <traj_future_start>, so it may replace token 0.
    candidate_mask = active_traj_token_mask.to(dtype=torch.bool) if autoregressive_prefix else active_traj_token_mask[:, 1:].to(dtype=torch.bool)
    if prefix_mode:
        body_order = torch.cumsum(candidate_mask.to(dtype=torch.int64), dim=1) - 1
        candidate_mask = candidate_mask & (body_order < prefix_tokens)

    candidate_count = int(candidate_mask.sum().detach().cpu())
    if candidate_count <= 0:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": 0.0,
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
            "scheduled_sampling_probability": float(probability),
        }, None

    selected_rows: list[int] = []
    if autoregressive_prefix:
        row_draw = torch.rand((candidate_mask.shape[0],), device=candidate_mask.device) < probability
        selected_rows = torch.nonzero(row_draw, as_tuple=False).flatten().tolist()
        selected_rows = [
            int(row_index)
            for row_index in selected_rows
            if bool(candidate_mask[int(row_index)].any().item())
        ]
        if not selected_rows:
            return batch["input_ids"], {
                "scheduled_sampling_candidates": float(candidate_count),
                "scheduled_sampling_replaced": 0.0,
                "scheduled_sampling_rate": 0.0,
                "scheduled_sampling_probability": float(probability),
                "scheduled_sampling_prefix_tokens": float(prefix_tokens),
                "scheduled_sampling_autoregressive_prefix": 1.0,
            }, None

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            predicted_tokens = None
            traj_positions_by_selected_row: list[torch.Tensor] = []
            if autoregressive_prefix:
                predicted_tokens, traj_positions_by_selected_row, _ = _generate_autoregressive_prefix_tokens(
                    model,
                    batch,
                    active_traj_token_mask,
                    selected_rows,
                    prefix_tokens=prefix_tokens,
                    vocab_start=vocab_start,
                    vocab_size=vocab_size,
                )
            if predicted_tokens is None and not autoregressive_prefix:
                teacher_forced_outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    return_hidden_states=False,
                    compute_meta_action=False,
                    compute_traj_aux=False,
                )
                shifted_logits = teacher_forced_outputs["logits"][:, :-1, vocab_start : vocab_start + vocab_size]
                predicted_tokens = shifted_logits.argmax(dim=-1).to(dtype=batch["input_ids"].dtype) + vocab_start
                del teacher_forced_outputs
    finally:
        if was_training:
            model.train()

    if autoregressive_prefix and predicted_tokens is None:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": float(candidate_count),
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
            "scheduled_sampling_probability": float(probability),
            "scheduled_sampling_prefix_tokens": float(prefix_tokens),
            "scheduled_sampling_autoregressive_prefix": 1.0,
            "scheduled_sampling_ar_failed": 1.0,
        }, None

    sampled_input_ids = batch["input_ids"].clone()
    kd_sample_scale = torch.ones((batch["input_ids"].shape[0],), dtype=torch.float32, device=batch["input_ids"].device)
    scheduled_kd_scale = float(getattr(config, "generated_prefix_topk_kd_scale", 1.0))
    match_count = 0
    valid_count = 0
    even_match_count = 0
    even_count = 0
    odd_match_count = 0
    odd_count = 0
    if autoregressive_prefix and selected_rows and predicted_tokens is not None and traj_positions_by_selected_row:
        replaced_count = 0
        for generated_row, row_index in enumerate(selected_rows):
            traj_positions = traj_positions_by_selected_row[generated_row][:prefix_tokens]
            if traj_positions.numel() <= 0:
                continue
            token_count = min(int(traj_positions.numel()), int(predicted_tokens.shape[1]))
            positions = traj_positions[:token_count]
            generated = predicted_tokens[generated_row, :token_count]
            targets = batch["input_ids"][row_index, positions]
            sampled_input_ids[row_index, positions] = generated
            valid = (generated >= vocab_start) & (generated < vocab_start + vocab_size)
            matches = generated == targets
            body_order = torch.arange(token_count, device=generated.device)
            even_mask = (body_order % 2) == 0
            odd_mask = ~even_mask
            valid_count += int(valid.sum().detach().cpu())
            match_count += int(matches.sum().detach().cpu())
            even_count += int(even_mask.sum().detach().cpu())
            odd_count += int(odd_mask.sum().detach().cpu())
            even_match_count += int(matches[even_mask].sum().detach().cpu()) if bool(even_mask.any().item()) else 0
            odd_match_count += int(matches[odd_mask].sum().detach().cpu()) if bool(odd_mask.any().item()) else 0
            replaced_count += token_count
        if replaced_count > 0 and selected_rows:
            kd_sample_scale[torch.tensor(selected_rows, dtype=torch.long, device=kd_sample_scale.device)] = scheduled_kd_scale
    else:
        if prefix_mode:
            row_draw = torch.rand((candidate_mask.shape[0], 1), device=candidate_mask.device) < probability
            replacement_mask = candidate_mask & row_draw
        else:
            replacement_draw = torch.rand(candidate_mask.shape, device=candidate_mask.device) < probability
            replacement_mask = candidate_mask & replacement_draw
        replaced_count = int(replacement_mask.sum().detach().cpu())
        if replaced_count > 0:
            sampled_tail = sampled_input_ids[:, 1:]
            sampled_tail[replacement_mask] = predicted_tokens[replacement_mask]
            target_tail = batch["input_ids"][:, 1:]
            matched = predicted_tokens[replacement_mask] == target_tail[replacement_mask]
            valid = (predicted_tokens[replacement_mask] >= vocab_start) & (
                predicted_tokens[replacement_mask] < vocab_start + vocab_size
            )
            match_count = int(matched.sum().detach().cpu())
            valid_count = int(valid.sum().detach().cpu())
            sample_has_replacement = replacement_mask.any(dim=1)
            kd_sample_scale[sample_has_replacement] = scheduled_kd_scale
    if replaced_count <= 0:
        return batch["input_ids"], {
            "scheduled_sampling_candidates": float(candidate_count),
            "scheduled_sampling_replaced": 0.0,
            "scheduled_sampling_rate": 0.0,
            "scheduled_sampling_probability": float(probability),
            "scheduled_sampling_prefix_tokens": float(prefix_tokens),
            "scheduled_sampling_autoregressive_prefix": float(autoregressive_prefix),
        }, None

    return sampled_input_ids, {
        "scheduled_sampling_candidates": float(candidate_count),
        "scheduled_sampling_replaced": float(replaced_count),
        "scheduled_sampling_rate": float(replaced_count / max(candidate_count, 1)),
        "scheduled_sampling_probability": float(probability),
        "scheduled_sampling_prefix_tokens": float(prefix_tokens),
        "scheduled_sampling_autoregressive_prefix": float(autoregressive_prefix),
        "scheduled_sampling_generated_valid_rate": float(valid_count / max(replaced_count, 1)),
        "scheduled_sampling_generated_match_rate": float(match_count / max(replaced_count, 1)),
        "scheduled_sampling_even_match_rate": float(even_match_count / max(even_count, 1)) if even_count else 0.0,
        "scheduled_sampling_odd_match_rate": float(odd_match_count / max(odd_count, 1)) if odd_count else 0.0,
        "scheduled_sampling_kd_sample_scale_mean": float(kd_sample_scale.mean().detach().cpu()),
    }, kd_sample_scale


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
    global_step: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one train step and return total loss plus scalar logs."""
    unwrapped_model = getattr(model, "module", model)
    if hasattr(unwrapped_model, "flex_enabled") and unwrapped_model.flex_enabled():
        flex_cfg = getattr(unwrapped_model, "flex_scene_config", None)
        batch = compress_batch_for_flex(
            batch,
            image_token_id=int(getattr(unwrapped_model, "image_token_id")),
            tokens_per_image=int(getattr(flex_cfg, "tokens_per_image")),
            pad_token_id=int(getattr(unwrapped_model, "pad_token_id", 0) or 0),
        )
    flex_stats = dict(batch.get("flex_stats") or {})
    device = batch["input_ids"].device
    active_traj_token_mask = _restrict_traj_token_mask_to_prefix(
        batch["traj_token_mask"],
        traj_body_prefix_tokens,
    )
    input_ids, scheduled_sampling_logs, scheduled_kd_sample_scale = _apply_scheduled_sampling(
        model,
        batch,
        active_traj_token_mask,
        scheduled_sampling_config,
        global_step,
    )
    needs_hard_hidden = bool(
        weights.feat_align > 0
        or weights.teacher_traj_hidden_align > 0
        or weights.teacher_boundary_hidden_align > 0
        or weights.boundary_action_xyz > 0
    )
    needs_meta_action = bool(weights.action_aux > 0)
    needs_traj_aux = bool(
        weights.traj_aux_reg > 0
        or weights.traj_aux_xyz_reg > 0
        or weights.traj_aux_final_reg > 0
        or weights.traj_aux_guided_kd > 0
        or weights.traj_aux_pseudo_ce > 0
    )
    needs_boundary_action = bool(weights.boundary_action_xyz > 0)
    hard_outputs = model(
        input_ids=input_ids,
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
        boundary_action_positions=batch.get("teacher_text_boundary_hidden_positions"),
        return_hidden_states=needs_hard_hidden,
        compute_meta_action=needs_meta_action,
        compute_traj_aux=needs_traj_aux,
        compute_boundary_action=needs_boundary_action,
    )

    hard_cot_ce = _zero(device)
    if weights.hard_cot_ce > 0:
        hard_cot_ce, _ = weighted_causal_ce(
            hard_outputs["logits"],
            batch["labels"],
            batch["hard_cot_weights"],
            batch["cot_span_mask"],
        )
    hard_traj_ce = _zero(device)
    if weights.traj_ce > 0:
        hard_traj_ce, _ = weighted_causal_ce(
            hard_outputs["logits"],
            batch["labels"],
            batch["traj_weights"],
            active_traj_token_mask,
            batch.get("traj_token_label_weights"),
        )
    format_ce = _zero(device)
    if weights.format_ce > 0:
        format_ce, _ = weighted_causal_ce(
            hard_outputs["logits"],
            batch["labels"],
            batch["hard_cot_weights"],
            batch["format_token_mask"],
        )
    action_aux = _zero(device)
    if needs_meta_action:
        action_aux = auxiliary_action_loss(
            hard_outputs["meta_action_logits"],
            batch["action_class_labels"],
            batch["action_aux_weight"],
        )
    traj_aux_reg = _zero(device)
    traj_aux_xyz_reg = _zero(device)
    traj_aux_final_reg = _zero(device)
    traj_aux_guided_kd = _zero(device)
    traj_aux_pseudo_ce = _zero(device)
    if needs_traj_aux:
        if weights.traj_aux_reg > 0:
            traj_aux_reg = trajectory_aux_regression_loss(
                hard_outputs.get("traj_aux_values"),
                batch["labels"],
                active_traj_token_mask,
                batch["traj_weights"],
                traj_decode_config,
                aux_config=traj_aux_interface_config,
            )
        if weights.traj_aux_xyz_reg > 0 or weights.traj_aux_final_reg > 0:
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
        if weights.traj_aux_guided_kd > 0:
            traj_aux_guided_kd = trajectory_aux_guided_kd_loss(
                hard_outputs["logits"],
                hard_outputs.get("traj_aux_values"),
                batch["labels"],
                active_traj_token_mask,
                batch["traj_weights"],
                traj_decode_config,
                aux_config=traj_aux_interface_config,
            )
        if weights.traj_aux_pseudo_ce > 0:
            traj_aux_pseudo_ce = trajectory_aux_pseudo_ce_loss(
                hard_outputs["logits"],
                hard_outputs.get("traj_aux_values"),
                batch["labels"],
                active_traj_token_mask,
                batch["traj_weights"],
                traj_decode_config,
                aux_config=traj_aux_interface_config,
            )
    traj_xyz_reg = _zero(device)
    traj_delta_reg = _zero(device)
    traj_final_reg = _zero(device)
    if weights.traj_xyz_reg > 0 or weights.traj_delta_reg > 0 or weights.traj_final_reg > 0:
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
    traj_control_reg = _zero(device)
    traj_control_delta_reg = _zero(device)
    if weights.traj_control_reg > 0 or weights.traj_control_delta_reg > 0:
        traj_control_reg, traj_control_delta_reg = trajectory_control_regression_losses(
            hard_outputs["logits"],
            batch["labels"],
            active_traj_token_mask,
            batch["traj_weights"],
            traj_decode_config,
        )
    hard_token_acc = _zero(device)
    hard_cot_acc = _zero(device)
    hard_traj_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], active_traj_token_mask)
    if weights.hard_cot_ce > 0:
        hard_token_mask = batch["cot_span_mask"] | batch["traj_span_mask"]
        hard_token_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], hard_token_mask)
        hard_cot_acc = masked_token_accuracy(hard_outputs["logits"], batch["labels"], batch["cot_span_mask"])
    else:
        hard_token_acc = hard_traj_acc
    traj_aux_tensor = hard_outputs.get("traj_aux_values")
    traj_aux_abs_max = float(traj_aux_tensor.detach().abs().max().cpu()) if traj_aux_tensor is not None else 0.0

    teacher_seq_ce = _zero(device)
    teacher_logit_kd = _zero(device)
    teacher_traj_ce = _zero(device)
    teacher_traj_topk_kd = _zero(device)
    teacher_traj_hidden_align = _zero(device)
    teacher_boundary_hidden_align = _zero(device)
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
    boundary_action_xyz = _zero(device)

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
            teacher_topk_positions=batch.get("teacher_topk_positions"),
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
    teacher_traj_kd_sample_weights = teacher_traj_sample_weights
    if scheduled_kd_sample_scale is not None:
        kd_scale = scheduled_kd_sample_scale.to(device=device, dtype=torch.float32)
        if teacher_traj_kd_sample_weights is None:
            teacher_traj_kd_sample_weights = kd_scale
        else:
            teacher_traj_kd_sample_weights = teacher_traj_kd_sample_weights.to(device=device, dtype=torch.float32) * kd_scale
    teacher_traj_label_token_weights = batch.get("teacher_traj_label_token_weights")
    if teacher_traj_label_token_weights is None:
        teacher_traj_label_token_weights = batch.get("traj_token_label_weights")
    if (
        weights.teacher_traj_ce is not None
        and weights.teacher_traj_ce > 0
        and batch.get("teacher_traj_labels") is not None
    ):
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
            teacher_traj_kd_sample_weights,
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
    if weights.teacher_boundary_hidden_align > 0 and batch.get("teacher_text_boundary_hidden") is not None:
        hidden_bridge_cfg = dict(traj_hidden_bridge_config or {})
        boundary_sample_weights = batch.get("teacher_text_boundary_hidden_available")
        if boundary_sample_weights is not None:
            boundary_sample_weights = boundary_sample_weights.float()
        boundary_student_hidden = None
        boundary_teacher_hidden = None
        bridge_student_hidden = hard_outputs.get("traj_hidden_bridge_states")
        if bridge_student_hidden is not None and hasattr(unwrapped_model, "project_teacher_traj_hidden"):
            bridge_teacher_hidden = unwrapped_model.project_teacher_traj_hidden(
                batch.get("teacher_text_boundary_hidden")
            )
            if bridge_teacher_hidden is not None:
                boundary_student_hidden = bridge_student_hidden
                boundary_teacher_hidden = bridge_teacher_hidden

        if boundary_student_hidden is None:
            direct_student_hidden = hard_outputs.get("traj_hidden_states", hard_outputs["hidden_states"])
            direct_teacher_hidden = batch.get("teacher_text_boundary_hidden")
            if (
                direct_student_hidden is not None
                and direct_teacher_hidden is not None
                and int(direct_student_hidden.shape[-1]) == int(direct_teacher_hidden.shape[-1])
            ):
                boundary_student_hidden = direct_student_hidden
                boundary_teacher_hidden = direct_teacher_hidden

        if boundary_student_hidden is not None and boundary_teacher_hidden is not None:
            teacher_boundary_hidden_align = token_hidden_positions_alignment_bridge_loss(
                boundary_student_hidden,
                boundary_teacher_hidden,
                batch.get("teacher_text_boundary_hidden_positions"),
                batch.get("teacher_text_boundary_hidden_mask"),
                boundary_sample_weights,
                cosine_weight=float(hidden_bridge_cfg.get("boundary_cosine_weight", hidden_bridge_cfg.get("cosine_weight", 0.8))),
                mse_weight=float(hidden_bridge_cfg.get("boundary_mse_weight", hidden_bridge_cfg.get("mse_weight", 0.2))),
            )
    if weights.boundary_action_xyz > 0 and batch.get("teacher_action_traj_xyz") is not None:
        boundary_action_xyz = boundary_action_xyz_loss(
            hard_outputs.get("boundary_action_xyz"),
            batch.get("teacher_action_traj_xyz"),
            batch.get("teacher_action_traj_mask"),
            batch.get("teacher_action_traj_available"),
            short_horizon_steps=int((traj_hidden_bridge_config or {}).get("boundary_action_short_horizon_steps", 16)),
            short_horizon_weight=float((traj_hidden_bridge_config or {}).get("boundary_action_short_horizon_weight", 2.0)),
            final_weight=float((traj_hidden_bridge_config or {}).get("boundary_action_final_weight", 0.25)),
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
            return_hidden_states=weights.feat_align > 0,
            compute_meta_action=False,
            compute_traj_aux=False,
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
            teacher_topk_positions=teacher_view.get("teacher_topk_positions"),
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

    teacher_traj_topk_kd_scale = float(scheduled_sampling_logs.get("scheduled_sampling_kd_sample_scale_mean", 1.0))
    teacher_traj_topk_kd_effective = teacher_traj_topk_kd

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
        + weights.teacher_boundary_hidden_align * teacher_boundary_hidden_align
        + weights.traj_xyz_reg * traj_xyz_reg
        + weights.traj_delta_reg * traj_delta_reg
        + weights.traj_final_reg * traj_final_reg
        + weights.traj_control_reg * traj_control_reg
        + weights.traj_control_delta_reg * traj_control_delta_reg
        + weights.traj_aux_guided_kd * traj_aux_guided_kd
        + weights.traj_aux_pseudo_ce * traj_aux_pseudo_ce
        + weights.boundary_action_xyz * boundary_action_xyz
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
        "teacher_traj_topk_kd_effective": float(teacher_traj_topk_kd_effective.detach().cpu()),
        "teacher_traj_topk_kd_scale": float(teacher_traj_topk_kd_scale),
        "teacher_traj_hidden_align": float(teacher_traj_hidden_align.detach().cpu()),
        "teacher_boundary_hidden_align": float(teacher_boundary_hidden_align.detach().cpu()),
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
        "boundary_action_xyz": float(boundary_action_xyz.detach().cpu()),
        "hard_token_acc": float(hard_token_acc.detach().cpu()),
        "hard_cot_acc": float(hard_cot_acc.detach().cpu()),
        "hard_traj_acc": float(hard_traj_acc.detach().cpu()),
        "teacher_cot_acc": float(teacher_cot_acc.detach().cpu()),
        "traj_aux_abs_max": traj_aux_abs_max,
        "traj_body_prefix_tokens": float(traj_body_prefix_tokens or 0),
        **flex_stats,
        **scheduled_sampling_logs,
        "total_loss": float(total.detach().cpu()),
        }
    )
    return total, metrics
