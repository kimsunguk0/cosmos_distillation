"""Batch surgery for FLEX visual-token compression experiments."""

from __future__ import annotations

from typing import Any

import torch

IGNORE_INDEX = -100


def _pad_value_for_key(key: str, tensor: torch.Tensor, pad_token_id: int) -> bool | int | float:
    if tensor.dtype == torch.bool:
        return False
    if key == "input_ids":
        return int(pad_token_id)
    if key in {"labels", "teacher_traj_labels"}:
        return IGNORE_INDEX
    if "mask" in key or "weight" in key:
        return 0.0 if torch.is_floating_point(tensor) else 0
    return 0.0 if torch.is_floating_point(tensor) else 0


def _kept_indices_for_row(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    image_token_id: int,
    tokens_per_image: int,
) -> torch.Tensor:
    active_len = int(attention_mask.to(dtype=torch.long).sum().item())
    keep = torch.ones(active_len, dtype=torch.bool, device=input_ids.device)
    cursor = 0
    while cursor < active_len:
        if int(input_ids[cursor].item()) != int(image_token_id):
            cursor += 1
            continue
        end = cursor + 1
        while end < active_len and int(input_ids[end].item()) == int(image_token_id):
            end += 1
        keep_start = cursor + min(max(int(tokens_per_image), 0), end - cursor)
        if keep_start < end:
            keep[keep_start:end] = False
        cursor = end
    return torch.nonzero(keep, as_tuple=False).flatten()


def _gather_and_pad_rows(
    tensor: torch.Tensor,
    keep_indices: list[torch.Tensor],
    *,
    key: str,
    max_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    pad_value = _pad_value_for_key(key, tensor, pad_token_id)
    out_shape = (int(tensor.shape[0]), max_len) + tuple(tensor.shape[2:])
    out = torch.full(out_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    for row_index, row_keep in enumerate(keep_indices):
        row_len = int(row_keep.numel())
        if row_len > 0:
            out[row_index, :row_len] = tensor[row_index].index_select(0, row_keep)
    return out


def _old_to_new_maps(
    batch_size: int,
    seq_len: int,
    keep_indices: list[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    maps = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)
    for row_index, row_keep in enumerate(keep_indices):
        maps[row_index, row_keep] = torch.arange(int(row_keep.numel()), dtype=torch.long, device=device)
    return maps


def _remap_positions(positions: torch.Tensor, old_to_new: torch.Tensor) -> torch.Tensor:
    remapped = torch.full_like(positions, -1)
    batch_size = int(positions.shape[0])
    seq_len = int(old_to_new.shape[1])
    for row_index in range(batch_size):
        row_positions = positions[row_index].long()
        valid = (row_positions >= 0) & (row_positions < seq_len)
        if bool(valid.any().item()):
            remapped[row_index, valid] = old_to_new[row_index].index_select(0, row_positions[valid])
    return remapped


def compress_batch_for_flex(
    batch: dict[str, Any],
    *,
    image_token_id: int,
    tokens_per_image: int,
    pad_token_id: int,
) -> dict[str, Any]:
    """Drop surplus image placeholders while keeping sequence-aligned tensors consistent."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    keep_indices = [
        _kept_indices_for_row(
            input_ids[row_index],
            attention_mask[row_index],
            image_token_id=image_token_id,
            tokens_per_image=tokens_per_image,
        )
        for row_index in range(batch_size)
    ]
    max_len = max(int(row_keep.numel()) for row_keep in keep_indices)
    old_to_new = _old_to_new_maps(batch_size, seq_len, keep_indices, device=input_ids.device)

    compressed: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and tuple(value.shape[:2]) == (batch_size, seq_len):
            compressed[key] = _gather_and_pad_rows(
                value,
                keep_indices,
                key=key,
                max_len=max_len,
                pad_token_id=pad_token_id,
            )
        elif key in {"teacher_topk_positions", "teacher_text_boundary_hidden_positions"} and isinstance(value, torch.Tensor):
            compressed[key] = _remap_positions(value, old_to_new)
        elif key == "teacher_view" and isinstance(value, dict):
            compressed[key] = compress_batch_for_flex(
                value,
                image_token_id=image_token_id,
                tokens_per_image=tokens_per_image,
                pad_token_id=pad_token_id,
            )
        else:
            compressed[key] = value

    original_image_tokens = int((input_ids == int(image_token_id)).sum().detach().cpu())
    compressed_image_tokens = int((compressed["input_ids"] == int(image_token_id)).sum().detach().cpu())
    compressed["flex_stats"] = {
        "flex_original_seq_len": float(seq_len),
        "flex_compressed_seq_len": float(max_len),
        "flex_original_image_tokens": float(original_image_tokens),
        "flex_compressed_image_tokens": float(compressed_image_tokens),
        "flex_image_token_compression": float(original_image_tokens / max(compressed_image_tokens, 1)),
    }
    return compressed
