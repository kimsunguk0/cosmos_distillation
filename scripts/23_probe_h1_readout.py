#!/usr/bin/env python3
"""Teacher-forced narrow H1 readout probe on top of a hidden-first checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import load_student_checkpoint, save_student_checkpoint
from src.model.student_wrapper import (
    StudentWrapperConfig,
    build_student_model,
    load_student_processor,
    load_student_tokenizer,
)
from src.training.collator import DistillationCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2_959.jsonl",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-checkpoint-dir",
        type=Path,
        default=None,
        help="Frozen baseline checkpoint used for baseline-preserving listwise losses.",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--max-train-samples", type=int, default=754)
    parser.add_argument("--max-val-samples", type=int, default=40)
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--baseline-cache-batch-size", type=int, default=4)
    parser.add_argument("--eval-every-steps", type=int, default=32)
    parser.add_argument("--row-learning-rate", type=float, default=3e-4)
    parser.add_argument("--norm-learning-rate", type=float, default=3e-6)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--hardneg-k", type=int, default=128)
    parser.add_argument(
        "--loss-mode",
        type=str,
        choices=("legacy_margin", "baseline_listwise"),
        default="legacy_margin",
    )
    parser.add_argument("--margin-floor-1499", type=float, default=-25.0)
    parser.add_argument("--margin-floor-hardneg", type=float, default=0.0)
    parser.add_argument("--loss-weight-1499", type=float, default=0.03)
    parser.add_argument("--loss-weight-hardneg", type=float, default=0.20)
    parser.add_argument("--traj-vocab-ce-weight", type=float, default=0.20)
    parser.add_argument("--loss-weight-baseline-improve", type=float, default=1.0)
    parser.add_argument("--loss-weight-dynamic-rank", type=float, default=0.50)
    parser.add_argument("--loss-weight-neg-kl", type=float, default=0.15)
    parser.add_argument("--loss-weight-balanced-traj-ce", type=float, default=0.0)
    parser.add_argument("--loss-weight-hidden-anchor", type=float, default=0.10)
    parser.add_argument("--balanced-ce-warmup-steps", type=int, default=50)
    parser.add_argument("--balanced-traj-ce-beta", type=float, default=0.5)
    parser.add_argument("--baseline-delta", type=float, default=0.0)
    parser.add_argument("--listwise-k", type=int, default=256)
    parser.add_argument("--dynamic-rank-k", type=int, default=256)
    parser.add_argument("--freq-neg-count", type=int, default=64)
    parser.add_argument("--trust-temperature", type=float, default=2.0)
    parser.add_argument("--row-delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--train-final-norm", action="store_true")
    parser.add_argument("--train-last-lora-layers", type=int, default=0)
    parser.add_argument("--disable-train-row-adapter", action="store_true")
    parser.add_argument(
        "--output-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "probe_h1_readout_from_h0d",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "probe_h1_readout_from_h0d.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _traj_vocab_range(tokenizer) -> tuple[int, int]:
    start_idx = int(getattr(tokenizer, "traj_token_start_idx", -1))
    end_idx = int(getattr(tokenizer, "traj_token_end_idx", -1))
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError("Tokenizer is missing valid trajectory token range.")
    return start_idx, end_idx - start_idx + 1


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = _move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


def _get_output_head(model):
    backbone = model.backbone
    if hasattr(backbone, "get_output_embeddings"):
        head = backbone.get_output_embeddings()
        if head is not None:
            return head
    head = getattr(backbone, "lm_head", None)
    if head is None:
        raise ValueError("Student backbone is missing output embedding head.")
    return head


def _get_hidden_backbone(model):
    backbone = model.backbone
    nested = getattr(backbone, "model", None)
    if nested is not None and hasattr(nested, "model"):
        return nested.model
    if nested is not None:
        return nested
    return backbone


def _enable_gradient_checkpointing(model) -> None:
    candidates = [model.backbone, _get_hidden_backbone(model), getattr(model.backbone, "model", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        fn = getattr(candidate, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                continue


def _manual_output_logits(
    hidden_states: torch.Tensor,
    model,
    *,
    center_traj_delta: bool = False,
    traj_start_id: int | None = None,
    traj_vocab_size: int | None = None,
) -> torch.Tensor:
    output_head = _get_output_head(model)
    token_adapter = getattr(output_head, "token_adapter", None)
    if token_adapter is None:
        return output_head(hidden_states)

    base_layer = token_adapter.base_layer
    base_logits = F.linear(hidden_states, base_layer.weight, base_layer.bias)
    active_adapters = list(getattr(output_head, "active_adapters", []) or [])
    if not active_adapters:
        return base_logits

    for adapter_name in active_adapters:
        delta = token_adapter.trainable_tokens_delta.get(adapter_name)
        if delta is None:
            continue
        token_indices = token_adapter.token_indices.get(adapter_name) or []
        if not token_indices:
            continue
        index_tensor = torch.as_tensor(token_indices, device=hidden_states.device, dtype=torch.long)
        delta_logits = F.linear(hidden_states, delta)
        if center_traj_delta and traj_start_id is not None and traj_vocab_size is not None:
            traj_end_id = int(traj_start_id) + int(traj_vocab_size)
            traj_mask = (index_tensor >= int(traj_start_id)) & (index_tensor < traj_end_id)
            if torch.any(traj_mask):
                traj_delta = delta_logits[:, traj_mask]
                delta_logits[:, traj_mask] = traj_delta - traj_delta.mean(dim=-1, keepdim=True)
        base_logits[:, index_tensor] = base_logits.index_select(1, index_tensor) + delta_logits
    return base_logits


def _row_delta_l2_penalty(model) -> torch.Tensor:
    output_head = _get_output_head(model)
    token_adapter = getattr(output_head, "token_adapter", None)
    if token_adapter is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    active_adapters = list(getattr(output_head, "active_adapters", []) or [])
    penalties = []
    for adapter_name in active_adapters:
        delta = token_adapter.trainable_tokens_delta.get(adapter_name)
        if delta is not None:
            penalties.append(delta.float().pow(2).mean())
    if not penalties:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(penalties).mean()


def _collect_prefix_shifted_logits(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor,
    *,
    prefix_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = traj_token_mask[:, 1:].to(dtype=torch.bool, device=hidden_states.device)

    selected_hidden: list[torch.Tensor] = []
    selected_labels: list[torch.Tensor] = []
    for row_index in range(shift_hidden.shape[0]):
        row_positions = torch.nonzero(
            shift_mask[row_index] & (shift_labels[row_index] != -100),
            as_tuple=False,
        ).flatten()[:prefix_tokens]
        if row_positions.numel() == 0:
            continue
        selected_hidden.append(shift_hidden[row_index, row_positions, :])
        selected_labels.append(shift_labels[row_index, row_positions])

    if not selected_hidden:
        return (
            torch.empty((0, int(hidden_states.shape[-1])), device=hidden_states.device, dtype=hidden_states.dtype),
            torch.empty((0,), device=hidden_states.device, dtype=torch.long),
        )

    return torch.cat(selected_hidden, dim=0), torch.cat(selected_labels, dim=0)


def _collect_prefix_shifted_grouped(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    traj_token_mask: torch.Tensor,
    *,
    prefix_tokens: int,
) -> list[dict[str, torch.Tensor]]:
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = traj_token_mask[:, 1:].to(dtype=torch.bool, device=hidden_states.device)

    grouped: list[dict[str, torch.Tensor]] = []
    for row_index in range(shift_hidden.shape[0]):
        row_positions = torch.nonzero(
            shift_mask[row_index] & (shift_labels[row_index] != -100),
            as_tuple=False,
        ).flatten()[:prefix_tokens]
        if row_positions.numel() == 0:
            grouped.append(
                {
                    "hidden": torch.empty(
                        (0, int(hidden_states.shape[-1])),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                    "labels": torch.empty((0,), device=hidden_states.device, dtype=torch.long),
                }
            )
            continue
        grouped.append(
            {
                "hidden": shift_hidden[row_index, row_positions, :],
                "labels": shift_labels[row_index, row_positions],
            }
        )
    return grouped


def _compute_traj_frequency_prior(
    records: list[dict[str, Any]],
    *,
    traj_vocab_size: int,
) -> torch.Tensor:
    counts = torch.ones((traj_vocab_size,), dtype=torch.float64)
    for row in records:
        token_ids = ((row.get("hard_target") or {}).get("traj_future_token_ids") or [])
        for token_id in token_ids:
            token_id = int(token_id)
            if 0 <= token_id < traj_vocab_size:
                counts[token_id] += 1.0
    probs = counts / counts.sum()
    return probs.to(dtype=torch.float32)


def _build_union_negatives(
    cur_logits: torch.Tensor,
    base_logits: torch.Tensor,
    gt_idx: torch.Tensor,
    *,
    freq_neg_idx: torch.Tensor,
    topk_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows, vocab_size = cur_logits.shape
    topk_k = max(1, min(int(topk_k), int(vocab_size) - 1))
    rows: list[torch.Tensor] = []
    max_len = 0
    for row_index in range(num_rows):
        gt_value = int(gt_idx[row_index].item())
        cur_masked = cur_logits[row_index].clone()
        cur_masked[gt_value] = float("-inf")
        base_masked = base_logits[row_index].clone()
        base_masked[gt_value] = float("-inf")
        cur_topk = torch.topk(cur_masked, k=topk_k, dim=-1).indices
        base_topk = torch.topk(base_masked, k=topk_k, dim=-1).indices
        union = torch.cat((cur_topk, base_topk, freq_neg_idx.to(device=cur_logits.device)), dim=0)
        union = union[union != gt_value].unique(sorted=False)
        rows.append(union)
        max_len = max(max_len, int(union.numel()))
    neg_idx = torch.zeros((num_rows, max_len), dtype=torch.long, device=cur_logits.device)
    neg_mask = torch.zeros((num_rows, max_len), dtype=torch.bool, device=cur_logits.device)
    for row_index, row_indices in enumerate(rows):
        if row_indices.numel() == 0:
            continue
        neg_idx[row_index, : row_indices.numel()] = row_indices
        neg_mask[row_index, : row_indices.numel()] = True
    return neg_idx, neg_mask


def _gather_masked_logits(
    logits: torch.Tensor,
    neg_idx: torch.Tensor,
    neg_mask: torch.Tensor,
) -> torch.Tensor:
    gathered = logits.gather(1, neg_idx)
    return gathered.masked_fill(~neg_mask, float("-inf"))


def _listwise_improve_loss(
    cur_logits: torch.Tensor,
    base_logits: torch.Tensor,
    gt_idx: torch.Tensor,
    *,
    freq_neg_idx: torch.Tensor,
    topk_k: int,
    delta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    neg_idx, neg_mask = _build_union_negatives(
        cur_logits,
        base_logits,
        gt_idx,
        freq_neg_idx=freq_neg_idx,
        topk_k=topk_k,
    )
    cur_neg = _gather_masked_logits(cur_logits, neg_idx, neg_mask)
    base_neg = _gather_masked_logits(base_logits, neg_idx, neg_mask)
    cur_gt = cur_logits.gather(-1, gt_idx.unsqueeze(-1)).squeeze(-1)
    base_gt = base_logits.gather(-1, gt_idx.unsqueeze(-1)).squeeze(-1)
    base_gap = base_gt - torch.logsumexp(base_neg, dim=-1)
    cur_gap = cur_gt - torch.logsumexp(cur_neg, dim=-1)
    loss = F.softplus((base_gap + float(delta)) - cur_gap).mean()
    return loss, {
        "base_gap_mean": float(base_gap.mean().detach().item()),
        "cur_gap_mean": float(cur_gap.mean().detach().item()),
        "neg_union_size_mean": float(neg_mask.float().sum(dim=-1).mean().detach().item()),
    }


def _dynamic_topk_rank_loss(
    cur_logits: torch.Tensor,
    gt_idx: torch.Tensor,
    *,
    topk_k: int,
    margin: float,
    tau: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    masked = cur_logits.clone()
    masked.scatter_(1, gt_idx.unsqueeze(-1), float("-inf"))
    topk_k = max(1, min(int(topk_k), int(masked.shape[-1]) - 1))
    neg_idx = torch.topk(masked, k=topk_k, dim=-1).indices
    cur_neg = cur_logits.gather(1, neg_idx)
    cur_gt = cur_logits.gather(-1, gt_idx.unsqueeze(-1))
    loss = torch.log1p(torch.exp((cur_neg - cur_gt + float(margin)) / float(tau)).sum(dim=-1)).mean()
    return loss, {
        "dynamic_hardneg_margin_mean": float((cur_gt - cur_neg).mean().detach().item()),
    }


def _negative_trust_kl(
    cur_logits: torch.Tensor,
    base_logits: torch.Tensor,
    gt_idx: torch.Tensor,
    *,
    freq_neg_idx: torch.Tensor,
    topk_k: int,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    neg_idx, neg_mask = _build_union_negatives(
        cur_logits,
        base_logits,
        gt_idx,
        freq_neg_idx=freq_neg_idx,
        topk_k=topk_k,
    )
    cur_neg = _gather_masked_logits(cur_logits, neg_idx, neg_mask)
    base_neg = _gather_masked_logits(base_logits, neg_idx, neg_mask)
    row_losses = []
    valid_rows = 0
    for row_index in range(cur_neg.shape[0]):
        valid_mask = neg_mask[row_index]
        if int(valid_mask.sum().item()) <= 1:
            continue
        cur_row = cur_neg[row_index, valid_mask] / float(temperature)
        base_row = base_neg[row_index, valid_mask] / float(temperature)
        log_p_cur = F.log_softmax(cur_row, dim=-1)
        p_base = F.softmax(base_row, dim=-1).detach()
        row_losses.append(F.kl_div(log_p_cur, p_base, reduction="batchmean"))
        valid_rows += 1
    if not row_losses:
        zero = torch.tensor(0.0, device=cur_logits.device)
        return zero, {"neg_trust_kl": 0.0, "neg_trust_valid_rows": 0.0}
    loss = torch.stack(row_losses).mean()
    return loss, {
        "neg_trust_kl": float(loss.detach().item()),
        "neg_trust_valid_rows": float(valid_rows),
    }


def _balanced_traj_ce(
    cur_logits: torch.Tensor,
    gt_idx: torch.Tensor,
    *,
    freq_prior: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    adjusted_logits = cur_logits - float(beta) * torch.log(freq_prior.to(device=cur_logits.device) + 1e-8)
    loss = F.cross_entropy(adjusted_logits, gt_idx)
    return loss, {"balanced_traj_ce": float(loss.detach().item())}


def _row_margin_1499_local(
    cur_logits: torch.Tensor,
    gt_idx: torch.Tensor,
    *,
    margin_floor_1499: float,
    local_1499_idx: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    token1499_logits = cur_logits[:, int(local_1499_idx)]
    gt_logits = cur_logits.gather(-1, gt_idx.unsqueeze(-1)).squeeze(-1)
    valid = gt_idx != int(local_1499_idx)
    if not torch.any(valid):
        zero = torch.tensor(0.0, device=cur_logits.device)
        return zero, {"target_vs_1499_margin_mean": 0.0}
    margin = gt_logits[valid] - token1499_logits[valid]
    loss = F.softplus(float(margin_floor_1499) - margin).mean()
    return loss, {"target_vs_1499_margin_mean": float(margin.mean().detach().item())}


def _centered_token_gram(hidden: torch.Tensor) -> torch.Tensor:
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    centered = F.normalize(centered, dim=-1)
    return centered @ centered.transpose(0, 1)


def _hidden_anchor_loss(
    current_grouped: list[dict[str, torch.Tensor]],
    sample_ids: list[str],
    baseline_prefix_cache: dict[str, dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    default_device = current_grouped[0]["hidden"].device if current_grouped else torch.device("cpu")
    for sample_id, item in zip(sample_ids, current_grouped):
        current_hidden = item["hidden"]
        if current_hidden.numel() == 0:
            continue
        cached = baseline_prefix_cache.get(str(sample_id))
        if cached is None:
            continue
        baseline_hidden = cached["prefix_hidden"].to(device=current_hidden.device, dtype=current_hidden.dtype)
        token_count = min(int(current_hidden.shape[0]), int(baseline_hidden.shape[0]))
        if token_count <= 1:
            continue
        cur_gram = _centered_token_gram(current_hidden[:token_count].float())
        base_gram = _centered_token_gram(baseline_hidden[:token_count].float())
        losses.append(F.mse_loss(cur_gram, base_gram))
    if not losses:
        zero = torch.tensor(0.0, device=default_device)
        return zero, {"hidden_anchor": 0.0}
    loss = torch.stack(losses).mean()
    return loss, {"hidden_anchor": float(loss.detach().item())}


@torch.inference_mode()
def _build_baseline_prefix_cache(
    *,
    records: list[dict[str, Any]],
    checkpoint_dir: Path,
    config: StudentWrapperConfig,
    tokenizer,
    processor,
    project_root: Path,
    device: torch.device,
    prefix_tokens: int,
    batch_size: int,
    traj_start_id: int,
    traj_vocab_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    collator = DistillationCollator(
        processor=processor,
        tokenizer=tokenizer,
        project_root=project_root,
        max_length=config.max_length,
        teacher_pair_target=False,
        enable_teacher_view=False,
        enable_action_aux=False,
        teacher_traj_cache_dir=Path("/data/teacher_cache/traj15"),
        teacher_traj_hidden_source="latent",
        teacher_traj_latent_suffix="lat32",
    )
    dataloader = _build_dataloader(records, collator, batch_size=batch_size, shuffle=False)
    baseline_model = build_student_model(config, tokenizer)
    load_student_checkpoint(checkpoint_dir, baseline_model, use_lora=True, adapter_trainable=False)
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    hidden_backbone = _get_hidden_backbone(baseline_model)
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = hidden_backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Baseline backbone did not return hidden states.")
        grouped = _collect_prefix_shifted_grouped(
            hidden_states[-1],
            batch["labels"],
            batch["traj_token_mask"],
            prefix_tokens=prefix_tokens,
        )
        for sample_id, item in zip(batch["sample_ids"], grouped):
            prefix_hidden = item["hidden"]
            prefix_labels = item["labels"]
            if prefix_labels.numel() == 0:
                continue
            prefix_logits = _manual_output_logits(prefix_hidden.float(), baseline_model).float()
            traj_logits = prefix_logits[:, traj_start_id : traj_start_id + traj_vocab_size]
            local_labels = (prefix_labels.long() - int(traj_start_id)).clamp(min=0, max=traj_vocab_size - 1)
            cache[str(sample_id)] = {
                "traj_logits": traj_logits.cpu().to(torch.float16),
                "prefix_hidden": prefix_hidden.float().cpu().to(torch.float16),
                "local_labels": local_labels.cpu().to(torch.int16),
            }
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def _margin_losses(
    full_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    traj_start_id: int,
    traj_vocab_size: int,
    traj_token_1499_id: int,
    margin_floor_1499: float,
    margin_floor_hardneg: float,
    hardneg_k: int,
    loss_weight_1499: float,
    loss_weight_hardneg: float,
    traj_vocab_ce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if full_logits.numel() == 0 or labels.numel() == 0:
        zero = torch.tensor(0.0, device=full_logits.device if full_logits.numel() else labels.device)
        return zero, {
            "loss_margin_1499": 0.0,
            "loss_hardneg": 0.0,
            "loss_traj_vocab_ce": 0.0,
            "target_vs_1499_margin_mean": 0.0,
            "target_vs_top1_margin_mean": 0.0,
            "hardneg_margin_mean": 0.0,
            "target_rank_mean": 0.0,
            "traj_vocab_target_rank_mean": 0.0,
            "target_logit_mean": 0.0,
            "token1499_logit_mean": 0.0,
            "top1_1499_ratio": 0.0,
            "traj_top1_1499_ratio": 0.0,
        }

    target_logits = full_logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    token1499_ids = torch.full_like(labels, int(traj_token_1499_id))
    token1499_logits = full_logits.gather(dim=-1, index=token1499_ids.unsqueeze(-1)).squeeze(-1)

    valid_1499 = labels != int(traj_token_1499_id)
    if torch.any(valid_1499):
        margin_1499 = target_logits[valid_1499] - token1499_logits[valid_1499]
        loss_margin_1499 = F.softplus(float(margin_floor_1499) - margin_1499).mean()
        margin_1499_mean = float(margin_1499.mean().detach().item())
    else:
        loss_margin_1499 = torch.tensor(0.0, device=full_logits.device)
        margin_1499_mean = 0.0

    traj_logits = full_logits[:, traj_start_id : traj_start_id + traj_vocab_size]
    traj_local_labels = (labels - int(traj_start_id)).clamp(min=0, max=traj_vocab_size - 1)
    traj_target_logits = traj_logits.gather(dim=-1, index=traj_local_labels.unsqueeze(-1)).squeeze(-1)

    masked_traj_logits = traj_logits.clone()
    masked_traj_logits.scatter_(1, traj_local_labels.unsqueeze(-1), float("-inf"))
    hardneg_k = max(1, min(int(hardneg_k), int(masked_traj_logits.shape[-1]) - 1))
    hardneg_logits, hardneg_indices = torch.topk(masked_traj_logits, k=hardneg_k, dim=-1)
    del hardneg_indices
    hardneg_margin = traj_target_logits.unsqueeze(-1) - hardneg_logits
    loss_hardneg = F.softplus(float(margin_floor_hardneg) - hardneg_margin).mean()

    if float(traj_vocab_ce_weight) > 0:
        loss_traj_vocab_ce = F.cross_entropy(traj_logits, traj_local_labels)
    else:
        loss_traj_vocab_ce = torch.tensor(0.0, device=full_logits.device)

    top1 = full_logits.argmax(dim=-1)
    traj_top1 = traj_logits.argmax(dim=-1)
    target_rank = (full_logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
    traj_target_rank = (traj_logits > traj_target_logits.unsqueeze(-1)).sum(dim=-1) + 1
    top1_logits = full_logits.gather(dim=-1, index=top1.unsqueeze(-1)).squeeze(-1)

    loss = (
        float(loss_weight_1499) * loss_margin_1499
        + float(loss_weight_hardneg) * loss_hardneg
        + float(traj_vocab_ce_weight) * loss_traj_vocab_ce
    )
    metrics = {
        "loss_margin_1499": float(loss_margin_1499.detach().item()),
        "loss_hardneg": float(loss_hardneg.detach().item()),
        "loss_traj_vocab_ce": float(loss_traj_vocab_ce.detach().item()),
        "target_vs_1499_margin_mean": margin_1499_mean,
        "target_vs_top1_margin_mean": float((target_logits - top1_logits).mean().detach().item()),
        "hardneg_margin_mean": float(hardneg_margin.mean().detach().item()),
        "target_rank_mean": float(target_rank.float().mean().detach().item()),
        "traj_vocab_target_rank_mean": float(traj_target_rank.float().mean().detach().item()),
        "target_logit_mean": float(target_logits.mean().detach().item()),
        "token1499_logit_mean": float(token1499_logits.mean().detach().item()),
        "top1_1499_ratio": float((top1 == int(traj_token_1499_id)).float().mean().detach().item()),
        "traj_top1_1499_ratio": float((traj_top1 == 1499).float().mean().detach().item()),
    }
    return loss, metrics


@torch.inference_mode()
def evaluate_probe(
    *,
    model,
    dataloader: DataLoader,
    device: torch.device,
    prefix_tokens: int,
    traj_start_id: int,
    traj_vocab_size: int,
    traj_token_1499_id: int,
    margin_floor_1499: float,
    margin_floor_hardneg: float,
    hardneg_k: int,
    loss_weight_1499: float,
    loss_weight_hardneg: float,
    traj_vocab_ce_weight: float,
) -> dict[str, float]:
    model.eval()
    hidden_backbone = _get_hidden_backbone(model)
    totals: dict[str, float] = {}
    count = 0
    top1_hist = Counter()
    traj_top1_hist = Counter()

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = hidden_backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Backbone did not return hidden states.")
        prefix_hidden, prefix_labels = _collect_prefix_shifted_logits(
            hidden_states[-1],
            batch["labels"],
            batch["traj_token_mask"],
            prefix_tokens=prefix_tokens,
        )
        if prefix_labels.numel() == 0:
            continue
        prefix_logits = _manual_output_logits(prefix_hidden.float(), model).float()
        _, metrics = _margin_losses(
            prefix_logits,
            prefix_labels.long(),
            traj_start_id=traj_start_id,
            traj_vocab_size=traj_vocab_size,
            traj_token_1499_id=traj_token_1499_id,
            margin_floor_1499=margin_floor_1499,
            margin_floor_hardneg=margin_floor_hardneg,
            hardneg_k=hardneg_k,
            loss_weight_1499=loss_weight_1499,
            loss_weight_hardneg=loss_weight_hardneg,
            traj_vocab_ce_weight=traj_vocab_ce_weight,
        )
        top1 = prefix_logits.argmax(dim=-1)
        traj_top1 = prefix_logits[:, traj_start_id : traj_start_id + traj_vocab_size].argmax(dim=-1)
        top1_hist.update(int(value) for value in top1.detach().cpu().tolist())
        traj_top1_hist.update(int(value) for value in traj_top1.detach().cpu().tolist())
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1

    if count == 0:
        return {"num_batches": 0}

    averaged = {key: value / count for key, value in totals.items()}
    averaged["num_batches"] = float(count)
    averaged["top1_histogram_top20"] = dict(top1_hist.most_common(20))
    averaged["traj_top1_histogram_top20"] = dict(traj_top1_hist.most_common(20))
    return averaged


def _build_dataloader(
    records: list[dict[str, Any]],
    collator: DistillationCollator,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        records,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=collator,
    )


def _infer_num_language_layers(model) -> int:
    candidate = getattr(getattr(model.backbone, "config", None), "num_hidden_layers", None)
    if candidate is None:
        candidate = getattr(
            getattr(getattr(model.backbone, "config", None), "text_config", None),
            "num_hidden_layers",
            None,
        )
    if candidate is not None:
        return int(candidate)

    max_index = -1
    for name, _ in model.named_parameters():
        marker = ".language_model.layers."
        if marker not in name:
            continue
        suffix = name.split(marker, 1)[1]
        layer_str = suffix.split(".", 1)[0]
        if layer_str.isdigit():
            max_index = max(max_index, int(layer_str))
    if max_index >= 0:
        return max_index + 1
    raise RuntimeError("Unable to infer language backbone layer count.")


def _unfreeze_h1_probe_params(
    model,
    *,
    train_final_norm: bool,
    train_last_lora_layers: int,
    train_row_adapter: bool,
) -> dict[str, int]:
    for param in model.parameters():
        param.requires_grad = False

    counts: dict[str, int] = {}
    total_layers = _infer_num_language_layers(model)
    lora_start = max(total_layers - int(train_last_lora_layers), 0)
    for name, parameter in model.named_parameters():
        if train_row_adapter and "embed_tokens.token_adapter.trainable_tokens_delta" in name:
            parameter.requires_grad = True
            counts["traj_token_rows"] = counts.get("traj_token_rows", 0) + int(parameter.numel())
        elif train_final_norm and name.endswith("language_model.norm.weight"):
            parameter.requires_grad = True
            counts["final_norm"] = counts.get("final_norm", 0) + int(parameter.numel())
        elif int(train_last_lora_layers) > 0 and "lora_" in name and ".language_model.layers." in name:
            suffix = name.split(".language_model.layers.", 1)[1]
            layer_str = suffix.split(".", 1)[0]
            if layer_str.isdigit() and int(layer_str) >= lora_start:
                parameter.requires_grad = True
                counts["last_lora_layers"] = counts.get("last_lora_layers", 0) + int(parameter.numel())
    return counts


def _optimizer_for_probe(
    model,
    *,
    row_lr: float,
    norm_lr: float,
    lora_lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    row_params = []
    norm_params = []
    lora_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "embed_tokens.token_adapter.trainable_tokens_delta" in name:
            row_params.append(parameter)
        elif "lora_" in name:
            lora_params.append(parameter)
        else:
            norm_params.append(parameter)

    param_groups = []
    summary: dict[str, int] = {}
    if row_params:
        param_groups.append({"params": row_params, "lr": float(row_lr), "weight_decay": float(weight_decay)})
        summary["traj_token_rows"] = sum(int(param.numel()) for param in row_params)
    if norm_params:
        param_groups.append({"params": norm_params, "lr": float(norm_lr), "weight_decay": 0.0})
        summary["final_norm"] = sum(int(param.numel()) for param in norm_params)
    if lora_params:
        param_groups.append({"params": lora_params, "lr": float(lora_lr), "weight_decay": float(weight_decay)})
        summary["last_lora_layers"] = sum(int(param.numel()) for param in lora_params)
    if not param_groups:
        raise RuntimeError("H1-probe has no trainable parameters.")
    return torch.optim.AdamW(param_groups), summary


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_records = load_jsonl(args.corpus_jsonl)
    train_records = [row for row in raw_records if str(row.get("split", "")).lower() == args.train_split.lower()]
    val_records = [row for row in raw_records if str(row.get("split", "")).lower() == args.val_split.lower()]
    if args.max_train_samples is not None:
        train_records = train_records[: int(args.max_train_samples)]
    if args.max_val_samples is not None:
        val_records = val_records[: int(args.max_val_samples)]
    if not train_records or not val_records:
        raise RuntimeError("H1-probe requires non-empty train and val splits.")

    config = StudentWrapperConfig(
        student_model_name="/workspace/base_models_weights/Cosmos-Reason2-2B",
        local_files_only=True,
    )
    tokenizer = load_student_tokenizer(config)
    processor = load_student_processor(config, tokenizer=tokenizer)
    traj_start_id, traj_vocab_size = _traj_vocab_range(tokenizer)
    traj_token_1499_id = traj_start_id + 1499

    collator = DistillationCollator(
        processor=processor,
        tokenizer=tokenizer,
        project_root=PROJECT_ROOT,
        max_length=config.max_length,
        teacher_pair_target=False,
        enable_teacher_view=False,
        enable_action_aux=False,
        teacher_traj_cache_dir=Path("/data/teacher_cache/traj15"),
        teacher_traj_hidden_source="latent",
        teacher_traj_latent_suffix="lat32",
    )

    train_loader = _build_dataloader(train_records, collator, batch_size=args.batch_size, shuffle=True)
    val_loader = _build_dataloader(val_records, collator, batch_size=args.batch_size, shuffle=False)

    freq_prior = _compute_traj_frequency_prior(train_records, traj_vocab_size=traj_vocab_size)
    freq_neg_idx = torch.topk(freq_prior, k=min(int(args.freq_neg_count), int(traj_vocab_size)), dim=-1).indices

    baseline_prefix_cache: dict[str, dict[str, torch.Tensor]] = {}
    if args.loss_mode == "baseline_listwise":
        baseline_checkpoint_dir = args.baseline_checkpoint_dir or args.checkpoint_dir
        baseline_prefix_cache = _build_baseline_prefix_cache(
            records=train_records + val_records,
            checkpoint_dir=baseline_checkpoint_dir,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            project_root=PROJECT_ROOT,
            device=device,
            prefix_tokens=int(args.prefix_tokens),
            batch_size=int(args.baseline_cache_batch_size),
            traj_start_id=traj_start_id,
            traj_vocab_size=traj_vocab_size,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model = build_student_model(config, tokenizer)
    load_student_checkpoint(args.checkpoint_dir, model, use_lora=True, adapter_trainable=True)
    model = model.to(device)
    trainable_counts = _unfreeze_h1_probe_params(
        model,
        train_final_norm=bool(args.train_final_norm),
        train_last_lora_layers=int(args.train_last_lora_layers),
        train_row_adapter=not bool(args.disable_train_row_adapter),
    )
    optimizer, optimizer_counts = _optimizer_for_probe(
        model,
        row_lr=float(args.row_learning_rate),
        norm_lr=float(args.norm_learning_rate),
        lora_lr=float(args.lora_learning_rate),
        weight_decay=float(args.weight_decay),
    )

    baseline_metrics = evaluate_probe(
        model=model,
        dataloader=val_loader,
        device=device,
        prefix_tokens=int(args.prefix_tokens),
        traj_start_id=traj_start_id,
        traj_vocab_size=traj_vocab_size,
        traj_token_1499_id=traj_token_1499_id,
        margin_floor_1499=float(args.margin_floor_1499),
        margin_floor_hardneg=float(args.margin_floor_hardneg),
        hardneg_k=int(args.hardneg_k),
        loss_weight_1499=float(args.loss_weight_1499),
        loss_weight_hardneg=float(args.loss_weight_hardneg),
        traj_vocab_ce_weight=float(args.traj_vocab_ce_weight),
    )

    history: list[dict[str, float | int]] = []
    best_val_traj_rank = float("inf")
    best_val_step = 0
    best_checkpoint_dir = args.output_checkpoint_dir / "best_val_traj_rank"
    train_iter = iter(train_loader)
    model.train()
    hidden_backbone = _get_hidden_backbone(model)
    track_backbone_grad = bool(args.train_final_norm) or int(args.train_last_lora_layers) > 0
    if track_backbone_grad:
        _enable_gradient_checkpointing(model)
    for step in range(1, int(args.max_steps) + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = _move_batch_to_device(batch, device)
        backbone_context = nullcontext() if track_backbone_grad else torch.no_grad()
        with backbone_context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = hidden_backbone(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Backbone did not return hidden states.")
        grouped = _collect_prefix_shifted_grouped(
            hidden_states[-1],
            batch["labels"],
            batch["traj_token_mask"],
            prefix_tokens=int(args.prefix_tokens),
        )
        prefix_hidden, prefix_labels = _collect_prefix_shifted_logits(
            hidden_states[-1],
            batch["labels"],
            batch["traj_token_mask"],
            prefix_tokens=int(args.prefix_tokens),
        )
        if prefix_labels.numel() == 0:
            continue
        if not track_backbone_grad:
            prefix_hidden = prefix_hidden.detach()
        prefix_logits = _manual_output_logits(
            prefix_hidden.float(),
            model,
            center_traj_delta=True,
            traj_start_id=traj_start_id,
            traj_vocab_size=traj_vocab_size,
        ).float()
        if args.loss_mode == "legacy_margin":
            loss, train_metrics = _margin_losses(
                prefix_logits,
                prefix_labels.long(),
                traj_start_id=traj_start_id,
                traj_vocab_size=traj_vocab_size,
                traj_token_1499_id=traj_token_1499_id,
                margin_floor_1499=float(args.margin_floor_1499),
                margin_floor_hardneg=float(args.margin_floor_hardneg),
                hardneg_k=int(args.hardneg_k),
                loss_weight_1499=float(args.loss_weight_1499),
                loss_weight_hardneg=float(args.loss_weight_hardneg),
                traj_vocab_ce_weight=float(args.traj_vocab_ce_weight),
            )
        else:
            traj_logits = prefix_logits[:, traj_start_id : traj_start_id + traj_vocab_size]
            local_labels = (prefix_labels.long() - int(traj_start_id)).clamp(min=0, max=traj_vocab_size - 1)
            base_traj_logits_list = []
            for sample_id, item in zip(batch["sample_ids"], grouped):
                if item["labels"].numel() == 0:
                    continue
                cached = baseline_prefix_cache[str(sample_id)]
                cached_logits = cached["traj_logits"].to(device=device, dtype=traj_logits.dtype)
                cached_labels = cached["local_labels"].to(device=device, dtype=torch.long)
                token_count = min(
                    int(item["labels"].numel()),
                    int(cached_logits.shape[0]),
                    int(cached_labels.shape[0]),
                )
                if token_count <= 0:
                    continue
                current_local = (item["labels"][:token_count].long() - int(traj_start_id)).clamp(
                    min=0,
                    max=traj_vocab_size - 1,
                )
                if not torch.equal(current_local, cached_labels[:token_count]):
                    raise ValueError(f"Baseline cache label mismatch for sample {sample_id}")
                base_traj_logits_list.append(cached_logits[:token_count])
            if not base_traj_logits_list:
                raise RuntimeError("Baseline listwise mode has no cached prefix logits for this batch.")
            base_traj_logits = torch.cat(base_traj_logits_list, dim=0)
            loss = torch.tensor(0.0, device=device)
            train_metrics = {}

            improve_loss, improve_metrics = _listwise_improve_loss(
                traj_logits,
                base_traj_logits,
                local_labels,
                freq_neg_idx=freq_neg_idx.to(device),
                topk_k=int(args.listwise_k),
                delta=float(args.baseline_delta),
            )
            loss = loss + float(args.loss_weight_baseline_improve) * improve_loss
            train_metrics["loss_baseline_improve_rank"] = float(improve_loss.detach().item())
            train_metrics.update(improve_metrics)

            dynamic_loss, dynamic_metrics = _dynamic_topk_rank_loss(
                traj_logits,
                local_labels,
                topk_k=int(args.dynamic_rank_k),
                margin=0.0,
                tau=1.0,
            )
            loss = loss + float(args.loss_weight_dynamic_rank) * dynamic_loss
            train_metrics["loss_dynamic_topk_rank"] = float(dynamic_loss.detach().item())
            train_metrics.update(dynamic_metrics)

            neg_kl_loss, neg_kl_metrics = _negative_trust_kl(
                traj_logits,
                base_traj_logits,
                local_labels,
                freq_neg_idx=freq_neg_idx.to(device),
                topk_k=int(args.listwise_k),
                temperature=float(args.trust_temperature),
            )
            loss = loss + float(args.loss_weight_neg_kl) * neg_kl_loss
            train_metrics["loss_neg_kl_trust"] = float(neg_kl_loss.detach().item())
            train_metrics.update(neg_kl_metrics)

            loss_1499, margin_1499_metrics = _row_margin_1499_local(
                traj_logits,
                local_labels,
                margin_floor_1499=float(args.margin_floor_1499),
                local_1499_idx=1499,
            )
            loss = loss + float(args.loss_weight_1499) * loss_1499
            train_metrics["loss_margin_1499"] = float(loss_1499.detach().item())
            train_metrics.update(margin_1499_metrics)

            balanced_weight = (
                float(args.loss_weight_balanced_traj_ce)
                if step > int(args.balanced_ce_warmup_steps)
                else 0.0
            )
            if balanced_weight > 0:
                balanced_ce_loss, balanced_ce_metrics = _balanced_traj_ce(
                    traj_logits,
                    local_labels,
                    freq_prior=freq_prior,
                    beta=float(args.balanced_traj_ce_beta),
                )
                loss = loss + balanced_weight * balanced_ce_loss
                train_metrics["loss_balanced_traj_ce"] = float(balanced_ce_loss.detach().item())
                train_metrics.update(balanced_ce_metrics)
            else:
                train_metrics["loss_balanced_traj_ce"] = 0.0

            hidden_anchor_loss, hidden_anchor_metrics = _hidden_anchor_loss(
                grouped,
                batch["sample_ids"],
                baseline_prefix_cache,
            )
            loss = loss + float(args.loss_weight_hidden_anchor) * hidden_anchor_loss
            train_metrics["loss_hidden_anchor"] = float(hidden_anchor_loss.detach().item())
            train_metrics.update(hidden_anchor_metrics)

            target_logits = prefix_logits.gather(dim=-1, index=prefix_labels.long().unsqueeze(-1)).squeeze(-1)
            token1499_ids = torch.full_like(prefix_labels.long(), int(traj_token_1499_id))
            token1499_logits = prefix_logits.gather(dim=-1, index=token1499_ids.unsqueeze(-1)).squeeze(-1)
            target_rank = (prefix_logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
            traj_target_logits = traj_logits.gather(dim=-1, index=local_labels.unsqueeze(-1)).squeeze(-1)
            traj_target_rank = (traj_logits > traj_target_logits.unsqueeze(-1)).sum(dim=-1) + 1
            top1 = prefix_logits.argmax(dim=-1)
            traj_top1 = traj_logits.argmax(dim=-1)
            top1_logits = prefix_logits.gather(dim=-1, index=top1.unsqueeze(-1)).squeeze(-1)
            masked_traj_logits = traj_logits.clone()
            masked_traj_logits.scatter_(1, local_labels.unsqueeze(-1), float("-inf"))
            hardneg_k = max(1, min(int(args.dynamic_rank_k), int(masked_traj_logits.shape[-1]) - 1))
            hardneg_logits = torch.topk(masked_traj_logits, k=hardneg_k, dim=-1).values
            train_metrics.update(
                {
                    "target_vs_top1_margin_mean": float((target_logits - top1_logits).mean().detach().item()),
                    "hardneg_margin_mean": float(
                        (traj_target_logits.unsqueeze(-1) - hardneg_logits).mean().detach().item()
                    ),
                    "target_rank_mean": float(target_rank.float().mean().detach().item()),
                    "traj_vocab_target_rank_mean": float(traj_target_rank.float().mean().detach().item()),
                    "target_logit_mean": float(target_logits.mean().detach().item()),
                    "token1499_logit_mean": float(token1499_logits.mean().detach().item()),
                    "top1_1499_ratio": float((top1 == int(traj_token_1499_id)).float().mean().detach().item()),
                    "traj_top1_1499_ratio": float((traj_top1 == 1499).float().mean().detach().item()),
                }
            )
        if float(args.row_delta_l2_weight) > 0:
            row_delta_penalty = _row_delta_l2_penalty(model)
            loss = loss + float(args.row_delta_l2_weight) * row_delta_penalty
            train_metrics["loss_row_delta_l2"] = float(row_delta_penalty.detach().item())
        else:
            train_metrics["loss_row_delta_l2"] = 0.0
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                float(args.grad_clip_norm),
            )
        optimizer.step()

        row: dict[str, float | int] = {
            "step": int(step),
            "train_total_loss": float(loss.detach().item()),
            **train_metrics,
        }
        if step == 1 or step % int(args.eval_every_steps) == 0 or step == int(args.max_steps):
            eval_metrics = evaluate_probe(
                model=model,
                dataloader=val_loader,
                device=device,
                prefix_tokens=int(args.prefix_tokens),
                traj_start_id=traj_start_id,
                traj_vocab_size=traj_vocab_size,
                traj_token_1499_id=traj_token_1499_id,
                margin_floor_1499=float(args.margin_floor_1499),
                margin_floor_hardneg=float(args.margin_floor_hardneg),
                hardneg_k=int(args.hardneg_k),
                loss_weight_1499=float(args.loss_weight_1499),
                loss_weight_hardneg=float(args.loss_weight_hardneg),
                traj_vocab_ce_weight=float(args.traj_vocab_ce_weight),
            )
            for key, value in eval_metrics.items():
                row[f"val_{key}"] = value
            val_traj_rank = float(eval_metrics.get("traj_vocab_target_rank_mean", float("inf")))
            if val_traj_rank < best_val_traj_rank:
                best_val_traj_rank = val_traj_rank
                best_val_step = int(step)
                best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_student_checkpoint(
                    best_checkpoint_dir,
                    model,
                    tokenizer,
                    processor,
                    use_lora=True,
                )
            model.train()
        history.append(row)

    final_metrics = evaluate_probe(
        model=model,
        dataloader=val_loader,
        device=device,
        prefix_tokens=int(args.prefix_tokens),
        traj_start_id=traj_start_id,
        traj_vocab_size=traj_vocab_size,
        traj_token_1499_id=traj_token_1499_id,
        margin_floor_1499=float(args.margin_floor_1499),
        margin_floor_hardneg=float(args.margin_floor_hardneg),
        hardneg_k=int(args.hardneg_k),
        loss_weight_1499=float(args.loss_weight_1499),
        loss_weight_hardneg=float(args.loss_weight_hardneg),
        traj_vocab_ce_weight=float(args.traj_vocab_ce_weight),
    )

    args.output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_student_checkpoint(
        args.output_checkpoint_dir,
        model,
        tokenizer,
        processor,
        use_lora=True,
    )

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "output_checkpoint_dir": str(args.output_checkpoint_dir),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "prefix_tokens": int(args.prefix_tokens),
        "max_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "baseline_cache_batch_size": int(args.baseline_cache_batch_size),
        "row_learning_rate": float(args.row_learning_rate),
        "norm_learning_rate": float(args.norm_learning_rate),
        "lora_learning_rate": float(args.lora_learning_rate),
        "margin_floor_1499": float(args.margin_floor_1499),
        "margin_floor_hardneg": float(args.margin_floor_hardneg),
        "hardneg_k": int(args.hardneg_k),
        "loss_weight_1499": float(args.loss_weight_1499),
        "loss_weight_hardneg": float(args.loss_weight_hardneg),
        "traj_vocab_ce_weight": float(args.traj_vocab_ce_weight),
        "row_delta_l2_weight": float(args.row_delta_l2_weight),
        "train_final_norm": bool(args.train_final_norm),
        "train_last_lora_layers": int(args.train_last_lora_layers),
        "traj_vocab_start_id": int(traj_start_id),
        "traj_vocab_size": int(traj_vocab_size),
        "traj_token_1499_id": int(traj_token_1499_id),
        "loss_mode": str(args.loss_mode),
        "baseline_checkpoint_dir": str(args.baseline_checkpoint_dir or args.checkpoint_dir),
        "loss_weight_baseline_improve": float(args.loss_weight_baseline_improve),
        "loss_weight_dynamic_rank": float(args.loss_weight_dynamic_rank),
        "loss_weight_neg_kl": float(args.loss_weight_neg_kl),
        "loss_weight_balanced_traj_ce": float(args.loss_weight_balanced_traj_ce),
        "loss_weight_hidden_anchor": float(args.loss_weight_hidden_anchor),
        "balanced_ce_warmup_steps": int(args.balanced_ce_warmup_steps),
        "balanced_traj_ce_beta": float(args.balanced_traj_ce_beta),
        "baseline_delta": float(args.baseline_delta),
        "listwise_k": int(args.listwise_k),
        "dynamic_rank_k": int(args.dynamic_rank_k),
        "freq_neg_count": int(args.freq_neg_count),
        "trust_temperature": float(args.trust_temperature),
        "trainable_counts": trainable_counts,
        "optimizer_counts": optimizer_counts,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "best_val_traj_rank": float(best_val_traj_rank),
        "best_val_step": int(best_val_step),
        "best_checkpoint_dir": str(best_checkpoint_dir),
        "history": history,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
