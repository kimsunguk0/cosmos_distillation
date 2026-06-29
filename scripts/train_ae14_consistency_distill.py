#!/usr/bin/env python3
"""Train AE14 via consistency distillation from AE28 teacher.

Pipeline:
  1. Load AE28 teacher bundle, create AE14 student (14 layers from 28)
  2. Load pre-dumped AE28 teacher features from disk
  3. Online backbone forward for KV cache generation
  4. 2-step distillation with reflow + consistency + hidden losses

Usage:
    python scripts/train_ae14_consistency_distill.py \
        --student-checkpoint-dir outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final \
        --ae28-checkpoint outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt \
        --teacher-dump-dir outputs/ae28_teacher_dumps/flex_k512_fp16 \
        --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl \
        --output-dir outputs/action_expert/ae14_consistency_2step \
        --num-student-layers 14 \
        --num-inference-steps 2 \
        --batch-size 4 \
        --steps 15000 \
        --lr 2e-5
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
VIS_ROOT = SUKIM_ROOT / "visualization"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC, VIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_teacher_kv_28layer_expert_compression import (  # noqa: E402
    build_28layer_expert,
    force_attention,
    layer_mapping,
    torch_dtype_from_name,
)


def _import_ae_script():
    spec = importlib.util.spec_from_file_location(
        "ae_train",
        str(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# AE14 creation from AE28
# ---------------------------------------------------------------------------

def build_ae14_from_ae28(
    ae28_bundle: nn.Module,
    num_student_layers: int,
    ae_dtype: torch.dtype,
    device: str,
    attn_implementation: str,
) -> tuple[nn.Module, list[int]]:
    """Create AE14 student from AE28 teacher via layer selection.

    Copies action_in_proj and action_out_proj from AE28.
    Expert layers are selected uniformly from AE28's 28 layers.

    Returns (ae14_bundle, selected_layer_indices).
    """
    ae28_expert = ae28_bundle.expert
    ae28_layers = int(ae28_expert.config.num_hidden_layers)

    # Layer mapping: 28 -> 14 (uniform selection)
    selected = layer_mapping(ae28_layers, num_student_layers, "linspace_round")
    print(json.dumps({
        "event": "ae14_layer_mapping",
        "ae28_layers": ae28_layers,
        "ae14_layers": num_student_layers,
        "selected_indices": selected,
    }), flush=True)

    # Build student expert with fewer layers
    new_config = copy.deepcopy(ae28_expert.config)
    new_config.num_hidden_layers = num_student_layers
    if hasattr(new_config, "layer_types") and getattr(new_config, "layer_types") is not None:
        new_config.layer_types = list(getattr(new_config, "layer_types"))[:num_student_layers]
    if hasattr(new_config, "_attn_implementation"):
        new_config._attn_implementation = attn_implementation
    if hasattr(new_config, "attn_implementation"):
        new_config.attn_implementation = attn_implementation

    from transformers import AutoModel
    student_expert = AutoModel.from_config(new_config)
    if hasattr(student_expert, "embed_tokens"):
        del student_expert.embed_tokens

    # Copy selected layers from AE28
    with torch.no_grad():
        for new_idx, old_idx in enumerate(selected):
            student_expert.layers[new_idx].load_state_dict(
                ae28_expert.layers[old_idx].state_dict(), strict=True
            )
        student_expert.norm.load_state_dict(ae28_expert.norm.state_dict(), strict=True)

    student_expert = student_expert.to(device=device, dtype=ae_dtype)
    force_attention(student_expert, attn_implementation)

    # Copy action projections from AE28
    action_in_proj = copy.deepcopy(ae28_bundle.action_in_proj).to(device=device, dtype=ae_dtype)
    action_out_proj = copy.deepcopy(ae28_bundle.action_out_proj).to(device=device, dtype=ae_dtype)

    ae = _import_ae_script()
    bundle = ae.AE28Bundle(
        expert=student_expert,
        action_in_proj=action_in_proj,
        action_out_proj=action_out_proj,
    )
    bundle.train()

    total_params = sum(p.numel() for p in bundle.parameters())
    trainable_params = sum(p.numel() for p in bundle.parameters() if p.requires_grad)
    print(json.dumps({
        "event": "ae14_created",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "expert_layers": num_student_layers,
    }), flush=True)

    return bundle, selected


# ---------------------------------------------------------------------------
# KV cache layer selection for AE14 (14 from 28 backbone layers)
# ---------------------------------------------------------------------------

def select_kv_cache_layers(prompt_cache, selected_layers: list[int]):
    """Create a new DynamicCache with only the selected layers from the backbone cache.

    Uses DynamicCache.layers[i].keys/.values API (transformers >= 4.49).
    """
    from transformers.cache_utils import DynamicCache
    new_cache = DynamicCache()
    for new_idx, old_idx in enumerate(selected_layers):
        layer = prompt_cache.layers[old_idx]
        new_cache.update(layer.keys, layer.values, layer_idx=new_idx)
    return new_cache


# ---------------------------------------------------------------------------
# Collapse 10-step teacher features to N-step supervision
# ---------------------------------------------------------------------------

def collapse_to_n_steps(dump: dict[str, torch.Tensor], n_steps: int) -> dict[str, torch.Tensor]:
    """Collapse 10-step teacher features to n_steps supervision targets.

    For 2-step: take teacher states at t=0, t=0.5, t=1.0
    """
    x_all_steps = dump["x_all_steps"]    # [11, 64, 2]
    v_steps_10 = dump["v_steps"]          # [10, 64, 2]
    hidden_10 = dump["last_hidden_steps"] # [10, 64, 2048]
    time_steps_10 = dump["time_steps"]    # [11]

    if n_steps == 10:
        return dump

    # Indices into 10-step trajectory for n_steps
    # For 2-step: take steps 0, 5 (t=0.0, t=0.5) as input times
    # x_all_steps indices: 0, 5, 10 (t=0.0, t=0.5, t=1.0)
    step_indices = [round(i * 10 / n_steps) for i in range(n_steps + 1)]
    v_indices = [round(i * 10 / n_steps) for i in range(n_steps)]  # velocity sample indices

    new_time_steps = time_steps_10[step_indices]  # [n_steps+1]
    new_x_all_steps = x_all_steps[step_indices]   # [n_steps+1, 64, 2]

    # Compute effective velocity for each collapsed step
    new_v_steps = []
    new_hidden_steps = []
    for i in range(n_steps):
        dt = float(new_time_steps[i + 1] - new_time_steps[i])
        # Effective velocity: (x_{t+dt} - x_t) / dt
        v_eff = (new_x_all_steps[i + 1] - new_x_all_steps[i]) / max(dt, 1e-6)
        new_v_steps.append(v_eff)
        # Use teacher hidden at the corresponding 10-step index
        new_hidden_steps.append(hidden_10[v_indices[i]])

    new_v_steps = torch.stack(new_v_steps, dim=0)        # [n_steps, 64, 2]
    new_hidden_steps = torch.stack(new_hidden_steps, dim=0)  # [n_steps, 64, 2048]

    # Reflow targets for collapsed steps
    x_start = new_x_all_steps[0]
    x_target = dump["sampled_action"]
    straight_delta = x_target - x_start
    reflow_target_v = straight_delta.unsqueeze(0).expand(n_steps, -1, -1)
    teacher_residual = new_v_steps - reflow_target_v

    return {
        "x_all_steps": new_x_all_steps,        # [n_steps+1, 64, 2]
        "v_steps": new_v_steps,                 # [n_steps, 64, 2]
        "last_hidden_steps": new_hidden_steps,  # [n_steps, 64, 2048]
        "time_steps": new_time_steps,           # [n_steps+1]
        "sampled_action": dump["sampled_action"],
        "target_action": dump["target_action"],
        "reflow_target_v_steps": reflow_target_v,
        "teacher_residual_steps": teacher_residual,
    }


# ---------------------------------------------------------------------------
# Multi-loss computation
# ---------------------------------------------------------------------------

def compute_distillation_losses(
    *,
    pred_v_steps: torch.Tensor,        # [B, n_steps, 64, 2]
    pred_hidden_steps: torch.Tensor,    # [B, n_steps, 64, 2048]
    teacher_v_steps: torch.Tensor,      # [B, n_steps, 64, 2]
    teacher_hidden_steps: torch.Tensor, # [B, n_steps, 64, 2048]
    x_all_steps: torch.Tensor,          # [B, n_steps+1, 64, 2]
    time_steps: torch.Tensor,           # [B, n_steps+1] or [n_steps+1]
    sampled_action: torch.Tensor,       # [B, 64, 2]
    target_action: torch.Tensor,        # [B, 64, 2]
    reflow_target_v_steps: torch.Tensor,   # [B, n_steps, 64, 2]
    teacher_residual_steps: torch.Tensor,  # [B, n_steps, 64, 2]
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute multi-objective distillation losses."""
    logs: dict[str, float] = {}
    total_loss = torch.tensor(0.0, device=pred_v_steps.device)
    n_steps = pred_v_steps.shape[1]

    # Ensure time_steps has batch dim
    if time_steps.dim() == 1:
        time_steps = time_steps.unsqueeze(0).expand(pred_v_steps.shape[0], -1)

    # 1. Velocity loss: MSE(student_v, teacher_v)
    w = loss_weights.get("v_weight", 1.0)
    if w > 0:
        v_loss = F.mse_loss(pred_v_steps.float(), teacher_v_steps.float())
        total_loss = total_loss + w * v_loss
        logs["v_loss"] = float(v_loss)

    # 2. Hidden state loss: MSE(student_hidden, teacher_hidden)
    w = loss_weights.get("hidden_weight", 1.0)
    if w > 0:
        hidden_loss = F.mse_loss(pred_hidden_steps.float(), teacher_hidden_steps.float())
        total_loss = total_loss + w * hidden_loss
        logs["hidden_loss"] = float(hidden_loss)

    # 3. Rollout trajectory loss: student Euler rollout vs teacher trajectory
    w = loss_weights.get("x_step_weight", 0.5)
    if w > 0:
        x = x_all_steps[:, 0].float()  # [B, 64, 2]
        rollout = [x]
        for step_idx in range(n_steps):
            dt = (time_steps[:, step_idx + 1] - time_steps[:, step_idx]).view(-1, 1, 1)
            x = x + dt * pred_v_steps[:, step_idx].float()
            rollout.append(x)
        rollout = torch.stack(rollout, dim=1)  # [B, n_steps+1, 64, 2]
        x_step_loss = F.mse_loss(rollout[:, 1:], x_all_steps[:, 1:].float())
        total_loss = total_loss + w * x_step_loss
        logs["x_step_loss"] = float(x_step_loss)

    # 4. Final action loss: student's rolled-out final action vs teacher's
    w = loss_weights.get("action_weight", 0.25)
    if w > 0:
        # Use last position from rollout
        student_final = rollout[:, -1] if "rollout" in dir() else x_all_steps[:, 0]
        action_loss = F.mse_loss(student_final, sampled_action.float())
        total_loss = total_loss + w * action_loss
        logs["action_loss"] = float(action_loss)

    # 5. GT action loss: student's final vs ground truth
    w = loss_weights.get("gt_action_weight", 0.25)
    if w > 0 and target_action is not None:
        student_final = rollout[:, -1] if "rollout" in dir() else x_all_steps[:, 0]
        gt_action_loss = F.mse_loss(student_final, target_action.float())
        total_loss = total_loss + w * gt_action_loss
        logs["gt_action_loss"] = float(gt_action_loss)

    # 6. Reflow velocity loss: student_v vs straight-line velocity
    w = loss_weights.get("reflow_v_weight", 0.5)
    if w > 0:
        reflow_v_loss = F.mse_loss(pred_v_steps.float(), reflow_target_v_steps.float())
        total_loss = total_loss + w * reflow_v_loss
        logs["reflow_v_loss"] = float(reflow_v_loss)

    # 7. Consistency loss: student residual matches teacher residual
    w = loss_weights.get("consistency_weight", 0.5)
    if w > 0:
        pred_residual = pred_v_steps.float() - reflow_target_v_steps.float()
        consistency_loss = F.mse_loss(pred_residual, teacher_residual_steps.float())
        total_loss = total_loss + w * consistency_loss
        logs["consistency_loss"] = float(consistency_loss)

    logs["total_loss"] = float(total_loss)
    return total_loss, logs


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step_distill(
    *,
    student_bundle: nn.Module,
    teacher_model: Any,
    prompt_cache: Any,
    context: dict[str, Any],
    teacher_features: dict[str, torch.Tensor],
    kv_layer_indices: list[int],
    loss_weights: dict[str, float],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One training step: run student AE14 on collapsed teacher trajectory, compute multi-loss."""
    dtype = next(student_bundle.parameters()).dtype
    n_steps = teacher_features["v_steps"].shape[0]
    batch_size = teacher_features["v_steps"].shape[0] if teacher_features["v_steps"].dim() == 4 else 1

    # Add batch dim if single sample
    def _ensure_batch(t, ndim_expected):
        if t.dim() == ndim_expected - 1:
            return t.unsqueeze(0)
        return t

    x_all_steps = _ensure_batch(teacher_features["x_all_steps"], 4).to(device).float()
    teacher_v = _ensure_batch(teacher_features["v_steps"], 4).to(device).float()
    teacher_hidden = _ensure_batch(teacher_features["last_hidden_steps"], 4).to(device).float()
    time_steps = teacher_features["time_steps"].to(device).float()
    sampled_action = _ensure_batch(teacher_features["sampled_action"], 3).to(device).float()
    target_action = _ensure_batch(teacher_features["target_action"], 3).to(device).float()
    reflow_target_v = _ensure_batch(teacher_features["reflow_target_v_steps"], 4).to(device).float()
    teacher_residual = _ensure_batch(teacher_features["teacher_residual_steps"], 4).to(device).float()

    batch_size = x_all_steps.shape[0]
    n_steps = teacher_v.shape[1]
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = teacher_model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False

    # Select KV layers for AE14
    student_cache = select_kv_cache_layers(prompt_cache, kv_layer_indices)

    # Run student through each step
    pred_v_list = []
    pred_hidden_list = []
    for step_idx in range(n_steps):
        x_t = x_all_steps[:, step_idx]  # [B, 64, 2]
        t_val = time_steps[step_idx]
        t = t_val.view(1, 1, 1).expand(batch_size, 1, 1).to(device=device, dtype=dtype)

        future_token_embeds = student_bundle.action_in_proj(
            x_t.to(dtype=dtype), t.to(dtype=dtype)
        )
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(batch_size, n_diffusion_tokens, -1)

        expert_attention_mask = context.get("attention_mask")
        if expert_attention_mask is not None:
            expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)

        out = student_bundle.expert(
            inputs_embeds=future_token_embeds,
            position_ids=context["position_ids"],
            past_key_values=student_cache,
            attention_mask=expert_attention_mask,
            use_cache=True,
            **kwargs,
        )
        # Crop cache back to prefill
        student_cache.crop(prefill_seq_len)

        last_hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
        v = student_bundle.action_out_proj(last_hidden).view(-1, *action_dims)

        pred_v_list.append(v)
        pred_hidden_list.append(last_hidden)

    pred_v_steps = torch.stack(pred_v_list, dim=1)        # [B, n_steps, 64, 2]
    pred_hidden_steps = torch.stack(pred_hidden_list, dim=1)  # [B, n_steps, 64, 2048]

    loss, logs = compute_distillation_losses(
        pred_v_steps=pred_v_steps,
        pred_hidden_steps=pred_hidden_steps,
        teacher_v_steps=teacher_v,
        teacher_hidden_steps=teacher_hidden,
        x_all_steps=x_all_steps,
        time_steps=time_steps,
        sampled_action=sampled_action,
        target_action=target_action,
        reflow_target_v_steps=reflow_target_v,
        teacher_residual_steps=teacher_residual,
        loss_weights=loss_weights,
    )

    return loss, logs


# ---------------------------------------------------------------------------
# Eval: sample paths with student AE14
# ---------------------------------------------------------------------------

def eval_ae14(
    *,
    student_bundle: nn.Module,
    teacher_model: Any,
    prompt_cache: Any,
    context: dict[str, Any],
    batch: dict[str, Any],
    kv_layer_indices: list[int],
    device: torch.device,
    inference_steps: int = 2,
    temperature: float = 1.0,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate AE14 using full Euler sampling with N steps."""
    ae = _import_ae_script()
    dtype = next(student_bundle.parameters()).dtype
    batch_size = int(batch["ego_history_xyz"].shape[0])
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = teacher_model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False

    student_cache = select_kv_cache_layers(prompt_cache, kv_layer_indices)

    def step_fn(*, x, t):
        future_token_embeds = student_bundle.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_attention_mask = context.get("attention_mask")
        if expert_attention_mask is not None:
            expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
        out = student_bundle.expert(
            inputs_embeds=future_token_embeds,
            position_ids=context["position_ids"],
            past_key_values=student_cache,
            attention_mask=expert_attention_mask,
            use_cache=True,
            **kwargs,
        )
        student_cache.crop(prefill_seq_len)
        last_hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
        return student_bundle.action_out_proj(last_hidden).view(-1, *action_dims)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        action = teacher_model.diffusion.sample(
            batch_size=batch_size,
            step_fn=step_fn,
            device=device,
            inference_step=inference_steps,
            temperature=temperature,
        )
        pred_xyz, pred_rot = teacher_model.action_space.action_to_traj(
            action,
            batch["ego_history_xyz"].to(device),
            batch["ego_history_rot"].to(device),
        )
    target_xyz = batch["target_xyz"].to(device)
    # pred_xyz: [B, 1, 1, 64, 3] or [B, 1, 64, 3], target_xyz: [B, 1, 64, 3]
    # ade_fde expects single-sample numpy [64, 3] arrays
    from probe_teacher_kv_28layer_expert_compression import ade_fde
    ade_list, fde_list = [], []
    for i in range(batch_size):
        p = pred_xyz[i]
        while p.dim() > 2:
            p = p[0]
        t = target_xyz[i]
        while t.dim() > 2:
            t = t[0]
        a, f = ade_fde(p.detach().float().cpu().numpy(), t.detach().float().cpu().numpy())
        ade_list.append(float(a))
        fde_list.append(float(f))
    return {"ade_m": float(np.mean(ade_list)), "fde_m": float(np.mean(fde_list))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AE14 consistency distillation")
    parser.add_argument("--student-checkpoint-dir", type=str, required=True)
    parser.add_argument("--ae28-checkpoint", type=str, required=True)
    parser.add_argument("--teacher-dump-dir", type=str, required=True)
    parser.add_argument("--corpus-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-student-layers", type=int, default=14)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--eval-temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--ae-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    # Loss weights
    parser.add_argument("--v-weight", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--x-step-weight", type=float, default=0.5)
    parser.add_argument("--action-weight", type=float, default=0.25)
    parser.add_argument("--gt-action-weight", type=float, default=0.25)
    parser.add_argument("--reflow-v-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    # Backbone args (passed through to ae script)
    parser.add_argument("--num-samples", type=int, default=18000)
    parser.add_argument("--val-samples", type=int, default=1900)
    parser.add_argument("--prefix-mode", type=str, default="teacher_forced")
    parser.add_argument("--preserve-flex-positions", action="store_true", default=True)
    parser.add_argument("--flex-selection-strategy", type=str, default="uniform")
    parser.add_argument("--flex-scene-deepstack", action="store_true", default=True)
    parser.add_argument("--qat-quantization", type=str, default="")
    parser.add_argument("--qat-calib-samples", type=int, default=256)
    parser.add_argument("--target-source", type=str, default="teacher")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--split-scan-all", action="store_true", default=True)
    parser.add_argument("--stage2-attention-mode", type=str, default="official_none")
    parser.add_argument("--student-model", type=str, default="")
    parser.add_argument("--teacher-checkpoint-path", type=str,
                        default=str(PROJECT_ROOT.parent.parent / "base_weights" / "Alpamayo-1.5-10B"))
    parser.add_argument("--ae-init-mode", type=str, default="student_backbone_init")
    parser.add_argument("--disable-student-deepstack", action="store_true", default=False)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--split-cache-json", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument("--mapping", type=str, default="linspace_round")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pt to resume training from")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dump_dir = Path(args.teacher_dump_dir)

    args.student_checkpoint_dir = Path(args.student_checkpoint_dir)
    args.corpus_jsonl = Path(args.corpus_jsonl)
    args.teacher_checkpoint_path = Path(args.teacher_checkpoint_path)
    ae28_path = Path(args.ae28_checkpoint)

    loss_weights = {
        "v_weight": args.v_weight,
        "hidden_weight": args.hidden_weight,
        "x_step_weight": args.x_step_weight,
        "action_weight": args.action_weight,
        "gt_action_weight": args.gt_action_weight,
        "reflow_v_weight": args.reflow_v_weight,
        "consistency_weight": args.consistency_weight,
    }

    print(json.dumps({
        "event": "train_start",
        "output_dir": str(output_dir),
        "num_student_layers": args.num_student_layers,
        "num_inference_steps": args.num_inference_steps,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "loss_weights": loss_weights,
    }), flush=True)

    ae = _import_ae_script()

    # Load student backbone (for online KV generation)
    student, student_tokenizer, student_processor, base_model = ae.load_student(args)

    # Load teacher model (for action space, diffusion, targets)
    def _torch_dtype(name):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.bfloat16)

    args.teacher_load_device = "cpu"
    print(json.dumps({"event": "load_teacher_start"}), flush=True)
    _load_fn = getattr(ae, "load_model_and_processor", None)
    if _load_fn is None:
        from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor as _load_fn
    teacher_model, _, _, _, _ = _load_fn(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=_torch_dtype(args.student_dtype),
        device=args.teacher_load_device,
        config_json=None,
        runtime_support=None,
        attn_implementation=args.attn_implementation,
        min_pixels=163840,
        max_pixels=196608,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_model.to(device)
    print(json.dumps({"event": "load_teacher_done"}), flush=True)

    # Build AE28 bundle and load weights
    ae28_bundle, ae28_selected = ae.build_bundle(teacher_model, args, student=student)
    ae.load_bundle_checkpoint(ae28_path, bundle=ae28_bundle)
    ae28_bundle.eval()
    for p in ae28_bundle.parameters():
        p.requires_grad_(False)
    ae28_bundle.to(device)

    # Build AE14 from AE28
    ae_dtype = torch_dtype_from_name(args.ae_dtype)
    student_bundle, ae14_selected = build_ae14_from_ae28(
        ae28_bundle, args.num_student_layers, ae_dtype, args.device, args.attn_implementation,
    )

    # KV cache layer mapping: AE14's layers correspond to backbone layers via ae14_selected
    # But ae14_selected maps into AE28's 28 layers.
    # AE28 uses backbone's 28 layers 1:1, so ae14_selected directly gives backbone layer indices.
    kv_layer_indices = ae14_selected
    print(json.dumps({
        "event": "kv_layer_mapping",
        "ae14_to_backbone": kv_layer_indices,
    }), flush=True)

    # Free AE28 from GPU (no longer needed for training)
    ae28_bundle.cpu()
    del ae28_bundle
    gc.collect()
    torch.cuda.empty_cache()

    # Optimizer & scheduler
    optimizer = AdamW(student_bundle.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    # Load data items + teacher dump index
    train_items, val_items, _ = ae.select_train_val_items(args)
    # Build dump lookup
    dump_index: dict[str, Path] = {}
    for p in dump_dir.glob("*.pt"):
        dump_index[p.stem] = p
    # Filter items to those with dumps available
    available_train = [it for it in train_items
                       if it["sample_id"].replace("/", "_").replace("\\", "_") in dump_index]
    available_val = [it for it in val_items
                     if it["sample_id"].replace("/", "_").replace("\\", "_") in dump_index]

    print(json.dumps({
        "event": "data_ready",
        "total_dumps": len(dump_index),
        "train_with_dumps": len(available_train),
        "val_with_dumps": len(available_val),
    }), flush=True)

    if not available_train:
        print(json.dumps({"event": "error", "msg": "No training samples found in dump_dir"}), flush=True)
        return

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        payload = ae.load_bundle_checkpoint(resume_path, bundle=student_bundle)
        start_step = int(payload.get("step", 0))
        # Fast-forward scheduler to resume step
        for _ in range(start_step):
            scheduler.step()
        print(json.dumps({
            "event": "resumed",
            "checkpoint": str(resume_path),
            "start_step": start_step,
            "lr_after_resume": optimizer.param_groups[0]["lr"],
        }), flush=True)

    # Training loop
    random.seed(args.seed)
    log_path = output_dir / "train_log.jsonl"
    best_ade = float("inf")
    step = start_step
    started = time.time()

    while step < args.steps:
        random.shuffle(available_train)
        batches = list(ae.iter_batches(available_train, args.batch_size))

        for batch_items in batches:
            if step >= args.steps:
                break

            # Load teacher dumps for this batch
            teacher_batch_features = []
            valid_items = []
            for item in batch_items:
                safe_id = item["sample_id"].replace("/", "_").replace("\\", "_")
                dump_path = dump_index.get(safe_id)
                if dump_path is None:
                    continue
                dump = torch.load(dump_path, map_location="cpu", weights_only=False)
                collapsed = collapse_to_n_steps(dump, args.num_inference_steps)
                teacher_batch_features.append(collapsed)
                valid_items.append(item)

            if not teacher_batch_features:
                continue

            # Build batch: online backbone forward for KV cache
            try:
                batch = ae.build_batch(
                    args=args,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    batch_items=valid_items,
                )
            except Exception as e:
                print(json.dumps({"event": "batch_error", "step": step, "error": str(e)}), flush=True)
                continue

            # Stack teacher features into batch
            def _stack_features(features_list):
                keys = features_list[0].keys()
                stacked = {}
                for k in keys:
                    vals = [f[k] for f in features_list]
                    if isinstance(vals[0], torch.Tensor):
                        # time_steps are shared, don't stack
                        if k == "time_steps":
                            stacked[k] = vals[0]
                        else:
                            stacked[k] = torch.stack(vals, dim=0)
                    else:
                        stacked[k] = vals[0]
                return stacked

            teacher_features = _stack_features(teacher_batch_features)

            # Train step
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=ae_dtype, enabled=device.type == "cuda"):
                loss, logs = train_step_distill(
                    student_bundle=student_bundle,
                    teacher_model=teacher_model,
                    prompt_cache=batch["cache"],
                    context=batch["context"],
                    teacher_features=teacher_features,
                    kv_layer_indices=kv_layer_indices,
                    loss_weights=loss_weights,
                    device=device,
                )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student_bundle.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1

            # Log
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - started
                eta = (args.steps - step) * elapsed / max(step, 1)
                log_entry = {
                    "event": "train",
                    "step": step,
                    "total_steps": args.steps,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_min": round(elapsed / 60, 1),
                    "eta_min": round(eta / 60, 1),
                    **logs,
                }
                print(json.dumps(log_entry), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # Eval
            if step % args.eval_every == 0:
                student_bundle.eval()
                eval_results = []
                eval_items = available_val[:args.eval_samples]
                eval_batches = list(ae.iter_batches(eval_items, max(args.batch_size, 1)))
                for eb_items in eval_batches[:64]:
                    try:
                        eb = ae.build_batch(
                            args=args,
                            student=student,
                            student_processor=student_processor,
                            student_tokenizer=student_tokenizer,
                            teacher_model=teacher_model,
                            batch_items=eb_items,
                        )
                        res = eval_ae14(
                            student_bundle=student_bundle,
                            teacher_model=teacher_model,
                            prompt_cache=eb["cache"],
                            context=eb["context"],
                            batch=eb,
                            kv_layer_indices=kv_layer_indices,
                            device=device,
                            inference_steps=args.num_inference_steps,
                            temperature=args.eval_temperature,
                            seed=args.seed + step,
                        )
                        eval_results.append(res)
                    except Exception as _eval_err:
                        if not eval_results:  # print first error for debugging
                            import traceback
                            print(json.dumps({"event": "eval_error", "error": str(_eval_err), "trace": traceback.format_exc()[-500:]}), flush=True)
                        continue
                    finally:
                        del eb
                if eval_results:
                    avg_ade = np.mean([r["ade_m"] for r in eval_results])
                    avg_fde = np.mean([r["fde_m"] for r in eval_results])
                    eval_log = {
                        "event": "eval",
                        "step": step,
                        "ade_m": round(float(avg_ade), 4),
                        "fde_m": round(float(avg_fde), 4),
                        "n_samples": len(eval_results),
                        "inference_steps": args.num_inference_steps,
                    }
                    print(json.dumps(eval_log), flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(eval_log) + "\n")

                    if avg_ade < best_ade:
                        best_ade = avg_ade
                        ae.save_checkpoint(
                            output_dir / "best.pt",
                            bundle=student_bundle,
                            payload={"step": step, "eval": eval_log},
                        )
                        print(json.dumps({"event": "best_saved", "step": step, "ade_m": round(float(avg_ade), 4)}), flush=True)

                student_bundle.train()
                gc.collect()
                torch.cuda.empty_cache()

            # Save checkpoint
            if step % args.save_every == 0:
                ae.save_checkpoint(
                    output_dir / f"step_{step:06d}.pt",
                    bundle=student_bundle,
                    payload={"step": step},
                )

            del batch, teacher_batch_features, teacher_features
            if step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # Save final
    ae.save_checkpoint(
        output_dir / "final.pt",
        bundle=student_bundle,
        payload={"step": step, "best_ade": best_ade},
    )
    print(json.dumps({
        "event": "train_done",
        "total_steps": step,
        "best_ade_m": round(best_ade, 4),
        "elapsed_min": round((time.time() - started) / 60, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
