#!/usr/bin/env python3
"""Cached AE28 overfit sanity check.

This script isolates the action expert/FM learning problem from expensive VLM
rollout. It builds a small set of student/teacher-forced KV batches once, drops
the student VLM from memory, and repeatedly trains AE28 on those cached KVs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AE84_PATH = PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"
spec = importlib.util.spec_from_file_location("ae84", AE84_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import {AE84_PATH}")
ae84 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae84)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=ae84.DEFAULT_CORPUS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--student-checkpoint-dir", type=Path, default=ae84.DEFAULT_STUDENT_CKPT)
    parser.add_argument("--student-model", default=ae84.resolve_student_model_path())
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=ae84.DEFAULT_TEACHER)
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--prefix-mode", choices=("student_free", "teacher_forced"), default="teacher_forced")
    parser.add_argument("--ae-init-mode", choices=("teacher_compressed", "scratch"), default="teacher_compressed")
    parser.add_argument("--mapping", choices=("linspace_round", "first_n"), default="linspace_round")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--train-timestep-sampler", choices=("uniform", "beta"), default="beta")
    parser.add_argument(
        "--stage2-attention-mode",
        choices=("official_none", "masked"),
        default="official_none",
        help=(
            "official_none matches alpamayo_base Stage-2 TrainableAlpamayoR1, "
            "which calls the expert with attention_mask=None. masked keeps the "
            "older local inference-style expert attention mask."
        ),
    )
    parser.add_argument("--num-time-samples", type=int, default=1)
    parser.add_argument(
        "--init-ae-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint produced by this script; loads bundle_state_dict before training.",
    )
    parser.add_argument(
        "--velocity-scale-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss on mean |pred_v| vs mean |target_v| to prevent under-scaled FM fields.",
    )
    parser.add_argument(
        "--action-recon-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary SmoothL1 on x1_hat = x_t + (1 - t) * pred_v against target action x1.",
    )
    parser.add_argument(
        "--traj-horizon-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary horizon-weighted ADE loss on the one-step reconstructed trajectory.",
    )
    parser.add_argument(
        "--traj-final-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary horizon-weighted FDE loss on the one-step reconstructed trajectory.",
    )
    parser.add_argument(
        "--traj-horizon-weights",
        default="0.25,0.5,1.0",
        help="Comma-separated weights for horizons 16,32,64 used by trajectory auxiliary losses.",
    )
    parser.add_argument("--expert-lr", type=float, default=1e-4)
    parser.add_argument("--proj-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-seed-mode", choices=("fixed", "step"), default="fixed")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "action_expert" / "student_ae28_official_cached_overfit",
    )
    return parser.parse_args()


def detach_cache(cache: Any) -> Any:
    for layer in getattr(cache, "layers", []):
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys.detach()
        if getattr(layer, "values", None) is not None:
            layer.values = layer.values.detach()
    return cache


def detach_batch(batch: dict[str, Any]) -> dict[str, Any]:
    batch["cache"] = detach_cache(batch["cache"])
    for key in ("target_action", "target_xyz", "ego_history_xyz", "ego_history_rot"):
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].detach()
    return batch


def iter_cached_batches(batches: list[dict[str, Any]], batch_size: int):
    del batch_size
    while True:
        for batch in batches:
            yield batch


def step_forward(
    *,
    bundle: Any,
    teacher_model: Any,
    batch: dict[str, Any],
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    dtype = next(bundle.parameters()).dtype
    n_diffusion_tokens = int(batch["context"]["n_diffusion_tokens"])
    action_dims = teacher_model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False
    future_token_embeds = bundle.action_in_proj(x_t.to(dtype=dtype), t.to(dtype=dtype))
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(x_t.shape[0], n_diffusion_tokens, -1)
    expert_attention_mask = batch["context"].get("attention_mask")
    if expert_attention_mask is not None:
        expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
    out = bundle.expert(
        inputs_embeds=future_token_embeds,
        position_ids=batch["context"]["position_ids"],
        past_key_values=batch["cache"],
        attention_mask=expert_attention_mask,
        use_cache=False,
        **kwargs,
    )
    # HF Cache objects are mutated even when use_cache=False. Restore the
    # cached prefix length after every expert call so the next FM/Euler step
    # sees the same prompt KV and the same fixed attention-mask width.
    if hasattr(batch["cache"], "crop"):
        batch["cache"].crop(int(batch["context"]["kv_cache_seq_len"]))
        detach_cache(batch["cache"])
    last_hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
    return bundle.action_out_proj(last_hidden).view(-1, *action_dims)


def train_step_cached(
    *,
    bundle: Any,
    teacher_model: Any,
    batch: dict[str, Any],
    num_time_samples: int,
    train_timestep_sampler: str,
    velocity_scale_loss_weight: float,
    action_recon_loss_weight: float,
    traj_horizon_loss_weight: float,
    traj_final_loss_weight: float,
    traj_horizon_weights: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = next(bundle.parameters()).dtype
    repeats = max(int(num_time_samples), 1)
    step_batch = batch
    target_action = batch["target_action"]
    target_xyz = batch["target_xyz"]
    ego_history_xyz = batch["ego_history_xyz"]
    ego_history_rot = batch["ego_history_rot"]
    if repeats > 1:
        # Train the same cached prefix against multiple independent FM
        # noise/timestep samples in one optimizer step.  The HF cache object is
        # mutable, so keep the stored cache pristine and repeat a throwaway copy.
        step_batch = dict(batch)
        step_batch["cache"] = copy.deepcopy(batch["cache"])
        step_batch["cache"].batch_repeat_interleave(repeats)
        step_batch["context"] = ae84.repeat_context(batch["context"], repeats)
        target_action = target_action.repeat_interleave(repeats, dim=0)
        target_xyz = target_xyz.repeat_interleave(repeats, dim=0)
        ego_history_xyz = ego_history_xyz.repeat_interleave(repeats, dim=0)
        ego_history_rot = ego_history_rot.repeat_interleave(repeats, dim=0)
    x1 = target_action.to(device=device, dtype=dtype)
    x0 = torch.randn_like(x1)
    t = ae84.sample_fm_timesteps(
        batch_size=int(x1.shape[0]),
        sampler=str(train_timestep_sampler),
        device=device,
        dtype=dtype,
    )
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    pred_v = step_forward(bundle=bundle, teacher_model=teacher_model, batch=step_batch, x_t=x_t, t=t)
    fm_loss = F.mse_loss(pred_v.float(), target_v.float())
    velocity_scale_loss = torch.zeros((), device=device, dtype=torch.float32)
    if float(velocity_scale_loss_weight) > 0.0:
        pred_scale = pred_v.float().abs().mean(dim=tuple(range(1, pred_v.ndim)))
        target_scale = target_v.float().abs().mean(dim=tuple(range(1, target_v.ndim)))
        velocity_scale_loss = F.smooth_l1_loss(pred_scale, target_scale)
    action_recon_loss = torch.zeros((), device=device, dtype=torch.float32)
    need_x1_hat = (
        float(action_recon_loss_weight) > 0.0
        or float(traj_horizon_loss_weight) > 0.0
        or float(traj_final_loss_weight) > 0.0
    )
    x1_hat = None
    if need_x1_hat:
        x1_hat = x_t.float() + (1.0 - t.float()) * pred_v.float()
    if float(action_recon_loss_weight) > 0.0:
        assert x1_hat is not None
        action_recon_loss = F.smooth_l1_loss(x1_hat, x1.float())
    traj_horizon_loss = torch.zeros((), device=device, dtype=torch.float32)
    traj_final_loss = torch.zeros((), device=device, dtype=torch.float32)
    if float(traj_horizon_loss_weight) > 0.0 or float(traj_final_loss_weight) > 0.0:
        assert x1_hat is not None
        pred_xyz, _pred_rot = teacher_model.action_space.action_to_traj(
            x1_hat.to(device=device, dtype=dtype),
            ego_history_xyz.to(device=device),
            ego_history_rot.to(device=device),
        )
        target_xyz_device = target_xyz.to(device=device, dtype=pred_xyz.dtype)
        horizon_weights = [float(value) for value in str(traj_horizon_weights).split(",") if value.strip()]
        if len(horizon_weights) != 3:
            raise ValueError("--traj-horizon-weights must contain exactly 3 comma-separated values.")
        horizons = (16, 32, 64)
        weighted_ade_terms: list[torch.Tensor] = []
        weighted_fde_terms: list[torch.Tensor] = []
        for horizon, weight in zip(horizons, horizon_weights, strict=True):
            n = min(int(horizon), int(pred_xyz.shape[1]), int(target_xyz_device.shape[1]))
            delta = pred_xyz[:, :n, :2].float() - target_xyz_device[:, :n, :2].float()
            point_dist = torch.sqrt(torch.sum(delta * delta, dim=-1) + 1e-6)
            weighted_ade_terms.append(float(weight) * point_dist.mean())
            weighted_fde_terms.append(float(weight) * point_dist[:, -1].mean())
        norm = max(float(sum(horizon_weights)), 1e-6)
        traj_horizon_loss = sum(weighted_ade_terms) / norm
        traj_final_loss = sum(weighted_fde_terms) / norm
    loss = (
        fm_loss
        + float(velocity_scale_loss_weight) * velocity_scale_loss
        + float(action_recon_loss_weight) * action_recon_loss
        + float(traj_horizon_loss_weight) * traj_horizon_loss
        + float(traj_final_loss_weight) * traj_final_loss
    )
    return loss, {
        "fm_loss": float(fm_loss.detach().cpu()),
        "velocity_scale_loss": float(velocity_scale_loss.detach().cpu()),
        "action_recon_loss": float(action_recon_loss.detach().cpu()),
        "traj_horizon_loss": float(traj_horizon_loss.detach().cpu()),
        "traj_final_loss": float(traj_final_loss.detach().cpu()),
        "num_time_samples": float(repeats),
        "target_action_abs_mean": float(x1.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target_v.detach().abs().mean().cpu()),
        "pred_v_abs_mean": float(pred_v.detach().abs().mean().cpu()),
        "train_t_mean": float(t.detach().float().mean().cpu()),
    }


def sample_paths_cached(
    *,
    bundle: Any,
    teacher_model: Any,
    batch: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    dtype = next(bundle.parameters()).dtype
    batch_size = int(batch["ego_history_xyz"].shape[0])
    eval_batch = dict(batch)
    eval_batch["cache"] = copy.deepcopy(batch["cache"])

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return step_forward(bundle=bundle, teacher_model=teacher_model, batch=eval_batch, x_t=x, t=t)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        action = teacher_model.diffusion.sample(batch_size=batch_size, step_fn=step_fn, device=device)
        pred_xyz, pred_rot = teacher_model.action_space.action_to_traj(
            action,
            batch["ego_history_xyz"].to(device),
            batch["ego_history_rot"].to(device),
        )
    return {
        "action": action.detach().float().cpu().numpy(),
        "pred_xyz": pred_xyz.detach().float().cpu().numpy(),
        "pred_rot": pred_rot.detach().float().cpu().numpy(),
    }


def evaluate_cached(
    *,
    args: argparse.Namespace,
    bundle: Any,
    teacher_model: Any,
    batches: list[dict[str, Any]],
    step: int,
) -> dict[str, Any]:
    bundle.eval()
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    eval_seed_base = int(args.seed) + 1000 + (0 if str(args.eval_seed_mode) == "fixed" else int(step))
    batch_index = 0
    for batch in batches:
        pred = sample_paths_cached(
            bundle=bundle,
            teacher_model=teacher_model,
            batch=batch,
            seed=eval_seed_base + batch_index,
            device=device,
        )
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        for row_index, sample_id in enumerate(batch["sample_ids"]):
            ade, fde = ae84.ade_fde(pred["pred_xyz"][row_index], target_xyz[row_index])
            h16_ade, h16_fde = ae84.ade_fde(pred["pred_xyz"][row_index][:16], target_xyz[row_index][:16])
            h32_ade, h32_fde = ae84.ade_fde(pred["pred_xyz"][row_index][:32], target_xyz[row_index][:32])
            rows.append(
                {
                    "sample_id": sample_id,
                    "ade_m": ade,
                    "fde_m": fde,
                    "h1p6_16wp_ade_m": h16_ade,
                    "h1p6_16wp_fde_m": h16_fde,
                    "h3p2_32wp_ade_m": h32_ade,
                    "h3p2_32wp_fde_m": h32_fde,
                    "pred_path_length_m": ae84.path_len(pred["pred_xyz"][row_index]),
                    "target_path_length_m": ae84.path_len(target_xyz[row_index]),
                }
            )
        batch_index += 1
    ades = [row["ade_m"] for row in rows]
    fdes = [row["fde_m"] for row in rows]
    h16_ades = [row["h1p6_16wp_ade_m"] for row in rows]
    h16_fdes = [row["h1p6_16wp_fde_m"] for row in rows]
    h32_ades = [row["h3p2_32wp_ade_m"] for row in rows]
    h32_fdes = [row["h3p2_32wp_fde_m"] for row in rows]
    out = {
        "event": "eval",
        "step": int(step),
        "eval_seed_mode": str(args.eval_seed_mode),
        "eval_seed_base": int(eval_seed_base),
        "eval_count": len(rows),
        "ade_mean_m": float(np.mean(ades)),
        "ade_p50_m": float(np.percentile(ades, 50)),
        "fde_mean_m": float(np.mean(fdes)),
        "fde_p50_m": float(np.percentile(fdes, 50)),
        "horizon": {
            "h1p6_16wp": {
                "ade_mean_m": float(np.mean(h16_ades)),
                "ade_p50_m": float(np.percentile(h16_ades, 50)),
                "fde_mean_m": float(np.mean(h16_fdes)),
                "fde_p50_m": float(np.percentile(h16_fdes, 50)),
            },
            "h3p2_32wp": {
                "ade_mean_m": float(np.mean(h32_ades)),
                "ade_p50_m": float(np.percentile(h32_ades, 50)),
                "fde_mean_m": float(np.mean(h32_fdes)),
                "fde_p50_m": float(np.percentile(h32_fdes, 50)),
            },
            "h6p4_64wp": {
                "ade_mean_m": float(np.mean(ades)),
                "ade_p50_m": float(np.percentile(ades, 50)),
                "fde_mean_m": float(np.mean(fdes)),
                "fde_p50_m": float(np.percentile(fdes, 50)),
            },
        },
        "rows": rows,
    }
    bundle.train()
    return out


def save_checkpoint(path: Path, *, bundle: Any, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bundle_state_dict": bundle.state_dict(), "payload": payload}, path)


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    summary_path = args.output_dir / "summary.json"
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    summary: dict[str, Any] = {
        "created_at_unix": time.time(),
        "args": jsonable_args(args) | {
            "corpus_jsonl": str(args.corpus_jsonl),
            "student_checkpoint_dir": str(args.student_checkpoint_dir),
            "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
            "output_dir": str(args.output_dir),
        },
        "status": "running",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        items = ae84.select_items(args)
        student, student_tokenizer, student_processor, base_model = ae84.load_student(args)
        summary["student_base_model"] = str(base_model)
        print(json.dumps({"event": "load_teacher_action_modules_start", "device": args.teacher_load_device}), flush=True)
        teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = ae84.load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=ae84.torch_dtype_from_name(args.ae_dtype),
            device=args.teacher_load_device,
            config_json=None,
            runtime_support=None,
            attn_implementation=args.attn_implementation,
            min_pixels=163840,
            max_pixels=196608,
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        ae84.force_attention(teacher_model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")
        bundle, selected_layers = ae84.build_bundle(teacher_model, args)
        summary["ae28_selected_teacher_layers"] = selected_layers
        summary["trainable_params"] = int(sum(p.numel() for p in bundle.parameters() if p.requires_grad))
        if args.init_ae_checkpoint is not None:
            init_path = Path(args.init_ae_checkpoint)
            if not init_path.exists():
                raise FileNotFoundError(f"--init-ae-checkpoint does not exist: {init_path}")
            payload = torch.load(init_path, map_location="cpu", weights_only=False)
            state_dict = payload.get("bundle_state_dict") if isinstance(payload, dict) else None
            if not isinstance(state_dict, dict):
                raise ValueError(f"Checkpoint does not contain bundle_state_dict: {init_path}")
            missing, unexpected = bundle.load_state_dict(state_dict, strict=False)
            summary["init_ae_checkpoint"] = str(init_path)
            summary["init_ae_checkpoint_missing_keys"] = list(missing)
            summary["init_ae_checkpoint_unexpected_keys"] = list(unexpected)
            print(
                json.dumps(
                    {
                        "event": "init_ae_checkpoint_loaded",
                        "path": str(init_path),
                        "missing_keys": list(missing),
                        "unexpected_keys": list(unexpected),
                    }
                ),
                flush=True,
            )
        if hasattr(teacher_model, "vlm"):
            delattr(teacher_model, "vlm")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cached_batches: list[dict[str, Any]] = []
        started_cache = time.perf_counter()
        for batch_items in ae84.iter_batches(items, int(args.batch_size)):
            batch = ae84.build_batch(
                args=args,
                student=student,
                student_processor=student_processor,
                student_tokenizer=student_tokenizer,
                teacher_model=teacher_model,
                batch_items=batch_items,
            )
            cached_batches.append(detach_batch(batch))
            print(
                json.dumps(
                    {
                        "event": "cache_batch_done",
                        "batch_index": len(cached_batches) - 1,
                        "batch_size": len(batch_items),
                        "kv_cache_seq_len": int(batch["context"]["kv_cache_seq_len"]),
                        "traj_start_hit_rate": batch["traj_start_hit_rate"],
                    }
                ),
                flush=True,
            )
        del student, student_processor, student_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        summary["cached_batch_count"] = len(cached_batches)
        summary["cache_build_sec"] = round(time.perf_counter() - started_cache, 3)

        optimizer = torch.optim.AdamW(
            [
                {"params": bundle.expert.parameters(), "lr": float(args.expert_lr)},
                {"params": bundle.action_in_proj.parameters(), "lr": float(args.proj_lr)},
                {"params": bundle.action_out_proj.parameters(), "lr": float(args.proj_lr)},
            ],
            weight_decay=float(args.weight_decay),
        )
        log_handle = log_path.open("a", encoding="utf-8")
        best_eval: dict[str, Any] | None = None

        ev = evaluate_cached(args=args, bundle=bundle, teacher_model=teacher_model, batches=cached_batches, step=0)
        print(json.dumps(ev), flush=True)
        log_handle.write(json.dumps(ev) + "\n")
        log_handle.flush()
        best_eval = ev

        batch_iter = iter_cached_batches(cached_batches, int(args.batch_size))
        started_train = time.perf_counter()
        for step in range(1, int(args.steps) + 1):
            batch = next(batch_iter)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = train_step_cached(
                bundle=bundle,
                teacher_model=teacher_model,
                batch=batch,
                num_time_samples=int(args.num_time_samples),
                train_timestep_sampler=str(args.train_timestep_sampler),
                velocity_scale_loss_weight=float(args.velocity_scale_loss_weight),
                action_recon_loss_weight=float(args.action_recon_loss_weight),
                traj_horizon_loss_weight=float(args.traj_horizon_loss_weight),
                traj_final_loss_weight=float(args.traj_final_loss_weight),
                traj_horizon_weights=str(args.traj_horizon_weights),
                device=device,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(bundle.parameters(), float(args.grad_clip_norm))
            optimizer.step()
            if step == 1 or step % int(args.log_every) == 0:
                row = {
                    "event": "train_step",
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    "elapsed_sec": round(time.perf_counter() - started_train, 3),
                    "sample_ids": batch["sample_ids"],
                    "traj_start_hit_rate": batch["traj_start_hit_rate"],
                    **stats,
                }
                print(json.dumps(row), flush=True)
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
            del loss
            if step % int(args.eval_every) == 0 or step == int(args.steps):
                ev = evaluate_cached(
                    args=args,
                    bundle=bundle,
                    teacher_model=teacher_model,
                    batches=cached_batches,
                    step=step,
                )
                print(json.dumps(ev), flush=True)
                log_handle.write(json.dumps(ev) + "\n")
                log_handle.flush()
                if best_eval is None or float(ev["ade_mean_m"]) < float(best_eval["ade_mean_m"]):
                    best_eval = ev
                    save_checkpoint(args.output_dir / "best.pt", bundle=bundle, payload={"step": step, "eval": ev, "args": vars(args)})

        save_checkpoint(args.output_dir / "final.pt", bundle=bundle, payload={"step": int(args.steps), "args": vars(args)})
        summary.update(
            {
                "status": "ok",
                "elapsed_sec": round(time.perf_counter() - started_train, 3),
                "best_eval": best_eval,
            }
        )
        log_handle.close()
    except Exception as exc:  # noqa: BLE001
        summary.update({"status": "failed", "error": repr(exc)})
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path), "status": summary["status"]}), flush=True)


if __name__ == "__main__":
    main()
