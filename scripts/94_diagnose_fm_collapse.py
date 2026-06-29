#!/usr/bin/env python3
"""Diagnostics for AE flow-matching velocity collapse.

E1: target action / flow target-v distribution.
E2: post-hoc bucket probe for an existing oracle-KV 32-sample overfit run.
E3: unconditional 1-sample overfit with no prompt KV.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra.utils as hyu  # noqa: E402


SCRIPT_84_PATH = PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"
spec = importlib.util.spec_from_file_location("script_84", SCRIPT_84_PATH)
script_84 = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
assert spec.loader is not None
spec.loader.exec_module(script_84)


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl"
DEFAULT_STUDENT_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "no_nav_camera_labeled_official_full444k"
    / "no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838"
    / "step_006250"
)
DEFAULT_TEACHER_CONFIG = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B" / "config.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "action_expert" / "fm_collapse_diagnostics"
DEFAULT_D1_RUN = PROJECT_ROOT / "outputs" / "action_expert" / "student_ae28" / "d1_oracle_kv_32_s1000_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("e1", "e2", "e3", "all"), default="all")
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--extra-num-samples", type=int, default=512)
    parser.add_argument("--target-source", choices=("teacher", "gt"), default="teacher")
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-draws", type=int, default=64)
    parser.add_argument("--train-timestep-sampler", choices=("uniform", "beta"), default="beta")

    # E3 options.
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--student-model", default=script_84.resolve_student_model_path())
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B")
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--e3-steps", type=int, default=500)
    parser.add_argument("--e3-variant", choices=("fixed", "random", "both"), default="both")
    parser.add_argument("--e3-sample-index", type=int, default=0)
    parser.add_argument("--expert-lr", type=float, default=1e-5)
    parser.add_argument("--proj-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=150)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--no-norm-bias-decay", action="store_true", default=True)

    # E2 options.
    parser.add_argument("--e2-run-dir", type=Path, default=DEFAULT_D1_RUN)
    parser.add_argument("--e2-checkpoints", default="initial,best,final")
    parser.add_argument("--e2-batch-size", type=int, default=2)
    parser.add_argument("--e2-probe-draws", type=int, default=16)
    parser.add_argument("--oracle-seed", type=int, default=7777)
    return parser.parse_args()


def instantiate_action_space(config_path: Path, device: torch.device | str = "cpu") -> nn.Module:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    action_space = hyu.instantiate(cfg["action_space_cfg"])
    return action_space.to(device)


def instantiate_action_projections(
    *,
    config_path: Path,
    action_dims: tuple[int, ...],
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    action_in_proj = hyu.instantiate(
        cfg["action_in_proj_cfg"],
        in_dims=action_dims,
        out_dim=int(hidden_size),
    ).to(device=device, dtype=dtype)
    action_out_proj = hyu.instantiate(
        cfg["action_out_proj_cfg"],
        in_features=int(hidden_size),
        out_features=int(action_dims[-1]),
    ).to(device=device, dtype=dtype)
    script_84.reset_module_parameters(action_in_proj)
    script_84.reset_module_parameters(action_out_proj)
    return action_in_proj.train(), action_out_proj.train()


def select_items_for_count(args: argparse.Namespace, count: int) -> list[dict[str, Any]]:
    select_args = SimpleNamespace(
        corpus_jsonl=args.corpus_jsonl,
        split=args.split,
        num_samples=int(count),
    )
    return script_84.select_items(select_args)


def load_target_action(
    *,
    item: dict[str, Any],
    action_space: nn.Module,
    target_source: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    row = item["row"]
    hist_xyz_np = script_84.load_ego_history_xyz(row, PROJECT_ROOT).astype(np.float32)
    hist_rot_np = script_84.normalize_history_rot(script_84.load_ego_history_rot(row, PROJECT_ROOT))
    if target_source == "gt":
        future_xyz_np = script_84.load_ego_future_xyz(row, PROJECT_ROOT).astype(np.float32)[:64]
        future_rot_np = script_84.load_ego_future_rot(row, PROJECT_ROOT).astype(np.float32)[:64]
    else:
        future_xyz_np, future_rot_np = script_84.raw_teacher_pred(Path(item["raw_json"]))
    hist_xyz = torch.from_numpy(hist_xyz_np[None]).to(device=device, dtype=torch.float32)
    hist_rot = torch.from_numpy(hist_rot_np[None]).to(device=device, dtype=torch.float32)
    future_xyz = torch.from_numpy(future_xyz_np[None]).to(device=device, dtype=torch.float32)
    future_rot = torch.from_numpy(future_rot_np[None]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        action = action_space.traj_to_action(hist_xyz, hist_rot, future_xyz, future_rot).detach()
    meta = {
        "sample_id": item["sample_id"],
        "target_xyz_abs_mean": float(np.abs(future_xyz_np).mean()),
        "target_path_length_m": float(script_84.path_len(future_xyz_np)),
    }
    return action[0].float().cpu(), meta


def sample_t(batch_size: int, sampler: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return script_84.sample_fm_timesteps(batch_size=batch_size, sampler=sampler, device=device, dtype=dtype)


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def bucket_name(value: float, q33: float, q66: float) -> str:
    if value <= q33:
        return "small"
    if value <= q66:
        return "medium"
    return "large"


def summarize_bucket(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("small", "medium", "large"):
        part = [r for r in rows if r["bucket"] == name]
        out[name] = {
            "count": len(part),
            "action_abs_mean": float(np.mean([r["action_abs_mean"] for r in part])) if part else None,
            "path_length_mean_m": float(np.mean([r["target_path_length_m"] for r in part])) if part else None,
            key: float(np.mean([r[key] for r in part])) if part and key in part[0] else None,
        }
    return out


def run_e1(args: argparse.Namespace, *, count: int, label: str) -> dict[str, Any]:
    device = torch.device("cpu")
    action_space = instantiate_action_space(args.teacher_config, device=device)
    items = select_items_for_count(args, count)
    actions: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for item in items:
        action, meta = load_target_action(
            item=item,
            action_space=action_space,
            target_source=str(args.target_source),
            device=device,
        )
        actions.append(action)
        arr = action.numpy()
        rows.append(
            {
                **meta,
                "action_abs_mean": float(np.abs(arr).mean()),
                "action_rms": float(np.sqrt(np.mean(arr * arr))),
                "accel_abs_mean": float(np.abs(arr[..., 0]).mean()),
                "curvature_abs_mean": float(np.abs(arr[..., 1]).mean()),
                "action_min": float(arr.min()),
                "action_max": float(arr.max()),
            }
        )
    action_tensor = torch.stack(actions, dim=0)
    action_np = action_tensor.numpy()
    action_abs_mean = np.asarray([r["action_abs_mean"] for r in rows], dtype=np.float64)
    path_lengths = np.asarray([r["target_path_length_m"] for r in rows], dtype=np.float64)
    q33, q66 = np.percentile(action_abs_mean, [33.333, 66.667])
    for row in rows:
        row["bucket"] = bucket_name(float(row["action_abs_mean"]), float(q33), float(q66))

    rng = torch.Generator(device="cpu").manual_seed(int(args.seed) + 17)
    flow_rows: list[dict[str, Any]] = []
    for draw_idx in range(max(1, int(args.noise_draws))):
        x1 = action_tensor.float()
        x0 = torch.randn(x1.shape, generator=rng, dtype=x1.dtype)
        t = sample_t(int(x1.shape[0]), str(args.train_timestep_sampler), device, x1.dtype)
        target_v = x1 - x0
        x_t = (1.0 - t) * x0 + t * x1
        snr_proxy = (t * x1.abs().mean(dim=(1, 2), keepdim=True)) / (
            (1.0 - t).clamp_min(1e-6) * x0.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        )
        for i, row in enumerate(rows):
            flow_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "bucket": row["bucket"],
                    "draw_idx": draw_idx,
                    "t": float(t[i].item()),
                    "target_v_abs_mean": float(target_v[i].abs().mean().item()),
                    "x0_abs_mean": float(x0[i].abs().mean().item()),
                    "x_t_abs_mean": float(x_t[i].abs().mean().item()),
                    "signal_to_noise_proxy": float(snr_proxy[i].item()),
                }
            )

    t_values = np.asarray([r["t"] for r in flow_rows], dtype=np.float64)
    target_v_abs = np.asarray([r["target_v_abs_mean"] for r in flow_rows], dtype=np.float64)
    snr_values = np.asarray([r["signal_to_noise_proxy"] for r in flow_rows], dtype=np.float64)
    result = {
        "event": "e1_distribution",
        "label": label,
        "target_source": str(args.target_source),
        "count": len(rows),
        "noise_draws": int(args.noise_draws),
        "action_shape": list(action_np.shape),
        "action_abs_mean_quantiles": quantiles(action_abs_mean),
        "path_length_quantiles_m": quantiles(path_lengths),
        "per_dim": {
            "accel": quantiles(action_np[..., 0].reshape(-1).astype(np.float64)),
            "curvature": quantiles(action_np[..., 1].reshape(-1).astype(np.float64)),
        },
        "bucket_thresholds_action_abs_mean": {"small_max": float(q33), "medium_max": float(q66)},
        "bucket_summary": summarize_bucket(rows, "action_rms"),
        "flow_target_v_abs_quantiles": quantiles(target_v_abs),
        "t_quantiles": quantiles(t_values),
        "t_fractions": {
            "lt_0p2": float(np.mean(t_values < 0.2)),
            "gt_0p8": float(np.mean(t_values > 0.8)),
            "gt_0p9": float(np.mean(t_values > 0.9)),
        },
        "signal_to_noise_proxy_quantiles": quantiles(snr_values),
        "flow_by_bucket": {
            name: {
                "count": sum(1 for r in flow_rows if r["bucket"] == name),
                "target_v_abs_mean": float(np.mean([r["target_v_abs_mean"] for r in flow_rows if r["bucket"] == name])),
                "snr_proxy_mean": float(np.mean([r["signal_to_noise_proxy"] for r in flow_rows if r["bucket"] == name])),
            }
            for name in ("small", "medium", "large")
        },
        "rows_head": rows[:16],
    }
    return result


def make_oracle_projection(
    *,
    in_dim: int,
    out_dim: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Linear:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    proj = nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        std = math.sqrt(2.0 / (in_dim + out_dim))
        proj.weight.copy_(torch.randn(proj.weight.shape, generator=generator) * std)
    proj = proj.to(device=device, dtype=dtype).eval()
    for param in proj.parameters():
        param.requires_grad_(False)
    return proj


def inject_oracle_kv(batch: dict[str, Any], proj: nn.Linear) -> dict[str, Any]:
    target_xyz = batch["target_xyz"]
    target = target_xyz.to(device=proj.weight.device, dtype=proj.weight.dtype)
    batch_size, horizon, _ = target.shape
    num_kv_heads = 8
    head_dim = 128
    oracle = proj(target).view(batch_size, horizon, num_kv_heads, head_dim)
    oracle_k = oracle.transpose(1, 2).contiguous()
    oracle_v = oracle_k.clone()

    cache = batch["cache"]
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise RuntimeError("Cache has no 'layers' attribute; cannot inject oracle KV.")
    for layer_cache in layers[:28]:
        layer_cache.keys = oracle_k.detach().clone()
        layer_cache.values = oracle_v.detach().clone()

    context = dict(batch["context"])
    context["kv_cache_seq_len"] = int(horizon)
    pos_dtype = context["position_ids"].dtype
    device = context["position_ids"].device
    n_diff = int(context["n_diffusion_tokens"])
    context["position_ids"] = (
        torch.arange(horizon, horizon + n_diff, device=device, dtype=pos_dtype)
        .view(1, 1, -1)
        .repeat(3, batch_size, 1)
    )
    context["attention_mask"] = None
    batch["context"] = context
    return batch


def build_d1_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        corpus_jsonl=args.corpus_jsonl,
        split=args.split,
        num_samples=int(args.num_samples),
        student_checkpoint_dir=args.student_checkpoint_dir,
        student_model=args.student_model,
        student_dtype=args.student_dtype,
        ae_dtype=args.ae_dtype,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        max_new_tokens=192,
        device=args.device,
        teacher_load_device=args.teacher_load_device,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        prefix_mode="student_free",
        target_source=args.target_source,
        train_backbone_lora=False,
        stage2_attention_mode="official_none",
        compressed_layers=28,
        mapping="linspace_round",
        ae_init_mode="student_backbone_init",
        expert_lr=args.expert_lr,
        proj_lr=args.proj_lr,
        weight_decay=args.weight_decay,
        no_norm_bias_decay=True,
    )


def bucket_thresholds_for_items(args: argparse.Namespace, items: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, float], dict[str, Any]]:
    action_space = instantiate_action_space(args.teacher_config, device="cpu")
    action_abs_by_id: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for item in items:
        action, meta = load_target_action(
            item=item,
            action_space=action_space,
            target_source=str(args.target_source),
            device=torch.device("cpu"),
        )
        action_abs = float(action.abs().mean())
        action_abs_by_id[item["sample_id"]] = action_abs
        rows.append(
            {
                **meta,
                "action_abs_mean": action_abs,
                "action_rms": float(torch.sqrt((action * action).mean())),
            }
        )
    values = np.asarray([row["action_abs_mean"] for row in rows], dtype=np.float64)
    q33, q66 = np.percentile(values, [33.333, 66.667])
    bucket_by_id: dict[str, str] = {}
    for row in rows:
        row["bucket"] = bucket_name(float(row["action_abs_mean"]), float(q33), float(q66))
        bucket_by_id[row["sample_id"]] = row["bucket"]
    meta = {
        "bucket_thresholds_action_abs_mean": {"small_max": float(q33), "medium_max": float(q66)},
        "action_abs_mean_quantiles": quantiles(values),
        "bucket_summary": summarize_bucket(rows, "action_rms"),
        "rows_head": rows[:16],
    }
    return bucket_by_id, action_abs_by_id, meta


def aggregate_probe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def part_stats(part: list[dict[str, Any]]) -> dict[str, Any]:
        if not part:
            return {
                "count": 0,
                "loss": None,
                "pred_v_abs_mean": None,
                "target_v_abs_mean": None,
                "pred_target_cosine": None,
                "optimal_scale_alpha": None,
                "pred_over_target_abs": None,
                "t_mean": None,
            }
        pred = np.asarray([r["pred_v_abs_mean"] for r in part], dtype=np.float64)
        target = np.asarray([r["target_v_abs_mean"] for r in part], dtype=np.float64)
        return {
            "count": len(part),
            "loss": float(np.mean([r["loss"] for r in part])),
            "pred_v_abs_mean": float(np.mean(pred)),
            "target_v_abs_mean": float(np.mean(target)),
            "pred_target_cosine": float(np.mean([r["pred_target_cosine"] for r in part])),
            "optimal_scale_alpha": float(np.mean([r["optimal_scale_alpha"] for r in part])),
            "pred_over_target_abs": float(np.mean(pred / np.clip(target, 1e-12, None))),
            "t_mean": float(np.mean([r["t"] for r in part])),
        }

    return {
        "overall": part_stats(rows),
        "by_bucket": {
            name: part_stats([r for r in rows if r["bucket"] == name])
            for name in ("small", "medium", "large")
        },
    }


def probe_bucket_metrics(
    *,
    args: argparse.Namespace,
    bundle: script_84.AE28Bundle,
    teacher_model: Any,
    batches: list[dict[str, Any]],
    bucket_by_id: dict[str, str],
    action_abs_by_id: dict[str, float],
    checkpoint_name: str,
) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = script_84.torch_dtype_from_name(args.ae_dtype)
    action_dims = tuple(teacher_model.action_space.get_action_space_dims())
    n_diffusion_tokens = int(action_dims[0])
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False

    rows: list[dict[str, Any]] = []
    torch.manual_seed(int(args.seed) + 9000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + 9000)
    started = time.perf_counter()
    bundle.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        for draw_idx in range(int(args.e2_probe_draws)):
            for batch in batches:
                prompt_cache = batch["cache"]
                context = batch["context"]
                target_action = batch["target_action"].to(device=device, dtype=dtype)
                x0 = torch.randn_like(target_action)
                t = sample_t(int(target_action.shape[0]), str(args.train_timestep_sampler), device, dtype)
                x_t = (1.0 - t) * x0 + t * target_action
                target_v = target_action - x0
                prefill_seq_len = int(context["kv_cache_seq_len"])
                future_token_embeds = bundle.action_in_proj(x_t, t)
                if future_token_embeds.dim() == 2:
                    future_token_embeds = future_token_embeds.view(target_action.shape[0], n_diffusion_tokens, -1)
                out = bundle.expert(
                    inputs_embeds=future_token_embeds,
                    position_ids=context["position_ids"],
                    past_key_values=prompt_cache,
                    attention_mask=None,
                    use_cache=True,
                    **kwargs,
                )
                prompt_cache.crop(prefill_seq_len)
                pred_v = bundle.action_out_proj(out.last_hidden_state[:, -n_diffusion_tokens:]).view(
                    -1, *action_dims
                )
                for sample_index, sample_id in enumerate(batch["sample_ids"]):
                    stats = vector_stats(pred_v[sample_index], target_v[sample_index])
                    rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "sample_id": sample_id,
                            "bucket": bucket_by_id[sample_id],
                            "target_action_abs_mean": action_abs_by_id[sample_id],
                            "draw_idx": draw_idx,
                            "t": float(t[sample_index].detach().float().mean().cpu()),
                            **stats,
                        }
                    )
    result = {
        "event": "e2_bucket_probe",
        "checkpoint": checkpoint_name,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "probe_draws": int(args.e2_probe_draws),
        "aggregate": aggregate_probe_rows(rows),
        "rows_head": rows[:24],
    }
    print(json.dumps(result), flush=True)
    return result


def run_e2(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    d1_args = build_d1_args(args)
    items = select_items_for_count(args, int(args.num_samples))
    bucket_by_id, action_abs_by_id, bucket_meta = bucket_thresholds_for_items(args, items)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    student, student_tokenizer, student_processor, base_model = script_84.load_student(d1_args)
    print(json.dumps({"event": "e2_load_teacher_action_modules_start", "device": args.teacher_load_device}), flush=True)
    teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = script_84.load_model_and_processor(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=script_84.torch_dtype_from_name(args.ae_dtype),
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
    script_84.force_attention(teacher_model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")
    bundle, selected_layers = script_84.build_bundle(teacher_model, d1_args, student=student)
    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    oracle_proj = make_oracle_projection(
        in_dim=3,
        out_dim=8 * 128,
        seed=int(args.oracle_seed),
        dtype=script_84.torch_dtype_from_name(args.ae_dtype),
        device=device,
    )
    batches: list[dict[str, Any]] = []
    for batch_items in script_84.iter_batches(items, int(args.e2_batch_size)):
        batch = script_84.build_batch(
            args=d1_args,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            batch_items=batch_items,
        )
        batches.append(inject_oracle_kv(batch, oracle_proj))
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint_results: list[dict[str, Any]] = []
    for name in [part.strip() for part in str(args.e2_checkpoints).split(",") if part.strip()]:
        if name != "initial":
            checkpoint_path = args.e2_run_dir / f"{name}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            bundle.load_state_dict(checkpoint["bundle_state_dict"], strict=True)
        checkpoint_results.append(
            probe_bucket_metrics(
                args=args,
                bundle=bundle,
                teacher_model=teacher_model,
                batches=batches,
                bucket_by_id=bucket_by_id,
                action_abs_by_id=action_abs_by_id,
                checkpoint_name=name,
            )
        )
    return {
        "event": "e2_oracle_kv_bucket_probe",
        "student_base_model": str(base_model),
        "selected_count": len(items),
        "selected_layers": selected_layers,
        "oracle_seed": int(args.oracle_seed),
        "bucket_meta": bucket_meta,
        "checkpoints": checkpoint_results,
    }


def split_decay_params(module: nn.Module, lr: float, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        is_norm = "norm" in lname or "layernorm" in lname or "rmsnorm" in lname or "ln_" in lname
        if param.dim() <= 1 or name.endswith(".bias") or is_norm:
            no_decay.append(param)
        else:
            decay.append(param)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> Any | None:
    if int(args.lr_warmup_steps) <= 0:
        return None
    import math

    lambdas = []
    for group in optimizer.param_groups:
        base_lr = float(group["lr"])
        min_ratio = min(1.0, float(args.min_lr) / max(base_lr, 1e-12))

        def lr_lambda(step_idx: int, *, base_min_ratio: float = min_ratio) -> float:
            if step_idx < int(args.lr_warmup_steps):
                return float(step_idx) / max(1, int(args.lr_warmup_steps))
            progress = (step_idx - int(args.lr_warmup_steps)) / max(1, int(args.e3_steps) - int(args.lr_warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(base_min_ratio, cosine * (1.0 - base_min_ratio) + base_min_ratio)

        lambdas.append(lr_lambda)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def vector_stats(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_f = pred.detach().float().reshape(-1)
    target_f = target.detach().float().reshape(-1)
    denom = torch.linalg.norm(pred_f) * torch.linalg.norm(target_f)
    cosine = torch.dot(pred_f, target_f) / denom.clamp_min(1e-12)
    alpha = torch.dot(pred_f, target_f) / torch.dot(target_f, target_f).clamp_min(1e-12)
    return {
        "loss": float(F.mse_loss(pred.float(), target.float()).detach().cpu()),
        "pred_v_abs_mean": float(pred.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target.detach().abs().mean().cpu()),
        "pred_target_cosine": float(cosine.detach().cpu()),
        "optimal_scale_alpha": float(alpha.detach().cpu()),
    }


def run_e3_variant(
    *,
    args: argparse.Namespace,
    variant: str,
    bundle: script_84.AE28Bundle,
    action_space: nn.Module,
    target_action: torch.Tensor,
    sample_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = script_84.torch_dtype_from_name(args.ae_dtype)
    action_dims = tuple(action_space.get_action_space_dims())
    n_tokens = int(action_dims[0])
    position_ids = torch.arange(n_tokens, dtype=torch.long, device=device).view(1, 1, -1).repeat(3, 1, 1)
    x1 = target_action.to(device=device, dtype=dtype).unsqueeze(0)

    opt_groups: list[dict[str, Any]] = []
    opt_groups.extend(split_decay_params(bundle.expert, float(args.expert_lr), float(args.weight_decay)))
    opt_groups.extend(split_decay_params(bundle.action_in_proj, float(args.proj_lr), float(args.weight_decay)))
    opt_groups.extend(split_decay_params(bundle.action_out_proj, float(args.proj_lr), float(args.weight_decay)))
    optimizer = torch.optim.AdamW(opt_groups)
    scheduler = make_scheduler(optimizer, args)

    fixed_gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu").manual_seed(int(args.seed) + 333)
    if device.type == "cuda":
        # torch.randn_like(generator=...) is not consistently available in this env.
        torch.cuda.manual_seed_all(int(args.seed) + 333)
        x0_fixed = torch.randn_like(x1)
    else:
        x0_fixed = torch.randn(x1.shape, generator=fixed_gen, dtype=x1.dtype, device=device)
    t_fixed = sample_t(1, str(args.train_timestep_sampler), device, dtype)

    log_steps = {1, 2, 5, 10, 25, 50, 100, 200, 300, 400, int(args.e3_steps)}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    bundle.train()
    for step in range(1, int(args.e3_steps) + 1):
        if variant == "fixed":
            x0 = x0_fixed
            t = t_fixed
        elif variant == "random":
            x0 = torch.randn_like(x1)
            t = sample_t(1, str(args.train_timestep_sampler), device, dtype)
        else:
            raise ValueError(f"Unknown E3 variant: {variant}")
        x_t = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0
        optimizer.zero_grad(set_to_none=True)
        embeds = bundle.action_in_proj(x_t, t)
        out = bundle.expert(
            inputs_embeds=embeds,
            position_ids=position_ids,
            past_key_values=None,
            attention_mask=None,
            use_cache=False,
        )
        pred_v = bundle.action_out_proj(out.last_hidden_state[:, -n_tokens:]).view(-1, *action_dims)
        loss = F.mse_loss(pred_v.float(), target_v.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(bundle.parameters()), float(args.grad_clip_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step in log_steps:
            row = {
                "event": "e3_train_step",
                "variant": variant,
                "sample_id": sample_id,
                "step": step,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                "t": float(t.detach().float().mean().cpu()),
                **vector_stats(pred_v, target_v),
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
    result = {
        "event": "e3_unconditional_one_sample",
        "variant": variant,
        "sample_id": sample_id,
        "steps": int(args.e3_steps),
        "status": "ok",
        "final": rows[-1] if rows else None,
        "rows": rows,
    }
    out_path = output_dir / f"e3_{variant}_summary.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_e3(args: argparse.Namespace, selected_item: dict[str, Any], target_action_cpu: torch.Tensor) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    action_space = instantiate_action_space(args.teacher_config, device=device)
    action_dims = tuple(action_space.get_action_space_dims())

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    load_args = argparse.Namespace(
        student_checkpoint_dir=args.student_checkpoint_dir,
        student_model=args.student_model,
        student_dtype=args.student_dtype,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        device=str(device),
    )
    student, _tokenizer, _processor, base_model = script_84.load_student(load_args)
    hidden_size = int(student.backbone.model.language_model.config.hidden_size)
    ae_dtype = script_84.torch_dtype_from_name(args.ae_dtype)
    expert = script_84.build_student_backbone_expert(
        student=student,
        dtype=ae_dtype,
        device=str(device),
        attn_implementation=("sdpa" if args.attn_implementation != "eager" else "eager"),
    )
    action_in_proj, action_out_proj = instantiate_action_projections(
        config_path=args.teacher_config,
        action_dims=action_dims,
        hidden_size=hidden_size,
        dtype=ae_dtype,
        device=device,
    )
    bundle = script_84.AE28Bundle(
        expert=expert,
        action_in_proj=action_in_proj,
        action_out_proj=action_out_proj,
    ).to(device).train()
    initial_state = {key: value.detach().clone() for key, value in bundle.state_dict().items()}
    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    variants = ("fixed", "random") if args.e3_variant == "both" else (args.e3_variant,)
    results = []
    for variant in variants:
        bundle.load_state_dict(initial_state, strict=True)
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        results.append(
            run_e3_variant(
                args=args,
                variant=variant,
                bundle=bundle,
                action_space=action_space,
                target_action=target_action_cpu,
                sample_id=selected_item["sample_id"],
                output_dir=args.output_dir,
            )
        )
    return {
        "event": "e3_unconditional",
        "student_base_model": str(base_model),
        "sample_id": selected_item["sample_id"],
        "variants": results,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    results: dict[str, Any] = {
        "created_at_unix": time.time(),
        "args": {
            **serializable_args,
            "corpus_jsonl": str(args.corpus_jsonl),
            "student_checkpoint_dir": str(args.student_checkpoint_dir),
            "teacher_config": str(args.teacher_config),
            "output_dir": str(args.output_dir),
        },
    }

    if args.mode in ("e1", "all"):
        e1_32 = run_e1(args, count=int(args.num_samples), label=f"first_{int(args.num_samples)}")
        print(json.dumps(e1_32), flush=True)
        results["e1"] = e1_32
        if int(args.extra_num_samples) > int(args.num_samples):
            e1_extra = run_e1(args, count=int(args.extra_num_samples), label=f"first_{int(args.extra_num_samples)}")
            print(json.dumps(e1_extra), flush=True)
            results["e1_extra"] = e1_extra

    if args.mode in ("e2", "all"):
        e2_result = run_e2(args)
        print(json.dumps(e2_result), flush=True)
        results["e2"] = e2_result

    if args.mode in ("e3", "all"):
        e3_items = select_items_for_count(args, max(int(args.e3_sample_index) + 1, int(args.num_samples)))
        selected_item = e3_items[int(args.e3_sample_index)]
        action_space_cpu = instantiate_action_space(args.teacher_config, device="cpu")
        target_action, target_meta = load_target_action(
            item=selected_item,
            action_space=action_space_cpu,
            target_source=str(args.target_source),
            device=torch.device("cpu"),
        )
        results["e3_target_meta"] = target_meta | {
            "target_action_abs_mean": float(target_action.abs().mean()),
            "target_action_rms": float(torch.sqrt((target_action * target_action).mean())),
        }
        e3_result = run_e3(args, selected_item, target_action)
        print(json.dumps(e3_result), flush=True)
        results["e3"] = e3_result

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
