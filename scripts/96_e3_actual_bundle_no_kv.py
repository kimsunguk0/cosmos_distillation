#!/usr/bin/env python3
"""E3 no-KV one-target sanity using the real 84.build_bundle path.

This differs from scripts/94_diagnose_fm_collapse.py's original E3 path by
loading the teacher/student, freezing teacher, and calling script_84.build_bundle()
exactly like the training script. It verifies that the projection requires_grad
fix is active in the actual AE construction path. For W1 it also supports
multi-draw random FM via --num-time-samples, matching script 84's train_step.
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
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


script_84 = import_script(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py", "script_84")
script_94 = import_script(PROJECT_ROOT / "scripts" / "94_diagnose_fm_collapse.py", "script_94")


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_STUDENT_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "no_nav_camera_labeled_official_full444k"
    / "no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838"
    / "step_006250"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "action_expert" / "v2_e3_actual_bundle_no_kv_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--target-source", choices=("teacher", "gt"), default="teacher")
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--student-model", default=script_84.resolve_student_model_path())
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B")
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--variant", choices=("fixed", "random", "both"), default="random")
    parser.add_argument("--graft-action-in", action="store_true", help="Replace bundle.action_in_proj with the frozen teacher original after build_bundle, then unfreeze it.")
    parser.add_argument("--graft-action-out", action="store_true", help="Replace bundle.action_out_proj with the frozen teacher original after build_bundle, then unfreeze it.")
    parser.add_argument("--num-time-samples", type=int, default=16)
    parser.add_argument("--eval-draws", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--expert-lr", type=float, default=1e-4)
    parser.add_argument("--proj-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--train-timestep-sampler", choices=("uniform", "beta"), default="beta")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_args(args: argparse.Namespace) -> argparse.Namespace:
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


def split_decay_params(module: torch.nn.Module, lr: float, weight_decay: float) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
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


def make_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
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
            progress = (step_idx - int(args.lr_warmup_steps)) / max(1, int(args.steps) - int(args.lr_warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(base_min_ratio, cosine * (1.0 - base_min_ratio) + base_min_ratio)

        lambdas.append(lr_lambda)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def vector_stats(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_f = pred.detach().float().reshape(-1)
    target_f = target.detach().float().reshape(-1)
    cosine = torch.dot(pred_f, target_f) / (torch.linalg.norm(pred_f) * torch.linalg.norm(target_f)).clamp_min(1e-12)
    alpha = torch.dot(pred_f, target_f) / torch.dot(target_f, target_f).clamp_min(1e-12)
    return {
        "loss": float(F.mse_loss(pred.float(), target.float()).detach().cpu()),
        "pred_v_abs_mean": float(pred.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target.detach().abs().mean().cpu()),
        "pred_target_cosine": float(cosine.detach().cpu()),
        "optimal_scale_alpha": float(alpha.detach().cpu()),
    }


def sample_random_batch(
    *,
    x1: torch.Tensor,
    draws: int,
    sampler: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1_b = x1.expand(int(draws), *x1.shape[1:]).to(device=device, dtype=dtype)
    x0 = torch.randn_like(x1_b)
    t = script_84.sample_fm_timesteps(
        batch_size=int(draws),
        sampler=str(sampler),
        device=device,
        dtype=dtype,
    )
    x_t = (1.0 - t) * x0 + t * x1_b
    target_v = x1_b - x0
    return x_t, t, target_v


@torch.no_grad()
def evaluate_random(
    *,
    args: argparse.Namespace,
    bundle: Any,
    action_dims: tuple[int, ...],
    x1: torch.Tensor,
    position_template: torch.Tensor,
    n_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    bundle.eval()
    pred_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    remaining = int(args.eval_draws)
    while remaining > 0:
        cur = min(int(args.eval_batch_size), remaining)
        x_t, t, target_v = sample_random_batch(
            x1=x1,
            draws=cur,
            sampler=str(args.train_timestep_sampler),
            device=device,
            dtype=dtype,
        )
        embeds = bundle.action_in_proj(x_t, t)
        out = bundle.expert(
            inputs_embeds=embeds,
            position_ids=position_template.repeat(1, cur, 1),
            past_key_values=None,
            attention_mask=None,
            use_cache=False,
        )
        pred_v = bundle.action_out_proj(out.last_hidden_state[:, -n_tokens:]).view(-1, *action_dims)
        pred_parts.append(pred_v.detach().cpu())
        target_parts.append(target_v.detach().cpu())
        remaining -= cur
    bundle.train()
    return vector_stats(torch.cat(pred_parts, dim=0), torch.cat(target_parts, dim=0))


def run_variant(
    *,
    args: argparse.Namespace,
    variant: str,
    bundle: Any,
    action_dims: tuple[int, ...],
    target_action: torch.Tensor,
    sample_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = script_84.torch_dtype_from_name(args.ae_dtype)
    n_tokens = int(action_dims[0])
    position_template = torch.arange(n_tokens, dtype=torch.long, device=device).view(1, 1, -1).repeat(3, 1, 1)
    x1 = target_action.to(device=device, dtype=dtype).unsqueeze(0)
    draws = max(int(args.num_time_samples), 1)

    opt_groups = []
    opt_groups.extend(split_decay_params(bundle.expert, float(args.expert_lr), float(args.weight_decay)))
    opt_groups.extend(split_decay_params(bundle.action_in_proj, float(args.proj_lr), float(args.weight_decay)))
    opt_groups.extend(split_decay_params(bundle.action_out_proj, float(args.proj_lr), float(args.weight_decay)))
    optimizer = torch.optim.AdamW(opt_groups)
    scheduler = make_scheduler(optimizer, args)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed) + 333)
        x0_fixed = torch.randn_like(x1)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 333)
        x0_fixed = torch.randn(x1.shape, generator=generator, dtype=x1.dtype, device=device)
    t_fixed = script_84.sample_fm_timesteps(
        batch_size=1,
        sampler=str(args.train_timestep_sampler),
        device=device,
        dtype=dtype,
    )

    log_steps = {1, 2, 5, 10, 25, 50, 100, 200, 300, 400, int(args.steps)}
    rows = []
    started = time.perf_counter()
    bundle.train()
    for step in range(1, int(args.steps) + 1):
        if variant == "fixed":
            x1_b = x1.expand(draws, *x1.shape[1:])
            x0 = x0_fixed.expand(draws, *x0_fixed.shape[1:])
            t = t_fixed.expand(draws, *t_fixed.shape[1:])
        else:
            x1_b = x1.expand(draws, *x1.shape[1:])
            x0 = torch.randn_like(x1_b)
            t = script_84.sample_fm_timesteps(
                batch_size=draws,
                sampler=str(args.train_timestep_sampler),
                device=device,
                dtype=dtype,
            )
        x_t = (1.0 - t) * x0 + t * x1_b
        target_v = x1_b - x0
        optimizer.zero_grad(set_to_none=True)
        embeds = bundle.action_in_proj(x_t, t)
        out = bundle.expert(
            inputs_embeds=embeds,
            position_ids=position_template.repeat(1, draws, 1),
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
                "event": "v2_e3_train_step",
                "variant": variant,
                "sample_id": sample_id,
                "step": step,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                "num_time_samples": draws,
                "t": float(t.detach().float().mean().cpu()),
                **vector_stats(pred_v, target_v),
            }
            print(json.dumps(row), flush=True)
            rows.append(row)
    final_eval = None
    if variant == "random" and int(args.eval_draws) > 0:
        final_eval = evaluate_random(
            args=args,
            bundle=bundle,
            action_dims=action_dims,
            x1=x1,
            position_template=position_template,
            n_tokens=n_tokens,
            device=device,
            dtype=dtype,
        )
        print(json.dumps({"event": "v2_e3_final_eval", "variant": variant, **final_eval}), flush=True)
    result = {
        "event": "v2_e3_actual_bundle_one_sample",
        "variant": variant,
        "sample_id": sample_id,
        "steps": int(args.steps),
        "num_time_samples": draws,
        "final": rows[-1],
        "final_eval": final_eval,
        "rows": rows,
    }
    (output_dir / f"v2_e3_{variant}_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = script_84.torch_dtype_from_name(args.ae_dtype)

    items = script_84.select_items(
        SimpleNamespace(corpus_jsonl=args.corpus_jsonl, split=args.split, num_samples=max(int(args.num_samples), int(args.sample_index) + 1))
    )
    item = items[int(args.sample_index)]
    targs = train_args(args)
    student, _tokenizer, _processor, base_model = script_84.load_student(targs)
    print(json.dumps({"event": "load_teacher_action_modules_start", "device": args.teacher_load_device}), flush=True)
    teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = script_84.load_model_and_processor(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=dtype,
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
    bundle, selected_layers = script_84.build_bundle(teacher_model, targs, student=student)
    if args.graft_action_in:
        bundle.action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=dtype).train()
        script_84.set_module_requires_grad(bundle.action_in_proj, True)
    if args.graft_action_out:
        bundle.action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=dtype).train()
        script_84.set_module_requires_grad(bundle.action_out_proj, True)
    trainable_summary = {
        "event": "v2_bundle_trainable_summary",
        "total_trainable_params": int(sum(param.numel() for param in bundle.parameters() if param.requires_grad)),
        "modules": {
            "expert": script_84.trainable_module_summary(bundle.expert, prefix="expert"),
            "action_in_proj": script_84.trainable_module_summary(bundle.action_in_proj, prefix="action_in_proj"),
            "action_out_proj": script_84.trainable_module_summary(bundle.action_out_proj, prefix="action_out_proj"),
        },
    }
    print(json.dumps(trainable_summary), flush=True)
    del student
    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_action, target_meta = script_94.load_target_action(
        item=item,
        action_space=teacher_model.action_space,
        target_source=str(args.target_source),
        device=torch.device("cpu"),
    )
    action_dims = tuple(teacher_model.action_space.get_action_space_dims())
    initial_state = {key: value.detach().clone() for key, value in bundle.state_dict().items()}
    variants = ("fixed", "random") if args.variant == "both" else (args.variant,)
    results = []
    for variant in variants:
        bundle.load_state_dict(initial_state, strict=True)
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        results.append(
            run_variant(
                args=args,
                variant=variant,
                bundle=bundle,
                action_dims=action_dims,
                target_action=target_action,
                sample_id=item["sample_id"],
                output_dir=args.output_dir,
            )
        )
    summary = {
        "created_at_unix": time.time(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "student_base_model": str(base_model),
        "selected_layers": selected_layers,
        "target_meta": target_meta | {
            "target_action_abs_mean": float(target_action.abs().mean()),
            "target_action_rms": float(torch.sqrt((target_action * target_action).mean())),
        },
        "trainable_summary": trainable_summary,
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
