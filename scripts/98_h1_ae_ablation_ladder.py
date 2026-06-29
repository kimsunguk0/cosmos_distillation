#!/usr/bin/env python3
"""H1/H3 ablation ladder for random FM learning.

Variants:
- proj_mlp: real PerWaypointActionInProjV2 -> small token-wise MLP head.
- expert_N: real action_in_proj -> first N student expert layers -> action_out_proj.

All variants use the same fixed target action x1 and random x0/t FM target
as E3, with no prompt KV.
"""
from __future__ import annotations

import argparse
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
from torch import nn


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
DEFAULT_TEACHER_CONFIG = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B" / "config.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "action_expert" / "h1_ablation_ladder_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="proj_mlp or expert_<N>, e.g. expert_2/expert_28")
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--target-source", choices=("teacher", "gt"), default="teacher")
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--student-model", default=script_84.resolve_student_model_path())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--draws-per-step", type=int, default=16)
    parser.add_argument("--eval-draws", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--train-timestep-sampler", choices=("uniform", "beta"), default="beta")
    parser.add_argument("--expert-lr", type=float, default=1e-5)
    parser.add_argument("--proj-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--head-width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


class ProjectionMlpHead(nn.Module):
    def __init__(self, *, action_in_proj: nn.Module, hidden_size: int, out_dim: int, width: int) -> None:
        super().__init__()
        self.action_in_proj = action_in_proj
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(width)),
            nn.SiLU(),
            nn.Linear(int(width), int(out_dim)),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        del position_ids
        return self.head(self.action_in_proj(x_t, t))


class ExpertFMModel(nn.Module):
    def __init__(self, *, expert: nn.Module, action_in_proj: nn.Module, action_out_proj: nn.Module) -> None:
        super().__init__()
        self.expert = expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.action_in_proj(x_t, t)
        out = self.expert(
            inputs_embeds=embeds,
            position_ids=position_ids,
            past_key_values=None,
            attention_mask=None,
            use_cache=False,
        )
        return self.action_out_proj(out.last_hidden_state[:, -x_t.shape[1] :]).view_as(x_t)


class ZeroAttention(nn.Module):
    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, None]:
        del args, kwargs
        return torch.zeros_like(hidden_states), None


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
    cosine = torch.dot(pred_f, target_f) / (
        torch.linalg.norm(pred_f) * torch.linalg.norm(target_f)
    ).clamp_min(1e-12)
    alpha = torch.dot(pred_f, target_f) / torch.dot(target_f, target_f).clamp_min(1e-12)
    per = F.cosine_similarity(pred.detach().float().flatten(1), target.detach().float().flatten(1), dim=1)
    return {
        "loss": float(F.mse_loss(pred.float(), target.float()).detach().cpu()),
        "pred_v_abs_mean": float(pred.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target.detach().abs().mean().cpu()),
        "pred_target_cosine": float(cosine.detach().cpu()),
        "pred_target_cosine_per_draw_mean": float(per.mean().detach().cpu()),
        "pred_target_cosine_per_draw_p05": float(torch.quantile(per.detach().cpu(), 0.05)),
        "optimal_scale_alpha": float(alpha.detach().cpu()),
    }


def sample_batch(
    *,
    x1: torch.Tensor,
    batch_size: int,
    sampler: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1_b = x1.unsqueeze(0).to(device=device, dtype=dtype).expand(int(batch_size), *x1.shape)
    x0 = torch.randn_like(x1_b)
    t = script_84.sample_fm_timesteps(
        batch_size=int(batch_size),
        sampler=str(sampler),
        device=device,
        dtype=dtype,
    )
    x_t = (1.0 - t) * x0 + t * x1_b
    target_v = x1_b - x0
    return x_t, t, target_v


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    x1: torch.Tensor,
    action_dims: tuple[int, ...],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    n_tokens = int(action_dims[0])
    position_template = torch.arange(n_tokens, dtype=torch.long, device=device).view(1, 1, -1).repeat(3, 1, 1)
    pred_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    remaining = int(args.eval_draws)
    while remaining > 0:
        cur = min(int(args.eval_batch_size), remaining)
        x_t, t, target_v = sample_batch(
            x1=x1,
            batch_size=cur,
            sampler=str(args.train_timestep_sampler),
            device=device,
            dtype=dtype,
        )
        position_ids = position_template.repeat(1, cur, 1)
        pred_parts.append(model(x_t, t, position_ids).detach().cpu())
        target_parts.append(target_v.detach().cpu())
        remaining -= cur
    return vector_stats(torch.cat(pred_parts, dim=0), torch.cat(target_parts, dim=0))


def truncate_expert(expert: nn.Module, n_layers: int) -> nn.Module:
    if not hasattr(expert, "layers"):
        raise AttributeError("Expert does not expose .layers")
    old_n = len(expert.layers)
    if int(n_layers) < old_n:
        expert.layers = nn.ModuleList(list(expert.layers[: int(n_layers)]))
    if hasattr(expert, "config"):
        expert.config.num_hidden_layers = int(n_layers)
        if hasattr(expert.config, "layer_types") and getattr(expert.config, "layer_types", None) is not None:
            expert.config.layer_types = list(getattr(expert.config, "layer_types"))[: int(n_layers)]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return expert


def disable_self_attention(expert: nn.Module) -> None:
    for layer in getattr(expert, "layers"):
        layer.self_attn = ZeroAttention()


def train_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        student_checkpoint_dir=args.student_checkpoint_dir,
        student_model=args.student_model,
        student_dtype=args.student_dtype,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        device=args.device,
    )


def build_model(args: argparse.Namespace, *, action_dims: tuple[int, ...], hidden_size: int | None) -> tuple[nn.Module, dict[str, Any]]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = script_84.torch_dtype_from_name(args.ae_dtype)
    if hidden_size is None:
        hidden_size = 2048
    action_in_proj, action_out_proj = script_94.instantiate_action_projections(
        config_path=args.teacher_config,
        action_dims=action_dims,
        hidden_size=int(hidden_size),
        dtype=dtype,
        device=device,
    )
    script_84.set_module_requires_grad(action_in_proj, True)
    script_84.set_module_requires_grad(action_out_proj, True)

    if args.variant == "proj_mlp":
        model = ProjectionMlpHead(
            action_in_proj=action_in_proj,
            hidden_size=int(hidden_size),
            out_dim=int(action_dims[-1]),
            width=int(args.head_width),
        ).to(device=device, dtype=dtype).train()
        return model, {
            "variant_kind": "projection_mlp_head",
            "expert_layers": 0,
            "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        }

    if not args.variant.startswith("expert_"):
        raise ValueError(f"Unknown variant: {args.variant}")
    parts = args.variant.split("_")
    n_layers = int(parts[1])
    no_attention = "noattn" in parts[2:]
    student, _tokenizer, _processor, base_model = script_84.load_student(train_args(args))
    expert = script_84.build_student_backbone_expert(
        student=student,
        dtype=dtype,
        device=str(device),
        attn_implementation=("sdpa" if args.attn_implementation != "eager" else "eager"),
    )
    del student
    expert = truncate_expert(expert, n_layers).train()
    if no_attention:
        disable_self_attention(expert)
    model = ExpertFMModel(expert=expert, action_in_proj=action_in_proj, action_out_proj=action_out_proj).train()
    return model, {
        "variant_kind": "expert",
        "student_base_model": str(base_model),
        "expert_layers": n_layers,
        "self_attention": "zero_update" if no_attention else "enabled",
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def optimizer_for(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if isinstance(model, ProjectionMlpHead):
        groups: list[dict[str, Any]] = []
        groups.extend(split_decay_params(model.action_in_proj, float(args.proj_lr), float(args.weight_decay)))
        groups.extend(split_decay_params(model.head, float(args.head_lr), float(args.weight_decay)))
        return torch.optim.AdamW(groups)
    if isinstance(model, ExpertFMModel):
        groups = []
        groups.extend(split_decay_params(model.expert, float(args.expert_lr), float(args.weight_decay)))
        groups.extend(split_decay_params(model.action_in_proj, float(args.proj_lr), float(args.weight_decay)))
        groups.extend(split_decay_params(model.action_out_proj, float(args.proj_lr), float(args.weight_decay)))
        return torch.optim.AdamW(groups)
    return torch.optim.AdamW(model.parameters(), lr=float(args.head_lr), weight_decay=float(args.weight_decay))


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
        SimpleNamespace(
            corpus_jsonl=args.corpus_jsonl,
            split=args.split,
            num_samples=max(int(args.num_samples), int(args.sample_index) + 1),
        )
    )
    item = items[int(args.sample_index)]
    action_space = script_94.instantiate_action_space(args.teacher_config, device="cpu")
    action_dims = tuple(action_space.get_action_space_dims())
    target_action, target_meta = script_94.load_target_action(
        item=item,
        action_space=action_space,
        target_source=str(args.target_source),
        device=torch.device("cpu"),
    )
    model, model_meta = build_model(args, action_dims=action_dims, hidden_size=2048)
    model.to(device=device, dtype=dtype).train()
    optimizer = optimizer_for(model, args)
    scheduler = make_scheduler(optimizer, args)

    n_tokens = int(action_dims[0])
    position_template = torch.arange(n_tokens, dtype=torch.long, device=device).view(1, 1, -1).repeat(3, 1, 1)
    x1 = target_action.to(device=device, dtype=dtype)
    log_steps = {1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 500, 750, int(args.steps)}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        model.train()
        x_t, t, target_v = sample_batch(
            x1=x1,
            batch_size=int(args.draws_per_step),
            sampler=str(args.train_timestep_sampler),
            device=device,
            dtype=dtype,
        )
        position_ids = position_template.repeat(1, int(args.draws_per_step), 1)
        pred = model(x_t, t, position_ids)
        loss = F.mse_loss(pred.float(), target_v.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step in log_steps:
            eval_stats = evaluate(
                model=model,
                x1=x1,
                action_dims=action_dims,
                args=args,
                device=device,
                dtype=dtype,
            )
            row = {
                "event": "h1_ablation_step",
                "variant": args.variant,
                "step": step,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "train_loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                **eval_stats,
            }
            print(json.dumps(row), flush=True)
            rows.append(row)

    final_eval = evaluate(model=model, x1=x1, action_dims=action_dims, args=args, device=device, dtype=dtype)
    summary = {
        "event": "h1_ablation_ladder",
        "created_at_unix": time.time(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sample_id": item["sample_id"],
        "target_meta": target_meta | {
            "target_action_shape": list(target_action.shape),
            "target_action_abs_mean": float(target_action.abs().mean()),
            "target_action_rms": float(torch.sqrt((target_action * target_action).mean())),
        },
        "model_meta": model_meta,
        "final_eval": final_eval,
        "rows": rows,
    }
    out_dir = args.output_dir / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path), "final_eval": final_eval}), flush=True)


if __name__ == "__main__":
    main()
