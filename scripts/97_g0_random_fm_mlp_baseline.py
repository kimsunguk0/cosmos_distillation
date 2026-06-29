#!/usr/bin/env python3
"""G0: independent MLP baseline for the 1-sample random FM task.

This intentionally avoids the AE/expert stack. It fixes one target action x1,
samples random x0/t exactly like the AE training path, and trains a small MLP
to predict target_v = x1 - x0 from (x_t, t).
"""
from __future__ import annotations

import argparse
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


def import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


script_84 = import_script(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py", "script_84")
script_94 = import_script(PROJECT_ROOT / "scripts" / "94_diagnose_fm_collapse.py", "script_94")


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_TEACHER_CONFIG = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B" / "config.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "action_expert" / "g0_random_fm_mlp_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--target-source", choices=("teacher", "gt"), default="teacher")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-timestep-sampler", choices=("uniform", "beta"), default="beta")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-draws", type=int, default=8192)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--t-freqs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


class RandomFMMlp(nn.Module):
    def __init__(self, *, x_dim: int, width: int, depth: int, t_freqs: int) -> None:
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(100.0), int(t_freqs)))
        self.register_buffer("freqs", freqs)
        t_dim = 1 + 2 * int(t_freqs)
        layers: list[nn.Module] = []
        in_dim = int(x_dim) + t_dim
        for layer_idx in range(int(depth)):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else int(width), int(width)))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(int(width), int(x_dim)))
        self.net = nn.Sequential(*layers)

    def t_embed(self, t: torch.Tensor) -> torch.Tensor:
        t_flat = t.detach() if not t.requires_grad else t
        t_flat = t_flat.reshape(t.shape[0], 1).float()
        angles = t_flat * self.freqs.view(1, -1).float()
        return torch.cat([t_flat, torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_flat = x_t.reshape(x_t.shape[0], -1).float()
        inp = torch.cat([x_flat, self.t_embed(t).to(device=x_flat.device)], dim=-1)
        return self.net(inp).view_as(x_t)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x1_b = x1.unsqueeze(0).expand(int(batch_size), *x1.shape)
    x0 = torch.randn_like(x1_b)
    t = script_84.sample_fm_timesteps(
        batch_size=int(batch_size),
        sampler=str(sampler),
        device=device,
        dtype=x1.dtype,
    )
    x_t = (1.0 - t) * x0 + t * x1_b
    target_v = x1_b - x0
    return x_t, t, target_v, x0


@torch.no_grad()
def evaluate(model: nn.Module, *, x1: torch.Tensor, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model.eval()
    x_t, t, target_v, x0 = sample_batch(
        x1=x1,
        batch_size=int(args.eval_draws),
        sampler=str(args.train_timestep_sampler),
        device=device,
    )
    pred = model(x_t, t)
    zero = torch.zeros_like(target_v)
    oracle = x1.unsqueeze(0).expand_as(x_t) - (x_t - t * x1.unsqueeze(0)) / (1.0 - t).clamp_min(1e-6)
    return {
        "model": vector_stats(pred, target_v),
        "zero_pred": vector_stats(zero, target_v),
        "oracle_formula": vector_stats(oracle, target_v),
        "t_mean": float(t.float().mean().cpu()),
        "t_min": float(t.float().min().cpu()),
        "t_p50": float(torch.quantile(t.float().flatten().cpu(), 0.50)),
        "t_p95": float(torch.quantile(t.float().flatten().cpu(), 0.95)),
        "x0_abs_mean": float(x0.detach().abs().mean().cpu()),
        "x_t_abs_mean": float(x_t.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target_v.detach().abs().mean().cpu()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    items = script_84.select_items(
        SimpleNamespace(corpus_jsonl=args.corpus_jsonl, split=args.split, num_samples=max(int(args.num_samples), int(args.sample_index) + 1))
    )
    item = items[int(args.sample_index)]
    action_space = script_94.instantiate_action_space(args.teacher_config, device="cpu")
    target_action, target_meta = script_94.load_target_action(
        item=item,
        action_space=action_space,
        target_source=str(args.target_source),
        device=torch.device("cpu"),
    )
    x1 = target_action.to(device=device, dtype=torch.float32)
    x_dim = int(x1.numel())
    model = RandomFMMlp(x_dim=x_dim, width=int(args.width), depth=int(args.depth), t_freqs=int(args.t_freqs)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    log_steps = {1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 500, 750, int(args.steps)}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        model.train()
        x_t, t, target_v, _x0 = sample_batch(
            x1=x1,
            batch_size=int(args.batch_size),
            sampler=str(args.train_timestep_sampler),
            device=device,
        )
        pred = model(x_t, t)
        loss = F.mse_loss(pred.float(), target_v.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step in log_steps:
            eval_stats = evaluate(model, x1=x1, args=args, device=device)
            row = {
                "event": "g0_mlp_step",
                "step": step,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "train_loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                **eval_stats["model"],
            }
            print(json.dumps(row), flush=True)
            rows.append(row)

    final_eval = evaluate(model, x1=x1, args=args, device=device)
    summary = {
        "event": "g0_random_fm_mlp_baseline",
        "created_at_unix": time.time(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sample_id": item["sample_id"],
        "target_meta": target_meta | {
            "target_action_shape": list(target_action.shape),
            "target_action_abs_mean": float(target_action.abs().mean()),
            "target_action_rms": float(torch.sqrt((target_action * target_action).mean())),
        },
        "model_param_count": int(sum(param.numel() for param in model.parameters())),
        "final_eval": final_eval,
        "rows": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path), "final_model": final_eval["model"]}), flush=True)


if __name__ == "__main__":
    main()
