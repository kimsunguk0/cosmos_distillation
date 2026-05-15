#!/usr/bin/env python3
"""Train h_cot_end-only small action heads on frozen hidden features."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(name: str, rel_path: str):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe70 = _load_module("hidden_to_action_probe_70", "scripts/70_train_hidden_to_action_probe.py")
abl73 = _load_module("hidden_to_action_ablation_73", "scripts/73_eval_hidden_to_action_ablation.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=PROJECT_ROOT / "outputs/probe_cache/hidden_to_action_v1")
    parser.add_argument("--checkpoint-name", default="bp3_200k_final")
    parser.add_argument("--prefix-type", default="teacher_prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-model", default="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
    parser.add_argument("--feature-name", choices=("h_cot_end", "h_traj_start", "h_prefix_mean_last16"), default="h_cot_end")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


class MLP2ActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64 * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 64, 2)


class MLP4ActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64 * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 64, 2)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualActionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.blocks = nn.Sequential(ResidualBlock(hidden_dim, dropout), ResidualBlock(hidden_dim, dropout), ResidualBlock(hidden_dim, dropout))
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64 * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.in_proj(x))
        return self.out(h).view(-1, 64, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_tensor(split: dict[str, Any], feature_name: str) -> torch.Tensor:
    return split[feature_name].float()


def pack(split: dict[str, Any], x: torch.Tensor) -> dict[str, Any]:
    return {
        "features": x.float(),
        "target_action": split["target_action"].float(),
        "target_traj": split["target_traj"].float(),
        "gt_future": split["gt_future"].float(),
        "ego_history_xyz": split["ego_history_xyz"].float(),
        "ego_history_rot": split["ego_history_rot"].float(),
        "rows": split["rows"],
    }


def train_head(
    *,
    name: str,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_pack: dict[str, Any],
    decoder: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = DataLoader(TensorDataset(train_x.float(), train_y.float()), batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    best_state = None
    best_score = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        metrics = probe70.evaluate_action_model(model=model, decoder=decoder, device=device, batch_size=int(args.batch_size), **val_pack)
        ade = float(metrics["ade_vs_teacher_action"]["mean"])
        fde = float(metrics["fde_vs_teacher_action"]["mean"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_ade": ade, "val_fde": fde}
        history.append(row)
        if ade < best_score:
            best_score = ade
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"event": "head_epoch", "head": name, **row}), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val_ade": best_score}


def compact(head: str, split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "head": head,
        "split": split_name,
        "num_samples": metrics.get("num_samples"),
        "ade_mean": (metrics.get("ade_vs_teacher_action") or {}).get("mean"),
        "fde_mean": (metrics.get("fde_vs_teacher_action") or {}).get("mean"),
        "ade_p95": (metrics.get("ade_vs_teacher_action") or {}).get("p95"),
        "fde_p95": (metrics.get("fde_vs_teacher_action") or {}).get("p95"),
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    t0 = time.time()

    train = abl73.load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_train", args.max_train_samples)
    val = abl73.load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_val", args.max_val_samples)
    test = abl73.load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_test", args.max_test_samples)
    decoder_config = probe70.helpers.resolve_traj_tokenizer_config_path(args.student_model)
    if decoder_config is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {args.student_model}")
    decoder = probe70.helpers.TrajectoryTokenDecoder(config_path=decoder_config)

    train_x = feature_tensor(train, args.feature_name)
    val_x = feature_tensor(val, args.feature_name)
    test_x = feature_tensor(test, args.feature_name)
    input_dim = int(train_x.shape[1])
    heads: dict[str, nn.Module] = {
        "mlp2": MLP2ActionHead(input_dim, int(args.hidden_dim), float(args.dropout)),
        "mlp4": MLP4ActionHead(input_dim, int(args.hidden_dim), float(args.dropout)),
        "residual3": ResidualActionHead(input_dim, int(args.hidden_dim), float(args.dropout)),
    }

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "schema_version": "hcot_action_head_smoke_v1",
        "checkpoint_name": args.checkpoint_name,
        "prefix_type": args.prefix_type,
        "feature_name": args.feature_name,
        "splits": {"probe_train": len(train["rows"]), "probe_val": len(val["rows"]), "probe_test": len(test["rows"])},
        "heads": {},
    }
    for name, model in heads.items():
        model, info = train_head(
            name=name,
            model=model,
            train_x=train_x,
            train_y=train["target_action"].float(),
            val_pack=pack(val, val_x),
            decoder=decoder,
            args=args,
            device=device,
        )
        val_metrics = probe70.evaluate_action_model(model=model, decoder=decoder, device=device, batch_size=int(args.batch_size), **pack(val, val_x))
        test_metrics = probe70.evaluate_action_model(model=model, decoder=decoder, device=device, batch_size=int(args.batch_size), **pack(test, test_x))
        details["heads"][name] = {"train_info": info, "probe_val": val_metrics, "probe_test": test_metrics}
        rows.append(compact(name, "probe_val", val_metrics))
        rows.append(compact(name, "probe_test", test_metrics))
        torch.save(
            {
                "state_dict": model.state_dict(),
                "head": name,
                "feature_name": args.feature_name,
                "input_dim": input_dim,
                "hidden_dim": int(args.hidden_dim),
            },
            args.output_dir / f"{name}_{args.feature_name}_action_head.pt",
        )

    csv_path = args.output_dir / "h_cot_end_action_head_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    details["compact"] = rows
    details["elapsed_sec"] = round(time.time() - t0, 3)
    details["csv"] = str(csv_path)
    (args.output_dir / "summary.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps({"event": "hcot_action_head_done", "summary": str(args.output_dir / "summary.json"), "csv": str(csv_path)}), flush=True)


if __name__ == "__main__":
    main()
