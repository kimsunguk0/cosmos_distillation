#!/usr/bin/env python3
"""Train a learned-query cross-attention action probe over prefix hidden sequences."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--prefix-type", default="teacher_prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-model", default="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--attn-dim", type=int, default=512)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class PrefixSeqDataset(Dataset):
    def __init__(self, split: dict[str, Any]) -> None:
        self.split = split

    def __len__(self) -> int:
        return int(self.split["prefix_hidden"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "prefix_hidden": self.split["prefix_hidden"][index],
            "prefix_mask": self.split["prefix_mask"][index],
            "target_action": self.split["target_action"][index],
            "target_traj": self.split["target_traj"][index],
            "ego_history_xyz": self.split["ego_history_xyz"][index],
            "ego_history_rot": self.split["ego_history_rot"][index],
            "gt_future": self.split["gt_future"][index],
        }


class KVQueryActionProbe(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int, num_queries: int, num_heads: int, mlp_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, attn_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, attn_dim) * 0.02)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(num_queries * attn_dim),
            nn.Linear(num_queries * attn_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 64 * 2),
        )

    def forward(self, prefix_hidden: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        keys = self.in_proj(self.input_norm(prefix_hidden.float()))
        queries = self.queries.unsqueeze(0).expand(prefix_hidden.shape[0], -1, -1)
        key_padding_mask = ~prefix_mask.bool()
        attended, _ = self.attn(queries, keys, keys, key_padding_mask=key_padding_mask, need_weights=False)
        return self.out(attended.flatten(1)).view(-1, 64, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(feature_root: Path, checkpoint_name: str, prefix_type: str, split_name: str) -> dict[str, Any]:
    split_dir = feature_root / checkpoint_name / prefix_type / split_name
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing prefix sequence manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: dict[str, list[torch.Tensor]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for shard_path_raw in manifest.get("shards") or []:
        shard_path = Path(shard_path_raw)
        if not shard_path.is_absolute():
            shard_path = PROJECT_ROOT / shard_path
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        rows.extend(list(payload.get("rows") or []))
        for key in ("prefix_hidden", "prefix_mask", "target_action", "target_traj", "ego_history_xyz", "ego_history_rot", "gt_future"):
            arrays[key].append(payload[key].clone())
    out: dict[str, Any] = {"rows": rows, "manifest": manifest}
    for key, values in arrays.items():
        out[key] = torch.cat(values, dim=0)
    return out


@torch.no_grad()
def evaluate(model: nn.Module, split: dict[str, Any], decoder: Any, device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval().to(device)
    dataset = PrefixSeqDataset(split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    action_losses: list[float] = []
    ade_all: list[float] = []
    fde_all: list[float] = []
    gt_ade_all: list[float] = []
    gt_fde_all: list[float] = []
    for batch in loader:
        prefix = batch["prefix_hidden"].to(device=device, dtype=torch.float32)
        mask = batch["prefix_mask"].to(device=device, dtype=torch.bool)
        target_action = batch["target_action"].to(device=device, dtype=torch.float32)
        target_traj = batch["target_traj"].to(device=device, dtype=torch.float32)
        ego_xyz = batch["ego_history_xyz"].to(device=device, dtype=torch.float32)
        ego_rot = batch["ego_history_rot"].to(device=device, dtype=torch.float32)
        gt = batch["gt_future"].to(device=device, dtype=torch.float32)
        pred_action = model(prefix, mask)
        action_loss = F.smooth_l1_loss(pred_action, target_action, reduction="none").mean(dim=(1, 2))
        pred_traj = probe70.action_to_traj(decoder, pred_action, ego_xyz, ego_rot)
        ade, fde = probe70.ade_fde(pred_traj, target_traj)
        gt_ade, gt_fde = probe70.ade_fde(pred_traj, gt)
        action_losses.extend(float(v) for v in action_loss.detach().cpu().tolist())
        ade_all.extend(float(v) for v in ade.detach().cpu().tolist())
        fde_all.extend(float(v) for v in fde.detach().cpu().tolist())
        gt_ade_all.extend(float(v) for v in gt_ade.detach().cpu().tolist())
        gt_fde_all.extend(float(v) for v in gt_fde.detach().cpu().tolist())
    return probe70.summarize_eval(action_losses, ade_all, fde_all, gt_ade_all, gt_fde_all, [], [], defaultdict(lambda: defaultdict(list)))


def compact(split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
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
    train = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_train")
    val = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_val")
    test = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_test")
    decoder_config = probe70.helpers.resolve_traj_tokenizer_config_path(args.student_model)
    if decoder_config is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {args.student_model}")
    decoder = probe70.helpers.TrajectoryTokenDecoder(config_path=decoder_config)
    input_dim = int(train["prefix_hidden"].shape[-1])
    model = KVQueryActionProbe(input_dim, int(args.attn_dim), int(args.num_queries), int(args.num_heads), int(args.mlp_hidden_dim), float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = DataLoader(PrefixSeqDataset(train), batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=False)
    best_state = None
    best_score = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            prefix = batch["prefix_hidden"].to(device=device, dtype=torch.float32)
            mask = batch["prefix_mask"].to(device=device, dtype=torch.bool)
            target = batch["target_action"].to(device=device, dtype=torch.float32)
            pred = model(prefix, mask)
            loss = F.smooth_l1_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate(model, val, decoder, device, int(args.batch_size))
        val_ade = float(val_metrics["ade_vs_teacher_action"]["mean"])
        val_fde = float(val_metrics["fde_vs_teacher_action"]["mean"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_ade": val_ade, "val_fde": val_fde}
        history.append(row)
        if val_ade < best_score:
            best_score = val_ade
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"event": "kv_probe_epoch", **row}), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics = evaluate(model, val, decoder, device, int(args.batch_size))
    test_metrics = evaluate(model, test, decoder, device, int(args.batch_size))
    rows = [compact("probe_val", val_metrics), compact("probe_test", test_metrics)]
    csv_path = args.output_dir / "kv_aware_probe_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "kv_aware_action_probe_v1",
        "checkpoint_name": args.checkpoint_name,
        "prefix_type": args.prefix_type,
        "feature_root": str(args.feature_root),
        "splits": {"probe_train": len(train["rows"]), "probe_val": len(val["rows"]), "probe_test": len(test["rows"])},
        "max_seq_tokens": int(train["prefix_hidden"].shape[1]),
        "history": history,
        "probe_val": val_metrics,
        "probe_test": test_metrics,
        "compact": rows,
        "csv": str(csv_path),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    torch.save({"state_dict": model.state_dict(), "config": vars(args), "input_dim": input_dim}, args.output_dir / "kv_aware_action_probe.pt")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "kv_aware_probe_done", "summary": str(args.output_dir / "summary.json"), "csv": str(csv_path)}), flush=True)


if __name__ == "__main__":
    main()
