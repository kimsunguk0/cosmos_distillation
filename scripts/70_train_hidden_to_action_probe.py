#!/usr/bin/env python3
"""Train small frozen-feature probes from student hidden state to teacher action trajectories."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_readiness_module():
    path = PROJECT_ROOT / "scripts" / "67_eval_backbone_readiness.py"
    spec = importlib.util.spec_from_file_location("backbone_readiness_67", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_readiness_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--prefix-type", default="teacher_prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-model", default="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 64 * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 64, 2)


class MLPProbe(nn.Module):
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


def load_split(feature_root: Path, checkpoint_name: str, prefix_type: str, split_name: str, max_samples: int = 0) -> dict[str, Any]:
    split_dir = feature_root / checkpoint_name / prefix_type / split_name
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: dict[str, list[torch.Tensor]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for shard_path_raw in manifest.get("shards") or []:
        shard_path = Path(shard_path_raw)
        if not shard_path.is_absolute():
            shard_path = PROJECT_ROOT / shard_path
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_rows = list(payload.get("rows") or [])
        take = len(shard_rows)
        if max_samples > 0:
            remaining = max(max_samples - len(rows), 0)
            if remaining <= 0:
                break
            take = min(take, remaining)
        rows.extend(shard_rows[:take])
        for key in (
            "hidden_feature",
            "ego_feature",
            "ego_history_xyz",
            "ego_history_rot",
            "target_action",
            "target_traj",
            "gt_future",
        ):
            if key in payload:
                arrays[key].append(payload[key][:take].clone())
    if not rows:
        raise RuntimeError(f"No rows loaded for {checkpoint_name}/{prefix_type}/{split_name}")
    out: dict[str, Any] = {"rows": rows, "manifest": manifest}
    for key, values in arrays.items():
        out[key] = torch.cat(values, dim=0)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ade_fde(pred_xyz: torch.Tensor, target_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(pred_xyz.shape[1], target_xyz.shape[1])
    dist = torch.linalg.norm(pred_xyz[:, :n, :2] - target_xyz[:, :n, :2], dim=-1)
    return dist.mean(dim=1), dist[:, -1]


def constant_velocity_prediction(history_xyz: torch.Tensor, *, dt: float = 0.1) -> torch.Tensor:
    last = history_xyz[:, -1, :2]
    prev = history_xyz[:, -2, :2] if history_xyz.shape[1] >= 2 else torch.zeros_like(last)
    velocity = (last - prev) / float(dt)
    steps = torch.arange(1, 65, dtype=history_xyz.dtype, device=history_xyz.device).view(1, 64, 1)
    xy = last[:, None, :] + velocity[:, None, :] * steps * float(dt)
    z = torch.zeros((history_xyz.shape[0], 64, 1), dtype=history_xyz.dtype, device=history_xyz.device)
    return torch.cat([xy, z], dim=-1)


@torch.no_grad()
def action_to_traj(decoder: Any, action: torch.Tensor, history_xyz: torch.Tensor, history_rot: torch.Tensor) -> torch.Tensor:
    pred_xyz, _pred_rot = decoder.action_space.action_to_traj(
        action.float(),
        history_xyz.float(),
        history_rot.float(),
    )
    return pred_xyz.float()


@torch.no_grad()
def evaluate_action_model(
    *,
    model: nn.Module,
    features: torch.Tensor,
    target_action: torch.Tensor,
    target_traj: torch.Tensor,
    gt_future: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    rows: list[dict[str, Any]],
    decoder: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    all_action_losses: list[float] = []
    all_ade: list[float] = []
    all_fde: list[float] = []
    all_gt_ade: list[float] = []
    all_gt_fde: list[float] = []
    per_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pred_norms: list[float] = []
    target_norms: list[float] = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        x = features[start:end].to(device=device, dtype=torch.float32)
        ta = target_action[start:end].to(device=device, dtype=torch.float32)
        tt = target_traj[start:end].to(device=device, dtype=torch.float32)
        gt = gt_future[start:end].to(device=device, dtype=torch.float32)
        hx = ego_history_xyz[start:end].to(device=device, dtype=torch.float32)
        hr = ego_history_rot[start:end].to(device=device, dtype=torch.float32)
        pred_action = model(x)
        action_loss = F.smooth_l1_loss(pred_action, ta, reduction="none").mean(dim=(1, 2))
        pred_traj = action_to_traj(decoder, pred_action, hx, hr)
        ade, fde = ade_fde(pred_traj, tt)
        gt_ade, gt_fde = ade_fde(pred_traj, gt)
        all_action_losses.extend(float(value) for value in action_loss.detach().cpu().tolist())
        all_ade.extend(float(value) for value in ade.detach().cpu().tolist())
        all_fde.extend(float(value) for value in fde.detach().cpu().tolist())
        all_gt_ade.extend(float(value) for value in gt_ade.detach().cpu().tolist())
        all_gt_fde.extend(float(value) for value in gt_fde.detach().cpu().tolist())
        pred_norms.extend(float(value) for value in torch.linalg.norm(pred_action, dim=-1).mean(dim=-1).detach().cpu().tolist())
        target_norms.extend(float(value) for value in torch.linalg.norm(ta, dim=-1).mean(dim=-1).detach().cpu().tolist())
        for local_index, row in enumerate(rows[start:end]):
            bucket = str(row.get("bucket") or "unknown")
            per_bucket[bucket]["ade"].append(float(ade[local_index].detach().cpu().item()))
            per_bucket[bucket]["fde"].append(float(fde[local_index].detach().cpu().item()))
    return summarize_eval(all_action_losses, all_ade, all_fde, all_gt_ade, all_gt_fde, pred_norms, target_norms, per_bucket)


@torch.no_grad()
def evaluate_constant_velocity(split: dict[str, Any]) -> dict[str, Any]:
    pred = constant_velocity_prediction(split["ego_history_xyz"].float())
    target = split["target_traj"].float()
    gt = split["gt_future"].float()
    ade, fde = ade_fde(pred, target)
    gt_ade, gt_fde = ade_fde(pred, gt)
    per_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(split["rows"]):
        bucket = str(row.get("bucket") or "unknown")
        per_bucket[bucket]["ade"].append(float(ade[index].item()))
        per_bucket[bucket]["fde"].append(float(fde[index].item()))
    return summarize_eval([], ade.tolist(), fde.tolist(), gt_ade.tolist(), gt_fde.tolist(), [], [], per_bucket)


def summarize(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {"mean": float(clean.mean()), "p50": float(np.percentile(clean, 50)), "p95": float(np.percentile(clean, 95))}


def summarize_eval(
    action_losses: list[float],
    ade: list[float],
    fde: list[float],
    gt_ade: list[float],
    gt_fde: list[float],
    pred_norms: list[float],
    target_norms: list[float],
    per_bucket: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    return {
        "num_samples": len(ade),
        "action_smooth_l1": summarize(action_losses),
        "ade_vs_teacher_action": summarize(ade),
        "fde_vs_teacher_action": summarize(fde),
        "ade_vs_gt": summarize(gt_ade),
        "fde_vs_gt": summarize(gt_fde),
        "pred_action_norm": summarize(pred_norms),
        "target_action_norm": summarize(target_norms),
        "buckets": {
            bucket: {
                "count": len(metrics.get("ade", [])),
                "ade_vs_teacher_action": summarize(metrics.get("ade", [])),
                "fde_vs_teacher_action": summarize(metrics.get("fde", [])),
            }
            for bucket, metrics in sorted(per_bucket.items())
        },
    }


def train_probe(
    *,
    name: str,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_data: dict[str, Any],
    decoder: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    dataset = TensorDataset(train_x.float(), train_y.float())
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    history: list[dict[str, Any]] = []
    best_state = None
    best_score = float("inf")
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
        val_metrics = evaluate_action_model(
            model=model,
            features=val_data["features"],
            target_action=val_data["target_action"],
            target_traj=val_data["target_traj"],
            gt_future=val_data["gt_future"],
            ego_history_xyz=val_data["ego_history_xyz"],
            ego_history_rot=val_data["ego_history_rot"],
            rows=val_data["rows"],
            decoder=decoder,
            device=device,
            batch_size=int(args.batch_size),
        )
        score = float(val_metrics["ade_vs_teacher_action"]["mean"] or float("inf"))
        history.append({"epoch": epoch, "train_action_loss": float(np.mean(losses)), "val_ade": score, "val_fde": val_metrics["fde_vs_teacher_action"]["mean"]})
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"event": "probe_epoch", "probe": name, "epoch": epoch, "train_loss": history[-1]["train_action_loss"], "val_ade": score}), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val_ade": best_score}


def compact_row(model_name: str, split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model_name,
        "split": split_name,
        "num_samples": metrics.get("num_samples"),
        "ade_mean": (metrics.get("ade_vs_teacher_action") or {}).get("mean"),
        "fde_mean": (metrics.get("fde_vs_teacher_action") or {}).get("mean"),
        "ade_p95": (metrics.get("ade_vs_teacher_action") or {}).get("p95"),
        "fde_p95": (metrics.get("fde_vs_teacher_action") or {}).get("p95"),
        "gt_ade_mean": (metrics.get("ade_vs_gt") or {}).get("mean"),
        "gt_fde_mean": (metrics.get("fde_vs_gt") or {}).get("mean"),
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    train = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_train", args.max_train_samples)
    val = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_val", args.max_val_samples)
    test = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_test", args.max_test_samples)
    decoder_config = helpers.resolve_traj_tokenizer_config_path(args.student_model)
    if decoder_config is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {args.student_model}")
    decoder = helpers.TrajectoryTokenDecoder(config_path=decoder_config)

    def pack(data: dict[str, Any], key: str) -> dict[str, Any]:
        return {
            "features": data[key].float(),
            "target_action": data["target_action"].float(),
            "target_traj": data["target_traj"].float(),
            "gt_future": data["gt_future"].float(),
            "ego_history_xyz": data["ego_history_xyz"].float(),
            "ego_history_rot": data["ego_history_rot"].float(),
            "rows": data["rows"],
        }

    results: dict[str, Any] = {
        "schema_version": "hidden_to_action_probe_results_v1",
        "checkpoint_name": args.checkpoint_name,
        "prefix_type": args.prefix_type,
        "feature_root": str(args.feature_root),
        "output_dir": str(args.output_dir),
        "splits": {
            "probe_train": {"num_samples": len(train["rows"])},
            "probe_val": {"num_samples": len(val["rows"])},
            "probe_test": {"num_samples": len(test["rows"])},
        },
        "models": {},
    }

    # Baseline 1: constant velocity.
    results["models"]["constant_velocity"] = {
        "probe_val": evaluate_constant_velocity(val),
        "probe_test": evaluate_constant_velocity(test),
    }

    # Baseline 2: ego-only MLP.
    ego_model = MLPProbe(int(train["ego_feature"].shape[1]), int(args.hidden_dim), float(args.dropout))
    ego_model, ego_train_info = train_probe(
        name="ego_only_mlp",
        model=ego_model,
        train_x=train["ego_feature"],
        train_y=train["target_action"],
        val_data=pack(val, "ego_feature"),
        decoder=decoder,
        args=args,
        device=device,
    )
    results["models"]["ego_only_mlp"] = {
        "train_info": ego_train_info,
        "probe_val": evaluate_action_model(model=ego_model, decoder=decoder, device=device, batch_size=int(args.batch_size), **pack(val, "ego_feature")),
        "probe_test": evaluate_action_model(model=ego_model, decoder=decoder, device=device, batch_size=int(args.batch_size), **pack(test, "ego_feature")),
    }
    torch.save({"state_dict": ego_model.state_dict(), "input_dim": int(train["ego_feature"].shape[1]), "kind": "mlp"}, args.output_dir / "ego_only_mlp.pt")

    # Hidden linear probe.
    linear = LinearProbe(int(train["hidden_feature"].shape[1]))
    linear, linear_info = train_probe(
        name="hidden_linear",
        model=linear,
        train_x=train["hidden_feature"],
        train_y=train["target_action"],
        val_data=pack(val, "hidden_feature"),
        decoder=decoder,
        args=args,
        device=device,
    )
    results["models"]["hidden_linear"] = {
        "train_info": linear_info,
        "probe_val": evaluate_action_model(model=linear, decoder=decoder, device=device, batch_size=int(args.batch_size), **pack(val, "hidden_feature")),
        "probe_test": evaluate_action_model(model=linear, decoder=decoder, device=device, batch_size=int(args.batch_size), **pack(test, "hidden_feature")),
    }
    torch.save({"state_dict": linear.state_dict(), "input_dim": int(train["hidden_feature"].shape[1]), "kind": "linear"}, args.output_dir / "hidden_linear.pt")

    # Hidden 2-layer MLP probe.
    mlp = MLPProbe(int(train["hidden_feature"].shape[1]), int(args.hidden_dim), float(args.dropout))
    mlp, mlp_info = train_probe(
        name="hidden_mlp",
        model=mlp,
        train_x=train["hidden_feature"],
        train_y=train["target_action"],
        val_data=pack(val, "hidden_feature"),
        decoder=decoder,
        args=args,
        device=device,
    )
    results["models"]["hidden_mlp"] = {
        "train_info": mlp_info,
        "probe_val": evaluate_action_model(model=mlp, decoder=decoder, device=device, batch_size=int(args.batch_size), **pack(val, "hidden_feature")),
        "probe_test": evaluate_action_model(model=mlp, decoder=decoder, device=device, batch_size=int(args.batch_size), **pack(test, "hidden_feature")),
    }
    torch.save({"state_dict": mlp.state_dict(), "input_dim": int(train["hidden_feature"].shape[1]), "kind": "mlp"}, args.output_dir / "hidden_mlp.pt")

    rows = []
    for model_name, model_results in results["models"].items():
        for split_name in ("probe_val", "probe_test"):
            rows.append(compact_row(model_name, split_name, model_results[split_name]))
    results["compact_table"] = rows
    results["elapsed_sec"] = round(time.time() - t0, 3)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "probe_results.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("model,split,num_samples,ade_mean,fde_mean,ade_p95,fde_p95,gt_ade_mean,gt_fde_mean\n")
        for row in rows:
            handle.write(",".join(str(row.get(key, "")) for key in ("model", "split", "num_samples", "ade_mean", "fde_mean", "ade_p95", "fde_p95", "gt_ade_mean", "gt_fde_mean")) + "\n")
    print(json.dumps({"event": "probe_done", "summary": str(summary_path), "csv": str(csv_path), "elapsed_sec": results["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
