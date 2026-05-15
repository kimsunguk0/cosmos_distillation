#!/usr/bin/env python3
"""Evaluate h_cot_end action heads across teacher-prefix and student-free prefixes."""

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
head75 = _load_module("hcot_action_head_75", "scripts/75_train_hcot_action_head_smoke.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=PROJECT_ROOT / "outputs/probe_cache/hidden_to_action_v1")
    parser.add_argument("--checkpoint-name", default="bp3_200k_final")
    parser.add_argument("--teacher-head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-model", default="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
    parser.add_argument("--feature-name", default="h_cot_end")
    parser.add_argument("--epochs", type=int, default=60)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pack(split: dict[str, Any], features: torch.Tensor) -> dict[str, Any]:
    return {
        "features": features.float(),
        "target_action": split["target_action"].float(),
        "target_traj": split["target_traj"].float(),
        "gt_future": split["gt_future"].float(),
        "ego_history_xyz": split["ego_history_xyz"].float(),
        "ego_history_rot": split["ego_history_rot"].float(),
        "rows": split["rows"],
    }


def compact(head: str, train_prefix: str, eval_prefix: str, split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "head": head,
        "train_prefix": train_prefix,
        "eval_prefix": eval_prefix,
        "split": split_name,
        "num_samples": metrics.get("num_samples"),
        "ade_mean": (metrics.get("ade_vs_teacher_action") or {}).get("mean"),
        "fde_mean": (metrics.get("fde_vs_teacher_action") or {}).get("mean"),
        "ade_p95": (metrics.get("ade_vs_teacher_action") or {}).get("p95"),
        "fde_p95": (metrics.get("fde_vs_teacher_action") or {}).get("p95"),
    }


def build_head(kind: str, input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if kind == "mlp2":
        return head75.MLP2ActionHead(input_dim, hidden_dim, dropout)
    if kind == "mlp4":
        return head75.MLP4ActionHead(input_dim, hidden_dim, dropout)
    if kind == "residual3":
        return head75.ResidualActionHead(input_dim, hidden_dim, dropout)
    raise ValueError(f"Unsupported head kind: {kind}")


def load_saved_head(path: Path, *, dropout: float) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = str(payload.get("head"))
    input_dim = int(payload.get("input_dim"))
    hidden_dim = int(payload.get("hidden_dim"))
    model = build_head(kind, input_dim, hidden_dim, dropout)
    model.load_state_dict(payload["state_dict"])
    return model


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    t0 = time.time()

    teacher_train = abl73.load_split(args.feature_root, args.checkpoint_name, "teacher_prefix", "probe_train", args.max_train_samples)
    teacher_val = abl73.load_split(args.feature_root, args.checkpoint_name, "teacher_prefix", "probe_val", args.max_val_samples)
    teacher_test = abl73.load_split(args.feature_root, args.checkpoint_name, "teacher_prefix", "probe_test", args.max_test_samples)
    student_train = abl73.load_split(args.feature_root, args.checkpoint_name, "student_free", "probe_train", args.max_train_samples)
    student_val = abl73.load_split(args.feature_root, args.checkpoint_name, "student_free", "probe_val", args.max_val_samples)
    student_test = abl73.load_split(args.feature_root, args.checkpoint_name, "student_free", "probe_test", args.max_test_samples)

    decoder_config = probe70.helpers.resolve_traj_tokenizer_config_path(args.student_model)
    if decoder_config is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {args.student_model}")
    decoder = probe70.helpers.TrajectoryTokenDecoder(config_path=decoder_config)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "schema_version": "hcot_prefix_shift_eval_v1",
        "checkpoint_name": args.checkpoint_name,
        "feature_name": args.feature_name,
        "heads": {},
    }

    for head_name in ("mlp4", "residual3"):
        path = args.teacher_head_dir / f"{head_name}_{args.feature_name}_action_head.pt"
        model = load_saved_head(path, dropout=float(args.dropout)).to(device)
        details["heads"][f"{head_name}_teacher_trained"] = {}
        for eval_prefix, split in (("teacher", teacher_test), ("student_free", student_test)):
            metrics = probe70.evaluate_action_model(
                model=model,
                decoder=decoder,
                device=device,
                batch_size=int(args.batch_size),
                **pack(split, split[args.feature_name]),
            )
            details["heads"][f"{head_name}_teacher_trained"][eval_prefix] = metrics
            rows.append(compact(head_name, "teacher", eval_prefix, "probe_test", metrics))

    input_dim = int(teacher_train[args.feature_name].shape[1])
    mixed_train_x = torch.cat([teacher_train[args.feature_name], student_train[args.feature_name]], dim=0).float()
    mixed_train_y = torch.cat([teacher_train["target_action"], student_train["target_action"]], dim=0).float()
    mixed_val_x = torch.cat([teacher_val[args.feature_name], student_val[args.feature_name]], dim=0).float()
    mixed_val_y = torch.cat([teacher_val["target_action"], student_val["target_action"]], dim=0).float()
    mixed_val = {
        "features": mixed_val_x,
        "target_action": mixed_val_y,
        # Eval conversion needs a coherent trajectory/history set. Use student-val for
        # checkpoint selection because student-free is the deployment prefix.
        **pack(student_val, student_val[args.feature_name]),
    }

    for head_name in ("mlp4", "residual3"):
        model = build_head(head_name, input_dim, int(args.hidden_dim), float(args.dropout))
        model, info = head75.train_head(
            name=f"{head_name}_mixed_prefix",
            model=model,
            train_x=mixed_train_x,
            train_y=mixed_train_y,
            val_pack=mixed_val,
            decoder=decoder,
            args=args,
            device=device,
        )
        details["heads"][f"{head_name}_mixed_trained"] = {"train_info": info}
        for eval_prefix, split in (("teacher", teacher_test), ("student_free", student_test)):
            metrics = probe70.evaluate_action_model(
                model=model,
                decoder=decoder,
                device=device,
                batch_size=int(args.batch_size),
                **pack(split, split[args.feature_name]),
            )
            details["heads"][f"{head_name}_mixed_trained"][eval_prefix] = metrics
            rows.append(compact(head_name, "teacher+student", eval_prefix, "probe_test", metrics))
        torch.save(
            {
                "state_dict": model.state_dict(),
                "head": head_name,
                "feature_name": args.feature_name,
                "input_dim": input_dim,
                "hidden_dim": int(args.hidden_dim),
                "train_prefix": "teacher+student",
            },
            args.output_dir / f"{head_name}_{args.feature_name}_mixed_action_head.pt",
        )

    csv_path = args.output_dir / "hcot_prefix_shift_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    details["compact"] = rows
    details["csv"] = str(csv_path)
    details["elapsed_sec"] = round(time.time() - t0, 3)
    (args.output_dir / "summary.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps({"event": "hcot_prefix_shift_done", "summary": str(args.output_dir / "summary.json"), "csv": str(csv_path)}), flush=True)


if __name__ == "__main__":
    main()
