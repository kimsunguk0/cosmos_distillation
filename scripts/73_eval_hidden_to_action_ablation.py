#!/usr/bin/env python3
"""Evaluate hidden-to-action probe buckets and feature ablations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_probe_module():
    path = PROJECT_ROOT / "scripts" / "70_train_hidden_to_action_probe.py"
    spec = importlib.util.spec_from_file_location("hidden_to_action_probe_70", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import probe helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe70 = _load_probe_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=PROJECT_ROOT / "outputs/probe_cache/hidden_to_action_v1")
    parser.add_argument("--checkpoint-name", default="bp3_200k_final")
    parser.add_argument("--prefix-type", default="teacher_prefix")
    parser.add_argument("--trained-probe-dir", type=Path, default=PROJECT_ROOT / "outputs/checkpoints/hidden_to_action_probe_v1/bp3_200k_final_teacher_prefix")
    parser.add_argument("--corpus-jsonl", type=Path, default=PROJECT_ROOT / "data/corpus/no_nav_teacher_pair_300chunks.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/reports/hidden_to_action_probe_v1/bp3_200k_final_teacher_prefix_ablation")
    parser.add_argument("--student-model", default="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def load_split(feature_root: Path, checkpoint_name: str, prefix_type: str, split_name: str, max_samples: int = 0) -> dict[str, Any]:
    split_dir = feature_root / checkpoint_name / prefix_type / split_name
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[torch.Tensor]] = defaultdict(list)
    keys = (
        "hidden_feature",
        "h_cot_end",
        "h_traj_start",
        "h_prefix_mean_last8",
        "h_prefix_mean_last16",
        "ego_feature",
        "ego_history_xyz",
        "ego_history_rot",
        "target_action",
        "target_traj",
        "gt_future",
    )
    for shard_raw in manifest.get("shards") or []:
        shard_path = Path(shard_raw)
        if not shard_path.is_absolute():
            shard_path = PROJECT_ROOT / shard_path
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_rows = list(payload.get("rows") or [])
        take = len(shard_rows)
        if max_samples > 0:
            remaining = max(int(max_samples) - len(rows), 0)
            if remaining <= 0:
                break
            take = min(take, remaining)
        rows.extend(shard_rows[:take])
        for key in keys:
            if key in payload:
                arrays[key].append(payload[key][:take].clone())
    if not rows:
        raise RuntimeError(f"No rows loaded for {checkpoint_name}/{prefix_type}/{split_name}")
    out: dict[str, Any] = {"rows": rows, "manifest": manifest}
    for key, values in arrays.items():
        out[key] = torch.cat(values, dim=0)
    return out


def load_cot_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cot = str((row.get("teacher_target") or {}).get("cot_text") or (row.get("hard_target") or {}).get("cot_text") or "")
            out[str(row.get("sample_id"))] = cot
    return out


def summarize(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {"mean": float(clean.mean()), "p50": float(np.percentile(clean, 50)), "p95": float(np.percentile(clean, 95))}


def _text_has(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def bucket_tags(target_xyz: np.ndarray, cot_text: str) -> list[str]:
    xy = np.asarray(target_xyz, dtype=np.float64)[:, :2]
    if xy.shape[0] < 3:
        return ["unknown"]
    deltas = np.diff(xy, axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    speeds = step_dist * 10.0
    total_length = float(step_dist.sum())
    final_y = float(xy[-1, 1] - xy[0, 1])
    good = step_dist > 0.05
    heading_delta = 0.0
    if good.sum() >= 2:
        headings = np.unwrap(np.arctan2(deltas[good, 1], deltas[good, 0]))
        heading_delta = float(headings[-1] - headings[0])
    initial_speed = float(np.mean(speeds[: min(5, len(speeds))])) if len(speeds) else 0.0
    final_speed = float(np.mean(speeds[-min(5, len(speeds)) :])) if len(speeds) else 0.0
    near_zero_steps = int((speeds < 0.3).sum())

    tags: set[str] = set()
    is_stop = total_length <= 5.0 or near_zero_steps >= max(6, len(speeds) // 5)
    is_left = heading_delta > 0.15 or final_y > 2.0
    is_right = heading_delta < -0.15 or final_y < -2.0
    if is_left:
        tags.update({"curve_all", "curve_left"})
    if is_right:
        tags.update({"curve_all", "curve_right"})
    if not is_left and not is_right and not is_stop:
        tags.add("straight")
    if is_stop:
        tags.add("stop")
    if final_speed < max(0.5, initial_speed * 0.5) and not is_stop:
        tags.add("slowdown")

    if _text_has(cot_text, (r"\bintersection\b", r"\bstop sign\b", r"\btraffic light\b", r"\bcrosswalk\b", r"\bjunction\b")):
        tags.add("intersection")
    if _text_has(cot_text, (r"\bcross traffic\b", r"\bcrossing\b", r"\bpedestrian\b", r"\bcyclist\b", r"\boncoming\b")):
        tags.add("cross_traffic")
    if _text_has(cot_text, (r"\blead vehicle\b", r"\bvehicle ahead\b", r"\bcar ahead\b", r"\bfollow", r"\bfront vehicle\b")):
        tags.add("lead_vehicle_close")
    if tags & {"stop", "slowdown", "lead_vehicle_close"} or _text_has(
        cot_text,
        (r"\bspeed\b", r"\bslow", r"\bdecel", r"\byield\b", r"\bcreep\b", r"\bstop\b", r"\btraffic\b", r"\bpedestrian\b"),
    ):
        tags.add("speed_sensitive")
    if not tags:
        tags.add("other")
    return sorted(tags)


@torch.no_grad()
def predict_ade_fde(
    *,
    model: torch.nn.Module | None,
    features: torch.Tensor | None,
    split: dict[str, Any],
    decoder: Any,
    device: torch.device,
    batch_size: int,
    constant_velocity: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    target = split["target_traj"].float()
    if constant_velocity:
        pred_traj = probe70.constant_velocity_prediction(split["ego_history_xyz"].float())
        ade, fde = probe70.ade_fde(pred_traj, target)
        return ade.cpu().numpy(), fde.cpu().numpy()
    if model is None or features is None:
        raise ValueError("model/features required unless constant_velocity=True")
    model.eval().to(device)
    all_ade: list[float] = []
    all_fde: list[float] = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        x = features[start:end].to(device=device, dtype=torch.float32)
        hx = split["ego_history_xyz"][start:end].to(device=device, dtype=torch.float32)
        hr = split["ego_history_rot"][start:end].to(device=device, dtype=torch.float32)
        tt = split["target_traj"][start:end].to(device=device, dtype=torch.float32)
        pred_action = model(x)
        pred_traj = probe70.action_to_traj(decoder, pred_action, hx, hr)
        ade, fde = probe70.ade_fde(pred_traj, tt)
        all_ade.extend(float(value) for value in ade.cpu().tolist())
        all_fde.extend(float(value) for value in fde.cpu().tolist())
    return np.asarray(all_ade, dtype=np.float64), np.asarray(all_fde, dtype=np.float64)


def load_mlp_probe(path: Path, *, dropout: float) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    input_dim = int(payload["input_dim"])
    hidden_dim = int(state["net.1.weight"].shape[0])
    model = probe70.MLPProbe(input_dim, hidden_dim, dropout)
    model.load_state_dict(state)
    return model


def write_bucket_breakdown(
    *,
    out_path: Path,
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    split: dict[str, Any],
    cot_map: dict[str, str],
) -> None:
    bucket_values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sample_tags: list[list[str]] = []
    for index, row in enumerate(split["rows"]):
        cot = cot_map.get(str(row.get("sample_id")), "")
        tags = bucket_tags(split["target_traj"][index].cpu().numpy(), cot)
        sample_tags.append(tags)
        for model_name, (ade, fde) in models.items():
            for tag in tags:
                bucket_values[model_name][tag]["ade"].append(float(ade[index]))
                bucket_values[model_name][tag]["fde"].append(float(fde[index]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "tag", "count", "ade_mean", "fde_mean", "ade_p50", "fde_p50", "ade_p95", "fde_p95"],
        )
        writer.writeheader()
        for model_name in sorted(bucket_values):
            for tag in sorted(bucket_values[model_name]):
                ade_s = summarize(bucket_values[model_name][tag]["ade"])
                fde_s = summarize(bucket_values[model_name][tag]["fde"])
                writer.writerow(
                    {
                        "model": model_name,
                        "tag": tag,
                        "count": len(bucket_values[model_name][tag]["ade"]),
                        "ade_mean": ade_s["mean"],
                        "fde_mean": fde_s["mean"],
                        "ade_p50": ade_s["p50"],
                        "fde_p50": fde_s["p50"],
                        "ade_p95": ade_s["p95"],
                        "fde_p95": fde_s["p95"],
                    }
                )


def feature_tensor(split: dict[str, Any], feature_name: str) -> torch.Tensor:
    if feature_name == "h_cot_end":
        return split["h_cot_end"].float()
    if feature_name == "h_traj_start":
        return split["h_traj_start"].float()
    if feature_name == "h_prefix_mean_last16":
        return split["h_prefix_mean_last16"].float()
    if feature_name == "h_cot_end+h_traj_start":
        return torch.cat([split["h_cot_end"], split["h_traj_start"]], dim=1).float()
    if feature_name == "all_concat":
        return split["hidden_feature"].float()
    raise ValueError(f"Unsupported feature_name: {feature_name}")


def pack_for_eval(split: dict[str, Any], features: torch.Tensor) -> dict[str, Any]:
    return {
        "features": features.float(),
        "target_action": split["target_action"].float(),
        "target_traj": split["target_traj"].float(),
        "gt_future": split["gt_future"].float(),
        "ego_history_xyz": split["ego_history_xyz"].float(),
        "ego_history_rot": split["ego_history_rot"].float(),
        "rows": split["rows"],
    }


def compact_row(feature: str, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature": feature,
        "split": split,
        "num_samples": metrics.get("num_samples"),
        "ade_mean": (metrics.get("ade_vs_teacher_action") or {}).get("mean"),
        "fde_mean": (metrics.get("fde_vs_teacher_action") or {}).get("mean"),
        "ade_p95": (metrics.get("ade_vs_teacher_action") or {}).get("p95"),
        "fde_p95": (metrics.get("fde_vs_teacher_action") or {}).get("p95"),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe70.set_seed(int(args.seed))
    device = torch.device(args.device if str(args.device) == "cpu" or torch.cuda.is_available() else "cpu")

    train = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_train", args.max_train_samples)
    val = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_val", args.max_val_samples)
    test = load_split(args.feature_root, args.checkpoint_name, args.prefix_type, "probe_test", args.max_test_samples)
    cot_map = load_cot_map(args.corpus_jsonl)
    decoder_config = probe70.helpers.resolve_traj_tokenizer_config_path(args.student_model)
    if decoder_config is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {args.student_model}")
    decoder = probe70.helpers.TrajectoryTokenDecoder(config_path=decoder_config)

    # Bucket breakdown for the already trained main probes.
    ego_model = load_mlp_probe(args.trained_probe_dir / "ego_only_mlp.pt", dropout=float(args.dropout))
    hidden_model = load_mlp_probe(args.trained_probe_dir / "hidden_mlp.pt", dropout=float(args.dropout))
    bucket_models = {
        "constant_velocity": predict_ade_fde(
            model=None,
            features=None,
            split=test,
            decoder=decoder,
            device=device,
            batch_size=int(args.batch_size),
            constant_velocity=True,
        ),
        "ego_only_mlp": predict_ade_fde(
            model=ego_model,
            features=test["ego_feature"].float(),
            split=test,
            decoder=decoder,
            device=device,
            batch_size=int(args.batch_size),
        ),
        "hidden_mlp_all_concat": predict_ade_fde(
            model=hidden_model,
            features=test["hidden_feature"].float(),
            split=test,
            decoder=decoder,
            device=device,
            batch_size=int(args.batch_size),
        ),
    }
    bucket_csv = args.output_dir / "bucket_breakdown_test.csv"
    write_bucket_breakdown(out_path=bucket_csv, models=bucket_models, split=test, cot_map=cot_map)

    feature_names = ["h_cot_end", "h_traj_start", "h_prefix_mean_last16", "h_cot_end+h_traj_start", "all_concat"]
    feature_rows: list[dict[str, Any]] = []
    train_info: dict[str, Any] = {}
    for feature_name in feature_names:
        train_x = feature_tensor(train, feature_name)
        val_x = feature_tensor(val, feature_name)
        test_x = feature_tensor(test, feature_name)
        model = probe70.MLPProbe(int(train_x.shape[1]), int(args.hidden_dim), float(args.dropout))
        model, info = probe70.train_probe(
            name=f"feature_ablation_{feature_name}",
            model=model,
            train_x=train_x,
            train_y=train["target_action"],
            val_data=pack_for_eval(val, val_x),
            decoder=decoder,
            args=args,
            device=device,
        )
        train_info[feature_name] = info
        val_metrics = probe70.evaluate_action_model(
            model=model,
            decoder=decoder,
            device=device,
            batch_size=int(args.batch_size),
            **pack_for_eval(val, val_x),
        )
        test_metrics = probe70.evaluate_action_model(
            model=model,
            decoder=decoder,
            device=device,
            batch_size=int(args.batch_size),
            **pack_for_eval(test, test_x),
        )
        feature_rows.append(compact_row(feature_name, "probe_val", val_metrics))
        feature_rows.append(compact_row(feature_name, "probe_test", test_metrics))
        torch.save(
            {"state_dict": model.state_dict(), "input_dim": int(train_x.shape[1]), "kind": "mlp", "feature": feature_name},
            args.output_dir / f"{feature_name.replace('+', '_plus_')}_mlp.pt",
        )

    feature_csv = args.output_dir / "feature_ablation_mlp.csv"
    with feature_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    summary = {
        "schema_version": "hidden_to_action_probe_ablation_v1",
        "checkpoint_name": args.checkpoint_name,
        "prefix_type": args.prefix_type,
        "feature_root": str(args.feature_root),
        "trained_probe_dir": str(args.trained_probe_dir),
        "splits": {
            "probe_train": len(train["rows"]),
            "probe_val": len(val["rows"]),
            "probe_test": len(test["rows"]),
        },
        "bucket_breakdown_csv": str(bucket_csv),
        "feature_ablation_csv": str(feature_csv),
        "train_info": train_info,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "ablation_done", "summary": str(args.output_dir / "summary.json"), "elapsed_sec": summary["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
