#!/usr/bin/env python3
"""Compare hidden-to-action visual ablations by semantic/trajectory buckets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    parser.add_argument("--corpus-jsonl", type=Path, default=PROJECT_ROOT / "data/corpus/no_nav_teacher_pair_300chunks.jsonl")
    parser.add_argument("--student-model", default="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", action="append", nargs=4, metavar=("NAME", "FEATURE_ROOT", "CHECKPOINT_NAME", "PROBE_DIR"), required=True)
    parser.add_argument("--prefix-type", default="student_free")
    parser.add_argument("--max-test-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {"mean": float(clean.mean()), "p50": float(np.percentile(clean, 50)), "p95": float(np.percentile(clean, 95))}


def load_mlp_probe(path: Path, *, dropout: float) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    input_dim = int(payload["input_dim"])
    hidden_dim = int(state["net.1.weight"].shape[0])
    model = probe70.MLPProbe(input_dim, hidden_dim, dropout)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def predict(
    *,
    model: torch.nn.Module,
    features: torch.Tensor,
    split: dict[str, Any],
    decoder: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    all_ade: list[float] = []
    all_fde: list[float] = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        pred_action = model(features[start:end].to(device=device, dtype=torch.float32))
        pred_traj = probe70.action_to_traj(
            decoder,
            pred_action,
            split["ego_history_xyz"][start:end].to(device=device, dtype=torch.float32),
            split["ego_history_rot"][start:end].to(device=device, dtype=torch.float32),
        )
        ade, fde = probe70.ade_fde(pred_traj, split["target_traj"][start:end].to(device=device, dtype=torch.float32))
        all_ade.extend(float(v) for v in ade.detach().cpu().tolist())
        all_fde.extend(float(v) for v in fde.detach().cpu().tolist())
    return np.asarray(all_ade, dtype=np.float64), np.asarray(all_fde, dtype=np.float64)


def add_bucket_values(
    rows_out: list[dict[str, Any]],
    *,
    mode_name: str,
    model_name: str,
    split: dict[str, Any],
    cot_map: dict[str, str],
    ade: np.ndarray,
    fde: np.ndarray,
) -> None:
    by_bucket: dict[str, dict[str, list[float]]] = {}
    requested = {"curve_all", "stop", "intersection", "cross_traffic", "lead_vehicle_close"}
    for index, row in enumerate(split["rows"]):
        cot = cot_map.get(str(row.get("sample_id")), "")
        tags = set(abl73.bucket_tags(split["target_traj"][index].cpu().numpy(), cot))
        if "curve_all" in tags:
            tags.add("curve")
        # Keep both requested hard buckets and an all-sample row.
        tags.add("all")
        for tag in sorted(tags):
            if tag not in requested and tag not in {"all", "curve"}:
                continue
            bucket = by_bucket.setdefault(tag, {"ade": [], "fde": []})
            bucket["ade"].append(float(ade[index]))
            bucket["fde"].append(float(fde[index]))
    for tag, metrics in sorted(by_bucket.items()):
        ade_s = summarize(metrics["ade"])
        fde_s = summarize(metrics["fde"])
        rows_out.append(
            {
                "mode": mode_name,
                "model": model_name,
                "bucket": tag,
                "count": len(metrics["ade"]),
                "ade_mean": ade_s["mean"],
                "fde_mean": fde_s["mean"],
                "ade_p50": ade_s["p50"],
                "fde_p50": fde_s["p50"],
                "ade_p95": ade_s["p95"],
                "fde_p95": fde_s["p95"],
            }
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    cot_map = abl73.load_cot_map(args.corpus_jsonl)
    decoder_config = probe70.helpers.resolve_traj_tokenizer_config_path(args.student_model)
    if decoder_config is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {args.student_model}")
    decoder = probe70.helpers.TrajectoryTokenDecoder(config_path=decoder_config)

    rows_out: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    for mode_name, feature_root_raw, checkpoint_name, probe_dir_raw in args.mode:
        feature_root = Path(feature_root_raw)
        probe_dir = Path(probe_dir_raw)
        split = abl73.load_split(feature_root, checkpoint_name, args.prefix_type, "probe_test", args.max_test_samples)
        hidden_model = load_mlp_probe(probe_dir / "hidden_mlp.pt", dropout=float(args.dropout))
        ego_model = load_mlp_probe(probe_dir / "ego_only_mlp.pt", dropout=float(args.dropout))
        for model_name, model, feature_name in (
            ("hidden_mlp", hidden_model, "hidden_feature"),
            ("ego_only_mlp", ego_model, "ego_feature"),
        ):
            ade, fde = predict(
                model=model,
                features=split[feature_name].float(),
                split=split,
                decoder=decoder,
                device=device,
                batch_size=int(args.batch_size),
            )
            add_bucket_values(
                rows_out,
                mode_name=mode_name,
                model_name=model_name,
                split=split,
                cot_map=cot_map,
                ade=ade,
                fde=fde,
            )
            compact.append(
                {
                    "mode": mode_name,
                    "model": model_name,
                    "num_samples": int(len(ade)),
                    "ade_mean": float(ade.mean()),
                    "fde_mean": float(fde.mean()),
                    "ade_p95": float(np.percentile(ade, 95)),
                    "fde_p95": float(np.percentile(fde, 95)),
                }
            )

    csv_path = args.output_dir / "student_free_visual_bucket_breakdown.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    summary = {
        "schema_version": "student_free_visual_bucket_breakdown_v1",
        "max_test_samples": int(args.max_test_samples),
        "compact": compact,
        "bucket_csv": str(csv_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "visual_bucket_done", "summary": str(args.output_dir / "summary.json"), "csv": str(csv_path)}), flush=True)


if __name__ == "__main__":
    main()
