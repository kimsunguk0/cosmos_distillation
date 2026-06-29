#!/usr/bin/env python3
"""Train a lightweight reranker on saved AE trajectory candidates.

The script is intentionally offline: it consumes benchmark rows.jsonl files plus
the prediction NPZ files that contain N sampled AE paths. Training uses GT only
to create the oracle path label; inference features are GT-free path/ensemble
geometry features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_BENCHMARK_ROOT = Path("outputs/benchmarks/semantic_val806_4models_20260612")
DEFAULT_MODEL_KEY = "student_noflex_ae28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--external-test-root", type=Path, default=None)
    parser.add_argument("--external-test-model-key", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reranker/ae_path_reranker_b0_val806"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--objective", choices=("oracle_ce", "weighted_mse"), default="weighted_mse")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--fde-weight", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--include-category", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@dataclass(frozen=True)
class CandidateRecord:
    sample_id: str
    category: str
    paths: np.ndarray
    target: np.ndarray
    path_ades: np.ndarray
    path_fdes: np.ndarray
    feature: np.ndarray
    oracle_idx: int


def stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_name(sample_id: str, train_frac: float, val_frac: float) -> str:
    value = stable_unit_interval(sample_id)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "val"
    return "test"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def squeeze_path(path: np.ndarray) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected path rank 2, got shape={arr.shape}")
    return arr


def ade_fde(path: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    path = squeeze_path(path)
    target = squeeze_path(target)
    n = min(int(path.shape[0]), int(target.shape[0]))
    diff = np.linalg.norm(path[:n, :2] - target[:n, :2], axis=-1)
    return float(diff.mean()), float(diff[-1])


def path_length(path: np.ndarray) -> float:
    path = squeeze_path(path)
    if int(path.shape[0]) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=-1).sum())


def medoid_index(paths: np.ndarray) -> int:
    xy = np.asarray(paths, dtype=np.float32)[..., :2]
    diff = xy[:, None, :, :] - xy[None, :, :, :]
    dist = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return int(np.argmin(dist.sum(axis=1)))


def finite_float(value: float) -> float:
    if math.isfinite(float(value)):
        return float(value)
    return 0.0


def path_feature(path: np.ndarray, paths: np.ndarray, dt: float) -> list[float]:
    path = squeeze_path(path)
    xy = path[:, :2].astype(np.float32)
    dxy = np.diff(xy, axis=0)
    step = np.linalg.norm(dxy, axis=-1)
    speed = step / max(float(dt), 1e-6)
    accel = np.diff(speed) / max(float(dt), 1e-6) if speed.shape[0] > 1 else np.zeros(1, dtype=np.float32)
    jerk = np.diff(accel) / max(float(dt), 1e-6) if accel.shape[0] > 1 else np.zeros(1, dtype=np.float32)
    heading = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0])) if dxy.shape[0] else np.zeros(1, dtype=np.float32)
    yaw_rate = np.diff(heading) / max(float(dt), 1e-6) if heading.shape[0] > 1 else np.zeros(1, dtype=np.float32)

    mean_path = np.asarray(paths, dtype=np.float32).mean(axis=0)
    median_len = float(np.median([path_length(candidate) for candidate in paths]))
    length = path_length(path)
    endpoint = xy[-1]
    start = xy[0]
    displacement = endpoint - start
    mean_dist = np.linalg.norm(xy - mean_path[:, :2], axis=-1)
    pair_dists = []
    for other in paths:
        pair_dists.append(float(np.linalg.norm(xy - other[:, :2], axis=-1).mean()))

    return [
        length,
        length - median_len,
        float(endpoint[0]),
        float(endpoint[1]),
        float(displacement[0]),
        float(displacement[1]),
        float(np.abs(xy[:, 1]).mean()),
        float(np.abs(xy[:, 1]).max()),
        float(speed.mean()) if speed.size else 0.0,
        float(speed.std()) if speed.size else 0.0,
        float(speed.max()) if speed.size else 0.0,
        float(accel.mean()) if accel.size else 0.0,
        float(np.abs(accel).mean()) if accel.size else 0.0,
        float(np.abs(accel).max()) if accel.size else 0.0,
        float(np.abs(jerk).mean()) if jerk.size else 0.0,
        float(np.abs(jerk).max()) if jerk.size else 0.0,
        float(np.abs(yaw_rate).mean()) if yaw_rate.size else 0.0,
        float(np.abs(yaw_rate).max()) if yaw_rate.size else 0.0,
        float(mean_dist.mean()),
        float(mean_dist[-1]),
        float(np.mean(pair_dists)),
        float(np.min(pair_dists)),
    ]


def build_features(paths: np.ndarray, category: str, categories: list[str], include_category: bool, dt: float) -> np.ndarray:
    base = np.asarray([path_feature(path, paths, dt=dt) for path in paths], dtype=np.float32)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if not include_category:
        return base
    one_hot = np.zeros((base.shape[0], len(categories)), dtype=np.float32)
    if category in categories:
        one_hot[:, categories.index(category)] = 1.0
    return np.concatenate([base, one_hot], axis=-1)


def resolve_prediction_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_records(
    root: Path,
    model_key: str,
    args: argparse.Namespace,
    categories: list[str] | None = None,
) -> tuple[list[CandidateRecord], list[str]]:
    rows_path = root / model_key / "rows.jsonl"
    rows = load_jsonl(rows_path)
    if categories is None:
        categories = sorted({str(row.get("category", "unknown")) for row in rows})
    records: list[CandidateRecord] = []
    for row in rows:
        npz_path = resolve_prediction_path(args.benchmark_root, str(row["prediction_npz"]))
        with np.load(npz_path) as data:
            paths = np.asarray(data["paths"], dtype=np.float32)
            target = squeeze_path(np.asarray(data["target_gt"], dtype=np.float32))
        if paths.ndim != 3:
            raise ValueError(f"Expected paths [K,T,3], got {paths.shape} at {npz_path}")
        path_ades = np.asarray([ade_fde(path, target)[0] for path in paths], dtype=np.float32)
        path_fdes = np.asarray([ade_fde(path, target)[1] for path in paths], dtype=np.float32)
        category = str(row.get("category", "unknown"))
        feature = build_features(
            paths=paths,
            category=category,
            categories=categories,
            include_category=bool(args.include_category),
            dt=float(args.dt),
        )
        records.append(
            CandidateRecord(
                sample_id=str(row["sample_id"]),
                category=category,
                paths=paths,
                target=target,
                path_ades=path_ades,
                path_fdes=path_fdes,
                feature=feature,
                oracle_idx=int(np.argmin(path_ades)),
            )
        )
    return records, categories


class PathScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, paths, feat = x.shape
        scores = self.net(x.reshape(batch * paths, feat)).reshape(batch, paths)
        return scores


def stack_features(records: list[CandidateRecord]) -> np.ndarray:
    return np.stack([record.feature for record in records], axis=0).astype(np.float32)


def stack_labels(records: list[CandidateRecord]) -> np.ndarray:
    return np.asarray([record.oracle_idx for record in records], dtype=np.int64)


def stack_paths(records: list[CandidateRecord]) -> np.ndarray:
    return np.stack([record.paths for record in records], axis=0).astype(np.float32)


def stack_targets(records: list[CandidateRecord]) -> np.ndarray:
    return np.stack([record.target for record in records], axis=0).astype(np.float32)


def normalize_splits(
    split_records: dict[str, list[CandidateRecord]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    x_train = stack_features(split_records["train"])
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    xs: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}
    for name, records in split_records.items():
        x = stack_features(records)
        xs[name] = ((x - mean) / std).astype(np.float32)
        ys[name] = stack_labels(records)
    return xs, ys, mean.astype(np.float32), std.astype(np.float32)


def selected_metrics(records: list[CandidateRecord], indices: list[int] | np.ndarray) -> dict[str, float]:
    idx = np.asarray(indices, dtype=np.int64)
    ades = [float(record.path_ades[int(i)]) for record, i in zip(records, idx, strict=True)]
    fdes = [float(record.path_fdes[int(i)]) for record, i in zip(records, idx, strict=True)]
    hit = [int(int(i) == int(record.oracle_idx)) for record, i in zip(records, idx, strict=True)]
    regret = [float(record.path_ades[int(i)] - record.path_ades[record.oracle_idx]) for record, i in zip(records, idx, strict=True)]
    return {
        "ade_mean_m": float(np.mean(ades)),
        "ade_p50_m": float(np.median(ades)),
        "fde_mean_m": float(np.mean(fdes)),
        "oracle_hit_rate": float(np.mean(hit)),
        "oracle_regret_mean_m": float(np.mean(regret)),
    }


def aggregate_metrics(records: list[CandidateRecord], method: str) -> dict[str, float]:
    ades: list[float] = []
    fdes: list[float] = []
    for record in records:
        if method == "mean_traj":
            path = record.paths.mean(axis=0)
        elif method == "medoid":
            path = record.paths[medoid_index(record.paths)]
        elif method == "first":
            path = record.paths[0]
        elif method == "oracle":
            path = record.paths[record.oracle_idx]
        else:
            raise ValueError(method)
        ade, fde = ade_fde(path, record.target)
        ades.append(ade)
        fdes.append(fde)
    return {
        "ade_mean_m": float(np.mean(ades)),
        "ade_p50_m": float(np.median(ades)),
        "fde_mean_m": float(np.mean(fdes)),
    }


def evaluate_model(model: PathScorer, x: np.ndarray, y: np.ndarray, records: list[CandidateRecord], device: str) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device=device))
        loss = nn.functional.cross_entropy(logits, torch.from_numpy(y).to(device=device)).item()
        pred = logits.argmax(dim=-1).cpu().numpy()
    metrics = selected_metrics(records, pred)
    metrics["loss"] = float(loss)
    return {"indices": pred.tolist(), "metrics": metrics}


def weighted_loss(
    model: PathScorer,
    x: torch.Tensor,
    paths_xy: torch.Tensor,
    targets_xy: torch.Tensor,
    softmax_temperature: float,
    fde_weight: float,
) -> torch.Tensor:
    scores = model(x)
    weights = torch.softmax(scores / max(float(softmax_temperature), 1e-6), dim=-1)
    pred = (weights[:, :, None, None] * paths_xy).sum(dim=1)
    mse = (pred - targets_xy).square().mean()
    fde = (pred[:, -1] - targets_xy[:, -1]).square().mean()
    return mse + float(fde_weight) * fde


def weighted_metrics(
    model: PathScorer,
    x: np.ndarray,
    records: list[CandidateRecord],
    device: str,
    softmax_temperature: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(x).to(device=device))
        weights = torch.softmax(scores / max(float(softmax_temperature), 1e-6), dim=-1).cpu().numpy()
        argmax = scores.argmax(dim=-1).cpu().numpy()
    ades: list[float] = []
    fdes: list[float] = []
    effective_paths: list[float] = []
    for record, weight in zip(records, weights, strict=True):
        pred = np.sum(weight[:, None, None] * record.paths, axis=0)
        ade, fde = ade_fde(pred, record.target)
        ades.append(ade)
        fdes.append(fde)
        effective_paths.append(float(1.0 / np.square(weight).sum()))
    return {
        "weights": weights.tolist(),
        "argmax_indices": argmax.tolist(),
        "metrics": {
            "ade_mean_m": float(np.mean(ades)),
            "ade_p50_m": float(np.median(ades)),
            "fde_mean_m": float(np.mean(fdes)),
            "effective_paths_mean": float(np.mean(effective_paths)),
        },
        "argmax_metrics": selected_metrics(records, argmax),
    }


def train_model(
    xs: dict[str, np.ndarray],
    ys: dict[str, np.ndarray],
    split_records: dict[str, list[CandidateRecord]],
    args: argparse.Namespace,
) -> tuple[PathScorer, dict[str, Any]]:
    torch.manual_seed(int(args.seed))
    model = PathScorer(input_dim=int(xs["train"].shape[-1]), hidden_dim=int(args.hidden_dim)).to(device=str(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_x = torch.from_numpy(xs["train"]).to(device=str(args.device))
    train_y = torch.from_numpy(ys["train"]).to(device=str(args.device))
    train_paths = torch.from_numpy(stack_paths(split_records["train"])[:, :, :, :2]).to(device=str(args.device))
    train_targets = torch.from_numpy(stack_targets(split_records["train"])[:, :, :2]).to(device=str(args.device))
    val_paths = torch.from_numpy(stack_paths(split_records["val"])[:, :, :, :2]).to(device=str(args.device))
    val_targets = torch.from_numpy(stack_targets(split_records["val"])[:, :, :2]).to(device=str(args.device))
    val_x = torch.from_numpy(xs["val"]).to(device=str(args.device))
    val_y = torch.from_numpy(ys["val"]).to(device=str(args.device))

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if str(args.objective) == "oracle_ce":
            logits = model(train_x)
            loss = nn.functional.cross_entropy(logits, train_y)
        elif str(args.objective) == "weighted_mse":
            loss = weighted_loss(
                model=model,
                x=train_x,
                paths_xy=train_paths,
                targets_xy=train_targets,
                softmax_temperature=float(args.softmax_temperature),
                fde_weight=float(args.fde_weight),
            )
        else:
            raise ValueError(str(args.objective))
        loss.backward()
        optimizer.step()

        if str(args.objective) == "oracle_ce":
            val = evaluate_model(model, xs["val"], ys["val"], split_records["val"], str(args.device))
            val_loss = float(val["metrics"]["loss"])
            val_ade = float(val["metrics"]["ade_mean_m"])
            val_hit = float(val["metrics"]["oracle_hit_rate"])
        else:
            model.eval()
            with torch.no_grad():
                vloss = weighted_loss(
                    model=model,
                    x=val_x,
                    paths_xy=val_paths,
                    targets_xy=val_targets,
                    softmax_temperature=float(args.softmax_temperature),
                    fde_weight=float(args.fde_weight),
                )
                logits = model(val_x)
                ce = nn.functional.cross_entropy(logits, val_y)
            val_weighted = weighted_metrics(
                model=model,
                x=xs["val"],
                records=split_records["val"],
                device=str(args.device),
                softmax_temperature=float(args.softmax_temperature),
            )
            val_loss = float(vloss.item())
            val_ade = float(val_weighted["metrics"]["ade_mean_m"])
            val_hit = float(val_weighted["argmax_metrics"]["oracle_hit_rate"])
        row = {
            "epoch": float(epoch),
            "train_loss": float(loss.item()),
            "val_loss": val_loss,
            "val_ade_mean_m": val_ade,
            "val_oracle_hit_rate": val_hit,
        }
        history.append(row)
        score = row["val_loss"] if str(args.objective) == "oracle_ce" else row["val_ade_mean_m"]
        if score < best_val_loss - 1e-5:
            best_val_loss = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(args.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "history": history}


def by_category(records: list[CandidateRecord], indices: list[int]) -> dict[str, dict[str, float]]:
    grouped: dict[str, tuple[list[CandidateRecord], list[int]]] = {}
    for record, index in zip(records, indices, strict=True):
        recs, idxs = grouped.setdefault(record.category, ([], []))
        recs.append(record)
        idxs.append(int(index))
    out: dict[str, dict[str, float]] = {}
    for category, (recs, idxs) in sorted(grouped.items()):
        out[category] = selected_metrics(recs, idxs)
        out[category]["count"] = float(len(recs))
    return out


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# 140 - AE Path Reranker Bootstrap",
        "",
        f"Model key: `{summary['model_key']}`",
        f"Source benchmark: `{summary['benchmark_root']}`",
        f"Output dir: `{summary['output_dir']}`",
        "",
        "## Split",
        "",
        "| split | count |",
        "|---|---:|",
    ]
    for name, count in summary["split_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## Test Metrics", "", "| method | ADE | FDE | oracle hit | regret |", "|---|---:|---:|---:|---:|"])
    for method, metrics in summary["test_metrics"].items():
        hit = metrics.get("oracle_hit_rate")
        regret = metrics.get("oracle_regret_mean_m")
        lines.append(
            f"| {method} | {metrics['ade_mean_m']:.4f} | {metrics['fde_mean_m']:.4f} | "
            f"{'' if hit is None else f'{hit:.4f}'} | {'' if regret is None else f'{regret:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a bootstrap reranker trained on saved semantic-val806 AE candidates.",
            "- Features are GT-free path and ensemble geometry features; GT is used only to label the oracle path during supervised training.",
            "- A production reranker needs a larger candidate corpus generated on train/heldout splits before treating these numbers as final.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def jsonable_settings(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, categories = load_records(args.benchmark_root, str(args.model_key), args)
    split_records: dict[str, list[CandidateRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        split_records[split_name(record.sample_id, float(args.train_frac), float(args.val_frac))].append(record)
    if args.external_test_root is not None:
        external_model_key = str(args.external_test_model_key or args.model_key)
        external_records, _ = load_records(args.external_test_root, external_model_key, args, categories=categories)
        split_records["test"] = external_records
    if not all(split_records.values()):
        raise RuntimeError({key: len(value) for key, value in split_records.items()})

    xs, ys, mean, std = normalize_splits(split_records)
    model, train_info = train_model(xs, ys, split_records, args)

    evals = {name: evaluate_model(model, xs[name], ys[name], split_records[name], str(args.device)) for name in split_records}
    weighted_evals = {
        name: weighted_metrics(
            model=model,
            x=xs[name],
            records=split_records[name],
            device=str(args.device),
            softmax_temperature=float(args.softmax_temperature),
        )
        for name in split_records
    }
    test_indices = evals["test"]["indices"]
    test_metrics: dict[str, dict[str, float]] = {
        "first_path": aggregate_metrics(split_records["test"], "first"),
        "mean_traj": aggregate_metrics(split_records["test"], "mean_traj"),
        "medoid": aggregate_metrics(split_records["test"], "medoid"),
        "oracle_best": aggregate_metrics(split_records["test"], "oracle"),
        "learned_argmax": evals["test"]["metrics"],
        "learned_weighted": weighted_evals["test"]["metrics"],
    }
    summary = {
        "event": "ae_path_reranker_bootstrap_done",
        "benchmark_root": str(args.benchmark_root),
        "model_key": str(args.model_key),
        "output_dir": str(args.output_dir),
        "settings": jsonable_settings(args),
        "categories": categories,
        "feature_dim": int(xs["train"].shape[-1]),
        "split_counts": {key: len(value) for key, value in split_records.items()},
        "train_info": train_info,
        "eval_metrics": {key: value["metrics"] for key, value in evals.items()},
        "weighted_eval_metrics": {key: value["metrics"] for key, value in weighted_evals.items()},
        "argmax_eval_metrics": {key: value["argmax_metrics"] for key, value in weighted_evals.items()},
        "test_metrics": test_metrics,
        "test_category_metrics_learned": by_category(split_records["test"], test_indices),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "categories": categories,
            "settings": jsonable_settings(args),
        },
        args.output_dir / "best_reranker.pt",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, args.output_dir / "report.md")
    print(json.dumps({"event": "done", "summary": str(args.output_dir / "summary.json"), "report": str(args.output_dir / "report.md")}), flush=True)


if __name__ == "__main__":
    main()
