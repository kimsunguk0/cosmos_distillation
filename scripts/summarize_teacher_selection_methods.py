#!/usr/bin/env python3
"""Re-aggregate saved teacher trajectory candidates by selection method."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_4models import ade_fde, medoid_index, path_len, squeeze_path, squeeze_paths, summarize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METHODS = ("first_path", "mean_traj", "medoid", "oracle_best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--model-key", default="teacher10b")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods: single,first_path,mean_traj,medoid,oracle_best.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def finite_mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return None
    return float(np.asarray(clean, dtype=np.float64).mean())


def finite_rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float64).mean())


def choose_path(
    paths: np.ndarray,
    path_ades: list[float],
    method: str,
) -> tuple[np.ndarray, int | None]:
    if method in {"single", "first_path"}:
        return paths[0], 0
    if method == "mean_traj":
        return paths.mean(axis=0), None
    if method == "medoid":
        idx = medoid_index(paths)
        return paths[idx], idx
    if method == "oracle_best":
        idx = int(np.nanargmin(np.asarray(path_ades, dtype=np.float64)))
        return paths[idx], idx
    raise ValueError(f"Unknown selection method: {method}")


def empty_metric_bucket() -> dict[str, list[Any]]:
    return {
        "ade_gt_m": [],
        "fde_gt_m": [],
        "path_length_m": [],
        "oracle_hit": [],
        "oracle_regret_ade_m": [],
    }


def metric_summary(bucket: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "ade_gt_m": summarize([float(v) for v in bucket["ade_gt_m"]]),
        "fde_gt_m": summarize([float(v) for v in bucket["fde_gt_m"]]),
        "path_length_m": summarize([float(v) for v in bucket["path_length_m"]]),
        "oracle_hit_rate": finite_rate([bool(v) for v in bucket["oracle_hit"]]),
        "oracle_regret_ade_m": {
            "mean": finite_mean([float(v) for v in bucket["oracle_regret_ade_m"]]),
        },
    }


def load_settings(benchmark_root: Path, model_key: str) -> dict[str, Any]:
    paths = [benchmark_root / model_key / "summary.json", benchmark_root / "summary.json"]
    out: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = payload.get("settings")
        if isinstance(settings, dict):
            out.update(settings)
    return out


def format_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    method_rows = summary["methods"]
    lines = [
        f"# Teacher Native Selection Summary: {summary['tag']}",
        "",
        f"- source: `{summary['benchmark_root']}`",
        f"- model: `{summary['model_key']}`",
        f"- samples: `{summary['count']}`",
        f"- observed candidate count histogram: `{summary['num_paths_histogram']}`",
        f"- settings: `{json.dumps(summary['settings'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "| method | ADE mean | FDE mean | ADE p50 | FDE p50 | ADE p95 | gap vs oracle ADE | oracle hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in method_rows.items():
        ade = metrics["ade_gt_m"]
        fde = metrics["fde_gt_m"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    format_num(ade.get("mean")),
                    format_num(fde.get("mean")),
                    format_num(ade.get("p50")),
                    format_num(fde.get("p50")),
                    format_num(ade.get("p95")),
                    format_num(metrics.get("gap_vs_oracle_ade_m")),
                    format_num(metrics.get("oracle_hit_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Category ADE Mean",
            "",
            "| category | count | first_path | mean_traj | medoid | oracle_best |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, payload in summary["by_category"].items():
        methods = payload["methods"]
        lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    str(payload["count"]),
                    format_num(methods.get("first_path", {}).get("ade_gt_m", {}).get("mean")),
                    format_num(methods.get("mean_traj", {}).get("ade_gt_m", {}).get("mean")),
                    format_num(methods.get("medoid", {}).get("ade_gt_m", {}).get("mean")),
                    format_num(methods.get("oracle_best", {}).get("ade_gt_m", {}).get("mean")),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_root = resolve_path(args.benchmark_root)
    model_key = str(args.model_key)
    methods = [part.strip() for part in str(args.methods).split(",") if part.strip()]
    rows_path = benchmark_root / model_key / "rows.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)

    buckets: dict[str, dict[str, list[Any]]] = {method: empty_metric_bucket() for method in methods}
    category_buckets: dict[str, dict[str, dict[str, list[Any]]]] = defaultdict(
        lambda: {method: empty_metric_bucket() for method in methods}
    )
    category_counts: Counter[str] = Counter()
    num_paths_histogram: Counter[int] = Counter()

    count = 0
    for row in iter_jsonl(rows_path):
        pred_path = resolve_path(row["prediction_npz"])
        with np.load(pred_path) as data:
            paths = squeeze_paths(data["paths"])
            target_gt = squeeze_path(data["target_gt"])
        path_ades = [ade_fde(path, target_gt)[0] for path in paths]
        oracle_idx = int(np.nanargmin(np.asarray(path_ades, dtype=np.float64)))
        oracle_ade = float(path_ades[oracle_idx])
        category = str(row.get("category") or "unknown")
        category_counts[category] += 1
        num_paths_histogram[int(paths.shape[0])] += 1
        count += 1

        for method in methods:
            selected, selected_idx = choose_path(paths, path_ades, method)
            ade, fde = ade_fde(selected, target_gt)
            for bucket in (buckets[method], category_buckets[category][method]):
                bucket["ade_gt_m"].append(float(ade))
                bucket["fde_gt_m"].append(float(fde))
                bucket["path_length_m"].append(path_len(selected))
                bucket["oracle_regret_ade_m"].append(float(ade) - oracle_ade)
                if selected_idx is not None:
                    bucket["oracle_hit"].append(int(selected_idx) == oracle_idx)

    overall = {method: metric_summary(bucket) for method, bucket in buckets.items()}
    oracle_mean = overall.get("oracle_best", {}).get("ade_gt_m", {}).get("mean")
    if oracle_mean is not None:
        for metrics in overall.values():
            method_mean = metrics.get("ade_gt_m", {}).get("mean")
            metrics["gap_vs_oracle_ade_m"] = None if method_mean is None else float(method_mean) - float(oracle_mean)

    by_category: dict[str, Any] = {}
    for category in sorted(category_buckets):
        methods_summary = {
            method: metric_summary(bucket)
            for method, bucket in category_buckets[category].items()
        }
        by_category[category] = {
            "count": int(category_counts[category]),
            "methods": methods_summary,
        }

    summary = {
        "tag": str(args.tag),
        "benchmark_root": str(benchmark_root),
        "model_key": model_key,
        "rows_jsonl": str(rows_path),
        "count": int(count),
        "settings": load_settings(benchmark_root, model_key),
        "num_paths_histogram": {str(k): int(v) for k, v in sorted(num_paths_histogram.items())},
        "methods": overall,
        "by_category": by_category,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(json.dumps({"event": "teacher_selection_summary_written", "json": str(args.output_json), "md": str(args.output_md)}))


if __name__ == "__main__":
    main()
