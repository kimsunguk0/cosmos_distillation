#!/usr/bin/env python3
"""Compute fixed-planning metric means and paired bootstrap confidence intervals.

This is intentionally rows.jsonl-based so it can be run after any benchmark that
writes per-sample metrics. Student rows may use either the newer
``*_vs_teacher_m`` names or the legacy ``*_10b_m`` names.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRICS = (
    "ade_gt_m",
    "fde_gt_m",
    "minade6_gt_m",
    "minfde6_gt_m",
    "ade_vs_teacher_m",
    "fde_vs_teacher_m",
    "minade6_vs_teacher_m",
    "minfde6_vs_teacher_m",
)

ALIASES = {
    "ade_vs_teacher_m": ("ade_vs_teacher_m", "ade_10b_m"),
    "fde_vs_teacher_m": ("fde_vs_teacher_m", "fde_10b_m"),
    "minade6_vs_teacher_m": ("minade6_vs_teacher_m", "minade6_10b_m"),
    "minfde6_vs_teacher_m": ("minfde6_vs_teacher_m", "minfde6_10b_m"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, help="Single rows.jsonl to summarize.")
    parser.add_argument("--baseline-rows", type=Path, help="Baseline rows.jsonl for paired delta.")
    parser.add_argument("--candidate-rows", type=Path, help="Candidate rows.jsonl for paired delta.")
    parser.add_argument("--metric", dest="metrics", action="append", help="Metric to include. Repeatable.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_value(row: dict[str, Any], key: str) -> float | None:
    for candidate in ALIASES.get(key, (key,)):
        value = row.get(candidate)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            return value_f
    return None


def summarize_values(values: list[float], *, bootstrap: int, seed: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None, "std": None}
    rng = np.random.default_rng(seed)
    if arr.size == 1 or bootstrap <= 0:
        low = high = float(arr.mean())
    else:
        draws = rng.integers(0, arr.size, size=(int(bootstrap), arr.size))
        means = arr[draws].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5]).tolist()
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def summarize_rows(rows: list[dict[str, Any]], metrics: tuple[str, ...], *, bootstrap: int, seed: int) -> dict[str, Any]:
    return {
        "count": len(rows),
        "category_counts": dict(sorted(Counter(str(row.get("category", "unknown")) for row in rows).items())),
        "metrics": {
            key: summarize_values(
                [value for row in rows if (value := metric_value(row, key)) is not None],
                bootstrap=bootstrap,
                seed=seed,
            )
            for key in metrics
        },
    }


def index_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id and sample_id not in out:
            out[sample_id] = row
    return out


def summarize_pair(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    baseline = index_by_sample(baseline_rows)
    candidate = index_by_sample(candidate_rows)
    sample_ids = sorted(set(baseline) & set(candidate))
    deltas: dict[str, Any] = {}
    for key in metrics:
        values: list[float] = []
        for sample_id in sample_ids:
            base_v = metric_value(baseline[sample_id], key)
            cand_v = metric_value(candidate[sample_id], key)
            if base_v is not None and cand_v is not None:
                values.append(float(cand_v - base_v))
        deltas[key] = summarize_values(values, bootstrap=bootstrap, seed=seed)
    return {
        "matched_samples": len(sample_ids),
        "baseline": summarize_rows(baseline_rows, metrics, bootstrap=bootstrap, seed=seed),
        "candidate": summarize_rows(candidate_rows, metrics, bootstrap=bootstrap, seed=seed),
        "delta_candidate_minus_baseline": deltas,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def write_markdown(path: Path, payload: dict[str, Any], metrics: tuple[str, ...]) -> None:
    lines = ["# Step B Planning Metric CI", ""]
    if "delta_candidate_minus_baseline" in payload:
        lines.append(f"Matched samples: {payload['matched_samples']}")
        lines.append("")
        lines.append("| metric | baseline mean | candidate mean | delta mean | delta 95% CI | n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key in metrics:
            base = payload["baseline"]["metrics"][key]
            cand = payload["candidate"]["metrics"][key]
            delta = payload["delta_candidate_minus_baseline"][key]
            lines.append(
                f"| {key} | {fmt(base['mean'])} | {fmt(cand['mean'])} | "
                f"{fmt(delta['mean'])} | [{fmt(delta['ci95_low'])}, {fmt(delta['ci95_high'])}] | {delta['n']} |"
            )
    else:
        lines.append(f"Rows: {payload['count']}")
        lines.append("")
        lines.append("| metric | mean | 95% CI | n |")
        lines.append("|---|---:|---:|---:|")
        for key in metrics:
            item = payload["metrics"][key]
            lines.append(
                f"| {key} | {fmt(item['mean'])} | [{fmt(item['ci95_low'])}, {fmt(item['ci95_high'])}] | {item['n']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = tuple(args.metrics or DEFAULT_METRICS)
    if args.rows is None and (args.baseline_rows is None or args.candidate_rows is None):
        raise SystemExit("Provide --rows or both --baseline-rows and --candidate-rows.")
    if args.rows is not None and (args.baseline_rows is not None or args.candidate_rows is not None):
        raise SystemExit("Use either --rows or paired --baseline-rows/--candidate-rows, not both.")
    if args.rows is not None:
        payload = summarize_rows(iter_jsonl(args.rows), metrics, bootstrap=args.bootstrap, seed=args.seed)
    else:
        payload = summarize_pair(
            iter_jsonl(args.baseline_rows),
            iter_jsonl(args.candidate_rows),
            metrics,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, payload, metrics)
    if not args.output_json and not args.output_md:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
