#!/usr/bin/env python3
"""Compare two decode summary JSON files with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric_value(row: dict[str, Any], key: str) -> float | None:
    if key == "bad_geometry":
        ade = finite(row.get("ade_m", row.get("student_vs_teacher_discrete_ade_m")))
        fde = finite(row.get("fde_m", row.get("student_vs_teacher_discrete_fde_m")))
        return 1.0 if ((ade is not None and ade >= 8.0) or (fde is not None and fde >= 25.0)) else 0.0
    if key == "motion_match":
        value = row.get("motion_match")
        return None if value is None else finite(value)
    aliases = {
        "ade_m": ("ade_m", "student_vs_teacher_discrete_ade_m", "ade_vs_teacher_m"),
        "fde_m": ("fde_m", "student_vs_teacher_discrete_fde_m", "fde_vs_teacher_m"),
        "unique_traj_ids": ("unique_traj_ids", "generated_unique_token_count"),
        "max_same_token_run": ("max_same_token_run", "generated_max_same_token_run"),
    }
    for candidate in aliases.get(key, (key,)):
        value = finite(row.get(candidate))
        if value is not None:
            return value
    return None


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise SystemExit(f"{path} does not contain a samples list")
    return payload


def summarize_delta(values: list[float], *, bootstrap: int, seed: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    if arr.size == 1 or bootstrap <= 0:
        low = high = float(arr.mean())
    else:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, arr.size, size=(int(bootstrap), arr.size))
        means = arr[draws].mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5]).tolist()
    return {"n": int(arr.size), "mean": float(arr.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def main() -> int:
    args = parse_args()
    baseline = load_summary(args.baseline_summary)
    candidate = load_summary(args.candidate_summary)
    baseline_rows = {
        str(row.get("sample_id")): row
        for row in baseline["samples"]
        if row.get("sample_id") is not None
    }
    candidate_rows = {
        str(row.get("sample_id")): row
        for row in candidate["samples"]
        if row.get("sample_id") is not None
    }
    sample_ids = sorted(set(baseline_rows) & set(candidate_rows))
    metrics = (
        "ade_m",
        "fde_m",
        "bad_geometry",
        "unique_traj_ids",
        "max_same_token_run",
        "motion_match",
        "target_set_jaccard",
    )
    metric_payload: dict[str, Any] = {}
    for key in metrics:
        base_values: list[float] = []
        cand_values: list[float] = []
        deltas: list[float] = []
        for sample_id in sample_ids:
            base = metric_value(baseline_rows[sample_id], key)
            cand = metric_value(candidate_rows[sample_id], key)
            if base is None or cand is None:
                continue
            base_values.append(base)
            cand_values.append(cand)
            deltas.append(cand - base)
        delta_summary = summarize_delta(deltas, bootstrap=int(args.bootstrap), seed=int(args.seed))
        metric_payload[key] = {
            "n": int(len(deltas)),
            "baseline_mean": float(np.mean(base_values)) if base_values else None,
            "candidate_mean": float(np.mean(cand_values)) if cand_values else None,
            "delta_candidate_minus_baseline": delta_summary["mean"],
            "delta_ci95_low": delta_summary["ci95_low"],
            "delta_ci95_high": delta_summary["ci95_high"],
        }

    payload = {
        "baseline_name": str(args.baseline_name),
        "candidate_name": str(args.candidate_name),
        "baseline_summary": str(args.baseline_summary),
        "candidate_summary": str(args.candidate_summary),
        "baseline_num_samples": baseline.get("num_samples"),
        "candidate_num_samples": candidate.get("num_samples"),
        "matched_samples": len(sample_ids),
        "metrics": metric_payload,
        "baseline_top_level": {key: baseline.get(key) for key in baseline if key != "samples"},
        "candidate_top_level": {key: candidate.get(key) for key in candidate if key != "samples"},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = [
        f"# Decode Summary Compare: {args.candidate_name} vs {args.baseline_name}",
        "",
        f"Matched samples: `{len(sample_ids)}`",
        "",
        "| metric | baseline mean | candidate mean | delta candidate-baseline | 95% CI | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in metric_payload.items():
        lines.append(
            f"| {key} | {fmt(item['baseline_mean'])} | {fmt(item['candidate_mean'])} | "
            f"{fmt(item['delta_candidate_minus_baseline'])} | "
            f"[{fmt(item['delta_ci95_low'])}, {fmt(item['delta_ci95_high'])}] | {item['n']} |"
        )
    lines.extend(
        [
            "",
            "## Top-Level",
            "",
            f"- baseline score: `{baseline.get('free_run_geometry_score', baseline.get('avg_ade_m'))}`",
            f"- candidate score: `{candidate.get('free_run_geometry_score', candidate.get('avg_ade_m'))}`",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "matched": len(sample_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
