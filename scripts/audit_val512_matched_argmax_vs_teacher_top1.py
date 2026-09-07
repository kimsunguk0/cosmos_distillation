#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SEGMENTS: list[tuple[str, int, int]] = [
    ("pos_001_016", 0, 16),
    ("pos_017_032", 16, 32),
    ("pos_033_064", 32, 64),
    ("pos_065_096", 64, 96),
    ("pos_097_128", 96, 128),
    ("pos_001_128", 0, 128),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def topk_path(row: dict[str, Any]) -> Path | None:
    target = row.get("teacher_traj_target") or {}
    for key in ("topk_logits_path", "topk_logprobs_path", "topk_ids_path"):
        raw = target.get(key)
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    return None


def load_topk(row: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = topk_path(row)
    if path is None:
        return None, None
    with np.load(path, allow_pickle=False) as data:
        if "topk_indices" not in data:
            return None, None
        topk = np.asarray(data["topk_indices"], dtype=np.int64)
        if topk.ndim != 2 or topk.shape[0] < 128:
            return None, None
        target = None
        if "target_token_ids" in data:
            target = np.asarray(data["target_token_ids"], dtype=np.int64).reshape(-1)[:128]
        return topk[:128], target


def load_cache_target(row: dict[str, Any]) -> np.ndarray | None:
    hard = row.get("hard_target") or {}
    inline = hard.get("traj_future_token_ids") or []
    if inline:
        arr = np.asarray([int(v) for v in inline], dtype=np.int64).reshape(-1)
        return arr[:128] if arr.size >= 128 else None
    path = hard.get("traj_future_token_ids_path") or (row.get("teacher_traj_target") or {}).get("token_ids_path")
    if path and Path(path).exists():
        arr = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
        return arr[:128] if arr.size >= 128 else None
    return None


def first_mismatch(matches: np.ndarray) -> int:
    misses = np.flatnonzero(~matches)
    return int(misses[0] + 1) if misses.size else 129


def rate(matches: np.ndarray) -> float:
    if matches.size == 0:
        return float("nan")
    return float(np.mean(matches))


def segment_rates(matches_stack: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, lo, hi in SEGMENTS:
        out[name] = rate(matches_stack[:, lo:hi].reshape(-1))
    out["accel_even"] = rate(matches_stack[:, 0::2].reshape(-1))
    out["curv_odd"] = rate(matches_stack[:, 1::2].reshape(-1))
    return out


def summarize_distances(diff_stack: np.ndarray, matches_stack: np.ndarray) -> dict[str, Any]:
    miss = ~matches_stack
    return {
        "all": quantiles(diff_stack.reshape(-1)),
        "mismatch_only": quantiles(diff_stack[miss]),
        "accel_even_all": quantiles(diff_stack[:, 0::2].reshape(-1)),
        "curv_odd_all": quantiles(diff_stack[:, 1::2].reshape(-1)),
        "accel_even_mismatch": quantiles(diff_stack[:, 0::2][~matches_stack[:, 0::2]]),
        "curv_odd_mismatch": quantiles(diff_stack[:, 1::2][~matches_stack[:, 1::2]]),
    }


def summarize_model(
    *,
    label: str,
    samples_by_id: dict[str, dict[str, Any]],
    teacher_topk_by_id: dict[str, np.ndarray],
    cache_target_by_id: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    match_top1_rows: list[np.ndarray] = []
    match_target_rows: list[np.ndarray] = []
    top5_rows: list[np.ndarray] = []
    top10_rows: list[np.ndarray] = []
    diff_rows: list[np.ndarray] = []
    first_mismatches: list[float] = []
    per_sample: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    missing = 0

    for sample_id, topk in teacher_topk_by_id.items():
        sample = samples_by_id.get(sample_id)
        cache = cache_target_by_id.get(sample_id)
        if sample is None or cache is None:
            missing += 1
            continue
        argmax = np.asarray(sample.get("tf_argmax_token_ids") or [], dtype=np.int64).reshape(-1)
        target = np.asarray(sample.get("target_traj_token_ids") or cache, dtype=np.int64).reshape(-1)
        if argmax.size < 128 or target.size < 128 or topk.shape[0] < 128:
            missing += 1
            continue
        argmax = argmax[:128]
        target = target[:128]
        teacher_top1 = topk[:128, 0]
        match_top1 = argmax == teacher_top1
        match_target = argmax == target
        top5 = np.any(topk[:128, :5] == argmax[:, None], axis=1)
        top10 = np.any(topk[:128, :10] == argmax[:, None], axis=1)
        diff = np.abs(argmax - teacher_top1)

        match_top1_rows.append(match_top1)
        match_target_rows.append(match_target)
        top5_rows.append(top5)
        top10_rows.append(top10)
        diff_rows.append(diff)
        first_mismatches.append(float(first_mismatch(match_top1)))
        per_sample.append(
            {
                "sample_id": sample_id,
                "match_teacher_top1": rate(match_top1),
                "match_cache_target": rate(match_target),
                "argmax_in_teacher_top5": rate(top5),
                "argmax_in_teacher_top10": rate(top10),
                "first_mismatch_teacher_top1": first_mismatch(match_top1),
                "bin_abs_diff_mean": float(np.mean(diff)),
                "bin_abs_diff_p50": float(np.quantile(diff, 0.5)),
                "bin_abs_diff_p95": float(np.quantile(diff, 0.95)),
            }
        )

    if not match_top1_rows:
        return {"label": label, "missing": missing, "num_samples": 0}, [], []

    match_top1_stack = np.stack(match_top1_rows, axis=0)
    match_target_stack = np.stack(match_target_rows, axis=0)
    top5_stack = np.stack(top5_rows, axis=0)
    top10_stack = np.stack(top10_rows, axis=0)
    diff_stack = np.stack(diff_rows, axis=0)

    for pos in range(128):
        position_rows.append(
            {
                "label": label,
                "position_1indexed": pos + 1,
                "argmax_vs_teacher_top1": rate(match_top1_stack[:, pos]),
                "argmax_vs_cache_target": rate(match_target_stack[:, pos]),
                "argmax_in_teacher_top5": rate(top5_stack[:, pos]),
                "argmax_in_teacher_top10": rate(top10_stack[:, pos]),
                "bin_abs_diff_mean": float(np.mean(diff_stack[:, pos])),
                "bin_abs_diff_p50": float(np.quantile(diff_stack[:, pos], 0.5)),
                "bin_abs_diff_p95": float(np.quantile(diff_stack[:, pos], 0.95)),
            }
        )

    summary = {
        "label": label,
        "num_samples": int(match_top1_stack.shape[0]),
        "missing": int(missing),
        "argmax_vs_teacher_top1": {
            "overall": rate(match_top1_stack.reshape(-1)),
            "segments": segment_rates(match_top1_stack),
        },
        "argmax_vs_cache_target": {
            "overall": rate(match_target_stack.reshape(-1)),
            "segments": segment_rates(match_target_stack),
        },
        "argmax_in_teacher_top5": rate(top5_stack.reshape(-1)),
        "argmax_in_teacher_top10": rate(top10_stack.reshape(-1)),
        "first_mismatch_teacher_top1": quantiles(first_mismatches),
        "bin_abs_diff_vs_teacher_top1": summarize_distances(diff_stack, match_top1_stack),
    }
    return summary, per_sample, position_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--model-samples", action="append", nargs=2, metavar=("LABEL", "JSONL"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    corpus = read_jsonl(args.corpus_jsonl)
    teacher_topk_by_id: dict[str, np.ndarray] = {}
    cache_target_by_id: dict[str, np.ndarray] = {}
    npz_target_by_id: dict[str, np.ndarray] = {}
    missing_topk = 0
    missing_target = 0
    target_top1_rows: list[np.ndarray] = []

    for row in corpus:
        sample_id = row.get("sample_id")
        if not sample_id:
            continue
        topk, npz_target = load_topk(row)
        cache_target = load_cache_target(row)
        if topk is None:
            missing_topk += 1
            continue
        if cache_target is None:
            missing_target += 1
            continue
        teacher_topk_by_id[str(sample_id)] = topk
        cache_target_by_id[str(sample_id)] = cache_target
        if npz_target is not None and npz_target.size >= 128:
            npz_target_by_id[str(sample_id)] = npz_target[:128]
        target_top1_rows.append(cache_target[:128] == topk[:128, 0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    all_per_sample: list[dict[str, Any]] = []
    all_position: list[dict[str, Any]] = []

    for label, samples_path in args.model_samples:
        rows = read_jsonl(Path(samples_path))
        samples_by_id = {str(row.get("sample_id")): row for row in rows if row.get("sample_id")}
        summary, per_sample, position_rows = summarize_model(
            label=label,
            samples_by_id=samples_by_id,
            teacher_topk_by_id=teacher_topk_by_id,
            cache_target_by_id=cache_target_by_id,
        )
        summaries[label] = summary
        for row in per_sample:
            row = {"label": label, **row}
            all_per_sample.append(row)
        all_position.extend(position_rows)

    target_top1_stack = np.stack(target_top1_rows, axis=0) if target_top1_rows else np.zeros((0, 128), dtype=bool)
    summary_out = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "num_corpus_rows": len(corpus),
        "num_rows_with_topk": len(teacher_topk_by_id),
        "missing_topk": missing_topk,
        "missing_target": missing_target,
        "cache_target_vs_teacher_top1": {
            "overall": rate(target_top1_stack.reshape(-1)) if target_top1_stack.size else float("nan"),
            "segments": segment_rates(target_top1_stack) if target_top1_stack.size else {},
        },
        "models": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_out, indent=2, sort_keys=True), encoding="utf-8")

    if all_per_sample:
        with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_per_sample[0].keys()))
            writer.writeheader()
            writer.writerows(all_per_sample)
    if all_position:
        with (args.output_dir / "position_curve.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_position[0].keys()))
            writer.writeheader()
            writer.writerows(all_position)

    print(json.dumps(summary_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
