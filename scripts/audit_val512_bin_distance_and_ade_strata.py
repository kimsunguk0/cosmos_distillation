#!/usr/bin/env python3
"""Bin-distance and geometry strata diagnostics for val512 student rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.checkpoint_eval import (  # noqa: E402
    TrajectoryTokenDecoder,
    load_ego_history_rot,
    resolve_traj_tokenizer_config_path,
)
from src.training.collator import load_ego_history_xyz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl",
    )
    parser.add_argument(
        "--current-10b-rows-jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/benchmarks/val512_full_metrics_10b_fullft_lora_20260715/teacher10b_vlm_discrete_n1/rows.jsonl",
    )
    parser.add_argument(
        "--fullft-summary",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/benchmarks/val512_greedy_10b_fullft_lora_20260715_0148/fullft_lr3e5_200k_best_greedy_summary.json",
    )
    parser.add_argument(
        "--lora-summary",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/benchmarks/val512_greedy_10b_fullft_lora_20260715_0148/lprime_lora_lr2e4_200k_best_greedy_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmarks/val512_bin_distance_ade_strata_20260716",
    )
    parser.add_argument("--student-model", type=Path, default=PROJECT_ROOT / "base_weights/Cosmos-Reason2-2B")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_summary_samples(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("sample_id")): row for row in data.get("samples") or [] if row.get("sample_id")}


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
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


def ade_fde(a: np.ndarray | None, b: np.ndarray | None) -> tuple[float, float]:
    if a is None or b is None:
        return float("nan"), float("nan")
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    count = min(len(aa), len(bb))
    if count <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(aa[:count, :2] - bb[:count, :2], axis=-1)
    return float(np.mean(dist)), float(dist[-1])


def topk_path(row: dict[str, Any]) -> Path | None:
    target = row.get("teacher_traj_target") or {}
    for key in ("topk_logits_path", "topk_logprobs_path", "topk_ids_path"):
        raw = target.get(key)
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    return None


def load_cache_tokens(row: dict[str, Any]) -> list[int]:
    hard = row.get("hard_target") or {}
    inline = hard.get("traj_future_token_ids") or []
    if inline:
        return [int(v) for v in inline]
    path = hard.get("traj_future_token_ids_path") or (row.get("teacher_traj_target") or {}).get("token_ids_path")
    if path:
        return [int(v) for v in np.load(path).reshape(-1).tolist()]
    return []


def bucket(value: float) -> str:
    bounds = [
        (0.0, 0.25, "0-0.25"),
        (0.25, 0.5, "0.25-0.5"),
        (0.5, 1.0, "0.5-1"),
        (1.0, 2.0, "1-2"),
        (2.0, 3.0, "2-3"),
        (3.0, 5.0, "3-5"),
        (5.0, 10.0, "5-10"),
        (10.0, float("inf"), "10+"),
    ]
    for lo, hi, name in bounds:
        if lo <= float(value) < hi:
            return name
    return "nan"


def summarize_model(
    *,
    label: str,
    samples: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
    current10b_tokens_by_id: dict[str, list[int]],
    teacher_top1_by_id: dict[str, np.ndarray],
    cache_vs_current_ade_by_id: dict[str, float],
    decoder: TrajectoryTokenDecoder,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_abs: list[float] = []
    mismatch_abs: list[float] = []
    odd_mismatch_abs: list[float] = []
    even_mismatch_abs: list[float] = []
    ade_by_bucket: dict[str, list[float]] = {}
    fde_by_bucket: dict[str, list[float]] = {}
    per_sample: list[dict[str, Any]] = []

    for sample_id, sample in samples.items():
        row = corpus_by_id.get(sample_id)
        teacher_top1 = teacher_top1_by_id.get(sample_id)
        current_tokens = current10b_tokens_by_id.get(sample_id)
        student_tokens = [int(v) for v in sample.get("generated_traj_tokens") or []]
        if row is None or teacher_top1 is None or current_tokens is None or len(student_tokens) != 128:
            continue
        student_arr = np.asarray(student_tokens, dtype=np.int64)
        diff = np.abs(student_arr - teacher_top1.astype(np.int64))
        matches = student_arr == teacher_top1
        all_abs.extend([float(v) for v in diff.tolist()])
        mismatch = diff[~matches]
        mismatch_abs.extend([float(v) for v in mismatch.tolist()])
        odd_mismatch_abs.extend([float(v) for v in diff[1::2][~matches[1::2]].tolist()])
        even_mismatch_abs.extend([float(v) for v in diff[0::2][~matches[0::2]].tolist()])

        history_xyz = load_ego_history_xyz(row, PROJECT_ROOT)
        history_rot = load_ego_history_rot(row, PROJECT_ROOT)
        student_xyz = decoder.decode(history_xyz, history_rot, student_tokens)
        current_xyz = decoder.decode(history_xyz, history_rot, current_tokens)
        ade, fde = ade_fde(student_xyz, current_xyz)
        cache_current_ade = float(cache_vs_current_ade_by_id.get(sample_id, float("nan")))
        b = bucket(cache_current_ade)
        ade_by_bucket.setdefault(b, []).append(ade)
        fde_by_bucket.setdefault(b, []).append(fde)
        per_sample.append(
            {
                "model": label,
                "sample_id": sample_id,
                "current10b_vs_cache_ade_bucket": b,
                "current10b_vs_cache_ade_m": cache_current_ade,
                "student_vs_current10b_ade_m": ade,
                "student_vs_current10b_fde_m": fde,
                "token_top1_match_rate": float(np.mean(matches)),
                "mismatch_abs_bin_diff_p50": float(np.median(mismatch)) if mismatch.size else 0.0,
                "mismatch_abs_bin_diff_mean": float(np.mean(mismatch)) if mismatch.size else 0.0,
            }
        )

    strata_rows = []
    strata_summary = {}
    for key in ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", "3-5", "5-10", "10+", "nan"]:
        if key not in ade_by_bucket:
            continue
        stats = {
            "n": len(ade_by_bucket[key]),
            "student_vs_current10b_ade": quantiles(ade_by_bucket[key]),
            "student_vs_current10b_fde": quantiles(fde_by_bucket[key]),
        }
        strata_summary[key] = stats
        strata_rows.append(
            {
                "model": label,
                "current10b_vs_cache_ade_bucket": key,
                "n": stats["n"],
                "student_vs_current10b_ade_mean": stats["student_vs_current10b_ade"].get("mean"),
                "student_vs_current10b_ade_p50": stats["student_vs_current10b_ade"].get("p50"),
                "student_vs_current10b_ade_p95": stats["student_vs_current10b_ade"].get("p95"),
            }
        )

    return (
        {
            "model": label,
            "bin_abs_diff_all_positions": quantiles(all_abs),
            "bin_abs_diff_mismatches_only": quantiles(mismatch_abs),
            "bin_abs_diff_mismatches_even_accel": quantiles(even_mismatch_abs),
            "bin_abs_diff_mismatches_odd_curv": quantiles(odd_mismatch_abs),
            "student_vs_current10b_by_current10b_cache_noise_bucket": strata_summary,
        },
        per_sample + strata_rows,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_jsonl(args.corpus_jsonl)
    corpus_by_id = {str(row.get("sample_id")): row for row in corpus}
    current10b_rows = load_jsonl(args.current_10b_rows_jsonl)
    current10b_tokens_by_id = {
        str(row.get("sample_id")): [int(v) for v in row.get("selected_traj_tokens") or []]
        for row in current10b_rows
    }
    fullft = load_summary_samples(args.fullft_summary)
    lora = load_summary_samples(args.lora_summary)

    teacher_top1_by_id: dict[str, np.ndarray] = {}
    cache_vs_current_ade_by_id: dict[str, float] = {}
    decoder = TrajectoryTokenDecoder(config_path=resolve_traj_tokenizer_config_path(args.student_model))

    for row in corpus:
        sample_id = str(row.get("sample_id") or "")
        path = topk_path(row)
        if path is not None:
            z = np.load(path)
            teacher_top1_by_id[sample_id] = np.asarray(z["topk_indices"], dtype=np.int64)[:, 0]
        cache_tokens = load_cache_tokens(row)
        current_tokens = current10b_tokens_by_id.get(sample_id)
        if len(cache_tokens) == 128 and current_tokens is not None and len(current_tokens) == 128:
            history_xyz = load_ego_history_xyz(row, PROJECT_ROOT)
            history_rot = load_ego_history_rot(row, PROJECT_ROOT)
            cache_xyz = decoder.decode(history_xyz, history_rot, cache_tokens)
            current_xyz = decoder.decode(history_xyz, history_rot, current_tokens)
            cache_vs_current_ade_by_id[sample_id] = ade_fde(current_xyz, cache_xyz)[0]

    fullft_summary, fullft_rows = summarize_model(
        label="FullFT_3e-5_200K_greedy",
        samples=fullft,
        corpus_by_id=corpus_by_id,
        current10b_tokens_by_id=current10b_tokens_by_id,
        teacher_top1_by_id=teacher_top1_by_id,
        cache_vs_current_ade_by_id=cache_vs_current_ade_by_id,
        decoder=decoder,
    )
    lora_summary, lora_rows = summarize_model(
        label="LoRA_2e-4_200K_greedy",
        samples=lora,
        corpus_by_id=corpus_by_id,
        current10b_tokens_by_id=current10b_tokens_by_id,
        teacher_top1_by_id=teacher_top1_by_id,
        cache_vs_current_ade_by_id=cache_vs_current_ade_by_id,
        decoder=decoder,
    )

    summary = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "current_10b_rows_jsonl": str(args.current_10b_rows_jsonl),
        "fullft_summary": str(args.fullft_summary),
        "lora_summary": str(args.lora_summary),
        "n_teacher_top1": len(teacher_top1_by_id),
        "n_current10b": len(current10b_tokens_by_id),
        "current10b_vs_cache_ade": quantiles(list(cache_vs_current_ade_by_id.values())),
        "models": {
            "fullft": fullft_summary,
            "lora": lora_summary,
        },
    }
    summary_path = args.output_dir / "summary.json"
    detail_path = args.output_dir / "details.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = fullft_rows + lora_rows
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": str(summary_path), "details": str(detail_path), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
