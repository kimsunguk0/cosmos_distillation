#!/usr/bin/env python3
"""Compare student greedy trajectory tokens against cached teacher top-1 bins.

This is an offline diagnostic.  Teacher top-1 is read from the first column of
the cached trajectory top-k NPZ, which is conditioned on the cached teacher
trajectory prefix used when the signal was extracted.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl",
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
        "--current-10b-rows-jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/benchmarks/val512_full_metrics_10b_fullft_lora_20260715/teacher10b_vlm_discrete_n1/rows.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmarks/val512_student_teacher_top1_alignment_20260715",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary_samples(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("sample_id")): row for row in data.get("samples") or [] if row.get("sample_id")}


def load_rows_as_samples(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        tokens = row.get("selected_traj_tokens") or row.get("generated_traj_tokens") or []
        if not sample_id or not tokens:
            continue
        out[sample_id] = {
            **row,
            "generated_traj_tokens": tokens,
            "ade_m": row.get("ade_gt_m"),
            "fde_m": row.get("fde_gt_m"),
        }
    return out


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
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


def first_mismatch(matches: np.ndarray) -> int:
    misses = np.flatnonzero(~matches)
    return int(misses[0] + 1) if misses.size else 129


def summarize_model(
    *,
    label: str,
    samples_by_id: dict[str, dict[str, Any]],
    teacher_top1_by_id: dict[str, np.ndarray],
    cache_target_by_id: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    position_hits: list[np.ndarray] = []
    position_cache_hits: list[np.ndarray] = []
    prefix_survival: list[np.ndarray] = []
    per_sample: list[dict[str, Any]] = []
    rates: list[float] = []
    cache_rates: list[float] = []
    first_mismatches: list[float] = []
    missing = 0

    for sample_id, teacher_top1 in teacher_top1_by_id.items():
        sample = samples_by_id.get(sample_id)
        if sample is None:
            missing += 1
            continue
        student = np.asarray(sample.get("generated_traj_tokens") or [], dtype=np.int64).reshape(-1)
        cache = cache_target_by_id.get(sample_id)
        if student.shape[0] != 128 or teacher_top1.shape[0] != 128:
            missing += 1
            continue
        matches = student == teacher_top1
        cache_matches = student == cache if cache is not None and cache.shape[0] == 128 else np.zeros_like(matches)
        survival = np.cumprod(matches.astype(np.int32)).astype(bool)
        rate = float(np.mean(matches))
        cache_rate = float(np.mean(cache_matches))
        fm = first_mismatch(matches)
        position_hits.append(matches)
        position_cache_hits.append(cache_matches)
        prefix_survival.append(survival)
        rates.append(rate)
        cache_rates.append(cache_rate)
        first_mismatches.append(float(fm))
        per_sample.append(
            {
                "sample_id": sample_id,
                "model": label,
                "teacher_top1_match_rate": rate,
                "cache_target_match_rate": cache_rate,
                "first_teacher_top1_mismatch_position_1based": fm,
                "ade_gt_m": float(sample.get("ade_m", float("nan"))),
                "fde_gt_m": float(sample.get("fde_m", float("nan"))),
            }
        )

    hit_arr = np.stack(position_hits, axis=0) if position_hits else np.zeros((0, 128), dtype=bool)
    cache_hit_arr = np.stack(position_cache_hits, axis=0) if position_cache_hits else np.zeros((0, 128), dtype=bool)
    survival_arr = np.stack(prefix_survival, axis=0) if prefix_survival else np.zeros((0, 128), dtype=bool)

    def seg(start: int, end: int) -> float:
        if hit_arr.size == 0:
            return float("nan")
        return float(np.mean(hit_arr[:, start:end]))

    summary = {
        "model": label,
        "n": int(hit_arr.shape[0]),
        "missing": int(missing),
        "teacher_top1_match_rate": quantiles(rates),
        "cache_target_match_rate": quantiles(cache_rates),
        "first_mismatch_position_1based": quantiles(first_mismatches),
        "position_segments": {
            "1_16": seg(0, 16),
            "17_32": seg(16, 32),
            "33_64": seg(32, 64),
            "65_96": seg(64, 96),
            "97_128": seg(96, 128),
            "odd_positions": float(np.mean(hit_arr[:, 0::2])) if hit_arr.size else float("nan"),
            "even_positions": float(np.mean(hit_arr[:, 1::2])) if hit_arr.size else float("nan"),
        },
        "prefix_survival": {
            "after_1": float(np.mean(survival_arr[:, 0])) if survival_arr.size else float("nan"),
            "after_2": float(np.mean(survival_arr[:, 1])) if survival_arr.size else float("nan"),
            "after_4": float(np.mean(survival_arr[:, 3])) if survival_arr.size else float("nan"),
            "after_8": float(np.mean(survival_arr[:, 7])) if survival_arr.size else float("nan"),
            "after_16": float(np.mean(survival_arr[:, 15])) if survival_arr.size else float("nan"),
            "after_32": float(np.mean(survival_arr[:, 31])) if survival_arr.size else float("nan"),
            "after_64": float(np.mean(survival_arr[:, 63])) if survival_arr.size else float("nan"),
            "after_128": float(np.mean(survival_arr[:, 127])) if survival_arr.size else float("nan"),
        },
    }
    return summary, per_sample, hit_arr


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(args.corpus_jsonl)
    fullft_samples = load_summary_samples(args.fullft_summary)
    lora_samples = load_summary_samples(args.lora_summary)
    current10b_samples = load_rows_as_samples(args.current_10b_rows_jsonl)

    teacher_top1_by_id: dict[str, np.ndarray] = {}
    cache_target_by_id: dict[str, np.ndarray] = {}
    teacher_top1_prob_positions: list[np.ndarray] = []
    cache_target_top1_positions: list[np.ndarray] = []
    missing_topk = 0
    for row in corpus:
        sample_id = str(row.get("sample_id") or "")
        path = topk_path(row)
        if path is None:
            missing_topk += 1
            continue
        z = np.load(path)
        topk_indices = np.asarray(z["topk_indices"], dtype=np.int64)
        topk_logprobs = np.asarray(z["topk_logprobs"], dtype=np.float64)
        target_ids = np.asarray(z["target_token_ids"], dtype=np.int64).reshape(-1)
        if topk_indices.shape[0] != 128 or target_ids.shape[0] != 128:
            missing_topk += 1
            continue
        teacher_top1 = topk_indices[:, 0]
        teacher_top1_by_id[sample_id] = teacher_top1
        cache_target_by_id[sample_id] = target_ids
        teacher_top1_prob_positions.append(np.exp(topk_logprobs[:, 0]))
        cache_target_top1_positions.append(target_ids == teacher_top1)

    fullft_summary, fullft_per_sample, fullft_hits = summarize_model(
        label="FullFT_3e-5_200K_greedy",
        samples_by_id=fullft_samples,
        teacher_top1_by_id=teacher_top1_by_id,
        cache_target_by_id=cache_target_by_id,
    )
    lora_summary, lora_per_sample, lora_hits = summarize_model(
        label="LoRA_2e-4_200K_greedy",
        samples_by_id=lora_samples,
        teacher_top1_by_id=teacher_top1_by_id,
        cache_target_by_id=cache_target_by_id,
    )
    current10b_summary, current10b_per_sample, current10b_hits = summarize_model(
        label="Current_10B_greedy_vs_cached_prefix_top1",
        samples_by_id=current10b_samples,
        teacher_top1_by_id=teacher_top1_by_id,
        cache_target_by_id=cache_target_by_id,
    )

    teacher_top1_prob_arr = (
        np.stack(teacher_top1_prob_positions, axis=0) if teacher_top1_prob_positions else np.zeros((0, 128))
    )
    cache_target_top1_arr = (
        np.stack(cache_target_top1_positions, axis=0) if cache_target_top1_positions else np.zeros((0, 128), dtype=bool)
    )

    position_rows: list[dict[str, Any]] = []
    for pos in range(128):
        position_rows.append(
            {
                "position_1based": pos + 1,
                "coord_axis": "x_or_accel" if pos % 2 == 0 else "y_or_curvature",
                "current10b_teacher_top1_match": float(np.mean(current10b_hits[:, pos]))
                if current10b_hits.size
                else float("nan"),
                "fullft_teacher_top1_match": float(np.mean(fullft_hits[:, pos])) if fullft_hits.size else float("nan"),
                "lora_teacher_top1_match": float(np.mean(lora_hits[:, pos])) if lora_hits.size else float("nan"),
                "cache_target_is_teacher_top1": float(np.mean(cache_target_top1_arr[:, pos]))
                if cache_target_top1_arr.size
                else float("nan"),
                "teacher_top1_prob_mean": float(np.mean(teacher_top1_prob_arr[:, pos]))
                if teacher_top1_prob_arr.size
                else float("nan"),
            }
        )

    summary = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "fullft_summary": str(args.fullft_summary),
        "lora_summary": str(args.lora_summary),
        "current_10b_rows_jsonl": str(args.current_10b_rows_jsonl),
        "n_teacher_topk": len(teacher_top1_by_id),
        "missing_topk": missing_topk,
        "teacher_cache_target_vs_teacher_top1": {
            "overall_rate": float(np.mean(cache_target_top1_arr)) if cache_target_top1_arr.size else float("nan"),
            "segments": {
                "1_16": float(np.mean(cache_target_top1_arr[:, 0:16])),
                "17_32": float(np.mean(cache_target_top1_arr[:, 16:32])),
                "33_64": float(np.mean(cache_target_top1_arr[:, 32:64])),
                "65_96": float(np.mean(cache_target_top1_arr[:, 64:96])),
                "97_128": float(np.mean(cache_target_top1_arr[:, 96:128])),
            }
            if cache_target_top1_arr.size
            else {},
        },
        "teacher_top1_prob_by_position": {
            "overall": quantiles(teacher_top1_prob_arr.reshape(-1).tolist()) if teacher_top1_prob_arr.size else {},
        },
        "models": {
            "current10b": current10b_summary,
            "fullft": fullft_summary,
            "lora": lora_summary,
        },
    }

    summary_path = args.output_dir / "summary.json"
    positions_path = args.output_dir / "position_curve.csv"
    per_sample_path = args.output_dir / "per_sample.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    with positions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(position_rows[0].keys()))
        writer.writeheader()
        writer.writerows(position_rows)

    per_rows = current10b_per_sample + fullft_per_sample + lora_per_sample
    with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_rows)

    print(json.dumps({"summary": str(summary_path), "position_curve": str(positions_path), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
