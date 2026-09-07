#!/usr/bin/env python3
"""Offline audits for cached Alpamayo trajectory teacher targets.

The script does not run any model.  It compares cached sampled teacher
trajectory tokens against an already decoded current-greedy 10B run, checks the
stored top-k probability mass, and measures cached-teacher geometry against GT.
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

from src.inference.checkpoint_eval import (  # noqa: E402
    TrajectoryTokenDecoder,
    load_ego_history_rot,
    resolve_traj_tokenizer_config_path,
)
from src.training.collator import (  # noqa: E402
    load_ego_future_xyz,
    load_ego_history_xyz,
    load_traj_future_token_ids,
)


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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmarks/val512_teacher_cache_offline_audit_20260715",
    )
    parser.add_argument("--student-model", type=Path, default=PROJECT_ROOT / "base_weights/Cosmos-Reason2-2B")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ade_fde(pred: np.ndarray | None, target: np.ndarray | None) -> tuple[float, float]:
    if pred is None or target is None:
        return float("nan"), float("nan")
    pred_arr = np.asarray(pred, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    count = min(len(pred_arr), len(target_arr))
    if count <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred_arr[:count, :2] - target_arr[:count, :2], axis=-1)
    return float(np.mean(dist)), float(dist[-1])


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
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


def load_cache_tokens(row: dict[str, Any]) -> list[int]:
    tokens = load_traj_future_token_ids(row.get("hard_target") or {}, PROJECT_ROOT)
    if tokens:
        return tokens
    target = row.get("teacher_traj_target") or {}
    path = target.get("token_ids_path")
    if path:
        return [int(v) for v in np.load(path).reshape(-1).tolist()]
    return []


def topk_path(row: dict[str, Any]) -> Path | None:
    target = row.get("teacher_traj_target") or {}
    for key in ("topk_logits_path", "topk_logprobs_path", "topk_ids_path"):
        raw = target.get(key)
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    return None


def summarize_target_rank(topk_indices: np.ndarray, target_ids: np.ndarray) -> dict[str, Any]:
    ranks: list[int] = []
    missing = 0
    top1_matches = 0
    for pos, target in enumerate(target_ids.reshape(-1)):
        matches = np.flatnonzero(topk_indices[pos] == int(target))
        if matches.size == 0:
            missing += 1
            continue
        rank = int(matches[0]) + 1
        ranks.append(rank)
        if rank == 1:
            top1_matches += 1
    return {
        "target_in_topk_rate": float(len(ranks) / max(len(target_ids), 1)),
        "target_top1_rate": float(top1_matches / max(len(target_ids), 1)),
        "target_missing_rate": float(missing / max(len(target_ids), 1)),
        "target_rank": quantiles([float(v) for v in ranks]),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.corpus_jsonl)
    current_by_id = {
        str(row.get("sample_id")): row
        for row in load_jsonl(args.current_10b_rows_jsonl)
        if row.get("sample_id")
    }

    config_path = resolve_traj_tokenizer_config_path(args.student_model)
    if config_path is None:
        raise SystemExit("Could not resolve trajectory tokenizer config.")
    decoder = TrajectoryTokenDecoder(config_path=config_path)

    per_sample: list[dict[str, Any]] = []
    all_cache_vs_gt_ade: list[float] = []
    all_cache_vs_gt_fde: list[float] = []
    all_current_vs_cache_ade: list[float] = []
    all_current_vs_cache_fde: list[float] = []
    all_token_match: list[float] = []
    all_top1_prob: list[float] = []
    all_target_prob: list[float] = []
    all_topk_mass: list[float] = []
    all_tail_mass: list[float] = []
    all_entropy: list[float] = []
    all_target_in_topk: list[float] = []
    all_target_top1: list[float] = []

    missing_current = 0
    missing_topk = 0
    decode_failures = 0

    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        cache_tokens = load_cache_tokens(row)
        if len(cache_tokens) != 128:
            decode_failures += 1
            continue

        history_xyz = load_ego_history_xyz(row, PROJECT_ROOT)
        history_rot = load_ego_history_rot(row, PROJECT_ROOT)
        gt_xyz = load_ego_future_xyz(row, PROJECT_ROOT)
        cache_xyz = decoder.decode(history_xyz, history_rot, cache_tokens)
        cache_ade, cache_fde = ade_fde(cache_xyz, gt_xyz)
        all_cache_vs_gt_ade.append(cache_ade)
        all_cache_vs_gt_fde.append(cache_fde)

        current_tokens = list((current_by_id.get(sample_id) or {}).get("selected_traj_tokens") or [])
        current_ade = float("nan")
        current_fde = float("nan")
        token_match = float("nan")
        if len(current_tokens) == 128:
            current_xyz = decoder.decode(history_xyz, history_rot, current_tokens)
            current_ade, current_fde = ade_fde(current_xyz, cache_xyz)
            token_match = float(np.mean(np.asarray(current_tokens, dtype=np.int32) == np.asarray(cache_tokens, dtype=np.int32)))
            all_current_vs_cache_ade.append(current_ade)
            all_current_vs_cache_fde.append(current_fde)
            all_token_match.append(token_match)
        else:
            missing_current += 1

        sample_topk: dict[str, Any] = {}
        topk_npz = topk_path(row)
        if topk_npz is None:
            missing_topk += 1
        else:
            z = np.load(topk_npz)
            topk_indices = np.asarray(z["topk_indices"], dtype=np.int64)
            topk_logprobs = np.asarray(z["topk_logprobs"], dtype=np.float64)
            target_ids = np.asarray(z["target_token_ids"], dtype=np.int64).reshape(-1)
            target_logprobs = np.asarray(z["target_token_logprobs"], dtype=np.float64).reshape(-1)
            valid = np.isfinite(topk_logprobs) & (topk_logprobs > -1.0e8)
            probs = np.where(valid, np.exp(topk_logprobs), 0.0)
            masses = probs.sum(axis=1)
            tails = np.clip(1.0 - masses, 0.0, 1.0)
            top1_probs = probs[:, 0]
            target_probs = np.exp(target_logprobs)
            rank_summary = summarize_target_rank(topk_indices, target_ids)

            all_top1_prob.extend([float(v) for v in top1_probs.tolist()])
            all_target_prob.extend([float(v) for v in target_probs.tolist()])
            all_topk_mass.extend([float(v) for v in masses.tolist()])
            all_tail_mass.extend([float(v) for v in tails.tolist()])
            if "entropy" in z.files:
                entropy = np.asarray(z["entropy"], dtype=np.float64).reshape(-1)
                all_entropy.extend([float(v) for v in entropy.tolist()])
            all_target_in_topk.append(float(rank_summary["target_in_topk_rate"]))
            all_target_top1.append(float(rank_summary["target_top1_rate"]))
            sample_topk = {
                "topk_mass_mean": float(np.mean(masses)),
                "topk_mass_min": float(np.min(masses)),
                "topk_mass_max": float(np.max(masses)),
                "tail_mass_mean": float(np.mean(tails)),
                "top1_prob_mean": float(np.mean(top1_probs)),
                "target_prob_mean": float(np.mean(target_probs)),
                **rank_summary,
            }

        per_sample.append(
            {
                "sample_id": sample_id,
                "category": str((row.get("metadata") or {}).get("semantic_scene_category") or "unknown"),
                "cache_vs_gt_ade_m": cache_ade,
                "cache_vs_gt_fde_m": cache_fde,
                "current10b_greedy_vs_cache_ade_m": current_ade,
                "current10b_greedy_vs_cache_fde_m": current_fde,
                "current10b_greedy_vs_cache_token_match": token_match,
                **sample_topk,
            }
        )

    hist_bins = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float("inf")]
    hist_labels = ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", "3-5", "5-10", "10+"]

    def histogram(values: list[float]) -> dict[str, int]:
        counts = {label: 0 for label in hist_labels}
        for value in values:
            if not math.isfinite(float(value)):
                continue
            for lo, hi, label in zip(hist_bins[:-1], hist_bins[1:], hist_labels, strict=True):
                if lo <= float(value) < hi:
                    counts[label] += 1
                    break
        return counts

    summary = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "current_10b_rows_jsonl": str(args.current_10b_rows_jsonl),
        "n_rows": len(rows),
        "n_per_sample": len(per_sample),
        "missing_current_10b": missing_current,
        "missing_topk": missing_topk,
        "decode_failures": decode_failures,
        "audit_1_current10b_greedy_vs_cache": {
            "ade_m": quantiles(all_current_vs_cache_ade),
            "fde_m": quantiles(all_current_vs_cache_fde),
            "token_match_rate": quantiles(all_token_match),
            "ade_histogram_m": histogram(all_current_vs_cache_ade),
        },
        "audit_2_acc_ceiling_from_cache_topk": {
            "top1_prob": quantiles(all_top1_prob),
            "target_token_prob": quantiles(all_target_prob),
            "target_in_topk_rate_per_sample": quantiles(all_target_in_topk),
            "target_top1_rate_per_sample": quantiles(all_target_top1),
            "entropy": quantiles(all_entropy),
        },
        "audit_3_topk_mass_raw_vs_warped": {
            "topk_prob_mass": quantiles(all_topk_mass),
            "tail_mass_1_minus_topk_sum": quantiles(all_tail_mass),
            "interpretation": (
                "mass_near_1_means_saved_probs_are_renormalized_or_nearly_full_support; "
                "mass_well_below_1_means_raw_distribution tail remains."
            ),
        },
        "audit_4_cache_vs_gt": {
            "ade_m": quantiles(all_cache_vs_gt_ade),
            "fde_m": quantiles(all_cache_vs_gt_fde),
            "ade_histogram_m": histogram(all_cache_vs_gt_ade),
        },
    }

    summary_path = args.output_dir / "summary.json"
    rows_path = args.output_dir / "per_sample.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for item in per_sample for key in item.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample)

    print(json.dumps({"summary": str(summary_path), "per_sample": str(rows_path), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
