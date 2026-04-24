#!/usr/bin/env python3
"""Build corpus variants with teacher trajectory KD weights.

The generated records keep GT CoT/GT trajectory as the hard target. They only
add per-body-token weights for teacher trajectory top-k KD, so C/D teachers can
contribute early-horizon signal without forcing their mismatched long horizon.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


POLICIES = ("b1b_abs_quality", "b1d_horizon")
EARLY_SLICE = slice(0, 40)
MID_SLICE = slice(40, 80)
LATE_SLICE = slice(80, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2_959.jsonl",
    )
    parser.add_argument(
        "--quality-jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "reports"
        / "teacher_vs_gt_quality"
        / "teacher_vs_gt_quality_table.jsonl",
    )
    parser.add_argument("--policy", choices=(*POLICIES, "all"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_kd_weighted_corpora",
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


def load_quality_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in load_jsonl(path)}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _bin(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    return str(row.get("teacher_abs_bin") or row.get("group") or "missing")


def _weights_for_policy(policy: str, quality_row: dict[str, Any] | None) -> list[float]:
    teacher_bin = _bin(quality_row)
    prefix_good = _bool((quality_row or {}).get("prefix_teacher_good"))

    if policy == "b1b_abs_quality":
        value = {
            "A_good": 1.0,
            "B_ok": 0.5,
            "D_middle": 0.25,
            "C_bad": 0.0,
        }.get(teacher_bin, 0.0)
        return [float(value)] * 128

    if policy != "b1d_horizon":
        raise ValueError(f"Unsupported policy: {policy}")

    if teacher_bin == "A_good":
        early, mid, late = 1.0, 1.0, 1.0
    elif teacher_bin == "B_ok":
        early, mid, late = 0.9, 0.7, 0.5
    elif teacher_bin == "D_middle":
        early, mid, late = (0.5 if prefix_good else 0.2), 0.2, 0.05
    elif teacher_bin == "C_bad":
        early, mid, late = (0.3 if prefix_good else 0.0), 0.05, 0.0
    else:
        early, mid, late = 0.0, 0.0, 0.0

    weights = [0.0] * 128
    weights[EARLY_SLICE] = [float(early)] * 40
    weights[MID_SLICE] = [float(mid)] * 40
    weights[LATE_SLICE] = [float(late)] * 48
    return weights


def _bucket_means(weights: list[float]) -> dict[str, float]:
    return {
        "early": sum(weights[EARLY_SLICE]) / 40.0,
        "mid": sum(weights[MID_SLICE]) / 40.0,
        "late": sum(weights[LATE_SLICE]) / 48.0,
    }


def build_policy_corpus(
    *,
    records: list[dict[str, Any]],
    quality_by_id: dict[str, dict[str, Any]],
    quality_path: Path,
    policy: str,
    output_path: Path,
    summary_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    prefix_counts = Counter()
    weight_sums: dict[str, dict[str, float]] = defaultdict(lambda: {"early": 0.0, "mid": 0.0, "late": 0.0})
    tag_counts: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            sample_id = str(record.get("sample_id") or "")
            quality_row = quality_by_id.get(sample_id)
            teacher_bin = _bin(quality_row)
            weights = _weights_for_policy(policy, quality_row)
            bucket_means = _bucket_means(weights)

            updated = dict(record)
            updated_weights = dict(updated.get("weights") or {})
            updated_weights["teacher_traj_token_kd_weights"] = [round(float(value), 4) for value in weights]
            updated_weights["teacher_traj_kd_policy"] = policy
            updated_weights["teacher_traj_abs_bin"] = teacher_bin
            updated_weights["teacher_traj_prefix_teacher_good"] = _bool(
                (quality_row or {}).get("prefix_teacher_good")
            )
            updated["weights"] = updated_weights
            updated["teacher_traj_kd_policy"] = {
                "policy": policy,
                "teacher_abs_bin": teacher_bin,
                "prefix_teacher_good": updated_weights["teacher_traj_prefix_teacher_good"],
                "early_weight": round(bucket_means["early"], 4),
                "mid_weight": round(bucket_means["mid"], 4),
                "late_weight": round(bucket_means["late"], 4),
                "source_quality_table": str(quality_path),
            }
            handle.write(json.dumps(updated, ensure_ascii=True) + "\n")

            counts[teacher_bin] += 1
            if updated_weights["teacher_traj_prefix_teacher_good"]:
                prefix_counts[teacher_bin] += 1
            for key, value in bucket_means.items():
                weight_sums[teacher_bin][key] += float(value)
            for tag in str((quality_row or {}).get("teacher_failure_tags") or "").split(";"):
                if tag:
                    tag_counts[tag] += 1

    mean_weights_by_bin = {}
    for teacher_bin, count in counts.items():
        mean_weights_by_bin[teacher_bin] = {
            key: round(value / max(count, 1), 4)
            for key, value in weight_sums[teacher_bin].items()
        }

    summary = {
        "policy": policy,
        "input_records": len(records),
        "output_corpus": str(output_path),
        "teacher_abs_bin_counts": dict(counts),
        "prefix_teacher_good_counts": dict(prefix_counts),
        "mean_horizon_weights_by_bin": mean_weights_by_bin,
        "teacher_failure_tag_counts": dict(tag_counts.most_common()),
        "horizon_buckets": {
            "early_tokens": [0, 39],
            "mid_tokens": [40, 79],
            "late_tokens": [80, 127],
            "note": "128 trajectory body tokens = 64 waypoints x 2 tokens; early covers first 2.0s.",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.input_corpus)
    quality_by_id = load_quality_map(args.quality_jsonl)
    policies = list(POLICIES) if args.policy == "all" else [str(args.policy)]
    for policy in policies:
        output_path = args.output_dir / f"distill_v3_2_959_{policy}.jsonl"
        summary_path = args.summary_dir / f"{policy}_summary.json"
        build_policy_corpus(
            records=records,
            quality_by_id=quality_by_id,
            quality_path=args.quality_jsonl,
            policy=policy,
            output_path=output_path,
            summary_path=summary_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
