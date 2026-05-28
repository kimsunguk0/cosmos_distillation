#!/usr/bin/env python3
"""Build a balanced KV-distillation corpus from the no-nav teacher-pair JSONL.

Reads ``no_nav_teacher_pair_300chunks.jsonl`` (or any compatible JSONL),
applies the same semantic scene bucketing used in
``85_build_semantic_scene_balanced_corpus.py``, then over/under-samples train
rows to the requested target size.  All non-train rows (val / test) are copied
through unchanged.

Outputs
-------
data/corpus/kv_distill_7k_balanced.jsonl  (default)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "corpus" / "kv_distill_7k_balanced.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "corpus" / "kv_distill_7k_balanced_summary.json"

SEMANTIC_CATEGORIES = [
    "traffic_right_turn",
    "traffic_left_turn",
    "right_turn_no_light",
    "left_turn_no_light",
    "red_light_stop",
    "stop_sign",
    "pedestrian_crosswalk",
    "cut_in_merge_yield",
    "lead_vehicle_follow",
    "parked_stopped_obstacle_nudge",
    "lane_change",
    "curve",
    "green_light_go_straight",
    "intersection_other",
    "slow_decel_other",
    "keep_lane_straight",
    "other",
]


# ---------------------------------------------------------------------------
# Helpers (reused from 85_build_semantic_scene_balanced_corpus.py)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL with teacher-pair rows (default: no_nav_teacher_pair_300chunks.jsonl).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output balanced JSONL path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to write a JSON summary of the balancing run.",
    )
    parser.add_argument(
        "--target-train-samples",
        type=int,
        default=7000,
        help="Total balanced train rows to produce (default: 7000).",
    )
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument(
        "--copy-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy non-train (val/test) rows through to the output (default: True).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def teacher_text(row: dict) -> str:
    return str(
        (row.get("teacher_target") or {}).get("cot_text")
        or (row.get("hard_target") or {}).get("cot_text")
        or ""
    ).lower()


def has(text: str, *subs: str) -> bool:
    return any(sub in text for sub in subs)


def scene_bucket(row: dict) -> str:
    """Assign a semantic scene category based on teacher CoT text."""
    text = teacher_text(row)
    priority = [
        ("traffic_right_turn", lambda t: has(t, "traffic light", "green light", "red light") and has(t, "turn right", "right turn")),
        ("traffic_left_turn", lambda t: has(t, "traffic light", "green light", "red light") and has(t, "turn left", "left turn")),
        ("right_turn_no_light", lambda t: has(t, "turn right", "right turn")),
        ("left_turn_no_light", lambda t: has(t, "turn left", "left turn")),
        ("red_light_stop", lambda t: has(t, "red light", "light is red", "traffic light is red")),
        ("stop_sign", lambda t: has(t, "stop sign", "all-way stop")),
        ("pedestrian_crosswalk", lambda t: has(t, "pedestrian", "crosswalk")),
        ("cut_in_merge_yield", lambda t: has(t, "cut-in", "cut in", "merge", "merges into our lane")),
        ("lead_vehicle_follow", lambda t: has(t, "lead vehicle", "directly ahead in our lane", "vehicle ahead", "follow the vehicle")),
        ("parked_stopped_obstacle_nudge", lambda t: has(t, "nudge", "parked car", "parked vehicle", "parked cars", "stopped vehicle", "blocking")),
        ("lane_change", lambda t: has(t, "lane change", "change lane", "change lanes")),
        ("curve", lambda t: has(t, "curve", "curvature")),
        ("green_light_go_straight", lambda t: has(t, "green light", "light is green", "traffic light is green")),
        ("intersection_other", lambda t: has(t, "intersection")),
        ("slow_decel_other", lambda t: has(t, "slow down", "decelerate", "deceleration", "slow")),
        ("keep_lane_straight", lambda t: has(t, "keep lane", "lane is clear", "keep speed", "straight")),
    ]
    for name, fn in priority:
        if fn(text):
            return name
    return "other"


def annotate(row: dict, *, category: str, repeat_index: int | None = None) -> dict:
    out = copy.deepcopy(row)
    metadata = out.setdefault("metadata", {})
    metadata["semantic_scene_category"] = category
    metadata["semantic_scene_category_source"] = "teacher_cot_semantic_rule_v1"
    if repeat_index is not None:
        metadata["semantic_scene_balance_repeat_index"] = int(repeat_index)
    weights = out.setdefault("weights", {})
    weights["semantic_scene_balance_category"] = category
    return out


def balanced_take(group: list[dict], target: int, rng: random.Random, category: str) -> list[dict]:
    if target <= 0 or not group:
        return []
    order = list(range(len(group)))
    out: list[dict] = []
    repeat_index = 0
    while len(out) < target:
        rng.shuffle(order)
        for index in order:
            out.append(annotate(group[index], category=category, repeat_index=repeat_index))
            if len(out) >= target:
                break
        repeat_index += 1
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = read_jsonl(args.input_jsonl)
    print(f"Loaded {len(rows)} rows from {args.input_jsonl}", flush=True)

    grouped_train: dict[str, list[dict]] = defaultdict(list)
    passthrough: list[dict] = []
    original_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        category = scene_bucket(row)
        split = str(row.get("split") or "unknown")
        original_counts[split][category] += 1
        if split == "train":
            grouped_train[category].append(row)
        elif args.copy_val:
            passthrough.append(annotate(row, category=category))

    missing = [c for c in SEMANTIC_CATEGORIES if not grouped_train.get(c)]
    if missing:
        print(f"WARNING: categories with zero train rows (will get 0 in output): {missing}", flush=True)

    base = int(args.target_train_samples) // len(SEMANTIC_CATEGORIES)
    remainder = int(args.target_train_samples) % len(SEMANTIC_CATEGORIES)
    targets = {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(SEMANTIC_CATEGORIES)
    }

    balanced_train: list[dict] = []
    for category in SEMANTIC_CATEGORIES:
        group = grouped_train.get(category, [])
        balanced_train.extend(balanced_take(group, targets[category], rng, category))
    rng.shuffle(balanced_train)

    output_rows = balanced_train + passthrough
    write_jsonl(args.output_jsonl, output_rows)
    print(f"Wrote {len(output_rows)} rows ({len(balanced_train)} train + {len(passthrough)} val/test) to {args.output_jsonl}", flush=True)

    sampled_counts = Counter(
        (row.get("metadata") or {}).get("semantic_scene_category") for row in balanced_train
    )
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "seed": int(args.seed),
        "target_train_samples": int(args.target_train_samples),
        "categories": SEMANTIC_CATEGORIES,
        "original_counts": {split: dict(counter) for split, counter in original_counts.items()},
        "sampled_train_counts": dict(sampled_counts),
        "sampled_train_total": len(balanced_train),
        "passthrough_non_train_total": len(passthrough),
        "oversample_factors": {
            category: round(float(sampled_counts[category]) / max(len(grouped_train.get(category, [])), 1), 4)
            for category in SEMANTIC_CATEGORIES
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
