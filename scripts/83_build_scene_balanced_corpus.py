#!/usr/bin/env python3
"""Build a scene-balanced no-nav distillation corpus.

The source no-nav corpus does not carry human-authored scene labels, so this
script attaches a coarse rule-based scene category from the teacher CoT text and
then oversamples/undersamples the train split to make the requested categories
as even as possible. Validation rows are copied through unchanged, with the
same category metadata attached for reporting.
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


DEFAULT_CATEGORIES = [
    "lead_vehicle",
    "stop_or_red_light",
    "lane_change",
    "straight_keep_lane",
    "slow_decel",
    "curve",
    "other",
    "right_turn",
    "left_turn",
]


PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "stop_or_red_light": [
        re.compile(r"\bstop\b"),
        re.compile(r"stop sign"),
        re.compile(r"red light"),
        re.compile(r"traffic light"),
        re.compile(r"brak(?:e|ing)"),
    ],
    "left_turn": [
        re.compile(r"left turn"),
        re.compile(r"turn left"),
        re.compile(r"left-turn"),
    ],
    "right_turn": [
        re.compile(r"right turn"),
        re.compile(r"turn right"),
        re.compile(r"right-turn"),
    ],
    "curve": [
        re.compile(r"\bcurve\b"),
        re.compile(r"curving"),
        re.compile(r"\bbend\b"),
        re.compile(r"\bbends\b"),
        re.compile(r"curvature"),
        re.compile(r"winding"),
    ],
    "slow_decel": [
        re.compile(r"\bslow\b"),
        re.compile(r"decel"),
        re.compile(r"reduce speed"),
        re.compile(r"\bcaution\b"),
        re.compile(r"\byield\b"),
    ],
    "lead_vehicle": [
        re.compile(r"lead vehicle"),
        re.compile(r"vehicle ahead"),
        re.compile(r"car ahead"),
        re.compile(r"\bfollow(?:ing)?\b"),
        re.compile(r"traffic ahead"),
    ],
    "lane_change": [
        re.compile(r"lane change"),
        re.compile(r"change lane"),
        re.compile(r"\bmerge\b"),
        re.compile(r"\bnudge\b"),
    ],
}


PRIORITY = [
    "stop_or_red_light",
    "left_turn",
    "right_turn",
    "curve",
    "slow_decel",
    "lead_vehicle",
    "lane_change",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--target-train-samples", type=int, default=200_000)
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument(
        "--copy-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy non-train rows through unchanged after adding category metadata.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def teacher_cot(row: dict) -> str:
    return str(
        row.get("teacher_target", {}).get("cot_text")
        or row.get("hard_target", {}).get("cot_text")
        or ""
    ).lower()


def classify_scene(row: dict) -> tuple[str, list[str]]:
    cot = teacher_cot(row)
    matches: list[str] = []
    for category, patterns in PATTERNS.items():
        if any(pattern.search(cot) for pattern in patterns):
            matches.append(category)
    for category in PRIORITY:
        if category in matches:
            return category, matches
    if any(token in cot for token in ["keep lane", "lane is clear", "clear ahead", "straight", "continue"]):
        return "straight_keep_lane", matches
    return "other", matches


def annotate(row: dict, *, category: str, matches: list[str], repeat_index: int | None = None) -> dict:
    out = copy.deepcopy(row)
    metadata = out.setdefault("metadata", {})
    metadata["scene_category"] = category
    metadata["scene_category_source"] = "teacher_cot_rule_v1"
    metadata["scene_category_matches"] = matches
    if repeat_index is not None:
        metadata["scene_balance_repeat_index"] = int(repeat_index)
    weights = out.setdefault("weights", {})
    weights["scene_balance_category"] = category
    return out


def balanced_take(group: list[tuple[dict, list[str]]], target: int, rng: random.Random, category: str) -> list[dict]:
    if target <= 0 or not group:
        return []
    order = list(range(len(group)))
    selected: list[dict] = []
    repeat_index = 0
    while len(selected) < target:
        rng.shuffle(order)
        for index in order:
            row, matches = group[index]
            selected.append(annotate(row, category=category, matches=matches, repeat_index=repeat_index))
            if len(selected) >= target:
                break
        repeat_index += 1
    return selected


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = read_jsonl(args.input_jsonl)

    categories = [str(category) for category in args.categories]
    if not categories:
        raise ValueError("--categories must not be empty")

    grouped_train: dict[str, list[tuple[dict, list[str]]]] = defaultdict(list)
    passthrough_rows: list[dict] = []
    original_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        category, matches = classify_scene(row)
        split = str(row.get("split") or "unknown")
        original_counts[split][category] += 1
        if split == "train":
            grouped_train[category].append((row, matches))
        elif args.copy_val:
            passthrough_rows.append(annotate(row, category=category, matches=matches))

    missing = [category for category in categories if not grouped_train.get(category)]
    if missing:
        raise RuntimeError(f"Cannot balance categories with zero train rows: {missing}")

    base = int(args.target_train_samples) // len(categories)
    remainder = int(args.target_train_samples) % len(categories)
    targets = {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }

    balanced_train: list[dict] = []
    for category in categories:
        balanced_train.extend(balanced_take(grouped_train[category], targets[category], rng, category))
    rng.shuffle(balanced_train)

    sampled_counts = Counter(row["metadata"]["scene_category"] for row in balanced_train)
    output_rows = balanced_train + passthrough_rows
    write_jsonl(args.output_jsonl, output_rows)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "seed": int(args.seed),
        "target_train_samples": int(args.target_train_samples),
        "categories": categories,
        "original_counts": {split: dict(counter) for split, counter in original_counts.items()},
        "sampled_train_counts": dict(sampled_counts),
        "sampled_train_total": len(balanced_train),
        "passthrough_non_train_total": len(passthrough_rows),
        "oversample_factors": {
            category: round(float(sampled_counts[category]) / max(len(grouped_train[category]), 1), 4)
            for category in categories
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
