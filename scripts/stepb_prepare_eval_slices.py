#!/usr/bin/env python3
"""Prepare frozen Step B eval slices from held-out rows.

The preferred split is chunk-disjoint. If the training/dev manifests already
touch every chunk, the script can fall back to clip-disjoint selection and
records that policy explicitly in the summary JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl"
DEFAULT_EXCLUDES = (
    PROJECT_ROOT / "data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl",
    PROJECT_ROOT / "data/corpus/benchmark_semantic_val_cap50_seed42.jsonl",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/corpus/benchmark_semantic_test_clipdisjoint_cap50_seed43.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=list(DEFAULT_EXCLUDES))
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train", help="Source split to sample from. Empty string disables filtering.")
    parser.add_argument("--per-category-cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--no-clip-fallback", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def category(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("semantic_scene_category") or row.get("category") or "unknown")


def collect_exclusions(paths: list[Path]) -> tuple[set[str], set[str], Counter[str]]:
    chunks: set[str] = set()
    clips: set[str] = set()
    counts: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            continue
        rows = 0
        for row in iter_jsonl(path):
            rows += 1
            if row.get("chunk_id") is not None:
                chunks.add(str(row.get("chunk_id")))
            if row.get("clip_id") is not None:
                clips.add(str(row.get("clip_id")))
        counts[str(path)] = rows
    return chunks, clips, counts


def collect_candidates(
    source: Path,
    *,
    split: str,
    excluded_chunks: set[str],
    excluded_clips: set[str],
    mode: str,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str], int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    scanned = 0
    for row in iter_jsonl(source):
        scanned += 1
        if split and str(row.get("split")) != split:
            continue
        cat = category(row)
        source_counts[cat] += 1
        chunk_id = str(row.get("chunk_id"))
        clip_id = str(row.get("clip_id"))
        if mode == "unused_chunk" and chunk_id in excluded_chunks:
            continue
        if mode == "unused_clip" and clip_id in excluded_clips:
            continue
        groups[cat].append(row)
    return groups, source_counts, scanned


def select_rows(groups: dict[str, list[dict[str, Any]]], *, cap: int, seed: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for cat in sorted(groups):
        rows = list(groups[cat])
        rng.shuffle(rows)
        chosen = rows[:cap]
        selected.extend(chosen)
        counts[cat] = len(chosen)
    selected.sort(key=lambda row: (category(row), str(row.get("sample_id"))))
    return selected, counts


def main() -> None:
    args = parse_args()
    excluded_chunks, excluded_clips, exclude_counts = collect_exclusions(args.exclude_jsonl)
    mode = "unused_chunk"
    groups, source_counts, scanned = collect_candidates(
        args.source_jsonl,
        split=args.split,
        excluded_chunks=excluded_chunks,
        excluded_clips=excluded_clips,
        mode=mode,
    )
    candidate_count = sum(len(rows) for rows in groups.values())
    if candidate_count == 0 and not args.no_clip_fallback:
        mode = "unused_clip"
        groups, source_counts, scanned = collect_candidates(
            args.source_jsonl,
            split=args.split,
            excluded_chunks=excluded_chunks,
            excluded_clips=excluded_clips,
            mode=mode,
        )
        candidate_count = sum(len(rows) for rows in groups.values())
    selected, selected_counts = select_rows(groups, cap=args.per_category_cap, seed=args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            row = dict(row)
            row["eval_slice"] = {
                "name": "stepb_frozen_test",
                "selection_mode": mode,
                "seed": args.seed,
                "per_category_cap": args.per_category_cap,
                "source_jsonl": str(args.source_jsonl),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source_jsonl": str(args.source_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "split": args.split,
        "selection_mode": mode,
        "seed": args.seed,
        "per_category_cap": args.per_category_cap,
        "exclude_jsonl_counts": dict(exclude_counts),
        "excluded_chunks": len(excluded_chunks),
        "excluded_clips": len(excluded_clips),
        "scanned_rows": scanned,
        "source_counts": dict(sorted(source_counts.items())),
        "candidate_counts": {cat: len(groups[cat]) for cat in sorted(groups)},
        "candidate_total": candidate_count,
        "selected_counts": dict(sorted(selected_counts.items())),
        "selected_total": len(selected),
        "rare_categories_below_cap": {
            cat: count for cat, count in sorted(selected_counts.items()) if count < args.per_category_cap
        },
    }
    summary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
