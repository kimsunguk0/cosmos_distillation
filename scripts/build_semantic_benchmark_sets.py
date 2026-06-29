#!/usr/bin/env python3
"""Build deterministic semantic-category benchmark JSONLs.

The source validation split is naturally imbalanced, so this builder uses a
per-category cap instead of forcing equal counts from train data. With the
current full444k semantic corpus, cap=50 yields 806 validation samples.
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
DEFAULT_OUTPUT = PROJECT_ROOT / "data/corpus/benchmark_semantic_val_cap50_seed42.jsonl"
DEFAULT_VIS_OUTPUT = PROJECT_ROOT / "data/corpus/benchmark_semantic_vis4_seed42.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vis-output-jsonl", type=Path, default=DEFAULT_VIS_OUTPUT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--per-category-cap", type=int, default=50)
    parser.add_argument("--vis-per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def category(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("semantic_scene_category") or "unknown")


def valid_row(row: dict[str, Any]) -> bool:
    materialized = Path(str((row.get("input") or {}).get("materialized_sample_path") or ""))
    raw_json = Path(str((row.get("teacher_cache") or {}).get("text_raw_json_path") or ""))
    return materialized.exists() and raw_json.exists()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    usable_counts: Counter[str] = Counter()
    scanned = 0
    for row in iter_jsonl(args.source_jsonl):
        scanned += 1
        if str(row.get("split")) != str(args.split):
            continue
        cat = category(row)
        source_counts[cat] += 1
        if not valid_row(row):
            continue
        usable_counts[cat] += 1
        by_cat[cat].append(row)

    selected: list[dict[str, Any]] = []
    vis_selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    vis_counts: Counter[str] = Counter()
    for cat in sorted(by_cat):
        pool = list(by_cat[cat])
        rng.shuffle(pool)
        cat_selected = pool[: int(args.per_category_cap)]
        selected.extend(cat_selected)
        selected_counts[cat] = len(cat_selected)
        cat_vis = cat_selected[: int(args.vis_per_category)]
        vis_selected.extend(cat_vis)
        vis_counts[cat] = len(cat_vis)

    selected.sort(key=lambda row: (category(row), str(row.get("sample_id"))))
    vis_selected.sort(key=lambda row: (category(row), str(row.get("sample_id"))))

    write_jsonl(args.output_jsonl, selected)
    write_jsonl(args.vis_output_jsonl, vis_selected)

    summary = {
        "source_jsonl": str(args.source_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "vis_output_jsonl": str(args.vis_output_jsonl),
        "split": str(args.split),
        "seed": int(args.seed),
        "per_category_cap": int(args.per_category_cap),
        "vis_per_category": int(args.vis_per_category),
        "scanned_rows": int(scanned),
        "selected_total": len(selected),
        "vis_total": len(vis_selected),
        "source_counts": dict(sorted(source_counts.items())),
        "usable_counts": dict(sorted(usable_counts.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "vis_counts": dict(sorted(vis_counts.items())),
        "rare_categories_below_cap": {
            cat: count
            for cat, count in sorted(selected_counts.items())
            if count < int(args.per_category_cap)
        },
    }
    summary_path = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "benchmark_sets_written", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
