#!/usr/bin/env python3
"""WP7 entrypoint: consistency gate execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.consistency import grade_action_pair, load_gt_path_action_class, summarize_pair_levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "strict_download_subset_manifest.parquet",
    )
    parser.add_argument(
        "--supervision-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "supervision_records" / "records.jsonl",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "canonical_samples",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "diagnostics" / "consistency_matrix.parquet",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "consistency_summary.md",
    )
    return parser.parse_args()


def load_supervision_map(path: Path) -> dict[str, dict]:
    """Read JSONL supervision records into a sample-id keyed dict."""
    result: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            result[record["sample_id"]] = record
    return result


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)
    supervision_map = load_supervision_map(args.supervision_jsonl)

    rows: list[dict] = []
    for _, row in manifest.iterrows():
        sample_id = row["sample_id"]
        supervision = supervision_map.get(sample_id, {})
        weak = supervision.get("weak_derived", {})
        human_record = weak.get("meta_action_from_human")
        human_action = human_record["value"] if human_record else None
        gt_action, gt_metrics = load_gt_path_action_class(args.canonical_root, sample_id)

        pair_level = None
        pair_notes: list[str] = []
        if human_action and gt_action:
            pair = grade_action_pair(human_action, gt_action)
            pair_level = pair.consistency_level
            pair_notes = pair.notes

        rows.append(
            {
                "sample_id": sample_id,
                "split": row["subset_split"] if "subset_split" in row else row["split"],
                "human_action_class": human_action,
                "gt_path_action_class": gt_action,
                "teacher_text_action_class": None,
                "teacher_traj_action_class": None,
                "human_reasoning__gt_path": pair_level,
                "teacher_text__gt_path": None,
                "human_reasoning__teacher_traj": None,
                "teacher_text__teacher_traj": None,
                "gt_path_metrics_json": json.dumps(gt_metrics) if gt_metrics else None,
                "notes_json": json.dumps(pair_notes),
            }
        )

    result = pd.DataFrame(rows)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_parquet, index=False)

    lines = [
        "# Consistency Summary",
        "",
        f"- Samples: `{len(result)}`",
        f"- Samples with GT path materialized: `{int(result['gt_path_action_class'].notna().sum())}`",
        f"- Samples with human action class: `{int(result['human_action_class'].notna().sum())}`",
        "",
        "## human_reasoning__gt_path",
    ]
    for label, count in summarize_pair_levels(result["human_reasoning__gt_path"]).items():
        lines.append(f"- `{label}`: `{count}`")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {args.output_parquet}")
    print(f"[done] {args.summary_md}")


if __name__ == "__main__":
    main()
