#!/usr/bin/env python3
"""WP8 entrypoint: safe multitask corpus build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.corpus_builder import validate_task_type
from src.data.teacher_cache import load_jsonl_by_key


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
        "--output-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "strict_human_long_cot.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "corpus_summary.json",
    )
    parser.add_argument(
        "--teacher-index-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "text" / "index.jsonl",
    )
    return parser.parse_args()


def load_supervision_map(path: Path) -> dict[str, dict]:
    """Read JSONL supervision records into a dict."""
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
    teacher_index = load_jsonl_by_key(args.teacher_index_jsonl)
    validate_task_type("human_long_cot")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts_by_split: dict[str, int] = {}
    teacher_soft_target_count = 0
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for _, row in manifest.iterrows():
            sample_id = row["sample_id"]
            supervision = supervision_map[sample_id]
            hard_human = supervision["hard_human"]
            weak_derived = supervision["weak_derived"]
            teacher_record = teacher_index.get(str(sample_id))
            teacher_output = teacher_record.get("output", {}) if teacher_record else {}
            teacher_ready = bool(
                teacher_record
                and teacher_record.get("status") == "ok"
                and teacher_output.get("teacher_short_reason")
            )
            task = {
                "sample_id": f"{sample_id}__human_long_cot",
                "source_sample_id": sample_id,
                "task_type": "human_long_cot",
                "input": {
                    "canonical_sample_path": f"data/processed/canonical_samples/{sample_id}",
                    "question": "Explain the chain of causation for the ego vehicle.",
                    "camera_names": [
                        "camera_cross_left_120fov",
                        "camera_front_wide_120fov",
                        "camera_cross_right_120fov",
                        "camera_front_tele_30fov",
                    ],
                },
                "target": {
                    "text": hard_human["human_coc"],
                },
                "soft_target": {
                    "teacher_short_reason": teacher_output.get("teacher_short_reason"),
                    "teacher_meta_action": teacher_output.get("teacher_meta_action"),
                    "teacher_answer": teacher_output.get("teacher_answer"),
                    "teacher_json_path": teacher_output.get("teacher_structured_json_path"),
                    "teacher_logit_cache_path": teacher_output.get("teacher_logit_cache_path"),
                },
                "provenance": {
                    "hard": "human",
                    "soft": "teacher_text" if teacher_ready else None,
                    "mixed_pair_allowed": False,
                },
                "weights": {
                    "hard_ce": 1.0,
                    "seq_kd": 0.6 if teacher_ready else 0.0,
                    "logit_kd": 0.5 if teacher_ready and teacher_output.get("teacher_logit_cache_path") else 0.0,
                    "feat": 0.15 if teacher_ready and teacher_output.get("teacher_hidden_path") else 0.0,
                },
                "derived": weak_derived,
                "split": row["subset_split"] if "subset_split" in row else row["split"],
            }
            handle.write(json.dumps(task, ensure_ascii=True) + "\n")
            split = task["split"]
            counts_by_split[split] = counts_by_split.get(split, 0) + 1
            teacher_soft_target_count += int(teacher_ready)

    summary = {
        "manifest_path": str(args.manifest_path),
        "output_jsonl": str(args.output_jsonl),
        "task_type": "human_long_cot",
        "counts_by_split": counts_by_split,
        "teacher_soft_target_count": teacher_soft_target_count,
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "forbidden_tasks_checked": [
            "teacher_reasoning_plus_gt_path",
            "human_reasoning_plus_teacher_path",
            "teacher_discrete_future_tokens_as_gt",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
