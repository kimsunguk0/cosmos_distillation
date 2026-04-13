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
    parser.add_argument(
        "--consistency-parquet",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "diagnostics" / "consistency_matrix.parquet",
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
    consistency_df = (
        pd.read_parquet(args.consistency_parquet).set_index("sample_id")
        if args.consistency_parquet.exists()
        else pd.DataFrame().set_index(pd.Index([], name="sample_id"))
    )
    validate_task_type("human_long_cot")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    counts_by_split: dict[str, int] = {}
    teacher_soft_target_count = 0
    text_quality_counts: dict[str, int] = {}
    structured_quality_counts: dict[str, int] = {}
    quality_multiplier_values: list[float] = []
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for _, row in manifest.iterrows():
            sample_id = row["sample_id"]
            supervision = supervision_map[sample_id]
            hard_human = supervision["hard_human"]
            weak_derived = supervision["weak_derived"]
            teacher_record = teacher_index.get(str(sample_id))
            teacher_output = teacher_record.get("output", {}) if teacher_record else {}
            consistency_row = consistency_df.loc[sample_id] if sample_id in consistency_df.index else None
            teacher_ready = bool(
                teacher_record
                and teacher_record.get("status") == "ok"
                and teacher_output.get("teacher_short_reason")
            )
            teacher_pair_level = None if consistency_row is None else consistency_row.get("teacher_text__gt_path")
            teacher_consistency_score = 0.0 if consistency_row is None else float(consistency_row.get("consistency_score", 0.0))
            selection_score = float(teacher_output.get("teacher_selection_score") or 0.0)
            quality_multiplier = float(teacher_output.get("teacher_quality_multiplier") or 1.0)
            hallucination_flags = list(teacher_output.get("teacher_hallucination_flags") or [])
            text_quality = str(teacher_output.get("teacher_text_quality") or "unknown")
            structured_quality = str(teacher_output.get("teacher_structured_quality") or "unknown")
            soft_target_allowed = teacher_ready and teacher_pair_level != "hard_fail"
            seq_weight = 0.6 * selection_score * quality_multiplier if soft_target_allowed else 0.0
            rank_weight = 0.1 * quality_multiplier if soft_target_allowed and teacher_output.get("teacher_action_class") else 0.0
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
                    "teacher_long_cot": teacher_output.get("teacher_long_cot"),
                    "teacher_short_reason": teacher_output.get("teacher_short_reason"),
                    "teacher_meta_action": teacher_output.get("teacher_meta_action"),
                    "teacher_answer": teacher_output.get("teacher_answer"),
                    "teacher_long_cot_source": teacher_output.get("teacher_long_cot_source"),
                    "teacher_long_cot_direct": teacher_output.get("teacher_long_cot_direct"),
                    "teacher_short_reason_source": teacher_output.get("teacher_short_reason_source"),
                    "teacher_short_reason_direct": teacher_output.get("teacher_short_reason_direct"),
                    "teacher_meta_action_source": teacher_output.get("teacher_meta_action_source"),
                    "teacher_meta_action_direct": teacher_output.get("teacher_meta_action_direct"),
                    "teacher_answer_source": teacher_output.get("teacher_answer_source"),
                    "teacher_answer_direct": teacher_output.get("teacher_answer_direct"),
                    "slot_channel_behavior": teacher_output.get("slot_channel_behavior"),
                    "teacher_text_quality": teacher_output.get("teacher_text_quality"),
                    "teacher_structured_quality": teacher_output.get("teacher_structured_quality"),
                    "teacher_direct_slot_reliability": teacher_output.get("teacher_direct_slot_reliability"),
                    "teacher_answer_short_reason_overlap": teacher_output.get("teacher_answer_short_reason_overlap"),
                    "teacher_quality_multiplier": quality_multiplier,
                    "teacher_action_class": teacher_output.get("teacher_action_class"),
                    "teacher_parse_status": teacher_output.get("teacher_parse_status"),
                    "teacher_selection_prompt": teacher_output.get("teacher_selection_prompt"),
                    "teacher_selection_score": selection_score,
                    "teacher_hallucination_flags": hallucination_flags,
                    "teacher_json_path": teacher_output.get("teacher_structured_json_path"),
                    "teacher_logit_cache_path": teacher_output.get("teacher_logit_cache_path"),
                    "teacher_hidden_path": teacher_output.get("teacher_hidden_path"),
                    "teacher_signal_target_field": teacher_output.get("teacher_signal_target_field"),
                    "teacher_signal_target_source": teacher_output.get("teacher_signal_target_source"),
                    "teacher_signal_cache_stale": teacher_output.get("teacher_signal_cache_stale"),
                },
                "provenance": {
                    "hard": "human",
                    "soft": "teacher_text" if soft_target_allowed else None,
                    "mixed_pair_allowed": False,
                },
                "weights": {
                    "hard_ce": 1.0,
                    "seq_kd": round(seq_weight, 4),
                    "logit_kd": round(0.5 * quality_multiplier, 4) if soft_target_allowed and teacher_output.get("teacher_logit_cache_path") else 0.0,
                    "feat": round(0.15 * quality_multiplier, 4) if soft_target_allowed and teacher_output.get("teacher_hidden_path") else 0.0,
                    "rank": round(rank_weight, 4),
                },
                "derived": weak_derived,
                "consistency_score": teacher_consistency_score,
                "split": row["subset_split"] if "subset_split" in row else row["split"],
            }
            handle.write(json.dumps(task, ensure_ascii=True) + "\n")
            split = task["split"]
            counts_by_split[split] = counts_by_split.get(split, 0) + 1
            teacher_soft_target_count += int(soft_target_allowed)
            if soft_target_allowed:
                text_quality_counts[text_quality] = text_quality_counts.get(text_quality, 0) + 1
                structured_quality_counts[structured_quality] = structured_quality_counts.get(structured_quality, 0) + 1
                quality_multiplier_values.append(quality_multiplier)

    summary = {
        "manifest_path": str(args.manifest_path),
        "output_jsonl": str(args.output_jsonl),
        "task_type": "human_long_cot",
        "counts_by_split": counts_by_split,
        "teacher_soft_target_count": teacher_soft_target_count,
        "teacher_text_quality_counts": text_quality_counts,
        "teacher_structured_quality_counts": structured_quality_counts,
        "average_teacher_quality_multiplier": round(sum(quality_multiplier_values) / len(quality_multiplier_values), 4)
        if quality_multiplier_values
        else 0.0,
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "consistency_parquet": str(args.consistency_parquet),
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
