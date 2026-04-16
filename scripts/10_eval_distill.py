#!/usr/bin/env python3
"""WP14 entrypoint: v3.2 corpus evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.metrics import exact_match_rate, jaccard_overlap, load_jsonl
from src.utils.runtime_paths import DEFAULT_STATE_ROOT, DEFAULT_TEACHER_CACHE_ROOT, remap_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2.jsonl",
    )
    parser.add_argument(
        "--teacher-index-jsonl",
        type=Path,
        default=DEFAULT_TEACHER_CACHE_ROOT / "text" / "index.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "eval_summary_v3_2.json",
    )
    parser.add_argument(
        "--consistency-parquet",
        type=Path,
        default=DEFAULT_STATE_ROOT / "split_semantic_gate.parquet",
    )
    return parser.parse_args()


def _path_exists(raw_path: str | Path | None) -> bool:
    if raw_path in (None, ""):
        return False
    path_str = remap_external_path(raw_path)
    return path_str is not None and Path(path_str).exists()


def main() -> None:
    args = parse_args()
    corpus = load_jsonl(args.corpus_jsonl)
    teacher_index = {record["sample_id"]: record for record in load_jsonl(args.teacher_index_jsonl)}
    consistency_df = pd.read_parquet(args.consistency_parquet) if args.consistency_parquet.exists() else pd.DataFrame()
    consistency_map = (
        consistency_df.set_index("sample_id").to_dict(orient="index")
        if not consistency_df.empty
        else {}
    )

    motion_pairs = []
    overlap_scores = []
    teacher_present = 0
    teacher_view_allowed = 0
    action_aux_allowed = 0
    teacher_topk_ready = 0
    traj_ready = 0
    teacher_human_levels = []
    teacher_gt_intent_levels = []
    for sample in corpus:
        sample_id = str(sample["sample_id"])
        teacher_record = teacher_index.get(sample_id)
        teacher_target = sample.get("teacher_target") or {}
        hard_target = sample.get("hard_target") or {}
        gate = sample.get("gate") or {}
        derived = sample.get("derived") or {}

        if teacher_record:
            teacher_present += 1
        teacher_view_allowed += int(bool(gate.get("teacher_view_allowed")))
        action_aux_allowed += int(bool(gate.get("action_aux_allowed")))
        teacher_topk_ready += int(_path_exists(teacher_target.get("topk_logits_path")))
        traj_ready += int(
            bool(hard_target.get("traj_future_token_ids")) and _path_exists(hard_target.get("traj_future_token_ids_path"))
        )

        teacher_motion = teacher_target.get("teacher_motion_class") or derived.get("teacher_motion_class")
        gt_motion = derived.get("gt_motion_class")
        if teacher_motion and gt_motion:
            motion_pairs.append((teacher_motion, gt_motion))
        teacher_cot = teacher_target.get("cot_text")
        human_cot = hard_target.get("cot_text")
        if teacher_cot and human_cot:
            overlap_scores.append(jaccard_overlap(teacher_cot, human_cot))

        consistency_row = consistency_map.get(sample_id, {})
        teacher_human_levels.append(str(consistency_row.get("teacher_vs_human_motion") or "missing"))
        teacher_gt_intent_levels.append(str(consistency_row.get("teacher_vs_gt_intent") or "missing"))

    summary = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "consistency_parquet": str(args.consistency_parquet),
        "records": len(corpus),
        "teacher_ready_records": teacher_present,
        "teacher_view_allowed_rate": (teacher_view_allowed / len(corpus)) if corpus else 0.0,
        "action_aux_allowed_rate": (action_aux_allowed / len(corpus)) if corpus else 0.0,
        "teacher_topk_ready_rate": (teacher_topk_ready / len(corpus)) if corpus else 0.0,
        "traj_ready_rate": (traj_ready / len(corpus)) if corpus else 0.0,
        "teacher_gt_motion_exact_match": exact_match_rate(motion_pairs),
        "teacher_human_cot_overlap": (sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0,
        "teacher_human_disagreement_rate": (
            sum(1 for level in teacher_human_levels if level not in {"pass", "soft_pass"}) / len(teacher_human_levels)
            if teacher_human_levels
            else 0.0
        ),
        "teacher_gt_intent_disagreement_rate": (
            sum(1 for level in teacher_gt_intent_levels if level not in {"pass", "soft_pass"}) / len(teacher_gt_intent_levels)
            if teacher_gt_intent_levels
            else 0.0
        ),
        "notes": [
            "Evaluation reflects v3.2 corpus coverage, gate policy, and teacher/GT consistency diagnostics.",
            "Checkpoint-level generation metrics should be measured with scripts/15_infer_student_smoke.py.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
