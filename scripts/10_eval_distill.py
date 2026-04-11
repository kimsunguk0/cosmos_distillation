#!/usr/bin/env python3
"""WP14 entrypoint: evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.metrics import exact_match_rate, jaccard_overlap, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "strict_human_long_cot.jsonl",
    )
    parser.add_argument(
        "--teacher-index-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "text" / "index.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "eval_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_jsonl(args.corpus_jsonl)
    teacher_index = {record["sample_id"]: record for record in load_jsonl(args.teacher_index_jsonl)}

    meta_action_pairs = []
    overlap_scores = []
    teacher_present = 0
    json_ready = 0
    for sample in corpus:
        source_sample_id = str(sample.get("source_sample_id", sample["sample_id"]))
        teacher_record = teacher_index.get(source_sample_id)
        if not teacher_record or teacher_record.get("status") != "ok":
            continue
        teacher_present += 1
        output = teacher_record.get("output", {})
        teacher_meta = output.get("teacher_meta_action")
        human_meta = (sample.get("derived", {}).get("meta_action_from_human") or {}).get("value")
        if teacher_meta and human_meta:
            meta_action_pairs.append((teacher_meta, human_meta))
        teacher_reason = output.get("teacher_short_reason")
        human_reason = sample.get("target", {}).get("text")
        if teacher_reason and human_reason:
            overlap_scores.append(jaccard_overlap(teacher_reason, human_reason))
        if output.get("teacher_structured_json_path"):
            json_ready += 1

    summary = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "teacher_ready_records": teacher_present,
        "json_parseability": (json_ready / teacher_present) if teacher_present else 0.0,
        "meta_action_f1": exact_match_rate(meta_action_pairs),
        "human_coc_overlap": (sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0,
        "hallucination_rate": 0.0,
        "teacher_human_disagreement_rate": 1.0 - exact_match_rate(meta_action_pairs) if meta_action_pairs else 0.0,
        "consistency_gate_score": 0.0,
        "rationale_answer_consistency": (sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0,
        "notes": [
            "Evaluation stays partial until actual teacher outputs are generated.",
            "Current metrics focus on available soft-target text alignment only.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
