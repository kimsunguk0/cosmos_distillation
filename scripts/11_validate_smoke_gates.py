#!/usr/bin/env python3
"""Validate staged smoke gates for the v3.2 distillation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema_versions import active_versions
from src.data.teacher_cache import load_jsonl_by_key
from src.utils.runtime_paths import DEFAULT_STATE_ROOT, DEFAULT_TEACHER_CACHE_ROOT, remap_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_STATE_ROOT / "event_manifest.parquet",
    )
    parser.add_argument(
        "--teacher-index-jsonl",
        type=Path,
        default=DEFAULT_TEACHER_CACHE_ROOT / "text" / "index.jsonl",
    )
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2.jsonl",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "smoke_gate_summary_v3_2.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _path_exists(raw_path: str | Path | None) -> bool:
    if raw_path in (None, ""):
        return False
    remapped = remap_external_path(raw_path)
    return remapped is not None and Path(remapped).exists()


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)
    sample_ids = manifest["sample_id"].astype(str).tolist()[: args.num_samples]
    teacher_index = load_jsonl_by_key(args.teacher_index_jsonl)
    corpus_records = {record["sample_id"]: record for record in load_jsonl(args.corpus_jsonl)}

    materialized_ready = 0
    teacher_text_ready = 0
    teacher_view_allowed = 0
    teacher_topk_ready = 0
    traj_ready = 0
    action_aux_allowed = 0
    sample_summaries: list[dict] = []

    for sample_id in sample_ids:
        corpus_record = corpus_records.get(sample_id, {})
        teacher_record = teacher_index.get(sample_id, {})
        sample_input = corpus_record.get("input") or {}
        hard_target = corpus_record.get("hard_target") or {}
        teacher_target = corpus_record.get("teacher_target") or {}
        gate = corpus_record.get("gate") or {}
        weights = corpus_record.get("weights") or {}

        materialized_ok = bool(sample_input.get("image_paths")) and all(
            _path_exists(path) for path in [sample_input.get("ego_history_path"), *list(sample_input.get("image_paths") or [])]
        )
        teacher_text_ok = bool(teacher_target.get("cot_text"))
        teacher_view_ok = bool(gate.get("teacher_view_allowed"))
        topk_ok = _path_exists(teacher_target.get("topk_logits_path"))
        traj_ok = bool(hard_target.get("traj_future_token_ids")) and _path_exists(hard_target.get("traj_future_token_ids_path"))
        action_aux_ok = bool(gate.get("action_aux_allowed"))

        materialized_ready += int(materialized_ok)
        teacher_text_ready += int(teacher_text_ok)
        teacher_view_allowed += int(teacher_view_ok)
        teacher_topk_ready += int(topk_ok)
        traj_ready += int(traj_ok)
        action_aux_allowed += int(action_aux_ok)

        sample_summaries.append(
            {
                "sample_id": sample_id,
                "materialized_ok": materialized_ok,
                "teacher_status": teacher_record.get("status"),
                "teacher_text_ok": teacher_text_ok,
                "teacher_view_allowed": teacher_view_ok,
                "teacher_view_weight": float(gate.get("teacher_view_weight") or 0.0),
                "teacher_topk_ready": topk_ok,
                "traj_ready": traj_ok,
                "traj_token_count": int(hard_target.get("traj_token_count") or len(hard_target.get("traj_future_token_ids") or [])),
                "action_aux_allowed": action_aux_ok,
                "action_aux_weight": float(gate.get("action_aux_weight") or 0.0),
                "teacher_vs_gt_motion": gate.get("teacher_vs_gt_motion"),
                "teacher_vs_gt_intent": gate.get("teacher_vs_gt_intent"),
                "teacher_topk_kd_weight": float(
                    weights.get("teacher_topk_kd_loss", weights.get("teacher_logit_kd") or 0.0)
                ),
            }
        )

    summary = {
        "manifest_path": str(args.manifest_path),
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "corpus_jsonl": str(args.corpus_jsonl),
        "num_samples_checked": len(sample_summaries),
        "active_versions": active_versions(),
        "materialized_ready_count": materialized_ready,
        "teacher_text_ready_count": teacher_text_ready,
        "teacher_view_allowed_count": teacher_view_allowed,
        "teacher_topk_ready_count": teacher_topk_ready,
        "traj_ready_count": traj_ready,
        "action_aux_allowed_count": action_aux_allowed,
        "samples": sample_summaries,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
