#!/usr/bin/env python3
"""Build a v3.2 multiview distillation corpus from externally-prepared assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.corpus_builder import validate_task_type
from src.utils.runtime_paths import (
    DEFAULT_MATERIALIZED_ROOT,
    DEFAULT_STATE_ROOT,
    DEFAULT_TEACHER_CACHE_ROOT,
    remap_external_path,
)


DEFAULT_QUESTION = (
    "Explain the chain of causation for the ego vehicle and then emit the future trajectory tokens."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-manifest",
        type=Path,
        default=DEFAULT_STATE_ROOT / "event_manifest.parquet",
    )
    parser.add_argument(
        "--semantic-gate-parquet",
        type=Path,
        default=DEFAULT_STATE_ROOT / "split_semantic_gate.parquet",
    )
    parser.add_argument(
        "--teacher-index-jsonl",
        type=Path,
        default=DEFAULT_TEACHER_CACHE_ROOT / "text" / "index.jsonl",
    )
    parser.add_argument(
        "--materialized-root",
        type=Path,
        default=DEFAULT_MATERIALIZED_ROOT,
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "corpus_v3_2_summary.json",
    )
    return parser.parse_args()


def load_jsonl_by_key(path: Path, key: str = "sample_id") -> dict[str, dict]:
    """Load a JSONL file into a mapping keyed by the requested field."""
    result: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(key)
            if value is not None:
                result[str(value)] = record
    return result


def load_gate_map(path: Path) -> dict[str, dict]:
    """Load the semantic gate parquet into a sample-id keyed mapping."""
    if not path.exists():
        return {}
    return {
        str(row["sample_id"]): {
            key: row[key]
            for key in row.index
        }
        for _, row in pd.read_parquet(path).iterrows()
    }


def _json_list(raw_value) -> list[str]:
    if raw_value in (None, ""):
        return []
    if isinstance(raw_value, list):
        return [str(value) for value in raw_value]
    return [str(value) for value in json.loads(raw_value)]


def _materialized_image_paths(sample_dir: Path, metadata: dict) -> list[str]:
    image_dir = sample_dir / "images"
    paths = sorted(
        image_dir.glob("cam*_f*.png"),
        key=lambda path: tuple(
            int(part[3:]) if part.startswith("cam") else int(part[1:])
            for part in path.stem.split("_")
        ),
    )
    if not paths:
        raise FileNotFoundError(f"Missing materialized images under {image_dir}")
    return [str(path) for path in paths]


def _load_materialized_sample(sample_dir: Path) -> tuple[dict, list[int], list[str]]:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    traj_token_path = sample_dir / "ego" / "traj_future_token_ids.npy"
    ego_history_path = sample_dir / "ego" / "ego_history_xyz.npy"
    if not traj_token_path.exists() or not ego_history_path.exists():
        raise FileNotFoundError(f"Missing ego tensors under {sample_dir}")
    traj_token_ids = np.load(traj_token_path).astype(int).tolist()
    image_paths = _materialized_image_paths(sample_dir, metadata)
    return metadata, traj_token_ids, image_paths


def _remapped_signal_path(signal_target: dict, field_name: str) -> str | None:
    raw_path = signal_target.get(field_name)
    remapped = remap_external_path(raw_path)
    if remapped is None:
        return None
    return remapped if Path(remapped).exists() else None


def main() -> None:
    args = parse_args()
    validate_task_type("reasoning_traj_multiview_v32")

    manifest = pd.read_parquet(args.event_manifest)
    teacher_index = load_jsonl_by_key(args.teacher_index_jsonl)
    gate_map = load_gate_map(args.semantic_gate_parquet)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    skipped_missing_materialized = 0
    skipped_missing_teacher = 0
    skipped_missing_traj = 0
    teacher_view_allowed_count = 0
    teacher_topk_ready_count = 0
    action_aux_allowed_count = 0
    counts_by_split: dict[str, int] = {}
    motion_gate_counts: dict[str, int] = {}

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for _, row in manifest.iterrows():
            sample_id = str(row["sample_id"])
            sample_dir = args.materialized_root / sample_id
            if not sample_dir.exists():
                skipped_missing_materialized += 1
                continue

            try:
                metadata, traj_token_ids, image_paths = _load_materialized_sample(sample_dir)
            except FileNotFoundError as exc:
                if "traj_future_token_ids" in str(exc):
                    skipped_missing_traj += 1
                else:
                    skipped_missing_materialized += 1
                continue

            teacher_record = teacher_index.get(sample_id)
            if teacher_record is None:
                skipped_missing_teacher += 1
                continue

            gate = gate_map.get(sample_id, {})
            teacher_output = teacher_record.get("output", {})
            teacher_signal_targets = teacher_output.get("teacher_signal_targets") or {}
            teacher_long_cot_signal = teacher_signal_targets.get("teacher_long_cot") or {}
            teacher_long_cot = teacher_output.get("teacher_long_cot")
            teacher_view_allowed = bool(gate.get("teacher_view_allowed")) and bool(teacher_long_cot)
            teacher_view_weight = float(gate.get("teacher_view_weight") or 0.0) if teacher_view_allowed else 0.0
            action_aux_allowed = bool(gate.get("action_aux_allowed"))
            action_aux_weight = float(gate.get("action_aux_weight") or 0.0) if action_aux_allowed else 0.0
            topk_logits_path = _remapped_signal_path(
                {
                    "path": teacher_long_cot_signal.get("topk_logits_path") or teacher_long_cot_signal.get("logits_path")
                },
                "path",
            )
            pooled_hidden_path = _remapped_signal_path(
                {
                    "path": teacher_long_cot_signal.get("hidden_path") or teacher_long_cot_signal.get("pooled_hidden_path")
                },
                "path",
            )
            teacher_topk_ready = bool(
                teacher_view_allowed
                and teacher_long_cot_signal.get("signal_ready")
                and topk_logits_path
            )

            gt_targets = metadata.get("gt_targets") or {}
            split = str(row.get("split", metadata.get("split", "train")))
            teacher_motion_gate = str(gate.get("teacher_vs_gt_motion") or "missing")
            teacher_intent_gate = str(gate.get("teacher_vs_gt_intent") or "missing")
            motion_gate_counts[teacher_motion_gate] = motion_gate_counts.get(teacher_motion_gate, 0) + 1

            record = {
                "sample_id": sample_id,
                "task_type": "reasoning_traj_multiview_v32",
                "schema_version": "cosmos_distillation_v3_2",
                "split": split,
                "input": {
                    "materialized_sample_path": str(sample_dir),
                    "metadata_path": str(sample_dir / "metadata.json"),
                    "image_paths": image_paths,
                    "ego_history_path": str(sample_dir / "ego" / "ego_history_xyz.npy"),
                    "question": DEFAULT_QUESTION,
                    "camera_indices": list(metadata.get("camera_indices") or []),
                    "num_frames_per_camera": int((metadata.get("config") or {}).get("num_frames_per_camera", 4)),
                },
                "hard_target": {
                    "cot_text": str(row.get("human_coc") or metadata.get("human_coc") or ""),
                    "traj_future_token_ids": traj_token_ids,
                    "traj_future_token_ids_path": str(sample_dir / "ego" / "traj_future_token_ids.npy"),
                    "traj_tokenizer_name": str(gt_targets.get("traj_tokenizer_name") or "DiscreteTrajectoryTokenizer"),
                    "traj_token_count": int(gt_targets.get("traj_future_token_count", len(traj_token_ids))),
                    "traj_waypoint_count": int(gt_targets.get("traj_future_waypoint_count", 64)),
                    "traj_frame": "local@t0",
                },
                "teacher_target": {
                    "cot_text": teacher_long_cot,
                    "source": teacher_output.get("teacher_long_cot_source"),
                    "topk_logits_path": topk_logits_path,
                    "pooled_hidden_path": pooled_hidden_path,
                    "signal_ready": bool(teacher_long_cot_signal.get("signal_ready")),
                    "teacher_view_allowed": teacher_view_allowed,
                    "teacher_view_weight": round(teacher_view_weight, 4),
                    "teacher_quality_multiplier": 1.0,
                    "teacher_motion_class": teacher_output.get("teacher_motion_class"),
                    "teacher_motion_confidence": teacher_output.get("teacher_motion_confidence"),
                    "teacher_intent_tags": list(teacher_output.get("teacher_intent_tags") or []),
                },
                "gate": {
                    "teacher_vs_gt_motion": teacher_motion_gate,
                    "teacher_vs_gt_intent": teacher_intent_gate,
                    "teacher_view_allowed": teacher_view_allowed,
                    "teacher_view_weight": round(teacher_view_weight, 4),
                    "action_aux_allowed": action_aux_allowed,
                    "action_aux_weight": round(action_aux_weight, 4),
                    "manual_motion_override_applied": bool(gate.get("manual_motion_override_applied")),
                    "complementary_motion_bridge_applied": bool(gate.get("complementary_motion_bridge_applied")),
                    "longitudinal_caution_bridge_applied": bool(gate.get("longitudinal_caution_bridge_applied")),
                    "notes": _json_list(gate.get("notes_json")),
                },
                "derived": {
                    "human_motion_class": gate.get("human_motion_class"),
                    "human_motion_confidence": float(gate.get("human_motion_confidence") or 0.0),
                    "human_intent_tags": _json_list(gate.get("human_intent_tags_json")),
                    "teacher_motion_class": gate.get("teacher_motion_class"),
                    "teacher_motion_confidence": float(gate.get("teacher_motion_confidence") or 0.0),
                    "teacher_intent_tags": _json_list(gate.get("teacher_intent_tags_json")),
                    "gt_motion_class": gate.get("gt_motion_class"),
                    "gt_metrics_json": gate.get("gt_metrics_json"),
                },
                "provenance": {
                    "hard_text": "human_refined_coc",
                    "soft_text": "teacher_long_cot" if teacher_long_cot else None,
                    "traj_target": "gt_future_trajectory_discrete_tokens",
                    "teacher_gt_joint_pair_forbidden": True,
                },
                "weights": {
                    "gt_cot_loss": 1.0,
                    "traj_loss": 1.0,
                    "teacher_cot_loss": round(teacher_view_weight, 4),
                    "teacher_topk_kd_loss": round(teacher_view_weight, 4) if teacher_topk_ready else 0.0,
                    "meta_action_loss": round(action_aux_weight, 4),
                    "feature_align_loss": 0.0,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            records_written += 1
            counts_by_split[split] = counts_by_split.get(split, 0) + 1
            teacher_view_allowed_count += int(teacher_view_allowed)
            teacher_topk_ready_count += int(bool(teacher_topk_ready))
            action_aux_allowed_count += int(action_aux_allowed)

    summary = {
        "event_manifest": str(args.event_manifest),
        "semantic_gate_parquet": str(args.semantic_gate_parquet),
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "materialized_root": str(args.materialized_root),
        "output_jsonl": str(args.output_jsonl),
        "records_written": records_written,
        "counts_by_split": counts_by_split,
        "teacher_view_allowed_count": teacher_view_allowed_count,
        "teacher_topk_ready_count": teacher_topk_ready_count,
        "action_aux_allowed_count": action_aux_allowed_count,
        "teacher_vs_gt_motion_counts": motion_gate_counts,
        "skipped_missing_materialized": skipped_missing_materialized,
        "skipped_missing_teacher": skipped_missing_teacher,
        "skipped_missing_traj": skipped_missing_traj,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
