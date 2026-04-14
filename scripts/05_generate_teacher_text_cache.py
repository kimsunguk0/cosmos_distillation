#!/usr/bin/env python3
"""WP5 entrypoint: teacher text cache request generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.teacher_cache import (
    build_hallucination_flags,
    diagnostics_bundle_for_sample,
    dump_json,
    field_text_hash,
    inspect_teacher_sample,
    load_jsonl_by_key,
    normalize_teacher_action_class,
    prompt_bundle_for_sample,
    selection_score_from_outputs,
    validate_teacher_runtime_bundle,
    versions_match,
    write_jsonl,
)
from src.data.schema_versions import KD_SCHEMA_VERSION, TEACHER_SIGNAL_CACHE_VERSION, active_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "text",
    )
    parser.add_argument(
        "--question",
        default="Explain the chain of causation for the ego vehicle.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_text_cache_summary.json",
    )
    return parser.parse_args()


def attach_teacher_metadata_defaults(output: dict) -> dict:
    """Backfill provenance and quality fields for legacy teacher outputs."""
    recovered_output = dict(output)
    long_cot = recovered_output.get("teacher_long_cot")
    short_reason = recovered_output.get("teacher_short_reason")
    meta_action = recovered_output.get("teacher_meta_action")
    answer = recovered_output.get("teacher_answer")
    overlap = bool(short_reason and answer and str(short_reason).strip().lower() == str(answer).strip().lower())
    collapse = bool(
        (short_reason and long_cot and str(short_reason).strip() == str(long_cot).strip())
        or (answer and long_cot and str(answer).strip() == str(long_cot).strip())
    )

    recovered_output.setdefault(
        "teacher_long_cot_source",
        "legacy_recovered_output" if long_cot else "missing",
    )
    recovered_output.setdefault("teacher_long_cot_direct", False)
    recovered_output.setdefault("teacher_structured_json", None)
    recovered_output.setdefault("teacher_structured_json_source", "missing")
    recovered_output.setdefault("teacher_structured_json_direct", False)
    recovered_output.setdefault(
        "teacher_short_reason_source",
        "legacy_recovered_output" if short_reason else "missing",
    )
    recovered_output.setdefault("teacher_short_reason_direct", False)
    recovered_output.setdefault(
        "teacher_meta_action_source",
        "normalized_label_from_text" if meta_action else "missing",
    )
    recovered_output.setdefault("teacher_meta_action_direct", False)
    recovered_output.setdefault(
        "teacher_answer_source",
        "legacy_recovered_output" if answer else "missing",
    )
    recovered_output.setdefault("teacher_answer_direct", False)
    recovered_output.setdefault(
        "slot_channel_behavior",
        {
            "cot_nonempty": bool(long_cot),
            "meta_action_nonempty": False,
            "answer_nonempty": False,
            "channel_collapse_detected": collapse,
            "all_text_from_same_channel": collapse,
        },
    )
    recovered_output.setdefault(
        "teacher_text_quality",
        "downweight" if long_cot else "reject",
    )
    recovered_output.setdefault(
        "teacher_structured_quality",
        "reject",
    )
    recovered_output.setdefault("teacher_direct_slot_reliability", "low")
    recovered_output.setdefault("teacher_answer_short_reason_overlap", overlap)
    recovered_output.setdefault("selected_direct_prompt_family", "long_cot_v1" if long_cot else None)
    recovered_output.setdefault("selected_normalization_mode", "legacy_recovered_output")
    recovered_output.setdefault(
        "sample_supervision_mode",
        "long_cot_only_fallback" if long_cot else "fallback_text_only",
    )

    quality_multiplier = 1.0
    if meta_action and not recovered_output.get("teacher_meta_action_direct"):
        quality_multiplier *= 0.75
    if answer and not recovered_output.get("teacher_answer_direct"):
        quality_multiplier *= 0.8
    if recovered_output["slot_channel_behavior"].get("channel_collapse_detected"):
        quality_multiplier *= 0.5
    if overlap:
        quality_multiplier *= 0.95
    if recovered_output.get("teacher_parse_status") == "json_synthesized_fallback":
        quality_multiplier = min(quality_multiplier, 0.25)
    recovered_output.setdefault("teacher_quality_multiplier", round(quality_multiplier, 4))
    recovered_signal_fields = ["teacher_short_reason", "teacher_answer"]
    if recovered_output.get("sample_supervision_mode") == "long_cot_only_fallback":
        recovered_signal_fields = ["teacher_long_cot"] if long_cot else []
    recovered_output.setdefault(
        "teacher_signal_schema",
        {
            "kd_schema_version": KD_SCHEMA_VERSION,
            "teacher_signal_cache_version": TEACHER_SIGNAL_CACHE_VERSION,
            "signal_fields": recovered_signal_fields,
            "seq_only_fields": [
                field_name
                for field_name, field_value in (
                    ("teacher_structured_json", recovered_output.get("teacher_structured_json")),
                    ("teacher_long_cot", long_cot),
                )
                if field_value and field_name not in recovered_signal_fields
            ],
            "label_only_fields": ["teacher_meta_action"] if recovered_output.get("teacher_meta_action") else [],
        },
    )
    recovered_output.setdefault("teacher_signal_targets", {})
    recovered_output.setdefault("teacher_logit_cache_path", None)
    recovered_output.setdefault("teacher_hidden_path", None)
    return recovered_output


def recover_existing_output(sample_id: str, cache_root: Path, existing_record: dict | None) -> tuple[str | None, dict | None]:
    """Recover an already-generated teacher output from index or raw cache artifacts."""
    if existing_record and versions_match(existing_record):
        output = existing_record.get("output") or {}
        if existing_record.get("status") == "ok" and any(
            output.get(field_name)
            for field_name in ("teacher_long_cot", "teacher_short_reason", "teacher_answer", "teacher_structured_json")
        ):
            action_record = normalize_teacher_action_class(
                meta_action=output.get("teacher_meta_action"),
                answer=output.get("teacher_answer"),
                short_reason=output.get("teacher_short_reason"),
                long_cot=output.get("teacher_long_cot"),
            )
            recovered_output = dict(output)
            recovered_output["teacher_long_cot"] = recovered_output.get("teacher_long_cot") or output.get("teacher_short_reason")
            recovered_output["teacher_answer"] = recovered_output.get("teacher_answer") or output.get("teacher_short_reason")
            recovered_output["teacher_meta_action"] = recovered_output.get("teacher_meta_action") or (
                action_record["value"] if action_record["value"] != "unknown" else None
            )
            recovered_output["teacher_action_class"] = recovered_output.get("teacher_action_class") or action_record["value"]
            recovered_output["teacher_action_confidence"] = recovered_output.get("teacher_action_confidence") or action_record["confidence"]
            recovered_output["teacher_action_source_field"] = recovered_output.get("teacher_action_source_field") or action_record.get("source_field")
            recovered_output["teacher_selection_score"] = recovered_output.get("teacher_selection_score") or selection_score_from_outputs(
                long_cot=recovered_output.get("teacher_long_cot"),
                json_payload=None,
                meta_action=recovered_output.get("teacher_meta_action"),
                answer=recovered_output.get("teacher_answer"),
            )
            recovered_output["teacher_parse_status"] = recovered_output.get("teacher_parse_status") or "json_unknown"
            recovered_output["teacher_hallucination_flags"] = recovered_output.get("teacher_hallucination_flags") or build_hallucination_flags(
                long_cot=recovered_output.get("teacher_long_cot"),
                json_payload=None,
                meta_action=recovered_output.get("teacher_meta_action"),
                answer=recovered_output.get("teacher_answer"),
            )
            recovered_output["teacher_selection_prompt"] = recovered_output.get("teacher_selection_prompt")
            return "ok", attach_teacher_metadata_defaults(recovered_output)

    raw_output_path = cache_root / "outputs" / f"{sample_id}.teacher_raw.json"
    if not raw_output_path.exists():
        return None, None

    raw_payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
    raw_versions = raw_payload.get("versions") or {}
    if any(raw_versions.get(key) != value for key, value in active_versions().items()):
        return None, None
    normalized = raw_payload.get("normalized_output") or {}
    action_record = normalize_teacher_action_class(
        meta_action=normalized.get("teacher_meta_action"),
        answer=normalized.get("teacher_answer"),
        short_reason=normalized.get("teacher_short_reason"),
        long_cot=normalized.get("teacher_long_cot"),
    )
    recovered_output = {
        "teacher_long_cot": normalized.get("teacher_long_cot"),
        "teacher_short_reason": normalized.get("teacher_short_reason") or normalized.get("teacher_long_cot"),
        "teacher_meta_action": normalized.get("teacher_meta_action") or (action_record["value"] if action_record["value"] != "unknown" else None),
        "teacher_answer": normalized.get("teacher_answer") or normalized.get("teacher_short_reason") or normalized.get("teacher_long_cot"),
        "teacher_structured_json": normalized.get("teacher_structured_json"),
        "teacher_action_class": normalized.get("teacher_action_class") or action_record["value"],
        "teacher_action_confidence": normalized.get("teacher_action_confidence") or action_record["confidence"],
        "teacher_action_source_field": normalized.get("teacher_action_source_field") or action_record.get("source_field"),
        "teacher_selection_prompt": normalized.get("teacher_selection_prompt"),
        "selected_direct_prompt_family": normalized.get("selected_direct_prompt_family"),
        "selected_normalization_mode": normalized.get("selected_normalization_mode"),
        "sample_supervision_mode": normalized.get("sample_supervision_mode"),
        "teacher_selection_score": normalized.get("teacher_selection_score")
        or selection_score_from_outputs(
            long_cot=normalized.get("teacher_long_cot"),
            json_payload=None,
            meta_action=normalized.get("teacher_meta_action"),
            answer=normalized.get("teacher_answer"),
        ),
        "teacher_parse_status": normalized.get("teacher_parse_status") or "json_unknown",
        "teacher_hallucination_flags": normalized.get("teacher_hallucination_flags")
        or build_hallucination_flags(
            long_cot=normalized.get("teacher_long_cot"),
            json_payload=None,
            meta_action=normalized.get("teacher_meta_action"),
            answer=normalized.get("teacher_answer"),
        ),
        "teacher_structured_json_path": str(raw_output_path),
        "teacher_signal_targets": {},
        "teacher_logit_cache_path": None,
        "teacher_hidden_path": None,
    }
    if any(recovered_output.get(field_name) for field_name in ("teacher_long_cot", "teacher_short_reason", "teacher_answer", "teacher_structured_json")):
        return "ok", attach_teacher_metadata_defaults(recovered_output)
    return None, attach_teacher_metadata_defaults(recovered_output)


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)
    supervision_map = load_jsonl_by_key(args.supervision_jsonl)
    existing_index = load_jsonl_by_key(args.cache_root / "index.jsonl")

    requests_dir = args.cache_root / "requests"
    diagnostics_dir = args.cache_root / "diagnostics"
    requests_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    prompt_names = [
        "long_cot_v1",
        "answer_vqa_v1",
        "meta_action_vqa_v1",
    ]
    records: list[dict] = []
    status_counts: dict[str, int] = {}

    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        readiness = inspect_teacher_sample(sample_id, args.canonical_root)
        supervision = supervision_map.get(sample_id)

        blockers = list(readiness.blockers)
        status = readiness.status
        if supervision is None:
            blockers.append("supervision_record_missing")
            status = "blocked"

        runtime_bundle_path = requests_dir / f"{sample_id}.runtime_request.json"
        diagnostics_bundle_path = diagnostics_dir / f"{sample_id}.diagnostics.json"
        bundle = prompt_bundle_for_sample(
            sample_id=sample_id,
            canonical_sample_path=args.canonical_root / sample_id,
            prompt_names=prompt_names,
            question=args.question,
        )
        validate_teacher_runtime_bundle(bundle)
        diagnostics_bundle = diagnostics_bundle_for_sample(
            sample_id=sample_id,
            canonical_sample_path=args.canonical_root / sample_id,
            manifest_row=row.to_dict(),
            supervision_record=supervision,
        )
        dump_json(runtime_bundle_path, bundle)
        dump_json(diagnostics_bundle_path, diagnostics_bundle)

        existing_record = existing_index.get(sample_id)
        recovered_status, recovered_output = recover_existing_output(sample_id, args.cache_root, existing_record)
        if recovered_status == "ok":
            status = "ready"
            blockers = []

        record = {
            "sample_id": sample_id,
            "clip_uuid": str(row["clip_uuid"]),
            "split": str(row["subset_split"] if "subset_split" in row else row["split"]),
            "status": recovered_status or ("ready_request_bundle" if status == "ready" else status),
            "blockers": blockers,
            "runtime_request_path": str(runtime_bundle_path),
            "diagnostics_bundle_path": str(diagnostics_bundle_path),
            "canonical_sample_path": str(args.canonical_root / sample_id),
            "versions": active_versions(),
            "output": recovered_output or {
                "teacher_long_cot": None,
                "teacher_structured_json": None,
                "teacher_short_reason": None,
                "teacher_meta_action": None,
                "teacher_answer": None,
                "teacher_long_cot_source": "missing",
                "teacher_long_cot_direct": False,
                "teacher_structured_json_source": "missing",
                "teacher_structured_json_direct": False,
                "teacher_short_reason_source": "missing",
                "teacher_short_reason_direct": False,
                "teacher_meta_action_source": "missing",
                "teacher_meta_action_direct": False,
                "teacher_answer_source": "missing",
                "teacher_answer_direct": False,
                "slot_channel_behavior": {
                    "cot_nonempty": False,
                    "meta_action_nonempty": False,
                    "answer_nonempty": False,
                    "channel_collapse_detected": False,
                    "all_text_from_same_channel": False,
                },
                "teacher_text_quality": "reject",
                "teacher_structured_quality": "reject",
                "teacher_direct_slot_reliability": "low",
                "teacher_answer_short_reason_overlap": False,
                "teacher_quality_multiplier": 1.0,
                "teacher_signal_schema": {
                    "kd_schema_version": KD_SCHEMA_VERSION,
                    "teacher_signal_cache_version": TEACHER_SIGNAL_CACHE_VERSION,
                    "signal_fields": ["teacher_short_reason", "teacher_answer"],
                    "seq_only_fields": ["teacher_structured_json", "teacher_long_cot"],
                    "label_only_fields": ["teacher_meta_action"],
                },
                "teacher_signal_targets": {},
                "teacher_structured_json_path": None,
                "teacher_logit_cache_path": None,
                "teacher_hidden_path": None,
            },
            "provenance": {
                "soft": "teacher_text",
                "source": "alpamayo_generate_text" if recovered_status == "ok" else "request_bundle_only",
                "teacher_is_gt": False,
            },
        }
        if record["status"] == "ok":
            for field_name in ("teacher_short_reason", "teacher_answer", "teacher_structured_json", "teacher_long_cot"):
                field_value = record["output"].get(field_name)
                if not field_value:
                    continue
                record["output"]["teacher_signal_targets"].setdefault(
                    field_name,
                    {
                        "field_name": field_name,
                        "text_hash": field_text_hash(field_value),
                        "signal_ready": False,
                        "signal_cache_stale": True,
                        "source": record["output"].get(f"{field_name}_source", "missing"),
                    },
                )
        records.append(record)
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

    index_path = args.cache_root / "index.jsonl"
    write_jsonl(index_path, records)

    summary = {
        "manifest_path": str(args.manifest_path),
        "supervision_jsonl": str(args.supervision_jsonl),
        "canonical_root": str(args.canonical_root),
        "cache_root": str(args.cache_root),
        "index_path": str(index_path),
        "request_bundle_count": len(records),
        "status_counts": status_counts,
        "ready_for_teacher_generation": status_counts.get("ready_request_bundle", 0),
        "notes": [
            "This step scaffolds teacher requests without generating teacher outputs.",
            "Alpamayo inference is deferred until image frames exist for the canonical sample.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
