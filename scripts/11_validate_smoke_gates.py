#!/usr/bin/env python3
"""Validate staged smoke gates for the distillation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema_versions import KD_SCHEMA_VERSION, active_versions
from src.data.teacher_cache import FORBIDDEN_RUNTIME_INPUT_KEYS, load_jsonl_by_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "teacher_262_manifest.parquet",
    )
    parser.add_argument(
        "--teacher-index-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "text_v2_262" / "index.jsonl",
    )
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "strict_human_long_cot_262.jsonl",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "smoke_gate_summary.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)
    sample_ids = manifest["sample_id"].astype(str).tolist()[: args.num_samples]
    teacher_index = load_jsonl_by_key(args.teacher_index_jsonl)
    corpus_records = {record["source_sample_id"]: record for record in load_jsonl(args.corpus_jsonl)}

    t0_pass = 0
    runtime_bundle_pass = 0
    soft_field_pass = 0
    logit_kd_positive = 0
    sample_summaries: list[dict] = []

    for sample_id in sample_ids:
        canonical_dir = PROJECT_ROOT / "data" / "processed" / "canonical_samples" / sample_id
        runtime_request = PROJECT_ROOT / teacher_index[sample_id]["runtime_request_path"]
        request_payload = json.loads(runtime_request.read_text(encoding="utf-8"))

        hist_local = np.load(canonical_dir / "ego_history_xyz_local_t0.npy")
        rot_local = np.load(canonical_dir / "ego_history_rot_local_t0.npy")
        t0_ok = bool(np.allclose(hist_local[-1], np.zeros(3), atol=1e-4) and np.allclose(rot_local[-1], np.eye(3), atol=1e-4))
        if t0_ok:
            t0_pass += 1

        forbidden_present = sorted(set(request_payload["inputs"]).intersection(FORBIDDEN_RUNTIME_INPUT_KEYS))
        runtime_ok = not forbidden_present
        if runtime_ok:
            runtime_bundle_pass += 1

        teacher_record = teacher_index[sample_id]
        output = teacher_record.get("output") or {}
        signal_targets = output.get("teacher_signal_targets") or {}
        signal_fields = list((output.get("teacher_signal_schema") or {}).get("signal_fields") or [])
        signal_ready = any((signal_targets.get(name) or {}).get("signal_ready") for name in signal_fields)

        corpus_record = corpus_records.get(sample_id, {})
        soft_target = corpus_record.get("soft_target") or {}
        weights = corpus_record.get("weights") or {}
        supervision_mode = soft_target.get("sample_supervision_mode")
        long_cot_fallback_ok = (
            supervision_mode == "long_cot_only_fallback"
            and soft_target.get("teacher_long_cot") is not None
            and soft_target.get("teacher_short_reason") is None
            and soft_target.get("teacher_answer") is None
        )
        soft_fields_present = long_cot_fallback_ok or all(
            soft_target.get(name) is not None for name in ("teacher_short_reason", "teacher_answer", "teacher_meta_action")
        )
        structured_present = soft_target.get("teacher_structured_json") is not None
        if soft_fields_present:
            soft_field_pass += 1
        if float(weights.get("logit_kd") or 0.0) > 0:
            logit_kd_positive += 1

        sample_summaries.append(
            {
                "sample_id": sample_id,
                "t0_ok": t0_ok,
                "runtime_bundle_ok": runtime_ok,
                "forbidden_runtime_keys": forbidden_present,
                "teacher_status": teacher_record.get("status"),
                "teacher_versions": teacher_record.get("versions"),
                "teacher_signal_schema": output.get("teacher_signal_schema"),
                "signal_ready": signal_ready,
                "sample_supervision_mode": supervision_mode,
                "soft_fields_present": soft_fields_present,
                "structured_present": structured_present,
                "logit_kd_weight": float(weights.get("logit_kd") or 0.0),
            }
        )

    summary = {
        "manifest_path": str(args.manifest_path),
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "corpus_jsonl": str(args.corpus_jsonl),
        "num_samples_checked": len(sample_summaries),
        "active_versions": active_versions(),
        "kd_schema_version": KD_SCHEMA_VERSION,
        "t0_gate_pass_count": t0_pass,
        "runtime_bundle_gate_pass_count": runtime_bundle_pass,
        "soft_field_gate_pass_count": soft_field_pass,
        "logit_kd_positive_count": logit_kd_positive,
        "samples": sample_summaries,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
