"""Pipeline schema versions and cache invalidation helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CANONICAL_LOCALIZATION_VERSION = "v2_explicit_t0_anchor"
PATH_SEMANTICS_VERSION = "v2_action_turn_stop_records"
TEACHER_RUNTIME_BUNDLE_VERSION = "v2_runtime_only_bundle"
TEACHER_PROMPT_FAMILY_VERSION = "v5_official_cot_plus_honest_vqa_probe"
TEACHER_SELECTION_POLICY_VERSION = "v5_long_cot_primary_honest_teacher_contract"
KD_SCHEMA_VERSION = "v5_long_cot_primary_meta_derived_answer_probe"
TEACHER_SIGNAL_CACHE_VERSION = "v3_field_scoped_sparse_topk"


def active_versions() -> dict[str, str]:
    """Return the active pipeline versions used for invalidation."""
    return {
        "canonical_localization_version": CANONICAL_LOCALIZATION_VERSION,
        "path_semantics_version": PATH_SEMANTICS_VERSION,
        "teacher_runtime_bundle_version": TEACHER_RUNTIME_BUNDLE_VERSION,
        "teacher_prompt_family_version": TEACHER_PROMPT_FAMILY_VERSION,
        "teacher_selection_policy_version": TEACHER_SELECTION_POLICY_VERSION,
        "kd_schema_version": KD_SCHEMA_VERSION,
        "teacher_signal_cache_version": TEACHER_SIGNAL_CACHE_VERSION,
    }


def stable_payload_hash(payload: Any) -> str:
    """Return a deterministic hash for a JSON-serializable payload."""
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
