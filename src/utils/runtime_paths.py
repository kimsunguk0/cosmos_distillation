"""Runtime path resolution for externally-provisioned assets."""

from __future__ import annotations

from pathlib import Path


DEFAULT_EXTERNAL_DATA_ROOT = Path("/data")
DEFAULT_MATERIALIZED_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "materialized"
DEFAULT_STATE_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "state"
DEFAULT_TEACHER_CACHE_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "teacher_cache"
DEFAULT_STUDENT_MODEL_CANDIDATES = (
    Path("/workspace/base_models_weights/Cosmos-Reason2-2B"),
    Path("/workspace/base_model_weights/Cosmos-Reason2-2B"),
    Path("/workspace/base_models_weights/cosmos-reason-2b"),
)

_PATH_REPLACEMENTS = (
    ("/workspace/sukim/alpamayo_teacher_prep/materialized", str(DEFAULT_MATERIALIZED_ROOT)),
    ("/workspace/sukim/alpamayo_teacher_prep/teacher_cache", str(DEFAULT_TEACHER_CACHE_ROOT)),
)


def resolve_student_model_path(explicit: str | None = None) -> str:
    """Return the best local student-model path, or fall back to the explicit value."""
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists():
            return str(explicit_path)
        return explicit
    for candidate in DEFAULT_STUDENT_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "nvidia/Cosmos-Reason2-2B"


def remap_external_path(raw_path: str | Path | None) -> str | None:
    """Rewrite stale Alpamayo-prep absolute paths onto the local mounted assets."""
    if raw_path in (None, ""):
        return None
    path_str = str(raw_path)
    path = Path(path_str).expanduser()
    if path.exists():
        return str(path)
    for old_prefix, new_prefix in _PATH_REPLACEMENTS:
        if path_str.startswith(old_prefix):
            remapped = Path(path_str.replace(old_prefix, new_prefix, 1))
            return str(remapped)
    return str(path)
