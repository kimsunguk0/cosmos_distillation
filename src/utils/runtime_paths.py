"""Runtime path resolution for externally-provisioned assets."""

from __future__ import annotations

import os
from pathlib import Path


def _candidate_paths(*raw_values: str | os.PathLike[str] | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for raw_value in raw_values:
        if raw_value in (None, ""):
            continue
        candidate = Path(raw_value).expanduser()
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _first_existing(candidates: tuple[Path, ...], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


DEFAULT_EXTERNAL_DATA_ROOT_CANDIDATES = _candidate_paths(
    os.environ.get("COSMOS_DATA_ROOT"),
    "/data",
    "/home/pm97/workspace/dataset/human_coc_dataset",
)
DEFAULT_EXTERNAL_DATA_ROOT = _first_existing(
    DEFAULT_EXTERNAL_DATA_ROOT_CANDIDATES,
    fallback=Path(os.environ.get("COSMOS_DATA_ROOT", "/data")).expanduser(),
)
DEFAULT_MATERIALIZED_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "materialized"
DEFAULT_STATE_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "state"
DEFAULT_TEACHER_CACHE_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "teacher_cache"
DEFAULT_STUDENT_MODEL_CANDIDATES = _candidate_paths(
    os.environ.get("COSMOS_STUDENT_MODEL"),
    "/workspace/base_models_weights/Cosmos-Reason2-2B",
    "/workspace/base_model_weights/Cosmos-Reason2-2B",
    "/workspace/base_models_weights/cosmos-reason-2b",
    "/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b",
    "/home/pm97/workspace/kwpark/weights/cosmos-reason-2b",
)
DEFAULT_ALPAMAYO_MODEL_CANDIDATES = _candidate_paths(
    os.environ.get("ALPAMAYO_MODEL_PATH"),
    "/workspace/base_models_weights/Alpamayo-1.5-10B",
    "/workspace/base_models_weights/Alpamayo-R1-10B",
    "/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B",
    "/home/pm97/workspace/sukim/base_weights/alpamayo15_vlm_weights",
)
DEFAULT_ALPAMAYO_SRC_CANDIDATES = _candidate_paths(
    os.environ.get("ALPAMAYO_SRC"),
    "/workspace/alpamayo_repos/alpamayo1.5/src",
    "/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src",
)

_PATH_REPLACEMENTS = (
    ("/data/materialized", str(DEFAULT_MATERIALIZED_ROOT)),
    ("/data/teacher_cache", str(DEFAULT_TEACHER_CACHE_ROOT)),
    ("/data/state", str(DEFAULT_STATE_ROOT)),
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
    env_path = os.environ.get("COSMOS_STUDENT_MODEL")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return str(candidate)
        return env_path
    for candidate in DEFAULT_STUDENT_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "nvidia/Cosmos-Reason2-2B"


def resolve_alpamayo_model_path(explicit: str | None = None) -> str | None:
    """Return the best local Alpamayo full-model path when available."""
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists():
            return str(explicit_path)
        return explicit
    env_path = os.environ.get("ALPAMAYO_MODEL_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return str(candidate)
        return env_path
    for candidate in DEFAULT_ALPAMAYO_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def resolve_alpamayo_src_path(explicit: str | None = None) -> str | None:
    """Return the best local Alpamayo source tree when available."""
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists():
            return str(explicit_path)
        return explicit
    env_path = os.environ.get("ALPAMAYO_SRC")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return str(candidate)
        return env_path
    for candidate in DEFAULT_ALPAMAYO_SRC_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


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
