"""Safe corpus contracts for v1 distillation."""

from __future__ import annotations

from dataclasses import dataclass, field


FORBIDDEN_TASK_TYPES = {
    "teacher_reasoning_plus_gt_path",
    "human_reasoning_plus_teacher_path",
    "teacher_discrete_future_tokens_as_gt",
}


@dataclass(slots=True)
class CorpusSample:
    sample_id: str
    task_type: str
    input_payload: dict
    target_payload: dict
    soft_target_payload: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)


def validate_task_type(task_type: str) -> None:
    """Reject tasks that are explicitly forbidden in v1."""
    if task_type in FORBIDDEN_TASK_TYPES:
        raise ValueError(f"Forbidden v1 task type: {task_type}")
