import pytest

from src.data.corpus_builder import validate_task_type


def test_forbidden_task_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_task_type("teacher_reasoning_plus_gt_path")
