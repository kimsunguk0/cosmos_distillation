import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts" / "05_run_teacher_text_inference.py"
SPEC = importlib.util.spec_from_file_location("teacher_inference_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_conflicting_collapsed_slots_are_suppressed_to_long_cot_only() -> None:
    prompt_outputs = {
        "long_cot_v1": {
            "cot": "Nudge to the left to clear the traffic cones blocking the right side of our lane",
            "meta_action": "",
            "answer": "",
            "json_payload": None,
            "json_summary": {"is_valid": False, "field_count": 0, "has_meta_action": False, "has_answer": False},
        },
        "concise_reason_answer_json_v1": {
            "cot": "Keep lane since the lane is clear ahead",
            "meta_action": "",
            "answer": "",
            "json_payload": None,
            "json_summary": {"is_valid": False, "field_count": 0, "has_meta_action": False, "has_answer": False},
        },
        "structured_json_v1": {
            "cot": "Nudge to the left to clear the construction cones blocking the right side of our lane",
            "meta_action": "",
            "answer": "",
            "json_payload": None,
            "json_summary": {"is_valid": False, "field_count": 0, "has_meta_action": False, "has_answer": False},
        },
        "short_reason_only_v2": {
            "cot": "Keep lane since the lane is clear ahead",
            "meta_action": "",
            "answer": "",
            "json_payload": None,
            "json_summary": {"is_valid": False, "field_count": 0, "has_meta_action": False, "has_answer": False},
        },
        "meta_action_only_v2": {
            "cot": "Keep lane since the lane is clear ahead",
            "meta_action": "",
            "answer": "",
            "json_payload": None,
            "json_summary": {"is_valid": False, "field_count": 0, "has_meta_action": False, "has_answer": False},
        },
        "answer_only_v2": {
            "cot": "Keep lane since the lane is clear with no lead vehicle ahead",
            "meta_action": "",
            "answer": "",
            "json_payload": None,
            "json_summary": {"is_valid": False, "field_count": 0, "has_meta_action": False, "has_answer": False},
        },
    }

    normalized, diagnostics = MODULE.normalize_teacher_output(
        prompt_outputs,
        human_action_class=None,
        weak_gt_action_class="change_lane_right",
    )

    assert normalized["teacher_long_cot"] == "Nudge to the left to clear the traffic cones blocking the right side of our lane"
    assert normalized["teacher_meta_action"] == "nudge_left"
    assert normalized["teacher_short_reason"] is None
    assert normalized["teacher_answer"] is None
    assert normalized["teacher_structured_json"] is None
    assert normalized["teacher_selection_prompt"] is None
    assert normalized["sample_supervision_mode"] == "long_cot_only_fallback"
    assert normalized["selected_direct_prompt_family"] == "long_cot_v1"
    assert normalized["selected_normalization_mode"] == "long_cot_only"
    assert normalized["teacher_quality_multiplier"] <= 0.25
    assert diagnostics["slot_conflict_checks"].keys() == {
        "teacher_short_reason",
        "teacher_answer",
        "teacher_meta_action",
    }
