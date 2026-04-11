from src.data.consistency import grade_action_pair


def test_identical_action_pair_passes() -> None:
    result = grade_action_pair("yield", "yield")
    assert result.consistency_level == "pass"
    assert result.is_consistent is True


def test_turn_conflict_hard_fails() -> None:
    result = grade_action_pair("left_turn", "right_turn")
    assert result.consistency_level == "hard_fail"
    assert result.is_consistent is False
