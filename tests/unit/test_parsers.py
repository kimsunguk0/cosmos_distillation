from src.data.parsers import extract_action_from_text


def test_extract_action_from_text_prefers_nudge_left_for_blocking_right_lane() -> None:
    parsed = extract_action_from_text(
        "A vehicle is blocking the right side of our lane, so the ego nudges left to go around it."
    )
    assert parsed.value == "nudge_left"


def test_extract_action_from_text_detects_encroaching_right_vehicle_as_nudge_left() -> None:
    parsed = extract_action_from_text(
        "A car is encroaching into our lane from the right, so we shift left slightly to avoid it."
    )
    assert parsed.value == "nudge_left"


def test_extract_action_from_text_detects_keep_lane_when_lane_is_clear() -> None:
    parsed = extract_action_from_text(
        "Keep lane since the lane is clear ahead with no lead vehicle in front."
    )
    assert parsed.value == "keep_lane"
