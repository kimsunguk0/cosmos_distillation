"""Prompt templates for teacher cache generation."""

from __future__ import annotations


PROMPTS = {
    "long_cot_v1": (
        "You are an AV reasoning teacher. Given synchronized multi-camera frames "
        "and ego-motion history, explain the chain of causation for the ego "
        "vehicle's situation. Focus on scene layout, relevant actors, temporal "
        "ordering, why the ego vehicle should act the way it does, and what makes "
        "the situation difficult. Return only the reasoning."
    ),
    "tagged_triplet_v2": (
        "You are an AV reasoning teacher. Complete the response using the exact "
        "special-token format below and do not output JSON, bullet lists, or "
        "numeric vectors.\n"
        "<|cot_start|>1-2 sentences explaining the driving situation and why the ego "
        "vehicle should act that way.<|cot_end|>"
        "<|meta_action_start|>one action label from "
        "[keep_lane, yield, stop, slow_down, creep, left_turn, right_turn, "
        "change_lane_left, change_lane_right, follow_lead, overtake, "
        "nudge_left, nudge_right, unknown]<|meta_action_end|>"
        "<|answer_start|>one short final driving answer in natural language."
        "<|answer_end|>"
    ),
    "tagged_triplet_brief_v2": (
        "Observe the scene and output exactly three tagged fields. Keep them short "
        "and concrete.\n"
        "<|cot_start|>brief causal explanation.<|cot_end|>"
        "<|meta_action_start|>single action label.<|meta_action_end|>"
        "<|answer_start|>single-sentence driving answer.<|answer_end|>\n"
        "If uncertain, still fill all three fields and use unknown only for the "
        "meta action."
    ),
    "short_reason_only_v2": (
        "Return one short sentence with at most 12 words describing the main "
        "causal reason for the ego vehicle's decision. Return natural language only."
    ),
    "meta_action_only_v2": (
        "Return only one action label for the ego vehicle. Choose exactly one from "
        "[keep_lane, yield, stop, slow_down, creep, left_turn, right_turn, "
        "change_lane_left, change_lane_right, follow_lead, overtake, nudge_left, "
        "nudge_right, unknown]. Do not add any extra words."
    ),
    "answer_only_v2": (
        "Return one short final driving answer in natural language, ideally under "
        "10 words. Do not output lists, JSON, or coordinates."
    ),
}


def get_prompt(name: str) -> str:
    """Return a named teacher prompt."""
    try:
        return PROMPTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt family: {name}") from exc
