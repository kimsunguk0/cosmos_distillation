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
    "concise_json_v1": (
        "You are an AV reasoning teacher. Summarize the key reason for the ego "
        "vehicle's decision in 2-5 sentences, then provide a short final answer. "
        "Return JSON with keys: rationale, meta_action, answer."
    ),
    "strict_schema_v1": (
        "You are an AV reasoning teacher. Return valid JSON only. Schema: "
        '{"scene_summary": str, "meta_action": str, "critical_objects": [{"type": '
        'str, "why": str}], "temporal_notices": [{"when": str, "what": str}], '
        '"final_answer": str, "confidence": float}. Do not include markdown.'
    ),
}


def get_prompt(name: str) -> str:
    """Return a named teacher prompt."""
    try:
        return PROMPTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt family: {name}") from exc
