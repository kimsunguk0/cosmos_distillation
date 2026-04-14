"""Prompt templates and generation configs for teacher cache generation."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.schema_versions import TEACHER_PROMPT_FAMILY_VERSION


@dataclass(frozen=True, slots=True)
class PromptGenerationConfig:
    temperature: float
    top_p: float
    max_generation_length: int
    num_candidates: int
    input_builder: str
    deterministic_second_pass: bool


@dataclass(frozen=True, slots=True)
class PromptConfig:
    name: str
    prompt: str
    generation: PromptGenerationConfig


PROMPT_CONFIGS = {
    "long_cot_v1": PromptConfig(
        name="long_cot_v1",
        prompt=(
            "Explain the chain of causation for the ego vehicle. Focus on the main "
            "scene factors, relevant actors, temporal ordering, and why the ego vehicle "
            "should act the way it does. Return only the reasoning."
        ),
        generation=PromptGenerationConfig(
            temperature=0.6,
            top_p=0.98,
            max_generation_length=256,
            num_candidates=3,
            input_builder="cot",
            deterministic_second_pass=True,
        ),
    ),
    "answer_vqa_v1": PromptConfig(
        name="answer_vqa_v1",
        prompt=(
            "What should the ego vehicle do right now? "
            "Answer in one short natural-language sentence."
        ),
        generation=PromptGenerationConfig(
            temperature=0.2,
            top_p=0.9,
            max_generation_length=48,
            num_candidates=2,
            input_builder="answer",
            deterministic_second_pass=True,
        ),
    ),
    "meta_action_vqa_v1": PromptConfig(
        name="meta_action_vqa_v1",
        prompt=(
            "Which action label best describes the ego vehicle's next maneuver? "
            "Choose exactly one from [keep_lane, yield, stop, slow_down, creep, "
            "left_turn, right_turn, change_lane_left, change_lane_right, follow_lead, "
            "overtake, nudge_left, nudge_right, unknown]. "
            "Answer with the label only."
        ),
        generation=PromptGenerationConfig(
            temperature=0.1,
            top_p=0.9,
            max_generation_length=16,
            num_candidates=2,
            input_builder="answer",
            deterministic_second_pass=True,
        ),
    ),
}


def get_prompt(name: str) -> str:
    """Return a named teacher prompt."""
    return get_prompt_config(name).prompt


def get_prompt_config(name: str) -> PromptConfig:
    """Return a named prompt config with generation settings."""
    try:
        return PROMPT_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt family: {name}") from exc


def prompt_family_version() -> str:
    """Expose the prompt-family version used in cache metadata."""
    return TEACHER_PROMPT_FAMILY_VERSION
