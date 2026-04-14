"""Tokenizer extension utilities."""

from __future__ import annotations

from typing import Any


REQUIRED_SPECIAL_TOKENS = [
    "<|cot_start|>",
    "<|cot_end|>",
    "<|meta_action_start|>",
    "<|meta_action_end|>",
    "<|question_start|>",
    "<|question_end|>",
    "<|answer_start|>",
    "<|answer_end|>",
    "<|traj_history_start|>",
    "<|traj_history_end|>",
    "<|traj_history|>",
    "<|traj_future_start|>",
    "<|traj_future_end|>",
    "<|route_start|>",
    "<|route_end|>",
]


def missing_special_tokens(existing_vocab: set[str]) -> list[str]:
    """Return required tokens missing from the provided vocab."""
    return [token for token in REQUIRED_SPECIAL_TOKENS if token not in existing_vocab]


def ensure_special_tokens(tokenizer: Any, extra_tokens: list[str] | None = None) -> list[str]:
    """Add required special tokens to a tokenizer and return the newly-added tokens."""
    vocab = set(tokenizer.get_vocab().keys())
    wanted = list(REQUIRED_SPECIAL_TOKENS)
    if extra_tokens:
        wanted.extend(extra_tokens)
    missing = [token for token in wanted if token not in vocab]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return missing
