"""Tokenizer extension utilities."""

from __future__ import annotations

from typing import Any

from src.utils.traj_tokens import DEFAULT_TRAJ_VOCAB_SIZE, discrete_traj_token, discrete_traj_tokens


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
    "<|traj_future|>",
    "<|route_start|>",
    "<|route_end|>",
]


def missing_special_tokens(
    existing_vocab: set[str],
    *,
    traj_vocab_size: int = DEFAULT_TRAJ_VOCAB_SIZE,
) -> list[str]:
    """Return required control or trajectory tokens missing from the provided vocab."""
    wanted = list(REQUIRED_SPECIAL_TOKENS) + discrete_traj_tokens(traj_vocab_size)
    return [token for token in wanted if token not in existing_vocab]


def ensure_special_tokens(
    tokenizer: Any,
    extra_tokens: list[str] | None = None,
    *,
    traj_vocab_size: int = DEFAULT_TRAJ_VOCAB_SIZE,
) -> list[str]:
    """Add required control tokens plus Alpamayo discrete trajectory tokens."""
    vocab = set(tokenizer.get_vocab().keys())
    traj_tokens = [token for token in discrete_traj_tokens(traj_vocab_size) if token not in vocab]
    control_tokens = [token for token in REQUIRED_SPECIAL_TOKENS if token not in vocab]
    extra_missing = []
    if extra_tokens:
        extra_missing = [token for token in extra_tokens if token not in vocab]

    added: list[str] = []
    if traj_tokens:
        tokenizer.add_tokens(traj_tokens)
        added.extend(traj_tokens)
    special_additions = control_tokens + extra_missing
    if special_additions:
        tokenizer.add_special_tokens({"additional_special_tokens": special_additions})
        added.extend(special_additions)

    if hasattr(tokenizer, "convert_tokens_to_ids"):
        tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
        tokenizer.traj_token_end_idx = tokenizer.convert_tokens_to_ids(f"<i{traj_vocab_size - 1}>")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added.append("<|pad|>")
    return added


def distill_trainable_token_ids(
    tokenizer: Any,
    *,
    traj_vocab_size: int = DEFAULT_TRAJ_VOCAB_SIZE,
) -> list[int]:
    """Return the custom control/traj token ids that should stay trainable under LoRA."""
    token_ids: list[int] = []
    for token in REQUIRED_SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.append(int(token_id))

    traj_start = getattr(tokenizer, "traj_token_start_idx", None)
    traj_end = getattr(tokenizer, "traj_token_end_idx", None)
    if traj_start is None or traj_end is None or int(traj_end) < int(traj_start):
        traj_start = tokenizer.convert_tokens_to_ids("<i0>")
        traj_end = tokenizer.convert_tokens_to_ids(f"<i{traj_vocab_size - 1}>")
    if isinstance(traj_start, int) and isinstance(traj_end, int) and traj_start >= 0 and traj_end >= traj_start:
        token_ids.extend(range(int(traj_start), int(traj_end) + 1))

    return sorted(set(token_ids))


def assert_traj_token_offsets(
    tokenizer: Any,
    *,
    traj_vocab_size: int = DEFAULT_TRAJ_VOCAB_SIZE,
) -> None:
    """Validate that discrete trajectory token ids are contiguous and aligned."""
    start = tokenizer.convert_tokens_to_ids(discrete_traj_token(0))
    end = tokenizer.convert_tokens_to_ids(discrete_traj_token(int(traj_vocab_size) - 1))
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < 0:
        raise ValueError("Tokenizer is missing discrete trajectory tokens <i0> or final trajectory token.")
    expected_end = int(start) + int(traj_vocab_size) - 1
    if int(end) != expected_end:
        raise ValueError(
            f"Trajectory token ids are not contiguous: <i0>={start}, "
            f"<i{int(traj_vocab_size) - 1}>={end}, expected_end={expected_end}."
        )
    for probe in (1, 2999, 3000, int(traj_vocab_size) - 1):
        if probe < 0 or probe >= int(traj_vocab_size):
            continue
        token = discrete_traj_token(probe)
        token_id = tokenizer.convert_tokens_to_ids(token)
        expected = int(start) + int(probe)
        if int(token_id) != expected:
            raise ValueError(f"Trajectory token offset mismatch for {token}: id={token_id}, expected={expected}.")
    tokenizer.traj_token_start_idx = int(start)
    tokenizer.traj_token_end_idx = int(end)
