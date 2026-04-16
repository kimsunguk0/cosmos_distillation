"""Trajectory-token helpers shared across corpus building and training."""

from __future__ import annotations

DEFAULT_TRAJ_VOCAB_SIZE = 4000


def discrete_traj_token(index: int) -> str:
    """Return the canonical Alpamayo/Cosmos discrete trajectory token string."""
    return f"<i{int(index)}>"


def discrete_traj_tokens(vocab_size: int = DEFAULT_TRAJ_VOCAB_SIZE) -> list[str]:
    """Return the full discrete trajectory vocabulary."""
    return [discrete_traj_token(index) for index in range(int(vocab_size))]


def format_traj_token_sequence(token_ids: list[int] | tuple[int, ...]) -> str:
    """Serialize trajectory token ids into a whitespace-delimited LM span."""
    return " ".join(discrete_traj_token(int(token_id)) for token_id in token_ids)
