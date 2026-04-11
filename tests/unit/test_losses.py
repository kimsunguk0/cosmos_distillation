import torch

from src.training.losses import STAGE_DEFAULTS, get_stage_weights, weighted_causal_ce


def test_stage_defaults_include_main_stage() -> None:
    assert "stage_b" in STAGE_DEFAULTS
    assert STAGE_DEFAULTS["stage_b"].seq_kd == 0.6


def test_get_stage_weights_matches_defaults() -> None:
    assert get_stage_weights("stage_a").aux == STAGE_DEFAULTS["stage_a"].aux


def test_weighted_causal_ce_returns_scalar_loss() -> None:
    logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    labels = torch.tensor([[-100, 0, 1]], dtype=torch.long)
    weights = torch.tensor([1.0], dtype=torch.float32)
    loss, token_count = weighted_causal_ce(logits, labels, weights)
    assert loss.ndim == 0
    assert int(token_count.item()) == 2
