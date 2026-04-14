import torch

from src.training.losses import STAGE_DEFAULTS, get_stage_weights, logit_kd_loss, weighted_causal_ce


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


def test_sparse_logit_kd_loss_returns_nonzero_when_topk_is_present() -> None:
    student_logits = torch.tensor(
        [[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 1, 0]], dtype=torch.long)
    teacher_topk_indices = torch.tensor([[[1], [0]]], dtype=torch.long)
    teacher_topk_logits = torch.tensor([[[2.0], [1.5]]], dtype=torch.float32)
    teacher_topk_mask = torch.tensor([[True, True]], dtype=torch.bool)
    weights = torch.tensor([1.0], dtype=torch.float32)
    loss = logit_kd_loss(
        student_logits,
        teacher_topk_indices,
        teacher_topk_logits,
        teacher_topk_mask,
        labels,
        weights,
    )
    assert float(loss) >= 0.0


def test_weighted_causal_ce_scales_with_batch1_weight() -> None:
    logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    labels = torch.tensor([[-100, 0, 1]], dtype=torch.long)
    full_loss, _ = weighted_causal_ce(logits, labels, torch.tensor([1.0], dtype=torch.float32))
    downweighted_loss, _ = weighted_causal_ce(logits, labels, torch.tensor([0.25], dtype=torch.float32))
    assert float(downweighted_loss) < float(full_loss)
