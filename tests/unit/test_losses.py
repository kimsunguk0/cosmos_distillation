import torch
import torch.nn.functional as F

from src.training.losses import (
    INVALID_SPARSE_LOGPROB_CUTOFF,
    STAGE_DEFAULTS,
    DistillationLossWeights,
    TrajectoryDecodeConfig,
    decoded_traj_geometry_losses,
    export_loss_weights,
    export_metric_logs,
    get_stage_weights,
    resolve_optional_loss_weight_value,
    resolve_loss_weight_value,
    teacher_logit_kd_loss,
    token_hidden_alignment_bridge_loss,
    token_hidden_alignment_loss,
    weighted_causal_ce,
)


def _reference_sparse_kd_loss(
    student_logits: torch.Tensor,
    cot_content_mask: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    temperature: float = 1.0,
    token_weights: torch.Tensor | None = None,
    teacher_topk_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_student = student_logits[:, :-1, :].contiguous()
    span_mask = cot_content_mask[:, 1:].to(dtype=torch.bool)
    vocab_size = shift_student.shape[-1]
    sample_losses = []
    active_sample_weights = []
    for sample_index in range(shift_student.shape[0]):
        teacher_indices = teacher_topk_indices[sample_index][teacher_topk_mask[sample_index]]
        teacher_values = teacher_topk_logprobs[sample_index][teacher_topk_mask[sample_index]]
        sample_token_weights = (
            token_weights[sample_index][teacher_topk_mask[sample_index]] if token_weights is not None else None
        )
        if teacher_topk_positions is not None:
            raw_positions = teacher_topk_positions[sample_index][teacher_topk_mask[sample_index]].long()
            valid_positions = (raw_positions > 0) & (raw_positions <= shift_student.shape[1])
            if not bool(valid_positions.any()):
                continue
            student_rows = raw_positions[valid_positions] - 1
            teacher_indices = teacher_indices[valid_positions]
            teacher_values = teacher_values[valid_positions]
            if sample_token_weights is not None:
                sample_token_weights = sample_token_weights[valid_positions]
        else:
            student_rows = torch.nonzero(span_mask[sample_index], as_tuple=False).flatten()
        aligned_tokens = min(student_rows.shape[0], teacher_indices.shape[0])
        if aligned_tokens <= 0:
            continue
        student_rows = student_rows[:aligned_tokens]
        teacher_indices = teacher_indices[:aligned_tokens]
        teacher_values = teacher_values[:aligned_tokens]
        if sample_token_weights is not None:
            sample_token_weights = sample_token_weights[:aligned_tokens]
        token_losses = []
        active_token_weights = []
        for token_index in range(aligned_tokens):
            keep = (
                (teacher_indices[token_index] >= 0)
                & (teacher_indices[token_index] < vocab_size)
                & torch.isfinite(teacher_values[token_index])
                & (teacher_values[token_index] > INVALID_SPARSE_LOGPROB_CUTOFF)
            )
            if not bool(keep.any()):
                continue
            gathered_student = torch.gather(
                shift_student[sample_index, student_rows[token_index]],
                dim=-1,
                index=teacher_indices[token_index][keep],
            )
            student_log_probs = F.log_softmax(gathered_student / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_values[token_index][keep] / temperature, dim=-1)
            token_losses.append(F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (temperature**2))
            if sample_token_weights is not None:
                active_token_weights.append(sample_token_weights[token_index])
        if not token_losses:
            continue
        stacked_losses = torch.stack(token_losses)
        if active_token_weights:
            stacked_weights = torch.stack(active_token_weights).to(dtype=stacked_losses.dtype)
            sample_losses.append((stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp(min=1e-6))
        else:
            sample_losses.append(stacked_losses.mean())
        if sample_weights is not None:
            active_sample_weights.append(sample_weights[sample_index].to(dtype=stacked_losses.dtype))
    if not sample_losses:
        return torch.tensor(0.0, dtype=student_logits.dtype)
    losses = torch.stack(sample_losses)
    if sample_weights is None:
        return losses.mean()
    weights = torch.stack(active_sample_weights).to(dtype=losses.dtype)
    return (losses * weights).mean()


def test_stage_defaults_include_main_stage() -> None:
    assert "stage_b" in STAGE_DEFAULTS
    assert STAGE_DEFAULTS["stage_b"].teacher_seq_ce == 0.25


def test_get_stage_weights_matches_defaults() -> None:
    assert get_stage_weights("stage_a").action_aux == STAGE_DEFAULTS["stage_a"].action_aux


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
    cot_content_mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    teacher_topk_indices = torch.tensor([[[1], [0]]], dtype=torch.long)
    teacher_topk_logits = torch.tensor([[[2.0], [1.5]]], dtype=torch.float32)
    teacher_topk_mask = torch.tensor([[True, True]], dtype=torch.bool)
    weights = torch.tensor([1.0], dtype=torch.float32)
    loss = teacher_logit_kd_loss(
        student_logits,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logits,
        teacher_topk_mask,
        weights,
    )
    assert float(loss) >= 0.0


def test_sparse_logit_kd_loss_ignores_invalid_sparse_entries() -> None:
    student_logits = torch.tensor(
        [[[0.0, 4.0, 1.0], [0.0, 4.0, 1.0]]],
        dtype=torch.float32,
    )
    cot_content_mask = torch.tensor([[False, True]], dtype=torch.bool)
    teacher_topk_mask = torch.tensor([[True]], dtype=torch.bool)
    weights = torch.tensor([1.0], dtype=torch.float32)
    valid_only = teacher_logit_kd_loss(
        student_logits,
        cot_content_mask,
        torch.tensor([[[1]]], dtype=torch.long),
        torch.tensor([[[0.0]]], dtype=torch.float32),
        teacher_topk_mask,
        weights,
    )
    with_invalid = teacher_logit_kd_loss(
        student_logits,
        cot_content_mask,
        torch.tensor([[[1, 2]]], dtype=torch.long),
        torch.tensor([[[0.0, -1.0e9]]], dtype=torch.float32),
        teacher_topk_mask,
        weights,
    )

    torch.testing.assert_close(with_invalid, valid_only)


def test_sparse_logit_kd_loss_matches_reference_loop_with_weights() -> None:
    student_logits = torch.tensor(
        [
            [
                [0.2, 1.3, -0.1, 0.7, 0.0],
                [1.1, -0.4, 0.3, 0.8, -0.2],
                [0.0, 0.5, 1.2, -0.6, 0.1],
                [0.4, -0.1, 0.2, 1.5, -0.3],
            ],
            [
                [-0.2, 0.4, 0.9, 0.1, 0.0],
                [0.7, 0.2, -0.5, 1.0, 0.3],
                [0.1, 1.4, 0.2, -0.3, 0.8],
                [0.3, -0.7, 0.6, 0.2, 1.1],
            ],
        ],
        dtype=torch.float32,
    )
    cot_content_mask = torch.tensor(
        [
            [False, True, False, True],
            [False, True, True, False],
        ],
        dtype=torch.bool,
    )
    teacher_topk_indices = torch.tensor(
        [
            [[1, 3, 99], [2, 4, 0], [4, -1, 1]],
            [[0, 2, 3], [4, 1, 2], [2, 99, 3]],
        ],
        dtype=torch.long,
    )
    teacher_topk_logprobs = torch.log(
        torch.tensor(
            [
                [[0.6, 0.3, 0.1], [0.5, 0.3, 0.2], [0.7, 0.2, 0.1]],
                [[0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.8, 0.1, 0.1]],
            ],
            dtype=torch.float32,
        )
    )
    teacher_topk_logprobs[0, 2, 1] = -1.0e9
    teacher_topk_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
        ],
        dtype=torch.bool,
    )
    sample_weights = torch.tensor([0.5, 1.5], dtype=torch.float32)
    token_weights = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.5, 0.5, 2.5],
        ],
        dtype=torch.float32,
    )

    vectorized = teacher_logit_kd_loss(
        student_logits,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        teacher_topk_mask,
        sample_weights,
        temperature=1.7,
        token_weights=token_weights,
    )
    reference = _reference_sparse_kd_loss(
        student_logits,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        teacher_topk_mask,
        sample_weights,
        temperature=1.7,
        token_weights=token_weights,
    )

    torch.testing.assert_close(vectorized, reference)


def test_sparse_logit_kd_loss_matches_reference_loop_with_positions() -> None:
    student_logits = torch.tensor(
        [
            [
                [0.5, -0.2, 1.2, 0.0],
                [0.1, 0.9, -0.4, 0.3],
                [1.1, 0.2, 0.0, -0.5],
                [-0.3, 0.4, 0.8, 0.2],
            ]
        ],
        dtype=torch.float32,
    )
    cot_content_mask = torch.tensor([[False, False, False, False]], dtype=torch.bool)
    teacher_topk_indices = torch.tensor([[[2, 1], [0, 3], [1, 2]]], dtype=torch.long)
    teacher_topk_logprobs = torch.log(torch.tensor([[[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]], dtype=torch.float32))
    teacher_topk_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
    teacher_topk_positions = torch.tensor([[1, 99, 3]], dtype=torch.long)
    token_weights = torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float32)

    vectorized = teacher_logit_kd_loss(
        student_logits,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        teacher_topk_mask,
        temperature=1.3,
        token_weights=token_weights,
        teacher_topk_positions=teacher_topk_positions,
    )
    reference = _reference_sparse_kd_loss(
        student_logits,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        teacher_topk_mask,
        temperature=1.3,
        token_weights=token_weights,
        teacher_topk_positions=teacher_topk_positions,
    )

    torch.testing.assert_close(vectorized, reference)


def test_sparse_logit_kd_tail_bucket_penalizes_out_of_support_mass() -> None:
    cot_content_mask = torch.tensor([[False, True]], dtype=torch.bool)
    teacher_topk_indices = torch.tensor([[[0]]], dtype=torch.long)
    teacher_topk_logprobs = torch.log(torch.tensor([[[0.9]]], dtype=torch.float32))
    teacher_topk_mask = torch.tensor([[True]], dtype=torch.bool)
    support_only_good = torch.log(torch.tensor([[[0.9, 0.033, 0.033, 0.034], [0.25, 0.25, 0.25, 0.25]]]))
    support_only_bad = torch.log(torch.tensor([[[0.1, 0.3, 0.3, 0.3], [0.25, 0.25, 0.25, 0.25]]]))

    support_good = teacher_logit_kd_loss(
        support_only_good,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        teacher_topk_mask,
    )
    support_bad = teacher_logit_kd_loss(
        support_only_bad,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        teacher_topk_mask,
    )
    tail_good = teacher_logit_kd_loss(
        support_only_good,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        teacher_topk_mask,
        include_tail_bucket=True,
    )
    tail_bad = teacher_logit_kd_loss(
        support_only_bad,
        cot_content_mask,
        teacher_topk_indices,
        teacher_topk_logprobs,
        teacher_topk_mask,
        include_tail_bucket=True,
    )

    torch.testing.assert_close(support_good, support_bad)
    assert float(tail_good) < 1.0e-4
    assert float(tail_bad) > float(tail_good) + 1.0


def test_weighted_causal_ce_scales_with_batch1_weight() -> None:
    logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    labels = torch.tensor([[-100, 0, 1]], dtype=torch.long)
    full_loss, _ = weighted_causal_ce(logits, labels, torch.tensor([1.0], dtype=torch.float32))
    downweighted_loss, _ = weighted_causal_ce(logits, labels, torch.tensor([0.25], dtype=torch.float32))
    assert float(downweighted_loss) < float(full_loss)


def test_weighted_causal_ce_supports_label_token_reweighting() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.5, 2.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 0, 1]], dtype=torch.long)
    token_mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    label_token_weights = torch.tensor([[1.0, 1.0, 3.0]], dtype=torch.float32)
    plain_loss, _ = weighted_causal_ce(logits, labels, torch.tensor([1.0], dtype=torch.float32), token_mask)
    reweighted_loss, _ = weighted_causal_ce(
        logits,
        labels,
        torch.tensor([1.0], dtype=torch.float32),
        token_mask,
        label_token_weights,
    )
    assert float(reweighted_loss) > float(plain_loss)


def test_resolve_loss_weight_value_accepts_user_facing_aliases() -> None:
    weights = {
        "gt_cot_loss": 1.1,
        "teacher_cot_loss": 0.2,
        "teacher_topk_kd_loss": 0.3,
        "traj_loss": 0.4,
        "output_format_loss": 0.5,
        "meta_action_loss": 0.6,
        "feature_align_loss": 0.7,
    }
    assert resolve_loss_weight_value(weights, "hard_cot_ce", 0.0) == 1.1
    assert resolve_loss_weight_value(weights, "teacher_seq_ce", 0.0) == 0.2
    assert resolve_loss_weight_value(weights, "teacher_logit_kd", 0.0) == 0.3
    assert resolve_loss_weight_value(weights, "traj_ce", 0.0) == 0.4
    assert resolve_loss_weight_value(weights, "format_ce", 0.0) == 0.5
    assert resolve_loss_weight_value(weights, "action_aux", 0.0) == 0.6
    assert resolve_loss_weight_value(weights, "feat_align", 0.0) == 0.7


def test_resolve_optional_loss_weight_value_supports_teacher_traj_alias() -> None:
    weights = {"teacher_traj_loss": 0.25}
    assert resolve_optional_loss_weight_value(weights, "teacher_traj_ce") == 0.25
    assert resolve_optional_loss_weight_value({}, "teacher_traj_ce") is None


def test_export_helpers_use_user_facing_names() -> None:
    exported_weights = export_loss_weights(DistillationLossWeights(1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    assert exported_weights["gt_cot_loss"] == 1.0
    assert exported_weights["teacher_cot_loss"] == 0.2
    assert exported_weights["teacher_topk_kd_loss"] == 0.3
    assert exported_weights["output_format_loss"] == 0.5

    exported_metrics = export_metric_logs(
        {
            "hard_cot_ce": 1.0,
            "teacher_seq_ce": 0.2,
            "teacher_logit_kd": 0.3,
            "format_ce": 0.4,
            "hard_traj_acc": 0.9,
            "total_loss": 2.0,
        }
    )
    assert exported_metrics["gt_cot_loss"] == 1.0
    assert exported_metrics["teacher_cot_loss"] == 0.2
    assert exported_metrics["teacher_topk_kd_loss"] == 0.3
    assert exported_metrics["output_format_loss"] == 0.4
    assert exported_metrics["traj_token_acc"] == 0.9
    assert exported_metrics["total_loss"] == 2.0


def test_export_loss_weights_includes_teacher_traj_when_explicit() -> None:
    exported_weights = export_loss_weights(
        DistillationLossWeights(1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, teacher_traj_ce=0.0)
    )
    assert exported_weights["teacher_traj_loss"] == 0.0


def test_decoded_traj_geometry_losses_are_small_for_matching_midbin_controls() -> None:
    config = TrajectoryDecodeConfig(
        traj_token_start_idx=2,
        num_bins=5,
        dims_min=(-1.0, -1.0),
        dims_max=(1.0, 1.0),
        accel_mean=0.0,
        accel_std=1.0,
        curvature_mean=0.0,
        curvature_std=1.0,
        dt=1.0,
        n_waypoints=2,
        short_horizon_steps=2,
    )
    logits = torch.full((1, 5, 7), -10.0, dtype=torch.float32)
    logits[0, 0, 4] = 10.0
    logits[0, 1, 4] = 10.0
    logits[0, 2, 4] = 10.0
    logits[0, 3, 4] = 10.0
    labels = torch.tensor([[-100, 4, 4, 4, 4]], dtype=torch.long)
    traj_token_mask = torch.tensor([[False, True, True, True, True]], dtype=torch.bool)
    history_xyz = torch.zeros((1, 2, 3), dtype=torch.float32)
    history_mask = torch.tensor([[True, True]], dtype=torch.bool)
    future_xyz = torch.zeros((1, 2, 3), dtype=torch.float32)
    future_mask = torch.tensor([[True, True]], dtype=torch.bool)

    xyz_loss, delta_loss, final_loss = decoded_traj_geometry_losses(
        logits,
        labels,
        traj_token_mask,
        history_xyz,
        history_mask,
        future_xyz,
        future_mask,
        config,
    )
    assert float(xyz_loss) < 1e-4
    assert float(delta_loss) < 1e-4
    assert float(final_loss) < 1e-4


def test_token_hidden_alignment_loss_respects_teacher_mask() -> None:
    student_hidden = torch.tensor(
        [[[1.0, 1.0], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]]],
        dtype=torch.float32,
    )
    teacher_hidden = torch.tensor(
        [[[1.0, 1.0], [100.0, 100.0]]],
        dtype=torch.float32,
    )
    token_mask = torch.tensor([[False, True, True, False]], dtype=torch.bool)
    teacher_token_mask = torch.tensor([[True, False]], dtype=torch.bool)
    loss = token_hidden_alignment_loss(
        student_hidden,
        teacher_hidden,
        token_mask,
        teacher_token_mask,
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert float(loss) < 1e-6


def test_token_hidden_alignment_bridge_loss_prefers_matching_vectors() -> None:
    student_hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0], [9.0, 9.0]]],
        dtype=torch.float32,
    )
    teacher_hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]],
        dtype=torch.float32,
    )
    token_mask = torch.tensor([[False, True, True, False]], dtype=torch.bool)
    teacher_token_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    loss = token_hidden_alignment_bridge_loss(
        student_hidden,
        teacher_hidden,
        token_mask,
        teacher_token_mask,
        torch.tensor([1.0], dtype=torch.float32),
    )
    assert float(loss) < 1e-4
