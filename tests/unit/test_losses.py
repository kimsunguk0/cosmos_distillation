import torch

from src.training.losses import (
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
    token_hidden_alignment_loss,
    weighted_causal_ce,
)


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
