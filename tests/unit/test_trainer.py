import torch

from src.training.losses import DistillationLossWeights, TrajectoryAuxInterfaceConfig
from src.training.trainer import run_train_step


class _FakeModel:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def __call__(self, **kwargs):
        return self._outputs.pop(0)


def _make_batch(*, include_teacher_view: bool = True) -> dict:
    zeros = torch.zeros((1,), dtype=torch.float32)
    batch = {
        "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 0, 0]], dtype=torch.long),
        "cot_span_mask": torch.zeros((1, 3), dtype=torch.bool),
        "traj_span_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "traj_token_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "format_token_mask": torch.zeros((1, 3), dtype=torch.bool),
        "hard_cot_weights": torch.ones((1,), dtype=torch.float32),
        "traj_weights": torch.ones((1,), dtype=torch.float32),
        "traj_token_label_weights": torch.tensor([[1.0, 1.0, 4.0]], dtype=torch.float32),
        "action_class_labels": torch.tensor([0], dtype=torch.long),
        "action_aux_weight": zeros,
        "teacher_view": {
            "input_ids": torch.tensor([[20, 21, 22]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, 0, 0]], dtype=torch.long),
            "cot_span_mask": torch.zeros((1, 3), dtype=torch.bool),
            "cot_content_mask": torch.zeros((1, 3), dtype=torch.bool),
            "traj_span_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
            "traj_token_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
            "teacher_view_weight": zeros,
            "teacher_logit_kd_weight": zeros,
            "traj_weights": torch.ones((1,), dtype=torch.float32),
            "traj_token_label_weights": torch.tensor([[1.0, 1.0, 4.0]], dtype=torch.float32),
            "teacher_quality_multiplier": torch.ones((1,), dtype=torch.float32),
        },
    }
    if not include_teacher_view:
        batch["teacher_view"] = None
    return batch


def _make_outputs() -> list[dict]:
    hard_logits = torch.tensor(
        [[[0.0, 0.0], [3.0, 0.0], [3.0, 0.0]]],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [[[0.0, 0.0], [0.0, 3.0], [0.0, 3.0]]],
        dtype=torch.float32,
    )
    meta_action_logits = torch.zeros((1, 2), dtype=torch.float32)
    hidden_states = torch.zeros((1, 3, 2), dtype=torch.float32)
    traj_aux_values = torch.zeros((1, 3, 2), dtype=torch.float32)
    return [
        {
            "logits": hard_logits,
            "meta_action_logits": meta_action_logits,
            "hidden_states": hidden_states,
            "traj_aux_values": traj_aux_values,
        },
        {
            "logits": teacher_logits,
            "meta_action_logits": meta_action_logits,
            "hidden_states": hidden_states,
            "traj_aux_values": traj_aux_values,
        },
    ]


def test_run_train_step_does_not_leak_teacher_traj_when_no_explicit_weight() -> None:
    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
    )

    with_teacher_view_loss, with_teacher_view_logs = run_train_step(
        _FakeModel(_make_outputs()[:1]),
        _make_batch(include_teacher_view=True),
        weights,
    )
    without_teacher_view_loss, without_teacher_view_logs = run_train_step(
        _FakeModel(_make_outputs()[:1]),
        _make_batch(include_teacher_view=False),
        weights,
    )

    assert abs(float(with_teacher_view_loss) - float(without_teacher_view_loss)) < 1e-6
    assert abs(float(with_teacher_view_logs["teacher_traj_loss"])) < 1e-6
    assert abs(float(without_teacher_view_logs["teacher_traj_loss"])) < 1e-6
    assert abs(float(with_teacher_view_logs["traj_loss"]) - float(without_teacher_view_logs["traj_loss"])) < 1e-6


def test_run_train_step_supports_explicit_teacher_traj_override() -> None:
    hard_only_weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
    )
    explicit_teacher_weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
        teacher_traj_ce=1.0,
    )

    explicit_batch = _make_batch(include_teacher_view=True)
    explicit_batch["teacher_view"]["traj_weights"] = torch.ones((1,), dtype=torch.float32)

    hard_only_loss, _ = run_train_step(_FakeModel(_make_outputs()[:1]), _make_batch(include_teacher_view=False), hard_only_weights)
    explicit_teacher_loss, explicit_teacher_logs = run_train_step(
        _FakeModel(_make_outputs()),
        explicit_batch,
        explicit_teacher_weights,
    )

    assert float(explicit_teacher_loss) > float(hard_only_loss)
    assert float(explicit_teacher_logs["teacher_traj_loss"]) > 0.0


def test_run_train_step_respects_traj_token_label_weights() -> None:
    batch = _make_batch()
    unweighted_batch = dict(batch)
    unweighted_batch["traj_token_label_weights"] = torch.ones_like(batch["traj_token_label_weights"])
    unweighted_batch["teacher_view"] = dict(batch["teacher_view"])
    unweighted_batch["teacher_view"]["traj_token_label_weights"] = torch.ones_like(
        batch["teacher_view"]["traj_token_label_weights"]
    )
    hard_logits = torch.tensor(
        [[[0.0, 0.0], [0.0, 4.0], [4.0, 0.0]]],
        dtype=torch.float32,
    )
    outputs = [
        {
            "logits": hard_logits,
            "meta_action_logits": torch.zeros((1, 2), dtype=torch.float32),
            "hidden_states": torch.zeros((1, 3, 2), dtype=torch.float32),
        }
    ]

    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
    )
    weighted_loss, _ = run_train_step(_FakeModel(outputs), batch, weights)
    plain_loss, _ = run_train_step(_FakeModel(outputs), unweighted_batch, weights)
    assert float(weighted_loss) > float(plain_loss)


def test_run_train_step_uses_body_only_mask_for_traj_ce() -> None:
    logits = torch.tensor(
        [[[20.0, 0.0], [20.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 0, 1]], dtype=torch.long),
        "cot_span_mask": torch.zeros((1, 3), dtype=torch.bool),
        "traj_span_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "traj_token_mask": torch.tensor([[False, True, False]], dtype=torch.bool),
        "format_token_mask": torch.tensor([[False, False, True]], dtype=torch.bool),
        "hard_cot_weights": torch.ones((1,), dtype=torch.float32),
        "traj_weights": torch.ones((1,), dtype=torch.float32),
        "action_class_labels": torch.tensor([0], dtype=torch.long),
        "action_aux_weight": torch.zeros((1,), dtype=torch.float32),
        "teacher_view": None,
    }
    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
    )
    loss, logs = run_train_step(
        _FakeModel(
            [
                {
                    "logits": logits,
                    "meta_action_logits": torch.zeros((1, 2)),
                    "hidden_states": torch.zeros((1, 3, 2)),
                    "traj_aux_values": torch.zeros((1, 3, 2)),
                }
            ]
        ),
        batch,
        weights,
    )

    assert abs(float(loss)) < 1e-4
    assert abs(float(logs["traj_loss"])) < 1e-4


def test_run_train_step_hidden_distill_works_without_teacher_bridge() -> None:
    logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    traj_hidden_states = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
            ]
        ],
        dtype=torch.float32,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 0, 0, 0]], dtype=torch.long),
        "cot_span_mask": torch.zeros((1, 4), dtype=torch.bool),
        "traj_span_mask": torch.tensor([[False, True, True, True]], dtype=torch.bool),
        "traj_token_mask": torch.tensor([[False, True, True, True]], dtype=torch.bool),
        "format_token_mask": torch.zeros((1, 4), dtype=torch.bool),
        "hard_cot_weights": torch.ones((1,), dtype=torch.float32),
        "traj_weights": torch.zeros((1,), dtype=torch.float32),
        "action_class_labels": torch.tensor([0], dtype=torch.long),
        "action_aux_weight": torch.zeros((1,), dtype=torch.float32),
        "teacher_view": None,
        "teacher_traj_hidden": torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.4, 0.3], [0.8, 0.7, 0.6, 0.5]]],
            dtype=torch.float32,
        ),
        "teacher_traj_hidden_mask": torch.tensor([[True, True, True]], dtype=torch.bool),
        "teacher_traj_available": torch.ones((1,), dtype=torch.bool),
        "teacher_traj_quality_multiplier": torch.ones((1,), dtype=torch.float32),
    }
    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=0.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
        teacher_traj_hidden_align=1.0,
    )
    loss, logs = run_train_step(
        _FakeModel(
            [
                {
                    "logits": logits,
                    "meta_action_logits": torch.zeros((1, 2), dtype=torch.float32),
                    "hidden_states": traj_hidden_states,
                    "traj_hidden_states": traj_hidden_states,
                    "traj_aux_values": torch.zeros((1, 4, 2), dtype=torch.float32),
                }
            ]
        ),
        batch,
        weights,
        traj_hidden_bridge_config={
            "cosine_weight": 0.8,
            "mse_weight": 0.2,
            "relation_weight": 0.5,
            "variance_weight": 0.1,
            "covariance_weight": 0.02,
            "variance_target": 0.5,
        },
    )

    assert float(loss) > 0.0
    assert float(logs["teacher_traj_hidden_align_loss"]) > 0.0
    assert float(logs["teacher_traj_hidden_relation"]) >= 0.0


def test_run_train_step_can_limit_traj_supervision_to_prefix() -> None:
    logits = torch.tensor(
        [[[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]]],
        dtype=torch.float32,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 0, 0, 0]], dtype=torch.long),
        "cot_span_mask": torch.zeros((1, 4), dtype=torch.bool),
        "traj_span_mask": torch.tensor([[False, True, True, True]], dtype=torch.bool),
        "traj_token_mask": torch.tensor([[False, True, True, True]], dtype=torch.bool),
        "format_token_mask": torch.zeros((1, 4), dtype=torch.bool),
        "hard_cot_weights": torch.ones((1,), dtype=torch.float32),
        "traj_weights": torch.ones((1,), dtype=torch.float32),
        "action_class_labels": torch.tensor([0], dtype=torch.long),
        "action_aux_weight": torch.zeros((1,), dtype=torch.float32),
        "teacher_view": None,
    }
    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
    )
    full_loss, _ = run_train_step(
        _FakeModel(
            [
                {
                    "logits": logits,
                    "meta_action_logits": torch.zeros((1, 2)),
                    "hidden_states": torch.zeros((1, 4, 2)),
                    "traj_aux_values": torch.zeros((1, 4, 2)),
                }
            ]
        ),
        batch,
        weights,
    )
    prefix_loss, prefix_logs = run_train_step(
        _FakeModel(
            [
                {
                    "logits": logits,
                    "meta_action_logits": torch.zeros((1, 2)),
                    "hidden_states": torch.zeros((1, 4, 2)),
                    "traj_aux_values": torch.zeros((1, 4, 2)),
                }
            ]
        ),
        batch,
        weights,
        traj_body_prefix_tokens=1,
    )

    assert float(prefix_loss) < float(full_loss)
    assert float(prefix_logs["traj_body_prefix_tokens"]) == 1.0


def test_run_train_step_supports_teacher_topk_on_hard_branch() -> None:
    batch = {
        "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 0, 0]], dtype=torch.long),
        "cot_span_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "cot_content_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "traj_span_mask": torch.zeros((1, 3), dtype=torch.bool),
        "traj_token_mask": torch.zeros((1, 3), dtype=torch.bool),
        "format_token_mask": torch.zeros((1, 3), dtype=torch.bool),
        "hard_cot_weights": torch.ones((1,), dtype=torch.float32),
        "traj_weights": torch.ones((1,), dtype=torch.float32),
        "teacher_pair_weight": torch.ones((1,), dtype=torch.float32),
        "teacher_pair_quality_multiplier": torch.ones((1,), dtype=torch.float32),
        "teacher_topk_indices": torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        "teacher_topk_logprobs": torch.tensor([[[0.0, -4.0], [0.0, -4.0]]], dtype=torch.float32),
        "teacher_topk_mask": torch.tensor([[True, True]], dtype=torch.bool),
        "action_class_labels": torch.tensor([0], dtype=torch.long),
        "action_aux_weight": torch.zeros((1,), dtype=torch.float32),
        "teacher_view": None,
    }
    outputs = [
        {
            "logits": torch.tensor(
                [[[0.0, 0.0], [0.0, 4.0], [0.0, 4.0]]],
                dtype=torch.float32,
            ),
            "meta_action_logits": torch.zeros((1, 2), dtype=torch.float32),
            "hidden_states": torch.zeros((1, 3, 2), dtype=torch.float32),
            "traj_aux_values": torch.zeros((1, 3, 2), dtype=torch.float32),
        }
    ]
    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=1.0,
        traj_ce=0.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
    )

    loss, logs = run_train_step(_FakeModel(outputs), batch, weights)

    assert float(loss) > 0.0
    assert float(logs["teacher_topk_kd_loss"]) > 0.0


def test_run_train_step_reports_traj_aux_loss() -> None:
    batch = {
        "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 3, 4]], dtype=torch.long),
        "cot_span_mask": torch.zeros((1, 3), dtype=torch.bool),
        "traj_span_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "traj_token_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "format_token_mask": torch.zeros((1, 3), dtype=torch.bool),
        "hard_cot_weights": torch.ones((1,), dtype=torch.float32),
        "traj_weights": torch.ones((1,), dtype=torch.float32),
        "action_class_labels": torch.tensor([0], dtype=torch.long),
        "action_aux_weight": torch.zeros((1,), dtype=torch.float32),
        "teacher_view": None,
    }
    outputs = [
        {
            "logits": torch.zeros((1, 3, 16), dtype=torch.float32),
            "meta_action_logits": torch.zeros((1, 2), dtype=torch.float32),
            "hidden_states": torch.zeros((1, 3, 2), dtype=torch.float32),
            "traj_aux_values": torch.tensor(
                [[[0.0, 0.0], [0.25, 0.0], [0.0, -0.25]]],
                dtype=torch.float32,
            ),
        }
    ]
    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=0.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
        traj_aux_reg=1.0,
    )
    from src.training.losses import TrajectoryDecodeConfig

    loss, logs = run_train_step(
        _FakeModel(outputs),
        batch,
        weights,
        TrajectoryDecodeConfig(
            traj_token_start_idx=3,
            num_bins=8,
            dims_min=(-1.0, -1.0),
            dims_max=(1.0, 1.0),
            accel_mean=0.0,
            accel_std=1.0,
            curvature_mean=0.0,
            curvature_std=1.0,
            dt=0.1,
            n_waypoints=1,
        ),
    )

    assert float(loss) > 0.0
    assert float(logs["traj_aux_loss"]) > 0.0


def test_run_train_step_supports_normalized_bounded_traj_aux_loss() -> None:
    batch = {
        "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "labels": torch.tensor([[-100, 3, 4]], dtype=torch.long),
        "cot_span_mask": torch.zeros((1, 3), dtype=torch.bool),
        "traj_span_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "traj_token_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        "format_token_mask": torch.zeros((1, 3), dtype=torch.bool),
        "hard_cot_weights": torch.ones((1,), dtype=torch.float32),
        "traj_weights": torch.ones((1,), dtype=torch.float32),
        "action_class_labels": torch.tensor([0], dtype=torch.long),
        "action_aux_weight": torch.zeros((1,), dtype=torch.float32),
        "teacher_view": None,
    }
    outputs = [
        {
            "logits": torch.zeros((1, 3, 16), dtype=torch.float32),
            "meta_action_logits": torch.zeros((1, 2), dtype=torch.float32),
            "hidden_states": torch.zeros((1, 3, 2), dtype=torch.float32),
            "traj_aux_values": torch.tensor(
                [[[0.0, 0.0], [50.0, 0.0], [0.0, -50.0]]],
                dtype=torch.float32,
            ),
        }
    ]
    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=0.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
        traj_aux_reg=1.0,
    )
    from src.training.losses import TrajectoryDecodeConfig

    loss, logs = run_train_step(
        _FakeModel(outputs),
        batch,
        weights,
        TrajectoryDecodeConfig(
            traj_token_start_idx=3,
            num_bins=8,
            dims_min=(-1.0, -1.0),
            dims_max=(1.0, 1.0),
            accel_mean=0.0,
            accel_std=1.0,
            curvature_mean=0.0,
            curvature_std=1.0,
            dt=0.1,
            n_waypoints=1,
        ),
        TrajectoryAuxInterfaceConfig(
            num_buckets=1,
            normalize_targets=True,
            tanh_bound=3.0,
            huber_delta=1.0,
            target_means=((0.0, 0.0),),
            target_stds=((0.5, 0.5),),
        ),
    )

    assert torch.isfinite(loss)
    assert float(logs["traj_aux_loss"]) > 0.0
