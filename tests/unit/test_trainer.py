import torch

from src.training.losses import DistillationLossWeights
from src.training.trainer import run_train_step


class _FakeModel:
    def __init__(self, outputs):
        self._outputs = list(outputs)

    def __call__(self, **kwargs):
        return self._outputs.pop(0)


def _make_batch() -> dict:
    zeros = torch.zeros((1,), dtype=torch.float32)
    return {
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
    return [
        {
            "logits": hard_logits,
            "meta_action_logits": meta_action_logits,
            "hidden_states": hidden_states,
        },
        {
            "logits": teacher_logits,
            "meta_action_logits": meta_action_logits,
            "hidden_states": hidden_states,
        },
    ]


def test_run_train_step_supports_hard_only_teacher_traj_override() -> None:
    legacy_model = _FakeModel(_make_outputs())
    hard_only_model = _FakeModel(_make_outputs())

    legacy_weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
    )
    hard_only_weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
        teacher_traj_ce=0.0,
    )

    legacy_loss, legacy_logs = run_train_step(legacy_model, _make_batch(), legacy_weights)
    hard_only_loss, hard_only_logs = run_train_step(hard_only_model, _make_batch(), hard_only_weights)

    assert float(legacy_loss) > float(hard_only_loss)
    assert abs(float(legacy_logs["teacher_traj_loss"]) - float(hard_only_logs["teacher_traj_loss"])) < 1e-6
    assert abs(float(legacy_logs["traj_loss"]) - float(hard_only_logs["traj_loss"])) < 1e-6


def test_run_train_step_respects_traj_token_label_weights() -> None:
    model = _FakeModel(_make_outputs())
    batch = _make_batch()
    unweighted_batch = dict(batch)
    unweighted_batch["traj_token_label_weights"] = torch.ones_like(batch["traj_token_label_weights"])
    unweighted_batch["teacher_view"] = dict(batch["teacher_view"])
    unweighted_batch["teacher_view"]["traj_token_label_weights"] = torch.ones_like(
        batch["teacher_view"]["traj_token_label_weights"]
    )

    weights = DistillationLossWeights(
        hard_cot_ce=0.0,
        teacher_seq_ce=0.0,
        teacher_logit_kd=0.0,
        traj_ce=1.0,
        format_ce=0.0,
        action_aux=0.0,
        feat_align=0.0,
    )
    weighted_loss, _ = run_train_step(_FakeModel(_make_outputs()), batch, weights)
    plain_loss, _ = run_train_step(model, unweighted_batch, weights)
    assert float(weighted_loss) > float(plain_loss)
