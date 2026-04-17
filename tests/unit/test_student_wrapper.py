from types import SimpleNamespace

import torch
from torch import nn

from src.model.student_wrapper import DistillStudentModel


class _DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        return SimpleNamespace(hidden_states=[hidden], logits=logits)


def test_distill_student_model_projects_traj_hidden_states() -> None:
    model = DistillStudentModel(
        _DummyBackbone(hidden_size=4),
        hidden_size=4,
        num_action_classes=3,
        traj_teacher_hidden_size=6,
    )
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
    )
    assert outputs["hidden_states"].shape == (1, 3, 4)
    assert outputs["traj_hidden_states"].shape == (1, 3, 6)
    assert outputs["traj_aux_values"].shape == (1, 3, 2)
    assert model.traj_hidden_projector is not None


def test_distill_student_model_supports_bucketed_traj_aux_head() -> None:
    model = DistillStudentModel(
        _DummyBackbone(hidden_size=4),
        hidden_size=4,
        num_action_classes=3,
        traj_aux_num_buckets=4,
    )
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
    )
    assert model.traj_aux_num_buckets == 4
    assert outputs["traj_aux_values"].shape == (1, 3, 8)


def test_distill_student_model_supports_shared_traj_hidden_bridge() -> None:
    model = DistillStudentModel(
        _DummyBackbone(hidden_size=4),
        hidden_size=4,
        num_action_classes=3,
        traj_teacher_hidden_size=6,
        traj_hidden_bridge_size=3,
    )
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
    )
    projected_teacher = model.project_teacher_traj_hidden(torch.randn(1, 2, 6))
    assert outputs["traj_hidden_bridge_states"] is not None
    assert outputs["traj_hidden_bridge_states"].shape == (1, 3, 3)
    assert projected_teacher is not None
    assert projected_teacher.shape == (1, 2, 3)
