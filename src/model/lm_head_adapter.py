"""Trainable row adapter for selected LM-head output tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class TrainableLMHeadTokenAdapter(nn.Module):
    """Add a small trainable delta to selected output rows of an LM head."""

    def __init__(
        self,
        base_head: nn.Module,
        token_indices: Sequence[int],
        *,
        init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_head = base_head
        indices = torch.as_tensor(sorted({int(idx) for idx in token_indices}), dtype=torch.long)
        if indices.numel() == 0:
            raise ValueError("TrainableLMHeadTokenAdapter requires at least one token index.")
        self.register_buffer("token_indices", indices, persistent=True)

        weight = getattr(base_head, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError("LM head adapter requires a linear-like output head with a 2D weight tensor.")
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1])
        if int(indices.max().item()) >= self.out_features:
            raise ValueError(
                f"Token index {int(indices.max().item())} is outside LM head vocab size {self.out_features}."
            )

        self.delta_weight = nn.Parameter(torch.zeros((indices.numel(), self.in_features), dtype=weight.dtype))
        if init_std > 0:
            nn.init.normal_(self.delta_weight, mean=0.0, std=float(init_std))

    @property
    def weight(self) -> torch.Tensor:
        return self.base_head.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base_head, "bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base_head(hidden_states)
        delta_logits = F.linear(hidden_states.to(dtype=self.delta_weight.dtype), self.delta_weight)
        delta_logits = delta_logits.to(dtype=logits.dtype)
        index = self.token_indices.to(device=logits.device)
        return logits.index_add(dim=-1, index=index, source=delta_logits)


def get_lm_head_token_adapter(model: Any) -> TrainableLMHeadTokenAdapter | None:
    """Return the installed LM-head row adapter, if present."""
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else getattr(model, "lm_head", None)
    if isinstance(head, TrainableLMHeadTokenAdapter):
        return head
    return None


def get_output_lm_head(model: Any) -> nn.Module:
    """Return the HF-style output head module for a backbone or PEFT model."""
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else getattr(model, "lm_head", None)
    if head is None:
        raise ValueError("Could not find output embeddings/lm_head on the student backbone.")
    return head


def attach_lm_head_token_adapter(
    model: Any,
    token_indices: Sequence[int],
    *,
    init_std: float = 0.0,
) -> TrainableLMHeadTokenAdapter:
    """Install a trainable selected-row adapter on a HF-style output head."""
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else getattr(model, "lm_head", None)
    if head is None:
        raise ValueError("Could not find output embeddings/lm_head on the student backbone.")

    base_head = head.base_head if isinstance(head, TrainableLMHeadTokenAdapter) else head
    adapter = TrainableLMHeadTokenAdapter(base_head, token_indices, init_std=init_std)
    setter = getattr(model, "set_output_embeddings", None)
    if callable(setter):
        setter(adapter)
    elif hasattr(model, "lm_head"):
        setattr(model, "lm_head", adapter)
    else:
        raise ValueError("Student backbone does not expose set_output_embeddings or lm_head.")
    return adapter


def enable_lm_head_token_rows(model: Any, token_indices: Sequence[int]) -> nn.Module:
    """Train only selected rows of the real LM head without extra forward activations."""
    head = get_output_lm_head(model)
    if isinstance(head, TrainableLMHeadTokenAdapter):
        head = head.base_head
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise TypeError("LM-head token rows require a linear-like head with a trainable weight parameter.")

    indices = torch.as_tensor(sorted({int(idx) for idx in token_indices}), dtype=torch.long)
    if indices.numel() == 0:
        raise ValueError("LM-head token rows require at least one token index.")
    if int(indices.max().item()) >= int(weight.shape[0]):
        raise ValueError(f"Token index {int(indices.max().item())} is outside LM head vocab size {int(weight.shape[0])}.")

    old_handle = getattr(head, "_distill_lm_head_token_rows_hook", None)
    if old_handle is not None:
        old_handle.remove()

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        active = indices.to(device=grad.device)
        masked = torch.zeros_like(grad)
        masked.index_copy_(0, active, grad.index_select(0, active))
        return masked

    weight.requires_grad = True
    weight._distill_lm_head_token_rows = True  # type: ignore[attr-defined]
    head._distill_lm_head_token_indices = indices  # type: ignore[attr-defined]
    head._distill_lm_head_token_rows_hook = weight.register_hook(_mask_grad)  # type: ignore[attr-defined]
    bias = getattr(head, "bias", None)
    if isinstance(bias, torch.nn.Parameter):
        bias.requires_grad = True
        bias._distill_lm_head_token_rows = True  # type: ignore[attr-defined]
    return head


def get_lm_head_token_row_count(model: Any) -> int:
    """Return the number of selected output rows configured for training/saving."""
    head = get_output_lm_head(model)
    if isinstance(head, TrainableLMHeadTokenAdapter):
        return int(head.token_indices.numel())
    indices = getattr(head, "_distill_lm_head_token_indices", None)
    if isinstance(indices, torch.Tensor):
        return int(indices.numel())
    token_adapter = getattr(head, "token_adapter", None)
    trainable_deltas = getattr(token_adapter, "trainable_tokens_delta", None)
    if trainable_deltas is not None:
        values = trainable_deltas.values() if hasattr(trainable_deltas, "values") else []
        for value in values:
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                return int(value.shape[0])
    return 0


def export_lm_head_token_rows_state(model: Any) -> dict[str, torch.Tensor] | None:
    """Serialize selected LM-head rows as absolute row weights."""
    head = get_output_lm_head(model)
    if isinstance(head, TrainableLMHeadTokenAdapter):
        return None
    indices = getattr(head, "_distill_lm_head_token_indices", None)
    weight = getattr(head, "weight", None)
    if not isinstance(indices, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return None
    indices = indices.detach().cpu().long()
    state: dict[str, torch.Tensor] = {
        "token_indices": indices,
        "weight": weight.detach().index_select(0, indices.to(device=weight.device)).cpu(),
    }
    bias = getattr(head, "bias", None)
    if isinstance(bias, torch.Tensor):
        state["bias"] = bias.detach().index_select(0, indices.to(device=bias.device)).cpu()
    return state


def load_lm_head_token_rows_state(
    model: Any,
    state: dict[str, torch.Tensor],
    *,
    trainable: bool = False,
) -> None:
    """Load selected LM-head rows into the real output head."""
    token_indices = state.get("token_indices")
    if not isinstance(token_indices, torch.Tensor):
        raise ValueError("LM-head token-row checkpoint is missing token_indices.")
    head = enable_lm_head_token_rows(model, token_indices.tolist()) if trainable else get_output_lm_head(model)
    if isinstance(head, TrainableLMHeadTokenAdapter):
        head = head.base_head
    weight = getattr(head, "weight", None)
    rows = state.get("weight")
    if not isinstance(weight, torch.Tensor) or not isinstance(rows, torch.Tensor):
        raise ValueError("LM-head token-row checkpoint is missing weight rows.")
    indices = token_indices.to(device=weight.device, dtype=torch.long)
    with torch.no_grad():
        weight.index_copy_(0, indices, rows.to(device=weight.device, dtype=weight.dtype))
        bias = getattr(head, "bias", None)
        bias_rows = state.get("bias")
        if isinstance(bias, torch.Tensor) and isinstance(bias_rows, torch.Tensor):
            bias.index_copy_(0, indices, bias_rows.to(device=bias.device, dtype=bias.dtype))
