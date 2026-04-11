"""Student wrapper contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.consistency import ACTION_CLASSES
from src.model.tokenizer_ext import REQUIRED_SPECIAL_TOKENS, ensure_special_tokens


@dataclass(slots=True)
class StudentWrapperConfig:
    student_model_name: str = "nvidia/Cosmos-Reason2-2B"
    teacher_model_name: str = "nvidia/Alpamayo-1.5-10B"
    max_length: int = 4096
    special_tokens: tuple[str, ...] = field(default_factory=lambda: tuple(REQUIRED_SPECIAL_TOKENS))
    trust_remote_code: bool = True
    local_files_only: bool = False


class DistillStudentModel(nn.Module):
    """Thin wrapper around a causal LM plus a meta-action classification head."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_action_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.meta_action_head = nn.Linear(hidden_size, num_action_classes)
        self.num_action_classes = num_action_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | Any]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        meta_action_logits = self.meta_action_head(pooled)
        return {
            "backbone_outputs": outputs,
            "logits": outputs.logits,
            "hidden_states": hidden,
            "meta_action_logits": meta_action_logits,
        }

    def resize_token_embeddings(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate resize call to the backbone."""
        return self.backbone.resize_token_embeddings(*args, **kwargs)


def load_student_tokenizer(config: StudentWrapperConfig):
    """Load and extend the student tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.student_model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    ensure_special_tokens(tokenizer, list(config.special_tokens))
    return tokenizer


def build_student_model(config: StudentWrapperConfig, tokenizer) -> DistillStudentModel:
    """Load the student model and resize embeddings for special tokens."""
    backbone = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    backbone.resize_token_embeddings(len(tokenizer))
    hidden_size = int(getattr(backbone.config, "hidden_size", getattr(backbone.config, "n_embd")))
    return DistillStudentModel(
        backbone=backbone,
        hidden_size=hidden_size,
        num_action_classes=len(ACTION_CLASSES),
    )
