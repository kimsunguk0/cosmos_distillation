"""Student wrapper contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - older transformers fallback
    Qwen3VLForConditionalGeneration = None

from src.data.consistency import ACTION_CLASSES
from src.model.tokenizer_ext import REQUIRED_SPECIAL_TOKENS, ensure_special_tokens


@dataclass(slots=True)
class StudentWrapperConfig:
    student_model_name: str = "nvidia/Cosmos-Reason2-2B"
    teacher_model_name: str = "nvidia/Alpamayo-1.5-10B"
    max_length: int = 4096
    min_pixels: int = 49152
    max_pixels: int = 196608
    torch_dtype: torch.dtype | None = None
    special_tokens: tuple[str, ...] = field(default_factory=lambda: tuple(REQUIRED_SPECIAL_TOKENS))
    trust_remote_code: bool = True
    local_files_only: bool = False
    traj_teacher_hidden_size: int | None = None


def _effective_local_files_only(config: StudentWrapperConfig) -> bool:
    return config.local_files_only or Path(config.student_model_name).expanduser().exists()


class DistillStudentModel(nn.Module):
    """Thin wrapper around a causal LM plus a meta-action classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_action_classes: int,
        *,
        traj_teacher_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.meta_action_head = nn.Linear(hidden_size, num_action_classes)
        # Training-time auxiliary trajectory interface head.
        # Channel 0 predicts accel-like control tokens, channel 1 predicts curvature-like tokens.
        self.traj_aux_head = nn.Linear(hidden_size, 2)
        self.num_action_classes = num_action_classes
        self.traj_teacher_hidden_size: int | None = None
        self.traj_hidden_projector: nn.Linear | None = None
        self.configure_traj_hidden_projector(traj_teacher_hidden_size)

    def configure_traj_hidden_projector(self, output_dim: int | None) -> None:
        """Attach or remove a trainable projector for teacher trajectory hidden alignment."""
        if output_dim in (None, 0):
            self.traj_teacher_hidden_size = None
            self.traj_hidden_projector = None
            return
        output_dim = int(output_dim)
        self.traj_teacher_hidden_size = output_dim
        if output_dim == self.hidden_size:
            self.traj_hidden_projector = None
            return
        projector = self.traj_hidden_projector
        if (
            projector is not None
            and projector.in_features == self.hidden_size
            and projector.out_features == output_dim
        ):
            return
        self.traj_hidden_projector = nn.Linear(self.hidden_size, output_dim, bias=False)

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
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and hasattr(outputs, "language_model_outputs"):
            hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Student backbone did not return hidden states.")
        hidden = hidden_states[-1]
        logits = getattr(outputs, "logits", None)
        if logits is None and hasattr(outputs, "language_model_outputs"):
            logits = getattr(outputs.language_model_outputs, "logits", None)
        if logits is None:
            raise ValueError("Student backbone did not return logits.")
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        traj_hidden = self.traj_hidden_projector(hidden) if self.traj_hidden_projector is not None else hidden
        meta_action_logits = self.meta_action_head(pooled)
        traj_aux_values = self.traj_aux_head(hidden)
        return {
            "backbone_outputs": outputs,
            "logits": logits,
            "hidden_states": hidden,
            "traj_hidden_states": traj_hidden,
            "pooled_hidden": pooled,
            "meta_action_logits": meta_action_logits,
            "traj_aux_values": traj_aux_values,
        }

    def resize_token_embeddings(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate resize call to the backbone."""
        return self.backbone.resize_token_embeddings(*args, **kwargs)


def load_student_tokenizer(config: StudentWrapperConfig):
    """Load and extend the student tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.student_model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=_effective_local_files_only(config),
    )
    ensure_special_tokens(tokenizer, list(config.special_tokens))
    return tokenizer


def load_student_processor(config: StudentWrapperConfig, tokenizer=None):
    """Load the student multimodal processor with bounded pixel budgets."""
    if tokenizer is None:
        tokenizer = load_student_tokenizer(config)
    processor = AutoProcessor.from_pretrained(
        config.student_model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=_effective_local_files_only(config),
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    processor.tokenizer = tokenizer
    return processor


def _resolve_backbone_loader(config: StudentWrapperConfig):
    base_config = AutoConfig.from_pretrained(
        config.student_model_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=_effective_local_files_only(config),
    )
    if getattr(base_config, "model_type", "") == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is not None:
            return Qwen3VLForConditionalGeneration, base_config
        return AutoModelForVision2Seq, base_config
    return AutoModelForCausalLM, base_config


def _hidden_size_from_config(config: Any) -> int:
    for attr in ("hidden_size", "n_embd"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    for attr in ("hidden_size", "n_embd"):
        value = getattr(text_config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("Could not infer hidden size from student config.")


def build_student_model(config: StudentWrapperConfig, tokenizer) -> DistillStudentModel:
    """Load the student model and resize embeddings for special tokens."""
    model_cls, resolved_config = _resolve_backbone_loader(config)
    backbone = model_cls.from_pretrained(
        config.student_model_name,
        dtype=config.torch_dtype,
        trust_remote_code=config.trust_remote_code,
        local_files_only=_effective_local_files_only(config),
    )
    backbone.resize_token_embeddings(len(tokenizer))
    hidden_size = _hidden_size_from_config(getattr(backbone, "config", resolved_config))
    return DistillStudentModel(
        backbone=backbone,
        hidden_size=hidden_size,
        num_action_classes=len(ACTION_CLASSES),
        traj_teacher_hidden_size=config.traj_teacher_hidden_size,
    )
