"""PEFT configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LoraConfigSpec:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def maybe_apply_lora(model: Any, spec: LoraConfigSpec, enabled: bool = False) -> Any:
    """Wrap a model with PEFT LoRA when available and requested."""
    if not enabled:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA requested but `peft` is not installed.") from exc

    peft_config = LoraConfig(
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        bias=spec.bias,
        target_modules=list(spec.target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)
