"""Checkpoint save/load helpers for full-state and LoRA-adapter training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _cast_float_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    float_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    casted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        if float_dtype is not None and torch.is_floating_point(tensor):
            tensor = tensor.to(float_dtype)
        casted[key] = tensor
    return casted


def _adapter_dir(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "lora_adapter"


def _manifest_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "checkpoint_manifest.json"


def _meta_head_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "meta_action_head.pt"


def _traj_aux_head_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "traj_aux_head.pt"


def _traj_hidden_projector_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "traj_hidden_projector.pt"


def _traj_hidden_bridge_student_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "traj_hidden_bridge_student.pt"


def _traj_hidden_bridge_teacher_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "traj_hidden_bridge_teacher.pt"


def _legacy_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "student_state.pt"


def detect_checkpoint_format(checkpoint_dir: Path) -> str:
    """Detect the checkpoint format saved under one directory."""
    manifest_path = _manifest_path(checkpoint_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return str(manifest.get("format", "unknown"))
    if _adapter_dir(checkpoint_dir).exists():
        return "lora_adapter"
    if _legacy_state_path(checkpoint_dir).exists():
        return "full_state_dict"
    raise FileNotFoundError(f"No recognizable checkpoint found under {checkpoint_dir}")


def save_student_checkpoint(
    checkpoint_dir: Path,
    model,
    tokenizer,
    processor,
    *,
    use_lora: bool,
    full_state_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, Any]:
    """Save a training checkpoint in a compact format."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    try:
        processor.save_pretrained(checkpoint_dir / "processor")
    except Exception:  # noqa: BLE001
        pass

    payload: dict[str, Any]
    if use_lora and hasattr(model.backbone, "save_pretrained"):
        adapter_dir = _adapter_dir(checkpoint_dir)
        model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
        torch.save(_cpu_state_dict(model.meta_action_head), _meta_head_path(checkpoint_dir))
        torch.save(_cpu_state_dict(model.traj_aux_head), _traj_aux_head_path(checkpoint_dir))
        payload = {
            "format": "lora_adapter",
            "adapter_dir": adapter_dir.name,
            "meta_action_head": _meta_head_path(checkpoint_dir).name,
            "traj_aux_head": _traj_aux_head_path(checkpoint_dir).name,
            "traj_aux_num_buckets": int(getattr(model, "traj_aux_num_buckets", 1) or 1),
        }
        if getattr(model, "traj_hidden_projector", None) is not None:
            torch.save(_cpu_state_dict(model.traj_hidden_projector), _traj_hidden_projector_path(checkpoint_dir))
            payload["traj_hidden_projector"] = _traj_hidden_projector_path(checkpoint_dir).name
            payload["traj_teacher_hidden_size"] = int(getattr(model, "traj_teacher_hidden_size", 0) or 0)
        if getattr(model, "traj_hidden_bridge_student", None) is not None:
            torch.save(_cpu_state_dict(model.traj_hidden_bridge_student), _traj_hidden_bridge_student_path(checkpoint_dir))
            torch.save(_cpu_state_dict(model.traj_hidden_bridge_teacher), _traj_hidden_bridge_teacher_path(checkpoint_dir))
            payload["traj_hidden_bridge_student"] = _traj_hidden_bridge_student_path(checkpoint_dir).name
            payload["traj_hidden_bridge_teacher"] = _traj_hidden_bridge_teacher_path(checkpoint_dir).name
            payload["traj_hidden_bridge_size"] = int(getattr(model, "traj_hidden_bridge_size", 0) or 0)
            payload["traj_teacher_hidden_size"] = int(getattr(model, "traj_teacher_hidden_size", 0) or 0)
    else:
        state_dict = _cast_float_state_dict(model.state_dict(), float_dtype=full_state_dtype)
        torch.save(state_dict, _legacy_state_path(checkpoint_dir))
        payload = {
            "format": "full_state_dict",
            "state_dict": _legacy_state_path(checkpoint_dir).name,
            "float_dtype": str(full_state_dtype) if full_state_dtype is not None else None,
            "traj_aux_num_buckets": int(getattr(model, "traj_aux_num_buckets", 1) or 1),
        }
        if getattr(model, "traj_hidden_projector", None) is not None:
            payload["traj_teacher_hidden_size"] = int(getattr(model, "traj_teacher_hidden_size", 0) or 0)
        if getattr(model, "traj_hidden_bridge_student", None) is not None:
            payload["traj_hidden_bridge_size"] = int(getattr(model, "traj_hidden_bridge_size", 0) or 0)
            payload["traj_teacher_hidden_size"] = int(getattr(model, "traj_teacher_hidden_size", 0) or 0)

    _manifest_path(checkpoint_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_student_checkpoint(
    checkpoint_dir: Path,
    model,
    *,
    use_lora: bool,
    adapter_trainable: bool = False,
) -> dict[str, Any]:
    """Load either adapter-only or legacy full-state checkpoints into a model."""
    checkpoint_format = detect_checkpoint_format(checkpoint_dir)

    if checkpoint_format == "lora_adapter":
        from peft import PeftModel
        manifest = json.loads(_manifest_path(checkpoint_dir).read_text(encoding="utf-8"))
        traj_aux_num_buckets = manifest.get("traj_aux_num_buckets")
        if traj_aux_num_buckets not in (None, 0):
            model.configure_traj_aux_head(int(traj_aux_num_buckets))
        traj_teacher_hidden_size = manifest.get("traj_teacher_hidden_size")
        if traj_teacher_hidden_size not in (None, 0):
            model.configure_traj_hidden_projector(int(traj_teacher_hidden_size))
        traj_hidden_bridge_size = manifest.get("traj_hidden_bridge_size")
        if traj_hidden_bridge_size not in (None, 0):
            bridge_teacher_hidden_size = int(
                traj_teacher_hidden_size
                or getattr(model, "traj_teacher_hidden_size", 0)
                or 0
            )
            model.configure_traj_hidden_bridge(
                teacher_hidden_size=bridge_teacher_hidden_size,
                bridge_size=int(traj_hidden_bridge_size),
            )

        model.backbone = PeftModel.from_pretrained(
            model.backbone,
            _adapter_dir(checkpoint_dir),
            is_trainable=adapter_trainable,
        )
        meta_head_path = _meta_head_path(checkpoint_dir)
        if meta_head_path.exists():
            try:
                meta_head_state = torch.load(meta_head_path, map_location="cpu", weights_only=True)
            except TypeError:
                meta_head_state = torch.load(meta_head_path, map_location="cpu")
            model.meta_action_head.load_state_dict(meta_head_state, strict=True)
        traj_aux_head_path = _traj_aux_head_path(checkpoint_dir)
        if traj_aux_head_path.exists():
            try:
                traj_aux_head_state = torch.load(traj_aux_head_path, map_location="cpu", weights_only=True)
            except TypeError:
                traj_aux_head_state = torch.load(traj_aux_head_path, map_location="cpu")
            aux_weight = traj_aux_head_state.get("weight")
            if isinstance(aux_weight, torch.Tensor):
                inferred_num_buckets = max(int(aux_weight.shape[0] // 2), 1)
                model.configure_traj_aux_head(inferred_num_buckets)
            model.traj_aux_head.load_state_dict(traj_aux_head_state, strict=True)
        traj_hidden_projector_path = _traj_hidden_projector_path(checkpoint_dir)
        if traj_hidden_projector_path.exists():
            try:
                projector_state = torch.load(
                    traj_hidden_projector_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                projector_state = torch.load(traj_hidden_projector_path, map_location="cpu")
            if getattr(model, "traj_hidden_projector", None) is None:
                raise ValueError("Checkpoint contains traj_hidden_projector but the model is not configured for it.")
            model.traj_hidden_projector.load_state_dict(projector_state, strict=True)
        traj_hidden_bridge_student_path = _traj_hidden_bridge_student_path(checkpoint_dir)
        traj_hidden_bridge_teacher_path = _traj_hidden_bridge_teacher_path(checkpoint_dir)
        if traj_hidden_bridge_student_path.exists() and traj_hidden_bridge_teacher_path.exists():
            try:
                student_bridge_state = torch.load(
                    traj_hidden_bridge_student_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                student_bridge_state = torch.load(traj_hidden_bridge_student_path, map_location="cpu")
            try:
                teacher_bridge_state = torch.load(
                    traj_hidden_bridge_teacher_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                teacher_bridge_state = torch.load(traj_hidden_bridge_teacher_path, map_location="cpu")
            if getattr(model, "traj_hidden_bridge_student", None) is None or getattr(model, "traj_hidden_bridge_teacher", None) is None:
                raise ValueError("Checkpoint contains traj_hidden_bridge modules but the model is not configured for them.")
            model.traj_hidden_bridge_student.load_state_dict(student_bridge_state, strict=True)
            model.traj_hidden_bridge_teacher.load_state_dict(teacher_bridge_state, strict=True)
        return {
            "format": checkpoint_format,
            "missing": [],
            "unexpected": [],
        }

    if checkpoint_format == "full_state_dict":
        manifest = json.loads(_manifest_path(checkpoint_dir).read_text(encoding="utf-8"))
        traj_aux_num_buckets = manifest.get("traj_aux_num_buckets")
        if traj_aux_num_buckets not in (None, 0):
            model.configure_traj_aux_head(int(traj_aux_num_buckets))
        traj_teacher_hidden_size = manifest.get("traj_teacher_hidden_size")
        if traj_teacher_hidden_size not in (None, 0):
            model.configure_traj_hidden_projector(int(traj_teacher_hidden_size))
        traj_hidden_bridge_size = manifest.get("traj_hidden_bridge_size")
        if traj_hidden_bridge_size not in (None, 0):
            bridge_teacher_hidden_size = int(
                traj_teacher_hidden_size
                or getattr(model, "traj_teacher_hidden_size", 0)
                or 0
            )
            model.configure_traj_hidden_bridge(
                teacher_hidden_size=bridge_teacher_hidden_size,
                bridge_size=int(traj_hidden_bridge_size),
            )
        state_path = _legacy_state_path(checkpoint_dir)
        load_kwargs = {"map_location": "cpu", "weights_only": True}
        try:
            state_dict = torch.load(state_path, mmap=True, **load_kwargs)
        except TypeError:
            state_dict = torch.load(state_path, map_location="cpu")
        try:
            load_result = model.load_state_dict(state_dict, strict=False, assign=True)
        except TypeError:
            load_result = model.load_state_dict(state_dict, strict=False)
        return {
            "format": checkpoint_format,
            "missing": list(load_result.missing_keys),
            "unexpected": list(load_result.unexpected_keys),
            "legacy_lora_expected": bool(use_lora),
        }

    raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
