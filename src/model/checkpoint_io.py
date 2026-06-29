"""Checkpoint save/load helpers for full-state and LoRA-adapter training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.model.flex_scene_encoder import FlexSceneConfig
from src.model.lm_head_adapter import (
    attach_lm_head_token_adapter,
    export_lm_head_token_rows_state,
    get_lm_head_token_adapter,
    get_lm_head_token_row_count,
    load_lm_head_token_rows_state,
)


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


def _boundary_action_head_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "boundary_action_head.pt"


def _traj_hidden_projector_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "traj_hidden_projector.pt"


def _traj_hidden_bridge_student_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "traj_hidden_bridge_student.pt"


def _traj_hidden_bridge_teacher_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "traj_hidden_bridge_teacher.pt"


def _lm_head_token_adapter_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "lm_head_token_adapter.pt"


def _lm_head_token_rows_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "lm_head_token_rows.pt"


def _flex_scene_encoder_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "flex_scene_encoder.pt"


def _flex_deepstack_projector_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "flex_deepstack_projector.pt"


def _flex_scene_config_from_manifest(manifest: dict[str, Any]) -> FlexSceneConfig | None:
    raw = manifest.get("flex_scene_config")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return None
    architecture = str(raw.get("architecture", "single_level") or "single_level")
    default_compression_mode = "per_image" if architecture in {"multi_level", "ml_flex", "ml-flex"} else "global"
    return FlexSceneConfig(
        enabled=True,
        architecture=architecture,
        tokens_per_image=int(raw.get("tokens_per_image", 32) or 32),
        expected_images_per_sample=int(raw.get("expected_images_per_sample", 16) or 16),
        input_hidden_size=int(raw.get("input_hidden_size", 2048) or 2048),
        hidden_size=int(raw.get("hidden_size", 1024) or 1024),
        num_layers=int(raw.get("num_layers", 2) or 2),
        num_heads=int(raw.get("num_heads", 8) or 8),
        mlp_ratio=float(raw.get("mlp_ratio", 4.0) or 4.0),
        dropout=float(raw.get("dropout", 0.0) or 0.0),
        use_camera_time_embeddings=bool(raw.get("use_camera_time_embeddings", False)),
        use_local_slot_embeddings=bool(raw.get("use_local_slot_embeddings", True)),
        max_camera_types=int(raw.get("max_camera_types", 16) or 16),
        compression_mode=str(raw.get("compression_mode", default_compression_mode) or default_compression_mode),
        selection_strategy=str(raw.get("selection_strategy", "first") or "first"),
        num_deepstack_levels=int(raw.get("num_deepstack_levels", 3) or 3),
    )


def _configure_flex_deepstack_projector_from_manifest(manifest: dict[str, Any], model) -> None:
    raw = manifest.get("flex_deepstack_projector_config")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return
    if not hasattr(model, "configure_flex_deepstack_projector"):
        raise ValueError("Checkpoint contains flex_deepstack_projector_config but model cannot configure it.")
    model.configure_flex_deepstack_projector(
        num_layers=int(raw.get("num_layers", 0) or 0),
        rank=int(raw.get("rank", 64) or 64),
        dropout=float(raw.get("dropout", 0.0) or 0.0),
    )


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
        if getattr(model, "boundary_action_head", None) is not None:
            torch.save(_cpu_state_dict(model.boundary_action_head), _boundary_action_head_path(checkpoint_dir))
        payload = {
            "format": "lora_adapter",
            "adapter_dir": adapter_dir.name,
            "meta_action_head": _meta_head_path(checkpoint_dir).name,
            "traj_aux_head": _traj_aux_head_path(checkpoint_dir).name,
            "traj_aux_num_buckets": int(getattr(model, "traj_aux_num_buckets", 1) or 1),
        }
        if getattr(model, "boundary_action_head", None) is not None:
            payload["boundary_action_head"] = _boundary_action_head_path(checkpoint_dir).name
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
        if getattr(model, "flex_scene_encoder", None) is not None:
            torch.save(_cpu_state_dict(model.flex_scene_encoder), _flex_scene_encoder_path(checkpoint_dir))
            payload["flex_scene_encoder"] = _flex_scene_encoder_path(checkpoint_dir).name
            flex_cfg = getattr(model, "flex_scene_config", None)
            if flex_cfg is not None:
                payload["flex_scene_config"] = {
                    "enabled": bool(getattr(flex_cfg, "enabled", False)),
                    "architecture": str(getattr(flex_cfg, "architecture", "single_level") or "single_level"),
                    "tokens_per_image": int(getattr(flex_cfg, "tokens_per_image", 0) or 0),
                    "expected_images_per_sample": int(getattr(flex_cfg, "expected_images_per_sample", 0) or 0),
                    "input_hidden_size": int(getattr(flex_cfg, "input_hidden_size", 0) or 0),
                    "hidden_size": int(getattr(flex_cfg, "hidden_size", 0) or 0),
                    "num_layers": int(getattr(flex_cfg, "num_layers", 0) or 0),
                    "num_heads": int(getattr(flex_cfg, "num_heads", 0) or 0),
                    "mlp_ratio": float(getattr(flex_cfg, "mlp_ratio", 0.0) or 0.0),
                    "dropout": float(getattr(flex_cfg, "dropout", 0.0) or 0.0),
                    "use_camera_time_embeddings": bool(
                        getattr(flex_cfg, "use_camera_time_embeddings", False)
                    ),
                    "use_local_slot_embeddings": bool(
                        getattr(flex_cfg, "use_local_slot_embeddings", True)
                    ),
                    "max_camera_types": int(getattr(flex_cfg, "max_camera_types", 0) or 0),
                    "compression_mode": str(getattr(flex_cfg, "compression_mode", "global") or "global"),
                    "selection_strategy": str(getattr(flex_cfg, "selection_strategy", "first") or "first"),
                    "num_deepstack_levels": int(getattr(flex_cfg, "num_deepstack_levels", 0) or 0),
                }
        if getattr(model, "flex_deepstack_projector", None) is not None:
            torch.save(_cpu_state_dict(model.flex_deepstack_projector), _flex_deepstack_projector_path(checkpoint_dir))
            payload["flex_deepstack_projector"] = _flex_deepstack_projector_path(checkpoint_dir).name
            projector_cfg = getattr(model, "flex_deepstack_projector_config", None) or {}
            payload["flex_deepstack_projector_config"] = {
                "enabled": True,
                "hidden_size": int(projector_cfg.get("hidden_size", 0) or 0),
                "num_layers": int(projector_cfg.get("num_layers", 0) or 0),
                "rank": int(projector_cfg.get("rank", 0) or 0),
                "dropout": float(projector_cfg.get("dropout", 0.0) or 0.0),
            }
        lm_head_adapter = get_lm_head_token_adapter(model.backbone)
        if lm_head_adapter is not None:
            torch.save(_cpu_state_dict(lm_head_adapter), _lm_head_token_adapter_path(checkpoint_dir))
            payload["lm_head_token_adapter"] = _lm_head_token_adapter_path(checkpoint_dir).name
            payload["lm_head_trainable_token_rows"] = int(lm_head_adapter.token_indices.numel())
        lm_head_rows_state = export_lm_head_token_rows_state(model.backbone)
        if lm_head_rows_state is not None:
            torch.save(_cast_float_state_dict(lm_head_rows_state, float_dtype=torch.bfloat16), _lm_head_token_rows_path(checkpoint_dir))
            payload["lm_head_token_rows"] = _lm_head_token_rows_path(checkpoint_dir).name
            payload["lm_head_trainable_token_rows"] = get_lm_head_token_row_count(model.backbone)
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
        if getattr(model, "flex_scene_encoder", None) is not None:
            payload["flex_scene_config"] = {
                "enabled": True,
                "architecture": str(getattr(model.flex_scene_config, "architecture", "single_level") or "single_level"),
                "tokens_per_image": int(getattr(model.flex_scene_config, "tokens_per_image", 0) or 0),
                "expected_images_per_sample": int(getattr(model.flex_scene_config, "expected_images_per_sample", 0) or 0),
                "input_hidden_size": int(getattr(model.flex_scene_config, "input_hidden_size", 0) or 0),
                "hidden_size": int(getattr(model.flex_scene_config, "hidden_size", 0) or 0),
                "num_layers": int(getattr(model.flex_scene_config, "num_layers", 0) or 0),
                "num_heads": int(getattr(model.flex_scene_config, "num_heads", 0) or 0),
                "mlp_ratio": float(getattr(model.flex_scene_config, "mlp_ratio", 0.0) or 0.0),
                "dropout": float(getattr(model.flex_scene_config, "dropout", 0.0) or 0.0),
                "use_camera_time_embeddings": bool(
                    getattr(model.flex_scene_config, "use_camera_time_embeddings", False)
                ),
                "use_local_slot_embeddings": bool(
                    getattr(model.flex_scene_config, "use_local_slot_embeddings", True)
                ),
                "max_camera_types": int(getattr(model.flex_scene_config, "max_camera_types", 0) or 0),
                "compression_mode": str(
                    getattr(model.flex_scene_config, "compression_mode", "global") or "global"
                ),
                "selection_strategy": str(
                    getattr(model.flex_scene_config, "selection_strategy", "first") or "first"
                ),
                "num_deepstack_levels": int(getattr(model.flex_scene_config, "num_deepstack_levels", 0) or 0),
            }
        if getattr(model, "flex_deepstack_projector", None) is not None:
            projector_cfg = getattr(model, "flex_deepstack_projector_config", None) or {}
            payload["flex_deepstack_projector_config"] = {
                "enabled": True,
                "hidden_size": int(projector_cfg.get("hidden_size", 0) or 0),
                "num_layers": int(projector_cfg.get("num_layers", 0) or 0),
                "rank": int(projector_cfg.get("rank", 0) or 0),
                "dropout": float(projector_cfg.get("dropout", 0.0) or 0.0),
            }
        lm_head_adapter = get_lm_head_token_adapter(model.backbone)
        if lm_head_adapter is not None:
            payload["lm_head_trainable_token_rows"] = int(lm_head_adapter.token_indices.numel())
        lm_head_rows_state = export_lm_head_token_rows_state(model.backbone)
        if lm_head_rows_state is not None:
            payload["lm_head_trainable_token_rows"] = get_lm_head_token_row_count(model.backbone)

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
        flex_scene_config = _flex_scene_config_from_manifest(manifest)
        if flex_scene_config is not None and hasattr(model, "configure_flex_scene"):
            model.configure_flex_scene(flex_scene_config)
        _configure_flex_deepstack_projector_from_manifest(manifest, model)
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
        boundary_action_head_path = _boundary_action_head_path(checkpoint_dir)
        if boundary_action_head_path.exists() and getattr(model, "boundary_action_head", None) is not None:
            try:
                boundary_action_head_state = torch.load(boundary_action_head_path, map_location="cpu", weights_only=True)
            except TypeError:
                boundary_action_head_state = torch.load(boundary_action_head_path, map_location="cpu")
            model.boundary_action_head.load_state_dict(boundary_action_head_state, strict=True)
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
        flex_scene_encoder_path = _flex_scene_encoder_path(checkpoint_dir)
        if flex_scene_encoder_path.exists():
            try:
                flex_state = torch.load(flex_scene_encoder_path, map_location="cpu", weights_only=True)
            except TypeError:
                flex_state = torch.load(flex_scene_encoder_path, map_location="cpu")
            if getattr(model, "flex_scene_encoder", None) is None:
                raise ValueError("Checkpoint contains flex_scene_encoder but the model is not configured for FLEX.")
            model.flex_scene_encoder.load_state_dict(flex_state, strict=True)
        flex_deepstack_projector_path = _flex_deepstack_projector_path(checkpoint_dir)
        if flex_deepstack_projector_path.exists():
            try:
                projector_state = torch.load(flex_deepstack_projector_path, map_location="cpu", weights_only=True)
            except TypeError:
                projector_state = torch.load(flex_deepstack_projector_path, map_location="cpu")
            if getattr(model, "flex_deepstack_projector", None) is None:
                raise ValueError(
                    "Checkpoint contains flex_deepstack_projector but the model is not configured for it."
                )
            model.flex_deepstack_projector.load_state_dict(projector_state, strict=True)
        lm_head_adapter_path = _lm_head_token_adapter_path(checkpoint_dir)
        if lm_head_adapter_path.exists():
            try:
                lm_head_adapter_state = torch.load(
                    lm_head_adapter_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                lm_head_adapter_state = torch.load(lm_head_adapter_path, map_location="cpu")
            token_indices = lm_head_adapter_state.get("token_indices")
            if not isinstance(token_indices, torch.Tensor):
                raise ValueError("LM-head token adapter checkpoint is missing token_indices.")
            lm_head_adapter = get_lm_head_token_adapter(model.backbone)
            if lm_head_adapter is None:
                lm_head_adapter = attach_lm_head_token_adapter(model.backbone, token_indices.tolist())
            lm_head_adapter.load_state_dict(lm_head_adapter_state, strict=True)
        lm_head_rows_path = _lm_head_token_rows_path(checkpoint_dir)
        if lm_head_rows_path.exists():
            try:
                lm_head_rows_state = torch.load(
                    lm_head_rows_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                lm_head_rows_state = torch.load(lm_head_rows_path, map_location="cpu")
            load_lm_head_token_rows_state(
                model.backbone,
                lm_head_rows_state,
                trainable=adapter_trainable,
            )
        return {
            "format": checkpoint_format,
            "missing": [],
            "unexpected": [],
        }

    if checkpoint_format == "full_state_dict":
        manifest = json.loads(_manifest_path(checkpoint_dir).read_text(encoding="utf-8"))
        flex_scene_config = _flex_scene_config_from_manifest(manifest)
        if flex_scene_config is not None and hasattr(model, "configure_flex_scene"):
            model.configure_flex_scene(flex_scene_config)
        _configure_flex_deepstack_projector_from_manifest(manifest, model)
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
