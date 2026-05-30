#!/usr/bin/env python3
"""Train a student-compatible Alpamayo-style AE28 action expert.

This is the formal-compatible action expert path:

  frozen student VLM 2B -> 28-layer student KV cache
  AE28 expert decoder + action_in_proj/action_out_proj
  FlowMatching target in Alpamayo action space [64, 2]

The teacher Alpamayo model is used to provide action_space / flow-matching
utilities, and optionally to initialize the AE modules. Teacher VLM weights may
be loaded on CPU; the student backbone and AE28 trainable modules live on GPU.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
VIS_ROOT = SUKIM_ROOT / "visualization"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC, VIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor  # noqa: E402
from probe_teacher_kv_28layer_expert_compression import (  # noqa: E402
    ade_fde,
    build_28layer_expert,
    force_attention,
    layer_mapping,
    path_len,
    torch_dtype_from_name,
)
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids  # noqa: E402
from src.training.collator import (  # noqa: E402
    _encode_messages,
    build_messages,
    build_user_prompt,
    fuse_history_tokens_in_input_ids,
    load_ego_history_xyz,
    load_sample_images,
    resolve_camera_indices,
)
from src.inference.checkpoint_eval import load_ego_history_rot  # noqa: E402
from src.utils.runtime_paths import remap_external_path, resolve_student_model_path  # noqa: E402


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_TEACHER = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B"
DEFAULT_STUDENT_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "no_nav_camera_labeled_official_200k"
    / "no_nav_official12500_topk_sched16_ar_ramp_p20_rowscale_evalfix_20260517"
    / "best_decode"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "action_expert" / "student_ae28_official"


class StopAfterToken(StoppingCriteria):
    """Stop one decode step after every row has generated a target token.

    The action expert consumes the VLM KV cache at the trajectory-start boundary.
    In HF generation, the sampled token is not guaranteed to be represented in
    the returned cache until the next decode step consumes it. This mirrors the
    official Alpamayo StopAfterEOS behavior: once all rows have emitted the
    boundary token, allow one more generation step so the boundary token is in
    the cache; later tokens are masked out by the expert attention mask.
    """

    def __init__(self, token_id: int, prompt_lengths: list[int]) -> None:
        self.token_id = int(token_id)
        self.prompt_lengths = [int(x) for x in prompt_lengths]
        self.token_found: torch.Tensor | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        batch_size = int(input_ids.shape[0])
        if self.token_found is None or int(self.token_found.numel()) != batch_size:
            self.token_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        if bool(self.token_found.all()):
            return True

        last_tokens = input_ids[:, -1]
        just_found = last_tokens == self.token_id
        for row in range(batch_size):
            prompt_len = self.prompt_lengths[min(row, len(self.prompt_lengths) - 1)]
            if int(input_ids.shape[1]) <= prompt_len:
                just_found[row] = False
        self.token_found = self.token_found | just_found
        return False


class AE28Bundle(nn.Module):
    def __init__(self, *, expert: nn.Module, action_in_proj: nn.Module, action_out_proj: nn.Module) -> None:
        super().__init__()
        self.expert = expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--mapping", choices=("linspace_round", "first_n"), default="linspace_round")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument(
        "--prefix-mode",
        choices=("student_free", "teacher_forced"),
        default="student_free",
        help=(
            "student_free generates CoT until <|traj_future_start|> before caching KV. "
            "teacher_forced caches KV from teacher CoT + <|traj_future_start|> directly."
        ),
    )
    parser.add_argument(
        "--ae-init-mode",
        choices=("teacher_compressed", "scratch", "student_backbone_init", "student_backbone_init_teacher_q"),
        default="teacher_compressed",
        help=(
            "teacher_compressed copies selected teacher expert layers and action projections. "
            "scratch keeps the Alpamayo-compatible AE structure but randomly initializes "
            "expert/action projection weights for student-KV training."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--num-time-samples", type=int, default=1)
    parser.add_argument(
        "--train-timestep-sampler",
        choices=("uniform", "beta"),
        default="beta",
        help=(
            "Flow-matching training timestep sampler. Alpamayo base Stage-2 uses "
            "beta with t = 0.999 - Beta(1.5, 1.0) * 0.999."
        ),
    )
    parser.add_argument(
        "--stage2-attention-mode",
        choices=("official_none", "masked"),
        default="official_none",
        help=(
            "official_none matches alpamayo_base Stage-2 TrainableAlpamayoR1, "
            "which calls the expert with attention_mask=None. masked keeps the "
            "older local inference-style expert attention mask."
        ),
    )
    parser.add_argument("--expert-lr", type=float, default=1e-5)
    parser.add_argument("--proj-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0,
        help="Linear warmup steps for cosine schedule. 0 disables the schedule (constant LR).")
    parser.add_argument("--min-lr", type=float, default=1e-6,
        help="Minimum learning rate at end of cosine decay.")
    parser.add_argument("--no-norm-bias-decay", action="store_true",
        help="Skip weight decay for biases, LayerNorm/RMSNorm scales (matches alpamayo SFT).")
    parser.add_argument("--train-backbone-lora", action="store_true",
        help="Joint-train student backbone LoRA params. Only valid in teacher_forced mode.")
    parser.add_argument("--backbone-lora-lr", type=float, default=5e-6,
        help="Learning rate for student backbone LoRA params when joint-trained.")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument(
        "--eval-seed-mode",
        choices=("fixed", "step"),
        default="step",
        help=(
            "Use a constant diffusion sampling seed at every eval, or include "
            "the training step in the eval seed. `fixed` is the right setting "
            "for overfit/reconstruction sanity checks."
        ),
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    return parser.parse_args()


def _resolve_path(raw: str | Path | None) -> Path | None:
    remapped = remap_external_path(raw)
    if remapped in (None, ""):
        return None
    path = Path(remapped)
    return path if path.exists() else None


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_raw_json(record: dict[str, Any]) -> Path | None:
    raw = ((record.get("teacher_cache") or {}).get("text_raw_json_path"))
    return _resolve_path(raw)


def select_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    scanned = 0
    for row in iter_jsonl(args.corpus_jsonl):
        scanned += 1
        if args.split and row.get("split") != args.split:
            continue
        raw_path = resolve_raw_json(row)
        sample_dir = _resolve_path((row.get("input") or {}).get("materialized_sample_path"))
        if raw_path is None or sample_dir is None:
            continue
        items.append(
            {
                "sample_id": str(row["sample_id"]),
                "row": row,
                "sample_dir": str(sample_dir),
                "raw_json": str(raw_path),
            }
        )
        if len(items) >= int(args.num_samples):
            break
    if not items:
        raise RuntimeError("No usable AE28 samples found.")
    print(
        json.dumps(
            {
                "event": "select_items_done",
                "selected_count": len(items),
                "scanned_count": scanned,
                "corpus_jsonl": str(args.corpus_jsonl),
            }
        ),
        flush=True,
    )
    return items


def raw_teacher_pred(raw_json: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    result = (payload.get("results") or [None])[0]
    if not isinstance(result, dict):
        raise ValueError(f"Missing results[0] in {raw_json}")
    xyz = np.asarray(result.get("pred_xyz"), dtype=np.float32).reshape(-1, 64, 3)[0]
    rot = np.asarray(result.get("pred_rot"), dtype=np.float32).reshape(-1, 64, 3, 3)[0]
    return xyz, rot


def _unwrap_singleton_text(value: Any) -> str:
    while isinstance(value, list) and value:
        value = value[0]
    return str(value or "").strip()


def teacher_cot_text(item: dict[str, Any]) -> str:
    row = item["row"]
    for section_name in ("teacher_target", "hard_target"):
        text = _unwrap_singleton_text((row.get(section_name) or {}).get("cot_text"))
        if text:
            return text
    try:
        payload = json.loads(Path(item["raw_json"]).read_text(encoding="utf-8"))
        result = (payload.get("results") or [None])[0]
        if isinstance(result, dict):
            text = _unwrap_singleton_text((result.get("extra") or {}).get("cot"))
            if text:
                return text
    except Exception:  # noqa: BLE001
        pass
    raise ValueError(f"Missing teacher CoT for sample {item.get('sample_id')}")


def normalize_history_rot(rot: np.ndarray) -> np.ndarray:
    """Normalize materialized history rotations to [T, 3, 3]."""
    arr = np.asarray(rot, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-2:] != (3, 3):
        raise ValueError(f"Expected ego_history_rot as [T,3,3] after squeeze, got shape={arr.shape}")
    return arr


def _to_device_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: _to_device_batch(value, device) for key, value in batch.items()}
    return batch


def unwrap_backbone(backbone: nn.Module) -> nn.Module:
    if hasattr(backbone, "get_base_model"):
        return backbone.get_base_model()
    return backbone


def get_rope_deltas(backbone: nn.Module) -> torch.Tensor:
    candidates = []
    base = unwrap_backbone(backbone)
    candidates.extend([base, getattr(base, "model", None), getattr(getattr(base, "base_model", None), "model", None)])
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "rope_deltas"):
            value = getattr(candidate, "rope_deltas")
            if value is not None:
                return value
    raise AttributeError("Could not find Qwen/Cosmos rope_deltas after student generate().")


def load_student(args: argparse.Namespace):
    checkpoint_dir = args.student_checkpoint_dir
    train_config_path = checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest = (
        json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")) if checkpoint_manifest_path.exists() else {}
    )
    base_model = str((train_config.get("args") or {}).get("student_model") or args.student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))
    tokenizer_dir = checkpoint_dir / "tokenizer"
    processor_dir = checkpoint_dir / "processor"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else base_model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(processor_dir if processor_dir.exists() else base_model, local_files_only=True)
    processor.tokenizer = tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    data_view = train_config.get("data_view") or {}
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", args.max_length)),
        torch_dtype=torch_dtype_from_name(args.student_dtype),
        local_files_only=Path(base_model).expanduser().exists(),
        attn_implementation=args.attn_implementation,
        traj_teacher_hidden_size=(
            int(data_view.get("teacher_traj_hidden_size"))
            if data_view.get("teacher_traj_hidden_size") not in (None, "", 0)
            else None
        ),
        traj_hidden_bridge_size=(
            int(checkpoint_manifest.get("traj_hidden_bridge_size"))
            if checkpoint_manifest.get("traj_hidden_bridge_size") not in (None, "", 0)
            else None
        ),
    )
    print(json.dumps({"event": "load_student_start", "checkpoint": str(checkpoint_dir), "base_model": base_model}), flush=True)
    model = build_student_model(wrapper_cfg, tokenizer)
    checkpoint_format = detect_checkpoint_format(checkpoint_dir)
    if checkpoint_format == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_info = load_student_checkpoint(checkpoint_dir, model, use_lora=use_lora, adapter_trainable=False)
    model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(
        json.dumps(
            {
                "event": "load_student_done",
                "checkpoint_format": checkpoint_format,
                "load_format": load_info.get("format"),
                "device": args.device,
            }
        ),
        flush=True,
    )
    return model, tokenizer, processor, base_model


def reset_module_parameters(module: nn.Module) -> None:
    """Reset trainable parameters while preserving structural buffers."""
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()
        elif hasattr(child, "weight") and child.__class__.__name__ == "RMSNorm":
            with torch.no_grad():
                child.weight.fill_(1.0)


def build_scratch_expert(
    *,
    teacher_expert: nn.Module,
    compressed_layers: int,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str,
) -> nn.Module:
    new_config = copy.deepcopy(teacher_expert.config)
    new_config.num_hidden_layers = int(compressed_layers)
    if hasattr(new_config, "layer_types") and getattr(new_config, "layer_types") is not None:
        new_config.layer_types = list(getattr(new_config, "layer_types"))[: int(compressed_layers)]
    if hasattr(new_config, "_attn_implementation"):
        new_config._attn_implementation = attn_implementation
    if hasattr(new_config, "attn_implementation"):
        new_config.attn_implementation = attn_implementation

    expert = AutoModel.from_config(new_config)
    if hasattr(expert, "embed_tokens"):
        del expert.embed_tokens
    expert = expert.to(device=device, dtype=dtype).train()
    force_attention(expert, attn_implementation)
    return expert



def _merged_layer_state_dict(layer: nn.Module) -> dict:
    """Extract state dict from a (possibly LoRA-wrapped) transformer layer.

    For LoRA layers, merges base_layer.weight + lora_B @ lora_A * scaling
    to produce the effective weight. Non-LoRA parameters are copied as-is.
    """
    merged: dict = {}
    for param_name, param in layer.named_parameters():
        parts = param_name.split(".")
        # e.g. self_attn.q_proj.base_layer.weight -> canonical: self_attn.q_proj.weight
        if "base_layer" in parts:
            idx = parts.index("base_layer")
            canonical = ".".join(parts[:idx] + parts[idx + 1:])
            # find the parent LoRA module to compute merged weight
            parent = layer
            for p in parts[:idx]:
                parent = getattr(parent, p)
            # parent is now the lora Linear module
            if hasattr(parent, "lora_A") and hasattr(parent, "lora_B"):
                # compute merged: base + lora_B @ lora_A * scaling
                base_w = param.data.float()
                for adapter_name in parent.lora_A:
                    scale = parent.scaling.get(adapter_name, 1.0)
                    lora_a = parent.lora_A[adapter_name].weight.data.float()
                    lora_b = parent.lora_B[adapter_name].weight.data.float()
                    base_w = base_w + (lora_b @ lora_a) * scale
                merged[canonical] = base_w.to(param.dtype)
            else:
                merged[canonical] = param.data
        elif any(x in parts for x in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "scaling")):
            # skip raw LoRA delta tensors — already merged above via base_layer path
            pass
        else:
            merged[param_name] = param.data
    return merged


def build_student_backbone_expert(
    *,
    student: Any,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str,
) -> nn.Module:
    """Init AE expert from student backbone transformer layers.

    Student backbone has 28 layers / hidden_size=2048 matching teacher expert dims.
    Q/K/V weights are calibrated for student KV, avoiding uniform-attention collapse
    that random init causes over 3000+ token KV caches.
    LoRA adapters (if present) are merged into base weights during copy.
    """
    student_lm = student.backbone.model.language_model
    new_config = copy.deepcopy(student_lm.config)
    if hasattr(new_config, "_attn_implementation"):
        new_config._attn_implementation = attn_implementation
    if hasattr(new_config, "attn_implementation"):
        new_config.attn_implementation = attn_implementation
    expert = AutoModel.from_config(new_config)
    if hasattr(expert, "embed_tokens"):
        del expert.embed_tokens
    with torch.no_grad():
        for i, src_layer in enumerate(student_lm.layers):
            sd = _merged_layer_state_dict(src_layer)
            expert.layers[i].load_state_dict(sd, strict=True)
        # norm has no LoRA — direct copy
        norm_sd = {k: v.data for k, v in student_lm.norm.named_parameters()}
        expert.norm.load_state_dict(norm_sd, strict=True)
    expert = expert.to(device=device, dtype=dtype).train()
    force_attention(expert, attn_implementation)
    return expert


def build_bundle(teacher_model: Any, args: argparse.Namespace, student: Any = None) -> tuple[AE28Bundle, list[int]]:
    ae_dtype = torch_dtype_from_name(args.ae_dtype)
    selected = layer_mapping(
        int(teacher_model.expert.config.num_hidden_layers),
        int(args.compressed_layers),
        args.mapping,
    )
    expert_attn = "sdpa" if args.attn_implementation != "eager" else "eager"
    if args.ae_init_mode == "teacher_compressed":
        expert = build_28layer_expert(
            teacher_expert=teacher_model.expert,
            selected_old_indices=selected,
            dtype=ae_dtype,
            device=args.device,
            attn_implementation=expert_attn,
        ).train()
        action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=ae_dtype).train()
    elif args.ae_init_mode == "scratch":
        expert = build_scratch_expert(
            teacher_expert=teacher_model.expert,
            compressed_layers=int(args.compressed_layers),
            dtype=ae_dtype,
            device=args.device,
            attn_implementation=expert_attn,
        )
        action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=ae_dtype).train()
        reset_module_parameters(action_in_proj)
        reset_module_parameters(action_out_proj)
    elif args.ae_init_mode == "student_backbone_init":
        if student is None:
            raise ValueError("student_backbone_init requires student model passed to build_bundle()")
        expert = build_student_backbone_expert(
            student=student,
            dtype=ae_dtype,
            device=args.device,
            attn_implementation=expert_attn,
        ).train()
        action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=ae_dtype).train()
        reset_module_parameters(action_in_proj)
        reset_module_parameters(action_out_proj)
    elif args.ae_init_mode == "student_backbone_init_teacher_q":
        if student is None:
            raise ValueError("student_backbone_init_teacher_q requires student model passed to build_bundle()")
        expert = build_student_backbone_expert(
            student=student,
            dtype=ae_dtype,
            device=args.device,
            attn_implementation=expert_attn,
        ).train()
        # Override q_proj from teacher expert layers (first_n mapping)
        teacher_layers = teacher_model.expert.layers
        n_layers = len(expert.layers)
        if len(teacher_layers) < n_layers:
            raise RuntimeError(
                f"Teacher expert has {len(teacher_layers)} layers, need at least {n_layers}"
            )
        with torch.no_grad():
            for new_idx in range(n_layers):
                t_q = teacher_layers[new_idx].self_attn.q_proj
                s_q = expert.layers[new_idx].self_attn.q_proj
                if t_q.weight.shape != s_q.weight.shape:
                    raise RuntimeError(
                        f"Q proj shape mismatch at layer {new_idx}: "
                        f"teacher={tuple(t_q.weight.shape)} student={tuple(s_q.weight.shape)}"
                    )
                s_q.weight.copy_(t_q.weight.to(device=args.device, dtype=ae_dtype))
                if getattr(t_q, "bias", None) is not None and getattr(s_q, "bias", None) is not None:
                    s_q.bias.copy_(t_q.bias.to(device=args.device, dtype=ae_dtype))
        action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=ae_dtype).train()
        reset_module_parameters(action_in_proj)
        reset_module_parameters(action_out_proj)
    else:
        raise ValueError(f"Unsupported ae-init-mode: {args.ae_init_mode}")
    force_attention(expert, expert_attn)
    return AE28Bundle(expert=expert, action_in_proj=action_in_proj, action_out_proj=action_out_proj).train(), selected


def build_batch(
    *,
    args: argparse.Namespace,
    student: Any,
    student_processor: Any,
    student_tokenizer: Any,
    teacher_model: Any,
    batch_items: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [item["row"] for item in batch_items]
    image_batch = [load_sample_images(row, PROJECT_ROOT) for row in rows]
    histories_xyz = [load_ego_history_xyz(row, PROJECT_ROOT).astype(np.float32) for row in rows]
    histories_rot = [normalize_history_rot(load_ego_history_rot(row, PROJECT_ROOT)) for row in rows]
    prompt_messages = []
    teacher_cot_texts = [teacher_cot_text(item) for item in batch_items]
    for row, images, hist_xyz, cot_text in zip(rows, image_batch, histories_xyz, teacher_cot_texts):
        camera_indices = resolve_camera_indices(row, PROJECT_ROOT, image_count=len(images))
        frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
        prompt_text = build_user_prompt(
            row,
            PROJECT_ROOT,
            ego_history_xyz=hist_xyz,
            prompt_text_style="official_alpamayo",
        )
        if args.prefix_mode == "teacher_forced":
            completion_text = f"{cot_text}<|cot_end|><|traj_future_start|>"
        elif args.prefix_mode == "student_free":
            completion_text = None
        else:
            raise ValueError(f"Unsupported prefix mode: {args.prefix_mode}")
        prompt_messages.append(
            build_messages(
                prompt_text,
                len(images),
                completion_text=completion_text,
                assistant_prefix="<|cot_start|>",
                image_prompt_style="camera_labeled",
                camera_indices=camera_indices,
                num_frames_per_camera=frames_per_camera,
            )
        )
    encoded = _encode_messages(
        student_processor,
        prompt_messages,
        image_batch,
        args.max_length,
        continue_final_message=True,
    )
    encoded["input_ids"] = fuse_history_tokens_in_input_ids(
        encoded["input_ids"],
        student_tokenizer,
        histories_xyz,
    )
    device = torch.device(args.device)
    encoded = _to_device_batch(encoded, device)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).to(dtype=torch.long).tolist()

    target_xyz_np: list[np.ndarray] = []
    target_rot_np: list[np.ndarray] = []
    for item in batch_items:
        xyz, rot = raw_teacher_pred(Path(item["raw_json"]))
        target_xyz_np.append(xyz)
        target_rot_np.append(rot)
    target_xyz = torch.from_numpy(np.stack(target_xyz_np, axis=0)).to(device=device, dtype=torch.float32)
    target_rot = torch.from_numpy(np.stack(target_rot_np, axis=0)).to(device=device, dtype=torch.float32)
    ego_history_xyz = torch.from_numpy(np.stack(histories_xyz, axis=0)).to(device=device, dtype=torch.float32)
    ego_history_rot = torch.from_numpy(np.stack(histories_rot, axis=0)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        target_action = teacher_model.action_space.traj_to_action(
            ego_history_xyz,
            ego_history_rot,
            target_xyz,
            target_rot,
        )

    traj_start_id = student_tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if not isinstance(traj_start_id, int) or traj_start_id < 0:
        raise ValueError("Student tokenizer is missing <|traj_future_start|>")

    model_kwargs = dict(encoded)
    input_ids = model_kwargs.pop("input_ids")
    if args.prefix_mode == "student_free":
        generation_config = copy.deepcopy(student.backbone.generation_config)
        generation_config.do_sample = False
        generation_config.num_return_sequences = 1
        generation_config.num_beams = 1
        generation_config.top_p = 1.0
        generation_config.top_k = None
        generation_config.temperature = 1.0
        generation_config.max_new_tokens = int(args.max_new_tokens)
        generation_config.output_logits = False
        generation_config.output_scores = False
        generation_config.output_hidden_states = False
        generation_config.return_dict_in_generate = True
        generation_config.pad_token_id = student_tokenizer.pad_token_id
        stopping = StoppingCriteriaList([StopAfterToken(traj_start_id, prompt_lengths)])
        with torch.no_grad(), torch.autocast(
            "cuda",
            dtype=torch_dtype_from_name(args.student_dtype),
            enabled=device.type == "cuda" and torch.cuda.is_available(),
        ):
            outputs = student.backbone.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                stopping_criteria=stopping,
                **model_kwargs,
            )
        rope_deltas = get_rope_deltas(student.backbone)
        sequences = outputs.sequences
        generated_ids = sequences[:, int(input_ids.shape[1]) :]
        generated_texts = student_tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=False)
        prefix_attention_mask = encoded.get("attention_mask")
        cache = outputs.past_key_values
    else:
        backbone_grad_ctx = nullcontext() if bool(getattr(args, "train_backbone_lora", False)) else torch.no_grad()
        with backbone_grad_ctx, torch.autocast(
            "cuda",
            dtype=torch_dtype_from_name(args.student_dtype),
            enabled=device.type == "cuda" and torch.cuda.is_available(),
        ):
            try:
                outputs = student.backbone(
                    input_ids=input_ids,
                    **model_kwargs,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
            except TypeError:
                outputs = student.backbone(
                    input_ids=input_ids,
                    **model_kwargs,
                    use_cache=True,
                    return_dict=True,
                )
        rope_deltas = getattr(outputs, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = get_rope_deltas(student.backbone)
        sequences = input_ids
        generated_texts = [f"{text}<|cot_end|><|traj_future_start|>" for text in teacher_cot_texts]
        prefix_attention_mask = encoded.get("attention_mask")
        cache = outputs.past_key_values

    offset = teacher_model._find_eos_offset(
        sequences=sequences,
        eos_token_id=int(traj_start_id),
        device=device,
        warn=False,
    )
    kv_cache_seq_len = int(cache.get_seq_length())
    n_diffusion_tokens = int(teacher_model.action_space.get_action_space_dims()[0])
    position_ids, attention_mask = teacher_model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas.to(device),
        kv_cache_seq_len=kv_cache_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=int(sequences.shape[0]),
        device=device,
        prefix_mask=prefix_attention_mask,
    )
    if str(args.stage2_attention_mode) == "official_none":
        position_ids = (
            torch.arange(n_diffusion_tokens, dtype=torch.long, device=device)
            .view(1, 1, -1)
            .repeat(3, int(sequences.shape[0]), 1)
            + rope_deltas.to(device)
            + kv_cache_seq_len
        )
        attention_mask = None
    return {
        "sample_ids": [item["sample_id"] for item in batch_items],
        "prefix_mode": str(args.prefix_mode),
        "cache": cache,
        "context": {
            "kv_cache_seq_len": kv_cache_seq_len,
            "n_diffusion_tokens": n_diffusion_tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "stage2_attention_mode": str(args.stage2_attention_mode),
        },
        "target_action": target_action.detach(),
        "target_xyz": target_xyz.detach(),
        "ego_history_xyz": ego_history_xyz.detach(),
        "ego_history_rot": ego_history_rot.detach(),
        "generated_text_preview": generated_texts[0][:240] if generated_texts else "",
        "traj_start_hit_rate": float(
            sum("<|traj_future_start|>" in text for text in generated_texts) / max(len(generated_texts), 1)
        ),
    }


def repeat_context(context: dict[str, Any], repeats: int) -> dict[str, Any]:
    if int(repeats) <= 1:
        return context
    repeated = dict(context)
    repeated["position_ids"] = context["position_ids"].repeat_interleave(int(repeats), dim=1)
    if context.get("attention_mask") is not None:
        repeated["attention_mask"] = context["attention_mask"].repeat_interleave(int(repeats), dim=0)
    return repeated


def sample_fm_timesteps(
    *,
    batch_size: int,
    sampler: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sampler == "uniform":
        t = torch.rand((batch_size,), device=device, dtype=dtype)
    elif sampler == "beta":
        # Matches alpamayo_base/src/alpamayo_r1/diffusion/flow_matching.py.
        beta = torch.distributions.beta.Beta(
            torch.tensor(1.5, dtype=torch.float32, device=device),
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        t = 0.999 - beta.sample((batch_size,)).to(device=device, dtype=dtype) * 0.999
    else:
        raise ValueError(f"Unknown train timestep sampler: {sampler}")
    return t.view(batch_size, 1, 1)


def train_step(
    *,
    bundle: AE28Bundle,
    teacher_model: Any,
    batch: dict[str, Any],
    num_time_samples: int,
    train_timestep_sampler: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = next(bundle.parameters()).dtype
    repeats = max(int(num_time_samples), 1)
    prompt_cache = batch["cache"]
    context = batch["context"]
    target_action = batch["target_action"]
    if repeats > 1:
        prompt_cache.batch_repeat_interleave(repeats)
        context = repeat_context(context, repeats)
        target_action = target_action.repeat_interleave(repeats, dim=0)

    x1 = target_action.to(device=device, dtype=dtype)
    x0 = torch.randn_like(x1)
    t = sample_fm_timesteps(
        batch_size=int(x1.shape[0]),
        sampler=str(train_timestep_sampler),
        device=device,
        dtype=dtype,
    )
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0

    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = teacher_model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False
    future_token_embeds = bundle.action_in_proj(x_t, t)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(x_t.shape[0], n_diffusion_tokens, -1)
    expert_attention_mask = context.get("attention_mask")
    if expert_attention_mask is not None:
        expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
    out = bundle.expert(
        inputs_embeds=future_token_embeds,
        position_ids=context["position_ids"],
        past_key_values=prompt_cache,
        attention_mask=expert_attention_mask,
        use_cache=True,
        **kwargs,
    )
    prompt_cache.crop(prefill_seq_len)
    last_hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
    pred_v = bundle.action_out_proj(last_hidden).view(-1, *action_dims)
    loss = F.mse_loss(pred_v.float(), target_v.float())
    return loss, {
        "target_action_abs_mean": float(x1.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target_v.detach().abs().mean().cpu()),
        "pred_v_abs_mean": float(pred_v.detach().abs().mean().cpu()),
        "train_t_mean": float(t.detach().float().mean().cpu()),
    }


def sample_paths(
    *,
    bundle: AE28Bundle,
    teacher_model: Any,
    batch: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    dtype = next(bundle.parameters()).dtype
    prompt_cache = batch["cache"]
    context = batch["context"]
    batch_size = int(batch["ego_history_xyz"].shape[0])
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = teacher_model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        future_token_embeds = bundle.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_attention_mask = context.get("attention_mask")
        if expert_attention_mask is not None:
            expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
        out = bundle.expert(
            inputs_embeds=future_token_embeds,
            position_ids=context["position_ids"],
            past_key_values=prompt_cache,
            attention_mask=expert_attention_mask,
            use_cache=True,
            **kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
        return bundle.action_out_proj(last_hidden).view(-1, *action_dims)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        action = teacher_model.diffusion.sample(batch_size=batch_size, step_fn=step_fn, device=device)
        pred_xyz, pred_rot = teacher_model.action_space.action_to_traj(
            action,
            batch["ego_history_xyz"].to(device),
            batch["ego_history_rot"].to(device),
        )
    return {
        "action": action.detach().float().cpu().numpy(),
        "pred_xyz": pred_xyz.detach().float().cpu().numpy(),
        "pred_rot": pred_rot.detach().float().cpu().numpy(),
    }


def iter_batches(items: list[dict[str, Any]], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(items), width):
        yield items[index : index + width]


def evaluate(
    *,
    args: argparse.Namespace,
    bundle: AE28Bundle,
    student: Any,
    student_processor: Any,
    student_tokenizer: Any,
    teacher_model: Any,
    items: list[dict[str, Any]],
    step: int,
) -> dict[str, Any]:
    bundle.eval()
    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    horizon_values: dict[str, dict[str, list[float]]] = {
        "h1p6_16wp": {"ade": [], "fde": []},
        "h3p2_32wp": {"ade": [], "fde": []},
        "h6p4_64wp": {"ade": [], "fde": []},
    }
    eval_seed_base = int(args.seed) + 1000 + (0 if str(args.eval_seed_mode) == "fixed" else int(step))
    for batch_index, batch_items in enumerate(iter_batches(items[: int(args.eval_samples)], int(args.eval_batch_size))):
        batch = build_batch(
            args=args,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            batch_items=batch_items,
        )
        pred = sample_paths(
            bundle=bundle,
            teacher_model=teacher_model,
            batch=batch,
            seed=eval_seed_base + batch_index,
            device=device,
        )
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        for row_index, sample_id in enumerate(batch["sample_ids"]):
            ade, fde = ade_fde(pred["pred_xyz"][row_index], target_xyz[row_index])
            horizon_metrics: dict[str, float] = {}
            for name, horizon in (("h1p6_16wp", 16), ("h3p2_32wp", 32), ("h6p4_64wp", 64)):
                n = min(horizon, int(pred["pred_xyz"][row_index].shape[0]), int(target_xyz[row_index].shape[0]))
                h_ade, h_fde = ade_fde(pred["pred_xyz"][row_index][:n], target_xyz[row_index][:n])
                horizon_values[name]["ade"].append(h_ade)
                horizon_values[name]["fde"].append(h_fde)
                horizon_metrics[f"{name}_ade_m"] = h_ade
                horizon_metrics[f"{name}_fde_m"] = h_fde
            rows.append(
                {
                    "sample_id": sample_id,
                    "ade_m": ade,
                    "fde_m": fde,
                    **horizon_metrics,
                    "pred_path_length_m": path_len(pred["pred_xyz"][row_index]),
                    "target_path_length_m": path_len(target_xyz[row_index]),
                }
            )
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ades = [row["ade_m"] for row in rows]
    fdes = [row["fde_m"] for row in rows]
    out = {
        "event": "eval",
        "step": int(step),
        "eval_seed_mode": str(args.eval_seed_mode),
        "eval_seed_base": int(eval_seed_base),
        "eval_count": len(rows),
        "ade_mean_m": float(np.mean(ades)) if ades else None,
        "ade_p50_m": float(np.percentile(ades, 50)) if ades else None,
        "fde_mean_m": float(np.mean(fdes)) if fdes else None,
        "fde_p50_m": float(np.percentile(fdes, 50)) if fdes else None,
        "horizon": {
            name: {
                "ade_mean_m": float(np.mean(values["ade"])) if values["ade"] else None,
                "ade_p50_m": float(np.percentile(values["ade"], 50)) if values["ade"] else None,
                "fde_mean_m": float(np.mean(values["fde"])) if values["fde"] else None,
                "fde_p50_m": float(np.percentile(values["fde"], 50)) if values["fde"] else None,
            }
            for name, values in horizon_values.items()
        },
        "rows": rows,
    }
    bundle.train()
    return out


def save_checkpoint(path: Path, *, bundle: AE28Bundle, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bundle_state_dict": bundle.state_dict(), "payload": payload}, path)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    summary_path = args.output_dir / "summary.json"
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    summary: dict[str, Any] = {
        "created_at_unix": time.time(),
        "args": vars(args) | {
            "corpus_jsonl": str(args.corpus_jsonl),
            "student_checkpoint_dir": str(args.student_checkpoint_dir),
            "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
            "output_dir": str(args.output_dir),
        },
        "status": "running",
    }
    try:
        items = select_items(args)
        summary["selected_count"] = len(items)
        summary["selected_sample_ids_head"] = [item["sample_id"] for item in items[:16]]

        student, student_tokenizer, student_processor, base_model = load_student(args)
        summary["student_base_model"] = str(base_model)

        print(json.dumps({"event": "load_teacher_action_modules_start", "device": args.teacher_load_device}), flush=True)
        teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=torch_dtype_from_name(args.ae_dtype),
            device=args.teacher_load_device,
            config_json=None,
            runtime_support=None,
            attn_implementation=args.attn_implementation,
            min_pixels=163840,
            max_pixels=196608,
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        force_attention(teacher_model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")
        bundle, selected_layers = build_bundle(teacher_model, args, student=student)
        summary["ae28_selected_teacher_layers"] = selected_layers
        summary["trainable_params"] = int(sum(p.numel() for p in bundle.parameters() if p.requires_grad))
        # Free teacher VLM weights from memory; action_space/diffusion/mask helpers stay on the parent.
        if hasattr(teacher_model, "vlm"):
            delattr(teacher_model, "vlm")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def _split_decay_params(mod: nn.Module, lr_val: float) -> list[dict[str, Any]]:
            if not args.no_norm_bias_decay:
                return [{"params": list(mod.parameters()), "lr": lr_val,
                         "weight_decay": float(args.weight_decay)}]
            decay, no_decay = [], []
            for pname, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                lname = pname.lower()
                is_norm = ("norm" in lname or "layernorm" in lname or "rmsnorm" in lname or "ln_" in lname)
                if p.dim() <= 1 or pname.endswith(".bias") or is_norm:
                    no_decay.append(p)
                else:
                    decay.append(p)
            groups: list[dict[str, Any]] = []
            if decay:
                groups.append({"params": decay, "lr": lr_val, "weight_decay": float(args.weight_decay)})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr_val, "weight_decay": 0.0})
            return groups

        opt_groups: list[dict[str, Any]] = []
        opt_groups.extend(_split_decay_params(bundle.expert, float(args.expert_lr)))
        opt_groups.extend(_split_decay_params(bundle.action_in_proj, float(args.proj_lr)))
        opt_groups.extend(_split_decay_params(bundle.action_out_proj, float(args.proj_lr)))

        # Joint-train student backbone LoRA params (only meaningful in teacher_forced mode).
        backbone_lora_trainable_count = 0
        if bool(args.train_backbone_lora):
            if str(args.prefix_mode) != "teacher_forced":
                raise ValueError("--train-backbone-lora requires --prefix-mode teacher_forced "
                                 "(stochastic generate() in student_free blocks gradient).")
            backbone_lora_params: list[nn.Parameter] = []
            for pname, p in student.backbone.named_parameters():
                lname = pname.lower()
                if ("lora_a" in lname or "lora_b" in lname or "lora_embedding_a" in lname or "lora_embedding_b" in lname):
                    p.requires_grad = True
                    backbone_lora_params.append(p)
                else:
                    p.requires_grad = False
            backbone_lora_trainable_count = sum(int(p.numel()) for p in backbone_lora_params)
            if backbone_lora_params:
                opt_groups.append({"params": backbone_lora_params,
                                   "lr": float(args.backbone_lora_lr),
                                   "weight_decay": 0.0})
            print(json.dumps({
                "event": "backbone_lora_unfrozen",
                "param_count": backbone_lora_trainable_count,
                "module_count": len(backbone_lora_params),
                "lr": float(args.backbone_lora_lr),
            }), flush=True)
        optimizer = torch.optim.AdamW(opt_groups)

        # Cosine LR schedule with warmup (matches alpamayo_base SFT lr_scheduler_type=cosine_warmup_with_min_lr).
        # Only enabled when --lr-warmup-steps > 0.
        scheduler = None
        if int(args.lr_warmup_steps) > 0:
            import math as _math
            warmup_steps_local = int(args.lr_warmup_steps)
            total_steps_local = int(args.steps)
            min_lr_local = float(args.min_lr)

            def _make_lambda(base_lr: float):
                min_ratio = min(1.0, min_lr_local / max(base_lr, 1e-12))
                def _lr_lambda(step_idx: int) -> float:
                    if step_idx < warmup_steps_local:
                        return float(step_idx) / max(1, warmup_steps_local)
                    progress = (step_idx - warmup_steps_local) / max(1, total_steps_local - warmup_steps_local)
                    cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
                    return max(min_ratio, cosine * (1.0 - min_ratio) + min_ratio)
                return _lr_lambda

            lambdas = [_make_lambda(g["lr"]) for g in opt_groups]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        log_handle = log_path.open("a", encoding="utf-8")
        best_eval: dict[str, Any] | None = None

        if not args.skip_initial_eval:
            ev = evaluate(
                args=args,
                bundle=bundle,
                student=student,
                student_processor=student_processor,
                student_tokenizer=student_tokenizer,
                teacher_model=teacher_model,
                items=items,
                step=0,
            )
            print(json.dumps(ev), flush=True)
            log_handle.write(json.dumps(ev) + "\n")
            log_handle.flush()
            best_eval = ev

        started = time.perf_counter()
        batches = list(iter_batches(items, int(args.batch_size)))
        for step in range(1, int(args.steps) + 1):
            batch_items = batches[(step - 1) % len(batches)]
            batch = build_batch(
                args=args,
                student=student,
                student_processor=student_processor,
                student_tokenizer=student_tokenizer,
                teacher_model=teacher_model,
                batch_items=batch_items,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, stats = train_step(
                bundle=bundle,
                teacher_model=teacher_model,
                batch=batch,
                num_time_samples=int(args.num_time_samples),
                train_timestep_sampler=str(args.train_timestep_sampler),
                device=device,
            )
            loss.backward()
            if bool(args.train_backbone_lora):
                params_for_clip = [p for p in bundle.parameters() if p.requires_grad]
                params_for_clip += [p for p in student.backbone.parameters() if p.requires_grad]
            else:
                params_for_clip = list(bundle.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, float(args.grad_clip_norm))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if step == 1 or step % int(args.log_every) == 0:
                row = {
                    "event": "train_step",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "traj_start_hit_rate": batch["traj_start_hit_rate"],
                    "generated_text_preview": batch["generated_text_preview"],
                    **stats,
                }
                print(json.dumps(row), flush=True)
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
            del batch, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            should_eval = step % int(args.eval_every) == 0 or step == int(args.steps)
            if should_eval:
                ev = evaluate(
                    args=args,
                    bundle=bundle,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    items=items,
                    step=step,
                )
                print(json.dumps(ev), flush=True)
                log_handle.write(json.dumps(ev) + "\n")
                log_handle.flush()
                if best_eval is None or float(ev.get("ade_mean_m") or 1e9) < float(best_eval.get("ade_mean_m") or 1e9):
                    best_eval = ev
                    save_checkpoint(args.output_dir / "best.pt", bundle=bundle, payload={"step": step, "eval": ev, "args": vars(args)})
            if args.save_every and step % int(args.save_every) == 0:
                save_checkpoint(args.output_dir / f"step_{step:06d}.pt", bundle=bundle, payload={"step": step, "args": vars(args)})

        save_checkpoint(args.output_dir / "final.pt", bundle=bundle, payload={"step": int(args.steps), "args": vars(args)})
        summary.update(
            {
                "status": "ok",
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "best_eval": best_eval,
            }
        )
        log_handle.close()
    except Exception as exc:  # noqa: BLE001
        summary.update({"status": "failed", "error": repr(exc)})
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path), "status": summary["status"]}), flush=True)


if __name__ == "__main__":
    main()
