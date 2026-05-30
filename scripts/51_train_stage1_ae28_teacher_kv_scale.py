#!/usr/bin/env python3
"""Scaled Stage 1 AE-28T trainer.

This is the serious Stage 1 path:

    teacher VLM 36-layer KV
      -> select 28 cache layers
      -> 28-layer action expert
      -> match cached original 36-layer teacher action trajectory

Compared with the first overfit script, this version:
  * trains with batches,
  * reuses each VLM batch for multiple flow-matching time/noise samples,
  * evaluates with fixed seeds in batched diffusion,
  * loads materialized samples lazily so larger corpora are practical.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, LogitsProcessorList, StoppingCriteriaList


SUKIM_ROOT = Path("/home/pm97/workspace/sukim")
DISTILL_ROOT = SUKIM_ROOT / "distillation" / "cosmos_distillation"
VIS_ROOT = SUKIM_ROOT / "visualization"
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
for path in (SUKIM_ROOT, DISTILL_ROOT, VIS_ROOT, ALPAMAYO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor  # noqa: E402
from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token  # noqa: E402
from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (  # noqa: E402
    build_model_inputs_batch,
    load_materialized_samples,
    load_model_and_processor,
)
from probe_teacher_kv_28layer_expert_compression import (  # noqa: E402
    ade_fde,
    build_28layer_expert,
    force_attention,
    layer_mapping,
    make_context,
    path_len,
    torch_dtype_from_name,
)


DEFAULT_CORPUS = DISTILL_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_TEACHER = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B"
DEFAULT_OUTPUT = DISTILL_ROOT / "outputs" / "action_expert" / "stage1_ae28_teacher_kv_scale"


class KVLayerMixer(nn.Module):
    """Learned 36->28 KV layer compressor.

    The initial point is close to the hand-picked layer selection, then each
    target layer can learn a separate key/value mixture over all teacher layers.
    """

    def __init__(
        self,
        *,
        old_layers: int,
        selected_old_indices: list[int],
        init_alpha: float,
    ) -> None:
        super().__init__()
        self.old_layers = int(old_layers)
        self.new_layers = len(selected_old_indices)
        base = torch.zeros((self.new_layers, self.old_layers), dtype=torch.float32)
        for new_idx, old_idx in enumerate(selected_old_indices):
            base[new_idx, int(old_idx)] = 1.0
        self.register_buffer("base_weights", base, persistent=True)
        init_alpha = min(max(float(init_alpha), 1e-4), 0.99)
        init_logit = float(np.log(init_alpha / (1.0 - init_alpha)))
        self.key_logits = nn.Parameter(torch.zeros((self.new_layers, self.old_layers), dtype=torch.float32))
        self.value_logits = nn.Parameter(torch.zeros((self.new_layers, self.old_layers), dtype=torch.float32))
        self.key_gate_logits = nn.Parameter(torch.full((self.new_layers, 1), init_logit, dtype=torch.float32))
        self.value_gate_logits = nn.Parameter(torch.full((self.new_layers, 1), init_logit, dtype=torch.float32))

    def _weights(self, logits: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        learned = torch.softmax(logits, dim=-1)
        gate = torch.sigmoid(gate_logits)
        return ((1.0 - gate) * self.base_weights.to(device=logits.device)) + (gate * learned)

    def key_weights(self) -> torch.Tensor:
        return self._weights(self.key_logits, self.key_gate_logits)

    def value_weights(self) -> torch.Tensor:
        return self._weights(self.value_logits, self.value_gate_logits)

    def stats(self) -> dict[str, float]:
        with torch.no_grad():
            kw = self.key_weights()
            vw = self.value_weights()
            key_gate = torch.sigmoid(self.key_gate_logits)
            value_gate = torch.sigmoid(self.value_gate_logits)
            key_entropy = -(kw * kw.clamp_min(1e-9).log()).sum(dim=-1)
            value_entropy = -(vw * vw.clamp_min(1e-9).log()).sum(dim=-1)
            return {
                "kv_mixer_key_gate_mean": float(key_gate.mean().detach().cpu()),
                "kv_mixer_value_gate_mean": float(value_gate.mean().detach().cpu()),
                "kv_mixer_key_entropy_mean": float(key_entropy.mean().detach().cpu()),
                "kv_mixer_value_entropy_mean": float(value_entropy.mean().detach().cpu()),
                "kv_mixer_key_max_weight_mean": float(kw.max(dim=-1).values.mean().detach().cpu()),
                "kv_mixer_value_max_weight_mean": float(vw.max(dim=-1).values.mean().detach().cpu()),
            }

    def forward(self, cache: Any, *, dtype: torch.dtype) -> Any:
        layers = list(getattr(cache, "layers", []))
        if len(layers) != self.old_layers:
            raise ValueError(f"KV mixer expected {self.old_layers} layers, got {len(layers)}")
        key_weights = self.key_weights().to(device=layers[0].keys.device, dtype=dtype)
        value_weights = self.value_weights().to(device=layers[0].values.device, dtype=dtype)
        mixed_cache = copy.copy(cache)
        new_layers = []
        for new_idx in range(self.new_layers):
            key_acc = None
            value_acc = None
            for old_idx, layer in enumerate(layers):
                key_term = layer.keys.to(dtype=dtype) * key_weights[new_idx, old_idx]
                value_term = layer.values.to(dtype=dtype) * value_weights[new_idx, old_idx]
                key_acc = key_term if key_acc is None else key_acc + key_term
                value_acc = value_term if value_acc is None else value_acc + value_term
            new_layer = copy.copy(layers[int(torch.argmax(self.base_weights[new_idx]).item())])
            new_layer.keys = key_acc
            new_layer.values = value_acc
            new_layers.append(new_layer)
        mixed_cache.layers = new_layers
        return mixed_cache


class AE28Bundle(nn.Module):
    def __init__(
        self,
        *,
        expert: nn.Module,
        action_in_proj: nn.Module,
        action_out_proj: nn.Module,
        kv_mixer: KVLayerMixer | None = None,
    ) -> None:
        super().__init__()
        self.expert = expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj
        self.kv_mixer = kv_mixer


def reset_module_parameters(module: nn.Module) -> None:
    """Reset parameters recursively where modules expose reset_parameters()."""
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def build_scratch_expert(
    *,
    teacher_expert: nn.Module,
    num_layers: int,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str,
) -> nn.Module:
    new_config = copy.deepcopy(teacher_expert.config)
    new_config.num_hidden_layers = int(num_layers)
    if hasattr(new_config, "layer_types") and getattr(new_config, "layer_types") is not None:
        new_config.layer_types = list(getattr(new_config, "layer_types"))[: int(num_layers)]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-time-samples", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--expert-lr", type=float, default=1e-5)
    parser.add_argument("--proj-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="sdpa")
    parser.add_argument("--vram-cap-gb", type=float, default=70.0)
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument("--mapping", choices=("linspace_round", "first_n"), default="linspace_round")
    parser.add_argument(
        "--ae-init-mode",
        choices=("teacher_compressed", "scratch_expert", "scratch_all"),
        default="teacher_compressed",
        help=(
            "teacher_compressed copies selected teacher expert layers/projections; "
            "scratch_expert random-initializes the expert but copies action projections; "
            "scratch_all random-initializes expert/action projections."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--io-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument(
        "--stage1-mode",
        choices=("endpoint", "official_fm", "teacher_velocity", "teacher_velocity_kv_mixer"),
        default="endpoint",
    )
    parser.add_argument(
        "--prefix-mode",
        choices=("generated", "teacher_forced"),
        default="generated",
        help=(
            "generated matches the older diagnostic path that samples CoT to "
            "<|traj_future_start|>. teacher_forced builds the KV from cached "
            "teacher CoT + <|traj_future_start|>, matching Alpamayo base Stage-2."
        ),
    )
    parser.add_argument("--teacher-velocity-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-aux-weight", type=float, default=0.05)
    parser.add_argument("--kv-mixer-lr", type=float, default=1e-3)
    parser.add_argument("--kv-mixer-init-alpha", type=float, default=0.02)
    parser.add_argument(
        "--train-noise-mode",
        choices=("random", "fixed_by_sample"),
        default="random",
        help="Use fresh flow-matching noise every step, or deterministic noise keyed by sample id.",
    )
    parser.add_argument(
        "--train-timestep-sampler",
        choices=("uniform", "beta"),
        default="uniform",
        help="Flow-matching training timestep sampler. beta matches the public Alpamayo base recipe.",
    )
    parser.add_argument(
        "--eval-velocity-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="During eval, also report velocity MSE on the same deterministic noise/t grid.",
    )
    parser.add_argument("--fixed-grid-salt", default="stage1_fixed_grid_v1")
    return parser.parse_args()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_mem() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    sync_cuda()
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
    }


def configure_vram_cap(device: str, cap_gb: float | None) -> dict[str, float | None]:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {"vram_cap_gb": None, "device_total_gb": None, "allocator_fraction": None}
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    total_gb = float(torch.cuda.get_device_properties(device_index).total_memory / (1024**3))
    cap = float(cap_gb or 0.0)
    if cap > 0:
        fraction = min(max(cap / total_gb, 0.01), 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    else:
        fraction = 1.0
    return {
        "vram_cap_gb": round(cap, 3) if cap > 0 else None,
        "device_total_gb": round(total_gb, 3),
        "allocator_fraction": round(fraction, 6),
    }


def check_vram_cap(args: argparse.Namespace, *, where: str) -> None:
    cap = float(getattr(args, "vram_cap_gb", 0.0) or 0.0)
    if cap <= 0 or not torch.cuda.is_available():
        return
    mem = cuda_mem() or {}
    max_reserved = float(mem.get("max_reserved_gb") or 0.0)
    if max_reserved > cap + 0.25:
        raise RuntimeError(f"VRAM cap exceeded at {where}: max_reserved_gb={max_reserved:.3f} cap_gb={cap:.3f}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_raw_json(record: dict[str, Any]) -> Path | None:
    raw = ((record.get("teacher_cache") or {}).get("text_raw_json_path"))
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None


def select_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    scanned = 0
    for row in iter_jsonl(args.corpus_jsonl):
        scanned += 1
        if args.split and row.get("split") != args.split:
            continue
        raw_path = resolve_raw_json(row)
        sample_dir = Path(str((row.get("input") or {}).get("materialized_sample_path") or ""))
        if raw_path is None or not sample_dir.exists():
            continue
        items.append(
            {
                "sample_id": str(row["sample_id"]),
                "sample_dir": str(sample_dir),
                "raw_json": str(raw_path),
                "clip_id": str(row.get("clip_id") or ""),
                "chunk_id": str(row.get("chunk_id") or ""),
            }
        )
        if len(items) >= int(args.num_samples):
            break
    if not items:
        raise RuntimeError("No Stage 1 items with raw teacher action outputs were found.")
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


def nested_text(value: Any) -> str:
    current = value
    while isinstance(current, list) and current:
        current = current[0]
    return str(current).strip() if isinstance(current, str) else ""


def raw_teacher_cot(raw_json: Path) -> str:
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    result = (payload.get("results") or [None])[0]
    if not isinstance(result, dict):
        raise ValueError(f"Missing results[0] in {raw_json}")
    cot = nested_text((result.get("extra") or {}).get("cot"))
    if not cot:
        raise ValueError(f"Missing teacher CoT in {raw_json}")
    return cot


def make_bundle(
    *,
    model: Any,
    selected_old_indices: list[int],
    dtype: torch.dtype,
    device: str,
    attn_implementation: str,
    use_kv_mixer: bool = False,
    kv_mixer_init_alpha: float = 0.02,
    ae_init_mode: str = "teacher_compressed",
) -> AE28Bundle:
    if ae_init_mode == "teacher_compressed":
        expert = build_28layer_expert(
            teacher_expert=model.expert,
            selected_old_indices=selected_old_indices,
            dtype=dtype,
            device=device,
            attn_implementation=attn_implementation,
        )
    elif ae_init_mode in {"scratch_expert", "scratch_all"}:
        expert = build_scratch_expert(
            teacher_expert=model.expert,
            num_layers=len(selected_old_indices),
            dtype=dtype,
            device=device,
            attn_implementation=attn_implementation,
        )
    else:
        raise ValueError(f"Unsupported ae_init_mode: {ae_init_mode}")
    action_in_proj = copy.deepcopy(model.action_in_proj).to(device=device, dtype=dtype).train()
    action_out_proj = copy.deepcopy(model.action_out_proj).to(device=device, dtype=dtype).train()
    if ae_init_mode == "scratch_all":
        reset_module_parameters(action_in_proj)
        reset_module_parameters(action_out_proj)
    for module in (expert, action_in_proj, action_out_proj):
        for param in module.parameters():
            param.requires_grad_(True)
    kv_mixer = None
    if use_kv_mixer:
        kv_mixer = KVLayerMixer(
            old_layers=int(model.expert.config.num_hidden_layers),
            selected_old_indices=selected_old_indices,
            init_alpha=float(kv_mixer_init_alpha),
        ).to(device=device)
    bundle = AE28Bundle(
        expert=expert,
        action_in_proj=action_in_proj,
        action_out_proj=action_out_proj,
        kv_mixer=kv_mixer,
    )
    bundle.train()
    return bundle


def select_cache_layers_inplace(cache: Any, selected_old_indices: list[int]) -> Any:
    cache.layers = [cache.layers[index] for index in selected_old_indices]
    return cache


def cast_cache_layers_inplace(cache: Any, dtype: torch.dtype) -> Any:
    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "keys") and layer.keys is not None and layer.keys.dtype != dtype:
            layer.keys = layer.keys.to(dtype=dtype)
        if hasattr(layer, "values") and layer.values is not None and layer.values.dtype != dtype:
            layer.values = layer.values.to(dtype=dtype)
    return cache


def repeat_context(context: dict[str, Any], repeats: int) -> dict[str, Any]:
    if int(repeats) <= 1:
        return context
    repeated = dict(context)
    repeated["position_ids"] = context["position_ids"].repeat_interleave(int(repeats), dim=1)
    if context.get("attention_mask") is not None:
        repeated["attention_mask"] = context["attention_mask"].repeat_interleave(int(repeats), dim=0)
    if "offset" in context:
        repeated["offset"] = context["offset"].repeat_interleave(int(repeats), dim=0)
    return repeated


def build_batch(
    *,
    model: Any,
    processor: Any,
    batch_items: list[dict[str, Any]],
    selected_old_indices: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_dirs = [Path(item["sample_dir"]) for item in batch_items]
    samples = load_materialized_samples(sample_dirs, int(args.io_workers))
    if str(args.prefix_mode) == "teacher_forced":
        messages_batch = []
        for item, sample in zip(batch_items, samples, strict=True):
            nav_text = sample.get("nav_text")
            if nav_text is not None and not str(nav_text).strip():
                nav_text = None
            messages = helper.create_message(
                frames=sample["image_frames"].flatten(0, 1),
                camera_indices=sample["camera_indices"],
                nav_text=nav_text,
            )
            cot = raw_teacher_cot(Path(item["raw_json"]))
            messages[-1]["content"][0]["text"] = f"<|cot_start|>{cot}<|cot_end|><|traj_future_start|>"
            messages_batch.append(messages)
        tokenized = processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None and bool(torch.all(attention_mask == 1).item()):
            tokenized.pop("attention_mask", None)
        model_inputs = helper.to_device(
            {
                "tokenized_data": tokenized,
                "ego_history_xyz": torch.cat([sample["ego_history_xyz"] for sample in samples], dim=0),
                "ego_history_rot": torch.cat([sample["ego_history_rot"] for sample in samples], dim=0),
            },
            args.device,
        )
    else:
        model_inputs = build_model_inputs_batch(processor=processor, samples=samples, device=args.device)

    target_xyz_np: list[np.ndarray] = []
    target_rot_np: list[np.ndarray] = []
    for item in batch_items:
        xyz, rot = raw_teacher_pred(Path(item["raw_json"]))
        target_xyz_np.append(xyz)
        target_rot_np.append(rot)
    target_xyz = torch.from_numpy(np.stack(target_xyz_np, axis=0)).to(args.device, dtype=torch.float32)
    target_rot = torch.from_numpy(np.stack(target_rot_np, axis=0)).to(args.device, dtype=torch.float32)
    with torch.inference_mode():
        target_action = model.action_space.traj_to_action(
            model_inputs["ego_history_xyz"][:, -1],
            model_inputs["ego_history_rot"][:, -1],
            target_xyz,
            target_rot,
        )

    tokenized_data = dict(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    fused_input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))

    dtype = next(model.parameters()).dtype
    sync_cuda()
    started = time.perf_counter()
    if str(args.prefix_mode) == "teacher_forced":
        with torch.no_grad(), torch.autocast(
            "cuda",
            dtype=dtype,
            enabled=str(args.device).startswith("cuda") and torch.cuda.is_available(),
        ):
            try:
                outputs = model.vlm(
                    input_ids=fused_input_ids,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                    **tokenized_data,
                )
            except TypeError:
                outputs = model.vlm(
                    input_ids=fused_input_ids,
                    use_cache=True,
                    return_dict=True,
                    **tokenized_data,
                )
        sync_cuda()
        outputs.rope_deltas = getattr(outputs, "rope_deltas", None)
        if outputs.rope_deltas is None:
            outputs.rope_deltas = model.vlm.model.rope_deltas
        outputs.sequences = fused_input_ids
    else:
        generation_config = copy.deepcopy(model.vlm.generation_config)
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
        generation_config.pad_token_id = model.tokenizer.pad_token_id
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=int(eos_token_id))])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=int(model.config.traj_token_start_idx),
                    traj_vocab_size=int(model.config.traj_vocab_size),
                )
            ]
        )
        # Use no_grad instead of inference_mode because learned KV mixing needs the
        # generated cache tensors to participate in autograd as constants.
        with torch.no_grad(), torch.autocast(
            "cuda",
            dtype=dtype,
            enabled=str(args.device).startswith("cuda") and torch.cuda.is_available(),
        ):
            outputs = model.vlm.generate(
                input_ids=fused_input_ids,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                **tokenized_data,
            )
        sync_cuda()
        outputs.rope_deltas = model.vlm.model.rope_deltas
        outputs.sequences = replace_padding_after_eos(
            token_ids=outputs.sequences.clone(),
            eos_token_id=int(eos_token_id),
            pad_token_id=model.tokenizer.pad_token_id,
        )
    if args.stage1_mode in {"teacher_velocity", "teacher_velocity_kv_mixer"}:
        cache = outputs.past_key_values
    else:
        cache = select_cache_layers_inplace(outputs.past_key_values, selected_old_indices)
    context = make_context(
        model=model,
        sequences=outputs.sequences,
        eos_token_id=int(eos_token_id),
        rope_deltas=outputs.rope_deltas,
        cache=cache,
        prefix_mask=tokenized_data.get("attention_mask"),
        device=torch.device(args.device),
    )
    generated_ids = outputs.sequences[:, int(fused_input_ids.shape[1]) :]
    generated_texts = model.tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=False)
    return {
        "sample_ids": [item["sample_id"] for item in batch_items],
        "cache": cache,
        "context": context,
        "target_action": target_action.detach(),
        "target_xyz": target_xyz.detach(),
        "ego_history_xyz": model_inputs["ego_history_xyz"].detach(),
        "ego_history_rot": model_inputs["ego_history_rot"].detach(),
        "meta": {
            "vlm_elapsed_sec": round(time.perf_counter() - started, 6),
            "cache_layer_count": len(getattr(cache, "layers", [])),
            "cache_seq_len": int(cache.get_seq_length()),
            "generated_len_mean": float(generated_ids.shape[1]),
            "generated_text_preview": generated_texts[0][:240] if generated_texts else "",
        },
    }


def stable_sample_seed(base_seed: int, sample_id: str, repeat_index: int, salt: str) -> int:
    key = f"{int(base_seed)}::{salt}::{sample_id}::{int(repeat_index)}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def sample_train_timestep(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sampler: str,
) -> torch.Tensor:
    if sampler == "beta":
        beta = torch.distributions.beta.Beta(
            torch.tensor(1.5, dtype=torch.float32, device=device),
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        t = 0.999 - beta.sample((batch_size,)).to(device=device) * 0.999
        return t.to(dtype=dtype).view(batch_size, 1, 1)
    return torch.rand((batch_size, 1, 1), device=device, dtype=dtype)


def make_flow_training_tensors(
    *,
    x1: torch.Tensor,
    sample_ids: list[str],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if str(args.train_noise_mode) != "fixed_by_sample":
        x0 = torch.randn_like(x1)
        t = sample_train_timestep(
            batch_size=int(x1.shape[0]),
            device=device,
            dtype=dtype,
            sampler=str(args.train_timestep_sampler),
        )
        x_t = (1.0 - t) * x0 + t * x1
        return x0, t, x_t, x1 - x0

    x0_rows: list[torch.Tensor] = []
    t_rows: list[float] = []
    for index, sample_id in enumerate(sample_ids):
        seed = stable_sample_seed(
            int(args.seed),
            str(sample_id),
            repeat_index=index,
            salt=str(args.fixed_grid_salt),
        )
        rng = np.random.default_rng(seed)
        x0_np = rng.standard_normal(tuple(x1.shape[1:]), dtype=np.float32)
        if str(args.train_timestep_sampler) == "beta":
            t_value = float(0.999 - rng.beta(1.5, 1.0) * 0.999)
        else:
            t_value = float(rng.random())
        x0_rows.append(torch.from_numpy(x0_np))
        t_rows.append(t_value)

    x0 = torch.stack(x0_rows, dim=0).to(device=device, dtype=dtype)
    t = torch.tensor(t_rows, device=device, dtype=dtype).view(int(x1.shape[0]), 1, 1)
    x_t = (1.0 - t) * x0 + t * x1
    return x0, t, x_t, x1 - x0


def train_velocity_forward(
    *,
    bundle: AE28Bundle,
    model: Any,
    prompt_cache: Any,
    context: dict[str, Any],
    target_action: torch.Tensor,
    sample_ids: list[str],
    args: argparse.Namespace,
    num_time_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = next(bundle.parameters()).dtype
    repeats = max(int(num_time_samples), 1)
    if repeats > 1:
        prompt_cache.batch_repeat_interleave(repeats)
        context = repeat_context(context, repeats)
        target_action = target_action.repeat_interleave(repeats, dim=0)
        sample_ids = [sample_id for sample_id in sample_ids for _ in range(repeats)]

    x1 = target_action.to(device=device, dtype=dtype)
    x0, t, x_t, target_v = make_flow_training_tensors(
        x1=x1,
        sample_ids=sample_ids,
        args=args,
        device=device,
        dtype=dtype,
    )

    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = model.action_space.get_action_space_dims()
    forward_kwargs: dict[str, Any] = {}
    if bool(getattr(model.config, "expert_non_causal_attention", False)):
        forward_kwargs["is_causal"] = False

    future_token_embeds = bundle.action_in_proj(x_t, t)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(x_t.shape[0], n_diffusion_tokens, -1)
    expert_out = bundle.expert(
        inputs_embeds=future_token_embeds,
        position_ids=context["position_ids"],
        past_key_values=prompt_cache,
        attention_mask=(
            None
            if context.get("attention_mask") is None
            else context["attention_mask"].to(dtype=future_token_embeds.dtype)
        ),
        use_cache=True,
        **forward_kwargs,
    )
    prompt_cache.crop(prefill_seq_len)
    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    pred_v = bundle.action_out_proj(last_hidden).view(-1, *action_dims)
    loss = F.mse_loss(pred_v.float(), target_v.float())
    return loss, {
        "target_action_abs_mean": float(x1.detach().abs().mean().cpu()),
        "pred_v_abs_mean": float(pred_v.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target_v.detach().abs().mean().cpu()),
        "train_t_mean": float(t.detach().float().mean().cpu()),
        "train_x0_abs_mean": float(x0.detach().abs().mean().cpu()),
    }


def _expert_velocity_forward(
    *,
    expert: nn.Module,
    action_in_proj: nn.Module,
    action_out_proj: nn.Module,
    prompt_cache: Any,
    context: dict[str, Any],
    x_t: torch.Tensor,
    t: torch.Tensor,
    model: Any,
) -> torch.Tensor:
    dtype = next(expert.parameters()).dtype
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = model.action_space.get_action_space_dims()
    forward_kwargs: dict[str, Any] = {}
    if bool(getattr(model.config, "expert_non_causal_attention", False)):
        forward_kwargs["is_causal"] = False

    future_token_embeds = action_in_proj(x_t.to(dtype=dtype), t.to(dtype=dtype))
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(x_t.shape[0], n_diffusion_tokens, -1)
    expert_out = expert(
        inputs_embeds=future_token_embeds,
        position_ids=context["position_ids"],
        past_key_values=prompt_cache,
        attention_mask=(
            None
            if context.get("attention_mask") is None
            else context["attention_mask"].to(dtype=future_token_embeds.dtype)
        ),
        use_cache=True,
        **forward_kwargs,
    )
    prompt_cache.crop(prefill_seq_len)
    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    return action_out_proj(last_hidden).view(-1, *action_dims)


def train_teacher_velocity_forward(
    *,
    bundle: AE28Bundle,
    model: Any,
    prompt_cache: Any,
    context: dict[str, Any],
    target_action: torch.Tensor,
    selected_old_indices: list[int],
    num_time_samples: int,
    teacher_velocity_weight: float,
    endpoint_aux_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = next(bundle.parameters()).dtype
    repeats = max(int(num_time_samples), 1)
    if repeats > 1:
        prompt_cache.batch_repeat_interleave(repeats)
        context = repeat_context(context, repeats)
        target_action = target_action.repeat_interleave(repeats, dim=0)

    x1 = target_action.to(device=device, dtype=dtype)
    x0 = torch.randn_like(x1)
    t = torch.rand((x1.shape[0], 1, 1), device=device, dtype=dtype)
    x_t = (1.0 - t) * x0 + t * x1
    endpoint_target_v = x1 - x0

    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=dtype,
        enabled=device.type == "cuda" and torch.cuda.is_available(),
    ):
        teacher_v = _expert_velocity_forward(
            expert=model.expert,
            action_in_proj=model.action_in_proj,
            action_out_proj=model.action_out_proj,
            prompt_cache=prompt_cache,
            context=context,
            x_t=x_t,
            t=t,
            model=model,
        ).detach()

    if bundle.kv_mixer is not None:
        prompt_cache = bundle.kv_mixer(prompt_cache, dtype=dtype)
    else:
        prompt_cache = select_cache_layers_inplace(prompt_cache, selected_old_indices)
        prompt_cache = cast_cache_layers_inplace(prompt_cache, dtype)
    student_v = _expert_velocity_forward(
        expert=bundle.expert,
        action_in_proj=bundle.action_in_proj,
        action_out_proj=bundle.action_out_proj,
        prompt_cache=prompt_cache,
        context=context,
        x_t=x_t,
        t=t,
        model=model,
    )
    teacher_velocity_loss = F.mse_loss(student_v.float(), teacher_v.float())
    endpoint_loss = F.mse_loss(student_v.float(), endpoint_target_v.float())
    loss = (float(teacher_velocity_weight) * teacher_velocity_loss) + (float(endpoint_aux_weight) * endpoint_loss)
    return loss, {
        "teacher_velocity_loss": float(teacher_velocity_loss.detach().cpu()),
        "endpoint_aux_loss": float(endpoint_loss.detach().cpu()),
        "teacher_v_abs_mean": float(teacher_v.detach().abs().mean().cpu()),
        "student_v_abs_mean": float(student_v.detach().abs().mean().cpu()),
        "target_action_abs_mean": float(x1.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(endpoint_target_v.detach().abs().mean().cpu()),
        **(bundle.kv_mixer.stats() if bundle.kv_mixer is not None else {}),
    }


def sample_bundle_paths_batch(
    *,
    bundle: AE28Bundle,
    model: Any,
    prompt_cache: Any,
    context: dict[str, Any],
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    dtype = next(bundle.parameters()).dtype
    batch_size = int(ego_history_xyz.shape[0])
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = model.action_space.get_action_space_dims()
    forward_kwargs: dict[str, Any] = {}
    if bool(getattr(model.config, "expert_non_causal_attention", False)):
        forward_kwargs["is_causal"] = False

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        future_token_embeds = bundle.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_out = bundle.expert(
            inputs_embeds=future_token_embeds,
            position_ids=context["position_ids"],
            past_key_values=prompt_cache,
            attention_mask=(
                None
                if context.get("attention_mask") is None
                else context["attention_mask"].to(dtype=future_token_embeds.dtype)
            ),
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        return bundle.action_out_proj(last_hidden).view(-1, *action_dims)

    sync_cuda()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=dtype,
        enabled=device.type == "cuda" and torch.cuda.is_available(),
    ):
        sampled_action = model.diffusion.sample(
            batch_size=batch_size,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
        )
        hist_xyz = ego_history_xyz[:, -1].to(device)
        hist_rot = ego_history_rot[:, -1].to(device)
        pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)
    sync_cuda()
    return {
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "sampled_action": sampled_action.detach().float().cpu().numpy().astype(np.float32),
        "pred_xyz": pred_xyz.detach().float().cpu().numpy().astype(np.float32),
        "pred_rot": pred_rot.detach().float().cpu().numpy().astype(np.float32),
    }


def sample_modules_paths_batch(
    *,
    expert: nn.Module,
    action_in_proj: nn.Module,
    action_out_proj: nn.Module,
    model: Any,
    prompt_cache: Any,
    context: dict[str, Any],
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    dtype = next(expert.parameters()).dtype
    batch_size = int(ego_history_xyz.shape[0])
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = model.action_space.get_action_space_dims()
    forward_kwargs: dict[str, Any] = {}
    if bool(getattr(model.config, "expert_non_causal_attention", False)):
        forward_kwargs["is_causal"] = False

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        future_token_embeds = action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_out = expert(
            inputs_embeds=future_token_embeds,
            position_ids=context["position_ids"],
            past_key_values=prompt_cache,
            attention_mask=(
                None
                if context.get("attention_mask") is None
                else context["attention_mask"].to(dtype=future_token_embeds.dtype)
            ),
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        return action_out_proj(last_hidden).view(-1, *action_dims)

    sync_cuda()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=dtype,
        enabled=device.type == "cuda" and torch.cuda.is_available(),
    ):
        sampled_action = model.diffusion.sample(
            batch_size=batch_size,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
        )
        hist_xyz = ego_history_xyz[:, -1].to(device)
        hist_rot = ego_history_rot[:, -1].to(device)
        pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)
    sync_cuda()
    return {
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "sampled_action": sampled_action.detach().float().cpu().numpy().astype(np.float32),
        "pred_xyz": pred_xyz.detach().float().cpu().numpy().astype(np.float32),
        "pred_rot": pred_rot.detach().float().cpu().numpy().astype(np.float32),
    }


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def evaluate(
    *,
    bundle: AE28Bundle,
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    selected_old_indices: list[int],
    args: argparse.Namespace,
    step: int,
) -> dict[str, Any]:
    bundle.eval()
    rows: list[dict[str, Any]] = []
    velocity_mse_rows: list[float] = []
    device = torch.device(args.device)
    eval_items = items[: int(args.eval_samples)]
    for batch_index, batch_items in enumerate(iter_batches(eval_items, int(args.eval_batch_size))):
        batch = build_batch(
            model=model,
            processor=processor,
            batch_items=batch_items,
            selected_old_indices=selected_old_indices,
            args=args,
        )
        if args.stage1_mode in {"teacher_velocity", "teacher_velocity_kv_mixer"}:
            teacher_pred = sample_modules_paths_batch(
                expert=model.expert,
                action_in_proj=model.action_in_proj,
                action_out_proj=model.action_out_proj,
                model=model,
                prompt_cache=batch["cache"],
                context=batch["context"],
                ego_history_xyz=batch["ego_history_xyz"],
                ego_history_rot=batch["ego_history_rot"],
                seed=int(args.seed) + 1000 + batch_index,
                device=device,
            )
            if bundle.kv_mixer is not None:
                with torch.no_grad():
                    batch["cache"] = bundle.kv_mixer(batch["cache"], dtype=next(bundle.parameters()).dtype)
            else:
                batch["cache"] = select_cache_layers_inplace(batch["cache"], selected_old_indices)
                batch["cache"] = cast_cache_layers_inplace(batch["cache"], next(bundle.parameters()).dtype)
            pred = sample_modules_paths_batch(
                expert=bundle.expert,
                action_in_proj=bundle.action_in_proj,
                action_out_proj=bundle.action_out_proj,
                model=model,
                prompt_cache=batch["cache"],
                context=batch["context"],
                ego_history_xyz=batch["ego_history_xyz"],
                ego_history_rot=batch["ego_history_rot"],
                seed=int(args.seed) + 1000 + batch_index,
                device=device,
            )
            target_xyz = teacher_pred["pred_xyz"]
        else:
            pred = sample_bundle_paths_batch(
                bundle=bundle,
                model=model,
                prompt_cache=batch["cache"],
                context=batch["context"],
                ego_history_xyz=batch["ego_history_xyz"],
                ego_history_rot=batch["ego_history_rot"],
                seed=int(args.seed) + 1000 + batch_index,
                device=device,
            )
            target_xyz = batch["target_xyz"].detach().cpu().numpy()
        for row_index, sample_id in enumerate(batch["sample_ids"]):
            ade, fde = ade_fde(pred["pred_xyz"][row_index], target_xyz[row_index])
            rows.append(
                {
                    "sample_id": sample_id,
                    "ade_m": ade,
                    "fde_m": fde,
                    "pred_path_length_m": path_len(pred["pred_xyz"][row_index]),
                    "target_path_length_m": path_len(target_xyz[row_index]),
                }
            )
        if bool(args.eval_velocity_grid) and args.stage1_mode == "endpoint":
            dtype = next(bundle.parameters()).dtype
            x1 = batch["target_action"].to(device=device, dtype=dtype)
            _x0, t, x_t, target_v = make_flow_training_tensors(
                x1=x1,
                sample_ids=batch["sample_ids"],
                args=args,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad(), torch.autocast(
                "cuda",
                dtype=dtype,
                enabled=device.type == "cuda" and torch.cuda.is_available(),
            ):
                pred_v = _expert_velocity_forward(
                    expert=bundle.expert,
                    action_in_proj=bundle.action_in_proj,
                    action_out_proj=bundle.action_out_proj,
                    prompt_cache=batch["cache"],
                    context=batch["context"],
                    x_t=x_t,
                    t=t,
                    model=model,
                )
            per_sample_mse = (pred_v.float() - target_v.float()).pow(2).flatten(1).mean(dim=1)
            velocity_mse_rows.extend([float(value.detach().cpu()) for value in per_sample_mse])
        del batch
        gc.collect()
    ades = [row["ade_m"] for row in rows]
    fdes = [row["fde_m"] for row in rows]
    out = {
        "step": int(step),
        "eval_count": len(rows),
        "ade_mean_m": float(np.mean(ades)) if ades else None,
        "ade_p50_m": float(np.percentile(ades, 50)) if ades else None,
        "ade_p95_m": float(np.percentile(ades, 95)) if ades else None,
        "fde_mean_m": float(np.mean(fdes)) if fdes else None,
        "fde_p50_m": float(np.percentile(fdes, 50)) if fdes else None,
        "fde_p95_m": float(np.percentile(fdes, 95)) if fdes else None,
        "rows": rows,
    }
    if velocity_mse_rows:
        out.update(
            {
                "velocity_grid_mse_mean": float(np.mean(velocity_mse_rows)),
                "velocity_grid_mse_p50": float(np.percentile(velocity_mse_rows, 50)),
                "velocity_grid_mse_p95": float(np.percentile(velocity_mse_rows, 95)),
            }
        )
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
    vram_cap_config = configure_vram_cap(args.device, args.vram_cap_gb)

    summary: dict[str, Any] = {
        "created_at_unix": time.time(),
        "args": vars(args) | {
            "corpus_jsonl": str(args.corpus_jsonl),
            "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
            "output_dir": str(args.output_dir),
        },
        "vram_cap_config": vram_cap_config,
        "status": "running",
    }
    try:
        print(
            json.dumps(
                {
                    "event": "select_items_start",
                    "corpus_jsonl": str(args.corpus_jsonl),
                    "split": args.split,
                    "num_samples": int(args.num_samples),
                }
            ),
            flush=True,
        )
        items = select_items(args)
        summary["selected_count"] = len(items)
        summary["selected_sample_ids_head"] = [item["sample_id"] for item in items[:32]]
        print(json.dumps({"event": "load_teacher_start", "checkpoint": str(args.teacher_checkpoint_path)}), flush=True)
        model, processor, _config, _config_path, _runtime = load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=torch_dtype_from_name(args.dtype),
            device=args.device,
            config_json=None,
            runtime_support=None,
            attn_implementation=args.attn_implementation,
            min_pixels=163840,
            max_pixels=196608,
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        force_attention(model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")
        selected_old_indices = layer_mapping(
            int(model.expert.config.num_hidden_layers),
            int(args.compressed_layers),
            args.mapping,
        )
        summary["layer_mapping"] = selected_old_indices

        print(json.dumps({"event": "build_bundle_start", "mapping": selected_old_indices}), flush=True)
        bundle = make_bundle(
            model=model,
            selected_old_indices=selected_old_indices,
            dtype=torch_dtype_from_name(args.dtype),
            device=args.device,
            attn_implementation="sdpa" if args.attn_implementation != "eager" else "eager",
            use_kv_mixer=args.stage1_mode == "teacher_velocity_kv_mixer",
            kv_mixer_init_alpha=float(args.kv_mixer_init_alpha),
            ae_init_mode=str(args.ae_init_mode),
        )
        trainable_params = sum(p.numel() for p in bundle.parameters() if p.requires_grad)
        summary["trainable_params"] = int(trainable_params)
        summary["ae_init_mode"] = str(args.ae_init_mode)
        optimizer_groups = [
            {"params": bundle.expert.parameters(), "lr": float(args.expert_lr)},
            {"params": bundle.action_in_proj.parameters(), "lr": float(args.proj_lr)},
            {"params": bundle.action_out_proj.parameters(), "lr": float(args.proj_lr)},
        ]
        if bundle.kv_mixer is not None:
            optimizer_groups.append(
                {
                    "params": bundle.kv_mixer.parameters(),
                    "lr": float(args.kv_mixer_lr),
                    "weight_decay": 0.0,
                }
            )
            summary["kv_mixer"] = {
                "old_layers": int(bundle.kv_mixer.old_layers),
                "new_layers": int(bundle.kv_mixer.new_layers),
                "init_alpha": float(args.kv_mixer_init_alpha),
                "lr": float(args.kv_mixer_lr),
            }
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=float(args.weight_decay),
        )
        print(
            json.dumps(
                {
                    "event": "train_ready",
                    "selected_count": len(items),
                    "batch_size": int(args.batch_size),
                    "num_time_samples": int(args.num_time_samples),
                    "trainable_params": int(trainable_params),
                    "vram_cap_config": vram_cap_config,
                    "cuda_mem": cuda_mem(),
                }
            ),
            flush=True,
        )
        check_vram_cap(args, where="train_ready")

        eval_history: list[dict[str, Any]] = []
        train_history_tail: list[dict[str, Any]] = []
        if not args.skip_initial_eval and int(args.eval_samples) > 0:
            print(json.dumps({"event": "eval_start", "step": 0}), flush=True)
            eval0 = evaluate(
                bundle=bundle,
                model=model,
                processor=processor,
                items=items,
                selected_old_indices=selected_old_indices,
                args=args,
                step=0,
            )
            eval_history.append(eval0)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "eval", **eval0}, ensure_ascii=True) + "\n")
            print(json.dumps({"event": "eval_done", **{k: v for k, v in eval0.items() if k != "rows"}}), flush=True)
            check_vram_cap(args, where="eval_step_0")

        device = torch.device(args.device)
        started = time.perf_counter()
        for step in range(1, int(args.steps) + 1):
            batch_items = random.sample(items, k=min(int(args.batch_size), len(items)))
            optimizer.zero_grad(set_to_none=True)
            batch = build_batch(
                model=model,
                processor=processor,
                batch_items=batch_items,
                selected_old_indices=selected_old_indices,
                args=args,
            )
            if args.stage1_mode in {"teacher_velocity", "teacher_velocity_kv_mixer"}:
                loss, loss_stats = train_teacher_velocity_forward(
                    bundle=bundle,
                    model=model,
                    prompt_cache=batch["cache"],
                    context=batch["context"],
                    target_action=batch["target_action"],
                    selected_old_indices=selected_old_indices,
                    num_time_samples=int(args.num_time_samples),
                    teacher_velocity_weight=float(args.teacher_velocity_weight),
                    endpoint_aux_weight=float(args.endpoint_aux_weight),
                    device=device,
                )
            else:
                loss, loss_stats = train_velocity_forward(
                    bundle=bundle,
                    model=model,
                    prompt_cache=batch["cache"],
                    context=batch["context"],
                    target_action=batch["target_action"],
                    sample_ids=batch["sample_ids"],
                    args=args,
                    num_time_samples=int(args.num_time_samples),
                    device=device,
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(bundle.parameters(), float(args.grad_clip_norm))
            optimizer.step()

            train_row = {
                "step": int(step),
                "batch_size": len(batch_items),
                "num_time_samples": int(args.num_time_samples),
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "elapsed_total_sec": round(time.perf_counter() - started, 3),
                "samples_per_sec": round((step * len(batch_items)) / max(time.perf_counter() - started, 1e-6), 3),
                **loss_stats,
                **batch["meta"],
                "cuda_mem": cuda_mem(),
            }
            train_history_tail.append(train_row)
            train_history_tail = train_history_tail[-20:]
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "train_step", **train_row}, ensure_ascii=True) + "\n")
            if step == 1 or step % int(args.log_every) == 0:
                print(json.dumps({"event": "train_step", **train_row}), flush=True)
            check_vram_cap(args, where=f"train_step_{step}")

            del batch
            del loss
            gc.collect()
            if torch.cuda.is_available() and step % 25 == 0:
                torch.cuda.empty_cache()

            if int(args.eval_every) > 0 and step % int(args.eval_every) == 0:
                print(json.dumps({"event": "eval_start", "step": step}), flush=True)
                eval_row = evaluate(
                    bundle=bundle,
                    model=model,
                    processor=processor,
                    items=items,
                    selected_old_indices=selected_old_indices,
                    args=args,
                    step=step,
                )
                eval_history.append(eval_row)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "eval", **eval_row}, ensure_ascii=True) + "\n")
                print(json.dumps({"event": "eval_done", **{k: v for k, v in eval_row.items() if k != "rows"}}), flush=True)
                check_vram_cap(args, where=f"eval_step_{step}")

            if int(args.save_every) > 0 and step % int(args.save_every) == 0:
                save_checkpoint(
                    args.output_dir / f"ae28_stage1_step{step:06d}.pt",
                    bundle=bundle,
                    payload={"step": step, "layer_mapping": selected_old_indices, "eval_history": eval_history},
                )

        summary["status"] = "ok"
        summary["train_history_tail"] = train_history_tail
        summary["eval_history"] = eval_history
        if args.save_checkpoint:
            checkpoint_path = args.output_dir / "ae28_stage1_final.pt"
            save_checkpoint(
                checkpoint_path,
                bundle=bundle,
                payload={
                    "layer_mapping": selected_old_indices,
                    "args": summary["args"],
                    "eval_history": eval_history,
                    "train_history_tail": train_history_tail,
                },
            )
            summary["checkpoint_path"] = str(checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "failed"
        summary["error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }
        print(json.dumps({"event": "failed", "error": str(exc)}), flush=True)
    finally:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        print(json.dumps({"event": "done", "status": summary.get("status"), "summary_path": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
