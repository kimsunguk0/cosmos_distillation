#!/usr/bin/env python3
"""Train a small student-KV -> AE36 KV adapter.

This probes the path:

  frozen student VLM 2B KV28
    -> small KV adapter (28 layers to 36 layers)
    -> frozen original Alpamayo AE36/action projections
    -> teacher action trajectory

Only the adapter is trained. The student backbone and original action expert are
kept frozen, so this is a direct test of whether a light KV translation layer can
make the official action expert read the 2B backbone.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
SCRIPT84_PATH = PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SUKIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SUKIM_ROOT))

spec = importlib.util.spec_from_file_location("ae84", SCRIPT84_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT84_PATH}")
ae84 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae84)


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_TEACHER = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B"
DEFAULT_STUDENT_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "no_nav_camera_labeled_official_200k"
    / "no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509"
    / "best_decode"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "action_expert" / "student_kv_adapter_ae36"


class StudentKVToAE36Adapter(nn.Module):
    """Layer-mix + per-channel affine adapter from student KV layers to AE36 KV layers."""

    def __init__(
        self,
        *,
        old_layers: int,
        new_layers: int,
        kv_heads: int = 8,
        head_dim: int = 128,
        init_alpha: float = 0.02,
        use_affine: bool = True,
        use_head_proj: bool = False,
    ) -> None:
        super().__init__()
        self.old_layers = int(old_layers)
        self.new_layers = int(new_layers)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.use_affine = bool(use_affine)
        self.use_head_proj = bool(use_head_proj)
        base = torch.zeros((self.new_layers, self.old_layers), dtype=torch.float32)
        if self.new_layers == 1:
            base[0, 0] = 1.0
        else:
            positions = torch.linspace(0, self.old_layers - 1, self.new_layers)
            for new_idx, position in enumerate(positions.tolist()):
                lo = int(np.floor(position))
                hi = int(np.ceil(position))
                if lo == hi:
                    base[new_idx, lo] = 1.0
                else:
                    hi_weight = float(position - lo)
                    base[new_idx, lo] = 1.0 - hi_weight
                    base[new_idx, hi] = hi_weight
        self.register_buffer("base_weights", base, persistent=True)
        init_alpha = min(max(float(init_alpha), 1e-4), 0.99)
        init_logit = float(np.log(init_alpha / (1.0 - init_alpha)))
        self.key_logits = nn.Parameter(torch.zeros((self.new_layers, self.old_layers), dtype=torch.float32))
        self.value_logits = nn.Parameter(torch.zeros((self.new_layers, self.old_layers), dtype=torch.float32))
        self.key_gate_logits = nn.Parameter(torch.full((self.new_layers, 1), init_logit, dtype=torch.float32))
        self.value_gate_logits = nn.Parameter(torch.full((self.new_layers, 1), init_logit, dtype=torch.float32))
        if self.use_affine:
            self.key_log_scale = nn.Parameter(torch.zeros((self.new_layers, self.kv_heads, self.head_dim)))
            self.value_log_scale = nn.Parameter(torch.zeros((self.new_layers, self.kv_heads, self.head_dim)))
            self.key_bias = nn.Parameter(torch.zeros((self.new_layers, self.kv_heads, self.head_dim)))
            self.value_bias = nn.Parameter(torch.zeros((self.new_layers, self.kv_heads, self.head_dim)))
        if self.use_head_proj:
            eye = torch.eye(self.head_dim, dtype=torch.float32)
            key_proj = eye[None, None, :, :].repeat(self.new_layers, self.kv_heads, 1, 1)
            value_proj = eye[None, None, :, :].repeat(self.new_layers, self.kv_heads, 1, 1)
            self.key_head_proj = nn.Parameter(key_proj)
            self.value_head_proj = nn.Parameter(value_proj)
            self.key_head_bias = nn.Parameter(torch.zeros((self.new_layers, self.kv_heads, self.head_dim)))
            self.value_head_bias = nn.Parameter(torch.zeros((self.new_layers, self.kv_heads, self.head_dim)))

    def _weights(self, logits: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        learned = torch.softmax(logits, dim=-1)
        gate = torch.sigmoid(gate_logits)
        return ((1.0 - gate) * self.base_weights.to(device=logits.device)) + (gate * learned)

    def stats(self) -> dict[str, float]:
        with torch.no_grad():
            kw = self._weights(self.key_logits, self.key_gate_logits)
            vw = self._weights(self.value_logits, self.value_gate_logits)
            key_gate = torch.sigmoid(self.key_gate_logits)
            value_gate = torch.sigmoid(self.value_gate_logits)
            return {
                "adapter_key_gate_mean": float(key_gate.mean().detach().cpu()),
                "adapter_value_gate_mean": float(value_gate.mean().detach().cpu()),
                "adapter_key_max_weight_mean": float(kw.max(dim=-1).values.mean().detach().cpu()),
                "adapter_value_max_weight_mean": float(vw.max(dim=-1).values.mean().detach().cpu()),
                "adapter_key_proj_delta_mean": float(
                    (
                        self.key_head_proj
                        - torch.eye(self.head_dim, device=self.key_head_proj.device)[None, None, :, :]
                    )
                    .detach()
                    .abs()
                    .mean()
                    .cpu()
                )
                if self.use_head_proj
                else 0.0,
                "adapter_value_proj_delta_mean": float(
                    (
                        self.value_head_proj
                        - torch.eye(self.head_dim, device=self.value_head_proj.device)[None, None, :, :]
                    )
                    .detach()
                    .abs()
                    .mean()
                    .cpu()
                )
                if self.use_head_proj
                else 0.0,
            }

    def forward(self, cache: Any, *, dtype: torch.dtype) -> Any:
        layers = list(getattr(cache, "layers", []))
        if len(layers) != self.old_layers:
            raise ValueError(f"KV adapter expected {self.old_layers} layers, got {len(layers)}")
        key_weights = self._weights(self.key_logits, self.key_gate_logits).to(device=layers[0].keys.device, dtype=dtype)
        value_weights = self._weights(self.value_logits, self.value_gate_logits).to(
            device=layers[0].values.device, dtype=dtype
        )
        adapted = copy.copy(cache)
        new_layers = []
        for new_idx in range(self.new_layers):
            key_acc = None
            value_acc = None
            for old_idx, layer in enumerate(layers):
                key_term = layer.keys.to(dtype=dtype) * key_weights[new_idx, old_idx]
                value_term = layer.values.to(dtype=dtype) * value_weights[new_idx, old_idx]
                key_acc = key_term if key_acc is None else key_acc + key_term
                value_acc = value_term if value_acc is None else value_acc + value_term
            if self.use_affine:
                key_scale = torch.exp(self.key_log_scale[new_idx]).to(device=key_acc.device, dtype=dtype)
                value_scale = torch.exp(self.value_log_scale[new_idx]).to(device=value_acc.device, dtype=dtype)
                key_bias = self.key_bias[new_idx].to(device=key_acc.device, dtype=dtype)
                value_bias = self.value_bias[new_idx].to(device=value_acc.device, dtype=dtype)
                key_acc = key_acc * key_scale[None, :, None, :] + key_bias[None, :, None, :]
                value_acc = value_acc * value_scale[None, :, None, :] + value_bias[None, :, None, :]
            if self.use_head_proj:
                key_proj = self.key_head_proj[new_idx].to(device=key_acc.device, dtype=dtype)
                value_proj = self.value_head_proj[new_idx].to(device=value_acc.device, dtype=dtype)
                key_bias = self.key_head_bias[new_idx].to(device=key_acc.device, dtype=dtype)
                value_bias = self.value_head_bias[new_idx].to(device=value_acc.device, dtype=dtype)
                key_acc = torch.einsum("bhtd,hdm->bhtm", key_acc, key_proj) + key_bias[None, :, None, :]
                value_acc = torch.einsum("bhtd,hdm->bhtm", value_acc, value_proj) + value_bias[None, :, None, :]
            base_idx = int(torch.argmax(self.base_weights[new_idx]).item())
            new_layer = copy.copy(layers[base_idx])
            new_layer.keys = key_acc
            new_layer.values = value_acc
            new_layers.append(new_layer)
        adapted.layers = new_layers
        return adapted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-corpus-jsonl", type=Path, default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--student-model", default=ae84.resolve_student_model_path())
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--prefix-mode", choices=("student_free", "teacher_forced"), default="student_free")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--train-timestep-sampler", choices=("uniform", "beta"), default="beta")
    parser.add_argument("--num-time-samples", type=int, default=1)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--ae-lr", type=float, default=1e-5)
    parser.add_argument("--action-proj-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--adapter-init-alpha", type=float, default=0.02)
    parser.add_argument("--no-affine", action="store_true")
    parser.add_argument("--head-proj", action="store_true", help="Add identity-initialized per-head KV projections.")
    parser.add_argument("--eval-seed-mode", choices=("fixed", "step"), default="fixed")
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--load-action-modules-from-checkpoint", action="store_true")
    parser.add_argument("--train-ae", action="store_true", help="Unfreeze the original AE36 decoder after adapter init.")
    parser.add_argument("--train-action-proj", action="store_true", help="Unfreeze action_in_proj/action_out_proj.")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--skip-initial-eval", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def _freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def select_eval_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.eval_split is None and args.eval_corpus_jsonl is None:
        return []
    eval_args = argparse.Namespace(**vars(args))
    eval_args.corpus_jsonl = args.eval_corpus_jsonl or args.corpus_jsonl
    eval_args.split = args.eval_split if args.eval_split is not None else args.split
    eval_args.num_samples = args.eval_samples
    return ae84.select_items(eval_args)


def _move_action_modules(model: Any, *, device: torch.device, dtype: torch.dtype, attn_implementation: str) -> None:
    model.expert.to(device=device, dtype=dtype).eval()
    model.action_in_proj.to(device=device, dtype=dtype).eval()
    model.action_out_proj.to(device=device, dtype=dtype).eval()
    model.action_space.to(device=device)
    if isinstance(model.diffusion, nn.Module):
        model.diffusion.to(device=device)
    _freeze(model.expert)
    _freeze(model.action_in_proj)
    _freeze(model.action_out_proj)
    ae84.force_attention(model.expert, "sdpa" if attn_implementation != "eager" else "eager")


def repeat_context(context: dict[str, Any], repeats: int) -> dict[str, Any]:
    if int(repeats) <= 1:
        return context
    repeated = dict(context)
    repeated["position_ids"] = context["position_ids"].repeat_interleave(int(repeats), dim=1)
    if context.get("attention_mask") is not None:
        repeated["attention_mask"] = context["attention_mask"].repeat_interleave(int(repeats), dim=0)
    return repeated


def train_step(
    *,
    adapter: StudentKVToAE36Adapter,
    model: Any,
    batch: dict[str, Any],
    num_time_samples: int,
    train_timestep_sampler: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = next(model.expert.parameters()).dtype
    repeats = max(int(num_time_samples), 1)
    adapted_cache = adapter(batch["cache"], dtype=dtype)
    context = batch["context"]
    target_action = batch["target_action"]
    if repeats > 1:
        adapted_cache.batch_repeat_interleave(repeats)
        context = repeat_context(context, repeats)
        target_action = target_action.repeat_interleave(repeats, dim=0)

    x1 = target_action.to(device=device, dtype=dtype)
    x0 = torch.randn_like(x1)
    t = ae84.sample_fm_timesteps(
        batch_size=int(x1.shape[0]),
        sampler=str(train_timestep_sampler),
        device=device,
        dtype=dtype,
    )
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0

    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False
    future_token_embeds = model.action_in_proj(x_t, t)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(x_t.shape[0], n_diffusion_tokens, -1)
    expert_attention_mask = context.get("attention_mask")
    if expert_attention_mask is not None:
        expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
    out = model.expert(
        inputs_embeds=future_token_embeds,
        position_ids=context["position_ids"],
        past_key_values=adapted_cache,
        attention_mask=expert_attention_mask,
        use_cache=True,
        **kwargs,
    )
    adapted_cache.crop(prefill_seq_len)
    pred_v = model.action_out_proj(out.last_hidden_state[:, -n_diffusion_tokens:]).view(-1, *action_dims)
    loss = F.mse_loss(pred_v.float(), target_v.float())
    return loss, {
        "target_action_abs_mean": float(x1.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target_v.detach().abs().mean().cpu()),
        "pred_v_abs_mean": float(pred_v.detach().abs().mean().cpu()),
        "train_t_mean": float(t.detach().float().mean().cpu()),
        **adapter.stats(),
    }


def sample_paths(
    *,
    adapter: StudentKVToAE36Adapter,
    model: Any,
    batch: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    dtype = next(model.expert.parameters()).dtype
    adapted_cache = adapter(batch["cache"], dtype=dtype)
    context = batch["context"]
    batch_size = int(batch["ego_history_xyz"].shape[0])
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        future_token_embeds = model.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_attention_mask = context.get("attention_mask")
        if expert_attention_mask is not None:
            expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
        out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=context["position_ids"],
            past_key_values=adapted_cache,
            attention_mask=expert_attention_mask,
            use_cache=True,
            **kwargs,
        )
        adapted_cache.crop(prefill_seq_len)
        return model.action_out_proj(out.last_hidden_state[:, -n_diffusion_tokens:]).view(-1, *action_dims)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        action = model.diffusion.sample(batch_size=batch_size, step_fn=step_fn, device=device)
        pred_xyz, pred_rot = model.action_space.action_to_traj(
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
    adapter: StudentKVToAE36Adapter,
    student: Any,
    student_processor: Any,
    student_tokenizer: Any,
    model: Any,
    items: list[dict[str, Any]],
    step: int,
) -> dict[str, Any]:
    adapter.eval()
    was_expert_training = bool(model.expert.training)
    was_in_proj_training = bool(model.action_in_proj.training)
    was_out_proj_training = bool(model.action_out_proj.training)
    model.expert.eval()
    model.action_in_proj.eval()
    model.action_out_proj.eval()
    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    for batch_index, batch_items in enumerate(iter_batches(items[: int(args.eval_samples)], int(args.eval_batch_size))):
        batch = ae84.build_batch(
            args=args,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=model,
            batch_items=batch_items,
        )
        pred = sample_paths(
            adapter=adapter,
            model=model,
            batch=batch,
            seed=int(args.seed) + 1000 + (0 if str(args.eval_seed_mode) == "fixed" else int(step)) + batch_index,
            device=device,
        )
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        for row_index, sample_id in enumerate(batch["sample_ids"]):
            ade, fde = ae84.ade_fde(pred["pred_xyz"][row_index], target_xyz[row_index])
            rows.append(
                {
                    "sample_id": sample_id,
                    "ade_m": ade,
                    "fde_m": fde,
                    "pred_path_length_m": ae84.path_len(pred["pred_xyz"][row_index]),
                    "target_path_length_m": ae84.path_len(target_xyz[row_index]),
                }
            )
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ades = [row["ade_m"] for row in rows]
    fdes = [row["fde_m"] for row in rows]
    adapter.train()
    model.expert.train(was_expert_training)
    model.action_in_proj.train(was_in_proj_training)
    model.action_out_proj.train(was_out_proj_training)
    return {
        "event": "eval",
        "step": int(step),
        "eval_count": len(rows),
        "ade_mean_m": float(np.mean(ades)) if ades else None,
        "ade_p50_m": float(np.percentile(ades, 50)) if ades else None,
        "fde_mean_m": float(np.mean(fdes)) if fdes else None,
        "fde_p50_m": float(np.percentile(fdes, 50)) if fdes else None,
        "rows": rows,
    }


def save_checkpoint(
    path: Path,
    *,
    adapter: StudentKVToAE36Adapter,
    payload: dict[str, Any],
    model: Any | None = None,
    include_action_modules: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"adapter_state_dict": adapter.state_dict(), "payload": payload}
    if include_action_modules:
        if model is None:
            raise ValueError("model is required when include_action_modules=True")
        state.update(
            {
                "expert_state_dict": model.expert.state_dict(),
                "action_in_proj_state_dict": model.action_in_proj.state_dict(),
                "action_out_proj_state_dict": model.action_out_proj.state_dict(),
            }
        )
    torch.save(state, path)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    ae_dtype = ae84.torch_dtype_from_name(args.ae_dtype)
    log_path = args.output_dir / "train_log.jsonl"
    summary_path = args.output_dir / "summary.json"
    summary: dict[str, Any] = {
        "created_at_unix": time.time(),
        "args": jsonable(vars(args) | {
            "corpus_jsonl": str(args.corpus_jsonl),
            "student_checkpoint_dir": str(args.student_checkpoint_dir),
            "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
            "output_dir": str(args.output_dir),
        }),
        "status": "running",
    }
    try:
        items = ae84.select_items(args)
        eval_items = select_eval_items(args)
        if not eval_items:
            eval_items = items
        summary["selected_count"] = len(items)
        summary["eval_selected_count"] = len(eval_items)
        summary["eval_split"] = str(args.eval_split) if args.eval_split is not None else str(args.split)
        student, student_tokenizer, student_processor, base_model = ae84.load_student(args)
        summary["student_base_model"] = str(base_model)

        print(json.dumps({"event": "load_teacher_action_modules_start", "device": args.teacher_load_device}), flush=True)
        model, _teacher_processor, _cfg, _cfg_path, _runtime = ae84.load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=ae_dtype,
            device=args.teacher_load_device,
            config_json=None,
            runtime_support=None,
            attn_implementation=args.attn_implementation,
            min_pixels=163840,
            max_pixels=196608,
        )
        model.eval()
        if hasattr(model, "vlm"):
            delattr(model, "vlm")
        gc.collect()
        _move_action_modules(model, device=device, dtype=ae_dtype, attn_implementation=args.attn_implementation)
        old_layers = int(getattr(getattr(student.backbone.config, "text_config", None), "num_hidden_layers", 28))
        new_layers = int(model.expert.config.num_hidden_layers)
        adapter = StudentKVToAE36Adapter(
            old_layers=old_layers,
            new_layers=new_layers,
            kv_heads=int(args.kv_heads),
            head_dim=int(args.head_dim),
            init_alpha=float(args.adapter_init_alpha),
            use_affine=not bool(args.no_affine),
            use_head_proj=bool(args.head_proj),
        ).to(device=device)
        adapter.train()
        if args.adapter_checkpoint is not None:
            state = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
            adapter.load_state_dict(state["adapter_state_dict"], strict=True)
            if bool(args.load_action_modules_from_checkpoint):
                if "expert_state_dict" in state:
                    model.expert.load_state_dict(state["expert_state_dict"], strict=True)
                if "action_in_proj_state_dict" in state:
                    model.action_in_proj.load_state_dict(state["action_in_proj_state_dict"], strict=True)
                if "action_out_proj_state_dict" in state:
                    model.action_out_proj.load_state_dict(state["action_out_proj_state_dict"], strict=True)
            summary["adapter_checkpoint"] = str(args.adapter_checkpoint)
            summary["loaded_action_modules_from_checkpoint"] = bool(args.load_action_modules_from_checkpoint)
        if bool(args.train_ae):
            model.expert.train()
            for param in model.expert.parameters():
                param.requires_grad_(True)
        if bool(args.train_action_proj):
            model.action_in_proj.train()
            model.action_out_proj.train()
            for param in model.action_in_proj.parameters():
                param.requires_grad_(True)
            for param in model.action_out_proj.parameters():
                param.requires_grad_(True)
        summary["student_kv_layers"] = old_layers
        summary["ae_kv_layers"] = new_layers
        summary["adapter_trainable_params"] = int(sum(p.numel() for p in adapter.parameters() if p.requires_grad))
        summary["expert_trainable_params"] = int(sum(p.numel() for p in model.expert.parameters() if p.requires_grad))
        summary["action_proj_trainable_params"] = int(
            sum(p.numel() for p in model.action_in_proj.parameters() if p.requires_grad)
            + sum(p.numel() for p in model.action_out_proj.parameters() if p.requires_grad)
        )
        param_groups: list[dict[str, Any]] = [
            {"params": [p for p in adapter.parameters() if p.requires_grad], "lr": float(args.adapter_lr)}
        ]
        expert_params = [p for p in model.expert.parameters() if p.requires_grad]
        if expert_params:
            param_groups.append({"params": expert_params, "lr": float(args.ae_lr)})
        action_proj_params = [
            p
            for module in (model.action_in_proj, model.action_out_proj)
            for p in module.parameters()
            if p.requires_grad
        ]
        if action_proj_params:
            param_groups.append({"params": action_proj_params, "lr": float(args.action_proj_lr)})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))
        trainable_for_clip = [param for group in param_groups for param in group["params"]]
        include_action_modules_in_ckpt = bool(args.train_ae or args.train_action_proj)
        log_handle = log_path.open("a", encoding="utf-8")
        best_eval: dict[str, Any] | None = None

        if not args.skip_initial_eval:
            ev = evaluate(
                args=args,
                adapter=adapter,
                student=student,
                student_processor=student_processor,
                student_tokenizer=student_tokenizer,
                model=model,
                items=eval_items,
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
            batch = ae84.build_batch(
                args=args,
                student=student,
                student_processor=student_processor,
                student_tokenizer=student_tokenizer,
                teacher_model=model,
                batch_items=batch_items,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, stats = train_step(
                adapter=adapter,
                model=model,
                batch=batch,
                num_time_samples=int(args.num_time_samples),
                train_timestep_sampler=str(args.train_timestep_sampler),
                device=device,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_for_clip, float(args.grad_clip_norm))
            optimizer.step()
            if step == 1 or step % int(args.log_every) == 0:
                row = {
                    "event": "train_step",
                    "step": int(step),
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

            if step % int(args.eval_every) == 0 or step == int(args.steps):
                ev = evaluate(
                    args=args,
                    adapter=adapter,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    model=model,
                    items=eval_items,
                    step=step,
                )
                print(json.dumps(ev), flush=True)
                log_handle.write(json.dumps(ev) + "\n")
                log_handle.flush()
                if best_eval is None or float(ev.get("ade_mean_m") or 1e9) < float(best_eval.get("ade_mean_m") or 1e9):
                    best_eval = ev
                    if not bool(args.no_checkpoints):
                        save_checkpoint(
                            args.output_dir / "best.pt",
                            adapter=adapter,
                            model=model,
                            include_action_modules=include_action_modules_in_ckpt,
                        payload={"step": step, "eval": ev, "args": jsonable(vars(args))},
                    )
            if (not bool(args.no_checkpoints)) and args.save_every and step % int(args.save_every) == 0:
                save_checkpoint(
                    args.output_dir / f"step_{step:06d}.pt",
                    adapter=adapter,
                    model=model,
                    include_action_modules=include_action_modules_in_ckpt,
                    payload={"step": step, "args": jsonable(vars(args))},
                )

        if not bool(args.no_checkpoints):
            save_checkpoint(
                args.output_dir / "final.pt",
                adapter=adapter,
                model=model,
                include_action_modules=include_action_modules_in_ckpt,
                payload={"step": int(args.steps), "args": jsonable(vars(args))},
            )
        summary.update({"status": "ok", "elapsed_sec": round(time.perf_counter() - started, 3), "best_eval": best_eval})
        log_handle.close()
    except Exception as exc:  # noqa: BLE001
        summary.update({"status": "failed", "error": repr(exc)})
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path), "status": summary["status"]}), flush=True)


if __name__ == "__main__":
    main()
