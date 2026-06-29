#!/usr/bin/env python3
"""F1/F2 probes for action_in_proj and timestep sensitivity.

F1: compare teacher original vs reset action_in_proj on identical (x_t, t).
F2: load D1 best.pt and measure whether action_in_proj/expert outputs react to
    t-only and x_t-only changes.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


script_84 = import_script(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py", "script_84")
script_94 = import_script(PROJECT_ROOT / "scripts" / "94_diagnose_fm_collapse.py", "script_94")


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_STUDENT_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "no_nav_camera_labeled_official_full444k"
    / "no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838"
    / "step_006250"
)
DEFAULT_D1_RUN = PROJECT_ROOT / "outputs" / "action_expert" / "student_ae28" / "d1_oracle_kv_32_s1000_seed42"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "action_expert" / "fm_collapse_diagnostics" / "f1_f2_action_in_proj_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-source", choices=("teacher", "gt"), default="teacher")
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--student-model", default=script_84.resolve_student_model_path())
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B")
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--d1-run-dir", type=Path, default=DEFAULT_D1_RUN)
    parser.add_argument("--checkpoint-name", default="best")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--oracle-seed", type=int, default=7777)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--t-values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--noise-scales", default="0.0,0.5,1.0,1.5")
    return parser.parse_args()


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float()
    return {
        "mean": float(xf.mean().cpu()),
        "std": float(xf.std(unbiased=False).cpu()),
        "abs_mean": float(xf.abs().mean().cpu()),
        "rms": float(torch.sqrt((xf * xf).mean()).cpu()),
        "min": float(xf.min().cpu()),
        "max": float(xf.max().cpu()),
    }


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    return float((torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf)).clamp_min(1e-12)).cpu())


def delta_stats(current: torch.Tensor, base: torch.Tensor) -> dict[str, float]:
    diff = current.detach().float() - base.detach().float()
    base_f = base.detach().float()
    return {
        "delta_abs_mean": float(diff.abs().mean().cpu()),
        "delta_rms": float(torch.sqrt((diff * diff).mean()).cpu()),
        "relative_delta_rms": float(
            (torch.sqrt((diff * diff).mean()) / torch.sqrt((base_f * base_f).mean()).clamp_min(1e-12)).cpu()
        ),
        "cosine_vs_base": cosine_flat(current, base),
    }


def d1_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        corpus_jsonl=args.corpus_jsonl,
        split=args.split,
        num_samples=int(args.num_samples),
        student_checkpoint_dir=args.student_checkpoint_dir,
        student_model=args.student_model,
        student_dtype=args.student_dtype,
        ae_dtype=args.ae_dtype,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        teacher_load_device=args.teacher_load_device,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        prefix_mode="student_free",
        target_source=args.target_source,
        train_backbone_lora=False,
        stage2_attention_mode="official_none",
        compressed_layers=28,
        mapping="linspace_round",
        ae_init_mode="student_backbone_init",
        expert_lr=1e-5,
        proj_lr=1e-4,
        weight_decay=0.01,
        no_norm_bias_decay=True,
    )


def select_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    return script_84.select_items(
        SimpleNamespace(corpus_jsonl=args.corpus_jsonl, split=args.split, num_samples=int(args.num_samples))
    )


def make_oracle_projection(*, seed: int, dtype: torch.dtype, device: torch.device) -> nn.Linear:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    proj = nn.Linear(3, 8 * 128, bias=False)
    with torch.no_grad():
        std = math.sqrt(2.0 / (3 + 8 * 128))
        proj.weight.copy_(torch.randn(proj.weight.shape, generator=generator) * std)
    proj = proj.to(device=device, dtype=dtype).eval()
    for param in proj.parameters():
        param.requires_grad_(False)
    return proj


def inject_oracle_kv(batch: dict[str, Any], proj: nn.Linear) -> dict[str, Any]:
    target = batch["target_xyz"].to(device=proj.weight.device, dtype=proj.weight.dtype)
    batch_size, horizon, _ = target.shape
    oracle = proj(target).view(batch_size, horizon, 8, 128)
    oracle_k = oracle.transpose(1, 2).contiguous()
    cache = batch["cache"]
    for layer_cache in getattr(cache, "layers")[:28]:
        layer_cache.keys = oracle_k.detach().clone()
        layer_cache.values = oracle_k.detach().clone()
    context = dict(batch["context"])
    context["kv_cache_seq_len"] = int(horizon)
    n_diff = int(context["n_diffusion_tokens"])
    device = context["position_ids"].device
    pos_dtype = context["position_ids"].dtype
    context["position_ids"] = (
        torch.arange(horizon, horizon + n_diff, device=device, dtype=pos_dtype)
        .view(1, 1, -1)
        .repeat(3, batch_size, 1)
    )
    context["attention_mask"] = None
    batch["context"] = context
    return batch


def freqs_report(teacher_in: nn.Module, reset_in: nn.Module) -> dict[str, Any]:
    teacher_buffers = {name: buf.detach().float().cpu() for name, buf in teacher_in.named_buffers() if name.endswith("freqs")}
    reset_buffers = {name: buf.detach().float().cpu() for name, buf in reset_in.named_buffers() if name.endswith("freqs")}
    rows = {}
    for name, teacher_buf in teacher_buffers.items():
        reset_buf = reset_buffers[name]
        rows[name] = {
            "shape": list(teacher_buf.shape),
            "teacher_min": float(teacher_buf.min()),
            "teacher_max": float(teacher_buf.max()),
            "reset_min": float(reset_buf.min()),
            "reset_max": float(reset_buf.max()),
            "max_abs_diff": float((teacher_buf - reset_buf).abs().max()),
        }
    return rows


def run_action_in_proj(module: nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    module.eval()
    with torch.no_grad():
        return module(x_t, t)


def run_expert_pred(
    *,
    bundle: Any,
    batch: dict[str, Any],
    x_t: torch.Tensor,
    t: torch.Tensor,
    action_dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = next(bundle.parameters()).dtype
    n_tokens = int(action_dims[0])
    context = batch["context"]
    prompt_cache = batch["cache"]
    prefill_seq_len = int(context["kv_cache_seq_len"])
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=x_t.device.type == "cuda"):
        embeds = bundle.action_in_proj(x_t.to(dtype=dtype), t.to(dtype=dtype))
        out = bundle.expert(
            inputs_embeds=embeds,
            position_ids=context["position_ids"],
            past_key_values=prompt_cache,
            attention_mask=None,
            use_cache=True,
        )
        prompt_cache.crop(prefill_seq_len)
        pred = bundle.action_out_proj(out.last_hidden_state[:, -n_tokens:]).view(-1, *action_dims)
    return embeds.detach(), pred.detach()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = script_84.torch_dtype_from_name(args.ae_dtype)

    items = select_items(args)
    train_args = d1_args(args)
    student, student_tokenizer, student_processor, base_model = script_84.load_student(train_args)
    print(json.dumps({"event": "load_teacher_start", "device": args.teacher_load_device}), flush=True)
    teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = script_84.load_model_and_processor(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=dtype,
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
    script_84.force_attention(teacher_model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")

    teacher_in = copy.deepcopy(teacher_model.action_in_proj).to(device=device, dtype=dtype).eval()
    bundle, selected_layers = script_84.build_bundle(teacher_model, train_args, student=student)
    reset_in = copy.deepcopy(bundle.action_in_proj).to(device=device, dtype=dtype).eval()
    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    oracle_proj = make_oracle_projection(seed=int(args.oracle_seed), dtype=dtype, device=device)
    batches = []
    target_actions = []
    for batch_items in script_84.iter_batches(items, int(args.batch_size)):
        batch = script_84.build_batch(
            args=train_args,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            batch_items=batch_items,
        )
        batch = inject_oracle_kv(batch, oracle_proj)
        batches.append(batch)
        target_actions.append(batch["target_action"].to(device=device, dtype=dtype))
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    x1_all = torch.cat(target_actions, dim=0)
    x0_all = torch.randn_like(x1_all)
    t_all = script_84.sample_fm_timesteps(
        batch_size=int(x1_all.shape[0]),
        sampler="beta",
        device=device,
        dtype=dtype,
    )
    x_t_all = (1.0 - t_all) * x0_all + t_all * x1_all

    teacher_emb = run_action_in_proj(teacher_in, x_t_all, t_all)
    reset_emb = run_action_in_proj(reset_in, x_t_all, t_all)
    f1 = {
        "event": "f1_action_in_proj_compare",
        "input": {
            "x_t": tensor_stats(x_t_all),
            "t": tensor_stats(t_all),
            "x1_target_action": tensor_stats(x1_all),
        },
        "freqs": freqs_report(teacher_in, reset_in),
        "teacher_original": tensor_stats(teacher_emb),
        "student_reset": tensor_stats(reset_emb),
        "teacher_vs_reset": {
            "cosine": cosine_flat(teacher_emb, reset_emb),
            **delta_stats(reset_emb, teacher_emb),
        },
    }
    print(json.dumps(f1), flush=True)

    checkpoint_path = args.d1_run_dir / f"{args.checkpoint_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle.load_state_dict(checkpoint["bundle_state_dict"], strict=True)
    bundle.eval()
    best_emb = run_action_in_proj(bundle.action_in_proj, x_t_all, t_all)
    f1["d1_best_action_in_proj"] = tensor_stats(best_emb)
    f1["teacher_vs_d1_best"] = {"cosine": cosine_flat(teacher_emb, best_emb), **delta_stats(best_emb, teacher_emb)}
    f1["reset_vs_d1_best"] = {"cosine": cosine_flat(reset_emb, best_emb), **delta_stats(best_emb, reset_emb)}

    action_dims = tuple(teacher_model.action_space.get_action_space_dims())
    t_values = [float(v.strip()) for v in str(args.t_values).split(",") if v.strip()]
    noise_scales = [float(v.strip()) for v in str(args.noise_scales).split(",") if v.strip()]

    # Prepare per-batch fixed tensors for sensitivity sweeps.
    batch_inputs = []
    offset = 0
    for batch in batches:
        batch_size = int(batch["target_action"].shape[0])
        x1 = x1_all[offset : offset + batch_size]
        x0 = x0_all[offset : offset + batch_size]
        offset += batch_size
        batch_inputs.append((batch, x1, x0))

    t_sweep_rows = []
    base_embs = []
    base_preds = []
    for t_index, t_value in enumerate(t_values):
        emb_parts = []
        pred_parts = []
        for batch, x1, x0 in batch_inputs:
            ref_t = torch.full((int(x1.shape[0]), 1, 1), 0.5, device=device, dtype=dtype)
            fixed_x_t = (1.0 - ref_t) * x0 + ref_t * x1
            t_tensor = torch.full((int(x1.shape[0]), 1, 1), t_value, device=device, dtype=dtype)
            embeds, pred = run_expert_pred(bundle=bundle, batch=batch, x_t=fixed_x_t, t=t_tensor, action_dims=action_dims)
            emb_parts.append(embeds)
            pred_parts.append(pred)
        emb_all = torch.cat(emb_parts, dim=0)
        pred_all = torch.cat(pred_parts, dim=0)
        if t_index == 0:
            base_embs = [emb_all]
            base_preds = [pred_all]
            deltas = {"delta_abs_mean": 0.0, "delta_rms": 0.0, "relative_delta_rms": 0.0, "cosine_vs_base": 1.0}
            pred_deltas = dict(deltas)
        else:
            deltas = delta_stats(emb_all, base_embs[0])
            pred_deltas = delta_stats(pred_all, base_preds[0])
        t_sweep_rows.append(
            {
                "t": t_value,
                "action_in_proj": tensor_stats(emb_all),
                "action_in_proj_vs_t0": deltas,
                "pred_v": tensor_stats(pred_all),
                "pred_v_vs_t0": pred_deltas,
            }
        )

    x_sweep_rows = []
    base_emb = None
    base_pred = None
    for scale_index, scale in enumerate(noise_scales):
        emb_parts = []
        pred_parts = []
        for batch, x1, x0 in batch_inputs:
            t_tensor = torch.full((int(x1.shape[0]), 1, 1), 0.5, device=device, dtype=dtype)
            x_t = (1.0 - t_tensor) * (float(scale) * x0) + t_tensor * x1
            embeds, pred = run_expert_pred(bundle=bundle, batch=batch, x_t=x_t, t=t_tensor, action_dims=action_dims)
            emb_parts.append(embeds)
            pred_parts.append(pred)
        emb_all = torch.cat(emb_parts, dim=0)
        pred_all = torch.cat(pred_parts, dim=0)
        if scale_index == 0:
            base_emb = emb_all
            base_pred = pred_all
            deltas = {"delta_abs_mean": 0.0, "delta_rms": 0.0, "relative_delta_rms": 0.0, "cosine_vs_base": 1.0}
            pred_deltas = dict(deltas)
        else:
            assert base_emb is not None and base_pred is not None
            deltas = delta_stats(emb_all, base_emb)
            pred_deltas = delta_stats(pred_all, base_pred)
        x_sweep_rows.append(
            {
                "noise_scale": scale,
                "action_in_proj": tensor_stats(emb_all),
                "action_in_proj_vs_scale0": deltas,
                "pred_v": tensor_stats(pred_all),
                "pred_v_vs_scale0": pred_deltas,
            }
        )

    f2 = {
        "event": "f2_t_xt_sensitivity",
        "checkpoint": str(checkpoint_path),
        "sample_count": int(x1_all.shape[0]),
        "t_sweep_same_x_t": t_sweep_rows,
        "x_t_sweep_same_t": x_sweep_rows,
    }
    print(json.dumps(f2), flush=True)

    summary = {
        "created_at_unix": time.time(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "student_base_model": str(base_model),
        "selected_layers": selected_layers,
        "f1": f1,
        "f2": f2,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
