#!/usr/bin/env python3
"""Stage 1 AE-28T overfit trainer.

Goal:
    teacher 36-layer VLM KV -> select 28 KV layers -> 28-layer action expert
    should reproduce the original 36-layer teacher action trajectory.

This deliberately does not involve the 2B student backbone yet. It isolates
whether the 36->28 action-expert compression can be recovered by distillation.
"""

from __future__ import annotations

import argparse
import copy
import gc
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


SUKIM_ROOT = Path("/home/pm97/workspace/sukim")
DISTILL_ROOT = SUKIM_ROOT / "distillation" / "cosmos_distillation"
VIS_ROOT = SUKIM_ROOT / "visualization"
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
for path in (SUKIM_ROOT, DISTILL_ROOT, VIS_ROOT, ALPAMAYO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpamayo1_5 import helper  # noqa: E402
from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (  # noqa: E402
    load_materialized_sample,
    load_model_and_processor,
)
from probe_teacher_kv_28layer_expert_compression import (  # noqa: E402
    ade_fde,
    build_28layer_expert,
    compress_cache,
    force_attention,
    layer_mapping,
    make_context,
    path_len,
    run_teacher_vlm_to_action_pre,
    torch_dtype_from_name,
)


DEFAULT_CORPUS = DISTILL_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_TEACHER = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B"
DEFAULT_OUTPUT = DISTILL_ROOT / "outputs" / "action_expert" / "stage1_ae28_teacher_kv_overfit"


class AE28Bundle(nn.Module):
    def __init__(self, *, expert: nn.Module, action_in_proj: nn.Module, action_out_proj: nn.Module) -> None:
        super().__init__()
        self.expert = expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="sdpa")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument("--mapping", choices=("linspace_round", "first_n"), default="linspace_round")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-cache-json", type=Path, default=None)
    return parser.parse_args()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_raw_json(record: dict[str, Any]) -> Path | None:
    raw = ((record.get("teacher_cache") or {}).get("text_raw_json_path"))
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None


def select_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in load_jsonl(args.corpus_jsonl):
        if args.split and row.get("split") != args.split:
            continue
        raw = resolve_raw_json(row)
        sample_dir = Path(str((row.get("input") or {}).get("materialized_sample_path") or ""))
        if raw is None or not sample_dir.exists():
            continue
        selected.append(row)
        if len(selected) >= int(args.num_samples):
            break
    if not selected:
        raise RuntimeError("No Stage 1 records with raw teacher action outputs were found.")
    return selected


def raw_teacher_pred(raw_json: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    result = (payload.get("results") or [None])[0]
    if not isinstance(result, dict):
        raise ValueError(f"Missing results[0] in {raw_json}")
    xyz = np.asarray(result.get("pred_xyz"), dtype=np.float32).reshape(-1, 64, 3)[0]
    rot = np.asarray(result.get("pred_rot"), dtype=np.float32).reshape(-1, 64, 3, 3)[0]
    return xyz, rot


def prepare_items(
    *,
    records: list[dict[str, Any]],
    model: Any,
    device: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        sample_id = str(record["sample_id"])
        sample_dir = Path(str((record.get("input") or {}).get("materialized_sample_path")))
        raw_path = resolve_raw_json(record)
        if raw_path is None:
            continue
        sample_cpu = load_materialized_sample(sample_dir)
        target_xyz_np, target_rot_np = raw_teacher_pred(raw_path)
        sample_gpu = helper.to_device(sample_cpu, device)
        target_xyz = torch.from_numpy(target_xyz_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        target_rot = torch.from_numpy(target_rot_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            target_action = model.action_space.traj_to_action(
                sample_gpu["ego_history_xyz"][:, -1],
                sample_gpu["ego_history_rot"][:, -1],
                target_xyz,
                target_rot,
            )
        items.append(
            {
                "sample_id": sample_id,
                "record": record,
                "sample_cpu": sample_cpu,
                "target_xyz": target_xyz.detach().cpu(),
                "target_rot": target_rot.detach().cpu(),
                "target_action": target_action.detach().cpu(),
                "raw_json": str(raw_path),
            }
        )
        print(
            json.dumps(
                {
                    "event": "prepared_item",
                    "index": index,
                    "total": len(records),
                    "sample_id": sample_id,
                    "target_path_length_m": path_len(target_xyz_np),
                }
            ),
            flush=True,
        )
    if not items:
        raise RuntimeError("Prepared zero Stage 1 items.")
    return items


def make_bundle(
    *,
    model: Any,
    selected_old_indices: list[int],
    dtype: torch.dtype,
    device: str,
    attn_implementation: str,
) -> AE28Bundle:
    expert = build_28layer_expert(
        teacher_expert=model.expert,
        selected_old_indices=selected_old_indices,
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
    )
    action_in_proj = copy.deepcopy(model.action_in_proj).to(device=device, dtype=dtype).train()
    action_out_proj = copy.deepcopy(model.action_out_proj).to(device=device, dtype=dtype).train()
    bundle = AE28Bundle(expert=expert, action_in_proj=action_in_proj, action_out_proj=action_out_proj)
    bundle.train()
    return bundle


def train_velocity_forward(
    *,
    bundle: AE28Bundle,
    model: Any,
    prompt_cache: Any,
    context: dict[str, Any],
    target_action: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = next(bundle.parameters()).dtype
    x1 = target_action.to(device=device, dtype=dtype)
    x0 = torch.randn_like(x1)
    t = torch.rand((x1.shape[0], 1, 1), device=device, dtype=dtype)
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0

    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = model.action_space.get_action_space_dims()
    forward_kwargs: dict[str, Any] = {}
    if bool(getattr(model.config, "expert_non_causal_attention", False)):
        forward_kwargs["is_causal"] = False

    future_token_embeds = bundle.action_in_proj(x_t, t)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(x_t.shape[0], n_diffusion_tokens, -1)
    attention_mask = context["attention_mask"].to(dtype=future_token_embeds.dtype)
    expert_out = bundle.expert(
        inputs_embeds=future_token_embeds,
        position_ids=context["position_ids"],
        past_key_values=prompt_cache,
        attention_mask=attention_mask,
        use_cache=True,
        **forward_kwargs,
    )
    prompt_cache.crop(prefill_seq_len)
    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    pred_v = bundle.action_out_proj(last_hidden).view(-1, *action_dims)
    loss = F.mse_loss(pred_v.float(), target_v.float())
    stats = {
        "target_action_abs_mean": float(x1.detach().abs().mean().cpu()),
        "pred_v_abs_mean": float(pred_v.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target_v.detach().abs().mean().cpu()),
    }
    return loss, stats


def build_selected_teacher_cache(
    *,
    model: Any,
    processor: Any,
    item: dict[str, Any],
    selected_old_indices: list[int],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    sample_gpu = helper.to_device(item["sample_cpu"], args.device)
    with torch.inference_mode():
        vlm = run_teacher_vlm_to_action_pre(
            model=model,
            processor=processor,
            sample=sample_gpu,
            device=args.device,
            max_new_tokens=int(args.max_new_tokens),
        )
    compressed_cache = compress_cache(vlm["past_key_values"], selected_old_indices)
    context = make_context(
        model=model,
        sequences=vlm["sequences"],
        eos_token_id=int(vlm["eos_token_id"]),
        rope_deltas=vlm["rope_deltas"],
        cache=compressed_cache,
        prefix_mask=vlm["prefix_mask"],
        device=torch.device(args.device),
    )
    meta = {
        "vlm_elapsed_sec": float(vlm["elapsed_sec"]),
        "generated_len": int(vlm["generated_ids"].shape[1]),
        "cache_seq_len": int(compressed_cache.get_seq_length()),
        "generated_text": str(vlm["generated_text"])[:300],
    }
    return compressed_cache, context, meta


def sample_bundle_path(
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
            attention_mask=context["attention_mask"].to(dtype=future_token_embeds.dtype),
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
            batch_size=1,
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
        "sampled_action": sampled_action.detach().float().cpu(),
        "pred_xyz": pred_xyz[0].detach().float().cpu().numpy().astype(np.float32),
        "pred_rot": pred_rot[0].detach().float().cpu().numpy().astype(np.float32),
    }


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
    device = torch.device(args.device)
    for index, item in enumerate(items[: int(args.eval_samples)], start=1):
        compressed_cache, context, meta = build_selected_teacher_cache(
            model=model,
            processor=processor,
            item=item,
            selected_old_indices=selected_old_indices,
            args=args,
        )
        sample_gpu = helper.to_device(item["sample_cpu"], args.device)
        pred = sample_bundle_path(
            bundle=bundle,
            model=model,
            prompt_cache=compressed_cache,
            context=context,
            ego_history_xyz=sample_gpu["ego_history_xyz"],
            ego_history_rot=sample_gpu["ego_history_rot"],
            seed=int(args.seed) + 1000 + index,
            device=device,
        )
        target_xyz = item["target_xyz"].numpy().reshape(64, 3)
        ade, fde = ade_fde(pred["pred_xyz"], target_xyz)
        rows.append(
            {
                "sample_id": item["sample_id"],
                "ade_m": ade,
                "fde_m": fde,
                "pred_path_length_m": path_len(pred["pred_xyz"]),
                "target_path_length_m": path_len(target_xyz),
                **meta,
            }
        )
        del compressed_cache
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
    bundle.train()
    return out


def save_checkpoint(path: Path, *, bundle: AE28Bundle, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bundle_state_dict": bundle.state_dict(),
            "payload": payload,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    summary_path = args.output_dir / "summary.json"
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    summary: dict[str, Any] = {
        "created_at_unix": time.time(),
        "args": vars(args) | {
            "corpus_jsonl": str(args.corpus_jsonl),
            "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
            "output_dir": str(args.output_dir),
            "target_cache_json": str(args.target_cache_json) if args.target_cache_json else None,
        },
        "status": "running",
    }
    try:
        records = select_records(args)
        summary["selected_sample_ids"] = [str(row["sample_id"]) for row in records]
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
        old_layers = int(model.expert.config.num_hidden_layers)
        selected_old_indices = layer_mapping(old_layers, int(args.compressed_layers), args.mapping)
        summary["layer_mapping"] = selected_old_indices

        print(json.dumps({"event": "prepare_items_start", "count": len(records)}), flush=True)
        items = prepare_items(records=records, model=model, device=args.device)
        summary["prepared_count"] = len(items)
        print(json.dumps({"event": "build_bundle_start", "mapping": selected_old_indices}), flush=True)
        bundle = make_bundle(
            model=model,
            selected_old_indices=selected_old_indices,
            dtype=torch_dtype_from_name(args.dtype),
            device=args.device,
            attn_implementation="sdpa" if args.attn_implementation != "eager" else "eager",
        )
        trainable_params = sum(p.numel() for p in bundle.parameters() if p.requires_grad)
        summary["trainable_params"] = int(trainable_params)
        optimizer = torch.optim.AdamW(
            [p for p in bundle.parameters() if p.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        eval_history: list[dict[str, Any]] = []
        train_history: list[dict[str, Any]] = []
        if int(args.eval_samples) > 0:
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

        device = torch.device(args.device)
        started = time.perf_counter()
        for step in range(1, int(args.steps) + 1):
            item = random.choice(items)
            optimizer.zero_grad(set_to_none=True)
            compressed_cache, context, meta = build_selected_teacher_cache(
                model=model,
                processor=processor,
                item=item,
                selected_old_indices=selected_old_indices,
                args=args,
            )
            loss, loss_stats = train_velocity_forward(
                bundle=bundle,
                model=model,
                prompt_cache=compressed_cache,
                context=context,
                target_action=item["target_action"],
                device=device,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(bundle.parameters(), float(args.grad_clip_norm))
            optimizer.step()
            train_row = {
                "step": int(step),
                "sample_id": item["sample_id"],
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "elapsed_total_sec": round(time.perf_counter() - started, 3),
                **loss_stats,
                **meta,
            }
            train_history.append(train_row)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "train_step", **train_row}, ensure_ascii=True) + "\n")
            if step == 1 or step % 10 == 0:
                print(json.dumps({"event": "train_step", **train_row}), flush=True)
            del compressed_cache
            del context
            del loss
            gc.collect()
            if torch.cuda.is_available() and step % 10 == 0:
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

        summary["status"] = "ok"
        summary["train_history_tail"] = train_history[-20:]
        summary["eval_history"] = eval_history
        if args.save_checkpoint:
            checkpoint_path = args.output_dir / "ae28_stage1_overfit_final.pt"
            save_checkpoint(
                checkpoint_path,
                bundle=bundle,
                payload={
                    "layer_mapping": selected_old_indices,
                    "args": summary["args"],
                    "eval_history": eval_history,
                    "train_history_tail": train_history[-20:],
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
