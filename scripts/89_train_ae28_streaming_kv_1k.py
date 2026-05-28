#!/usr/bin/env python3
"""Streaming-KV AE28 pilot.

For 1k+ samples, keeping every KV cache resident on GPU is too expensive. This
script builds one batch KV at a time, trains several inner FM updates on that
cached batch, then releases it and moves on. It keeps the useful cached-overfit
behavior without scaling VRAM linearly with dataset size.
"""

from __future__ import annotations

import argparse
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AE84_PATH = PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"
AE88_PATH = PROJECT_ROOT / "scripts" / "88_train_ae28_cached_overfit.py"

spec84 = importlib.util.spec_from_file_location("ae84", AE84_PATH)
if spec84 is None or spec84.loader is None:
    raise RuntimeError(f"Could not import {AE84_PATH}")
ae84 = importlib.util.module_from_spec(spec84)
spec84.loader.exec_module(ae84)

spec88 = importlib.util.spec_from_file_location("ae88", AE88_PATH)
if spec88 is None or spec88.loader is None:
    raise RuntimeError(f"Could not import {AE88_PATH}")
ae88 = importlib.util.module_from_spec(spec88)
spec88.loader.exec_module(ae88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=ae84.DEFAULT_CORPUS)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--num-train-samples", type=int, default=1000)
    parser.add_argument("--num-val-samples", type=int, default=64)
    parser.add_argument("--num-train-eval-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--inner-updates", type=int, default=4)
    parser.add_argument("--eval-every-updates", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--student-checkpoint-dir", type=Path, default=ae84.DEFAULT_STUDENT_CKPT)
    parser.add_argument("--student-model", default=ae84.resolve_student_model_path())
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=ae84.DEFAULT_TEACHER)
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--prefix-mode", choices=("student_free", "teacher_forced"), default="student_free")
    parser.add_argument("--ae-init-mode", choices=("teacher_compressed", "scratch"), default="teacher_compressed")
    parser.add_argument("--mapping", choices=("linspace_round", "first_n"), default="linspace_round")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--train-timestep-sampler", choices=("uniform", "beta"), default="beta")
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
    parser.add_argument("--num-time-samples", type=int, default=1)
    parser.add_argument(
        "--velocity-scale-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss on mean |pred_v| vs mean |target_v|; forwarded to cached AE train step.",
    )
    parser.add_argument(
        "--action-recon-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary SmoothL1 on reconstructed action; forwarded to cached AE train step.",
    )
    parser.add_argument(
        "--traj-horizon-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary horizon-weighted ADE loss on one-step reconstructed trajectory.",
    )
    parser.add_argument(
        "--traj-final-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary horizon-weighted FDE loss on one-step reconstructed trajectory.",
    )
    parser.add_argument(
        "--traj-horizon-weights",
        default="0.25,0.5,1.0",
        help="Comma-separated weights for horizons 16,32,64 used by trajectory auxiliary losses.",
    )
    parser.add_argument("--expert-lr", type=float, default=1e-4)
    parser.add_argument("--proj-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-seed-mode", choices=("fixed", "step"), default="fixed")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "action_expert" / "student_ae28_streaming_kv",
    )
    return parser.parse_args()


def select_split_items(args: argparse.Namespace, split: str, count: int) -> list[dict[str, Any]]:
    # ae84.select_items validates path existence for every candidate row. That
    # is fine for tiny overfit runs, but for 50k+ streaming runs it spends
    # minutes doing metadata lookups before the GPU sees a batch. The corpus was
    # already materialized/QC'd, so keep selection to a single cheap JSONL scan
    # and let the batch builder surface any genuinely missing files.
    items: list[dict[str, Any]] = []
    scanned = 0
    with Path(args.corpus_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            row = json.loads(line)
            if split and row.get("split") != split:
                continue
            raw = ((row.get("teacher_cache") or {}).get("text_raw_json_path"))
            sample_dir = ((row.get("input") or {}).get("materialized_sample_path"))
            raw_path = str(raw) if raw else None
            sample_path = str(sample_dir) if sample_dir else None
            if not raw_path or not sample_path:
                continue
            items.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "row": row,
                    "sample_dir": str(sample_path),
                    "raw_json": str(raw_path),
                }
            )
            if len(items) >= int(count):
                break
    if not items:
        raise RuntimeError(f"No usable AE28 samples found for split={split!r}.")
    print(
        json.dumps(
            {
                "event": "select_items_done_fast",
                "split": split,
                "selected_count": len(items),
                "scanned_count": scanned,
                "corpus_jsonl": str(args.corpus_jsonl),
            }
        ),
        flush=True,
    )
    return items


def build_one_batch(
    *,
    args: argparse.Namespace,
    student: Any,
    student_processor: Any,
    student_tokenizer: Any,
    teacher_model: Any,
    batch_items: list[dict[str, Any]],
) -> dict[str, Any]:
    batch = ae84.build_batch(
        args=args,
        student=student,
        student_processor=student_processor,
        student_tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        batch_items=batch_items,
    )
    return ae88.detach_batch(batch)


def evaluate_streaming(
    *,
    name: str,
    args: argparse.Namespace,
    bundle: Any,
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
    eval_seed_base = int(args.seed) + 1000 + (0 if str(args.eval_seed_mode) == "fixed" else int(step))
    for batch_index, batch_items in enumerate(ae84.iter_batches(items, int(args.batch_size))):
        batch = build_one_batch(
            args=args,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            batch_items=batch_items,
        )
        pred = ae88.sample_paths_cached(
            bundle=bundle,
            teacher_model=teacher_model,
            batch=batch,
            seed=eval_seed_base + batch_index,
            device=device,
        )
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        for row_index, sample_id in enumerate(batch["sample_ids"]):
            ade, fde = ae84.ade_fde(pred["pred_xyz"][row_index], target_xyz[row_index])
            h16_ade, h16_fde = ae84.ade_fde(pred["pred_xyz"][row_index][:16], target_xyz[row_index][:16])
            h32_ade, h32_fde = ae84.ade_fde(pred["pred_xyz"][row_index][:32], target_xyz[row_index][:32])
            rows.append(
                {
                    "sample_id": sample_id,
                    "ade_m": ade,
                    "fde_m": fde,
                    "h1p6_16wp_ade_m": h16_ade,
                    "h1p6_16wp_fde_m": h16_fde,
                    "h3p2_32wp_ade_m": h32_ade,
                    "h3p2_32wp_fde_m": h32_fde,
                    "pred_path_length_m": ae84.path_len(pred["pred_xyz"][row_index]),
                    "target_path_length_m": ae84.path_len(target_xyz[row_index]),
                }
            )
        del batch, pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ades = [row["ade_m"] for row in rows]
    fdes = [row["fde_m"] for row in rows]
    h16_ades = [row["h1p6_16wp_ade_m"] for row in rows]
    h16_fdes = [row["h1p6_16wp_fde_m"] for row in rows]
    h32_ades = [row["h3p2_32wp_ade_m"] for row in rows]
    h32_fdes = [row["h3p2_32wp_fde_m"] for row in rows]
    out = {
        "event": f"eval_{name}",
        "step": int(step),
        "eval_count": len(rows),
        "ade_mean_m": float(np.mean(ades)) if ades else None,
        "ade_p50_m": float(np.percentile(ades, 50)) if ades else None,
        "fde_mean_m": float(np.mean(fdes)) if fdes else None,
        "fde_p50_m": float(np.percentile(fdes, 50)) if fdes else None,
        "horizon": {
            "h1p6_16wp": {
                "ade_mean_m": float(np.mean(h16_ades)) if h16_ades else None,
                "ade_p50_m": float(np.percentile(h16_ades, 50)) if h16_ades else None,
                "fde_mean_m": float(np.mean(h16_fdes)) if h16_fdes else None,
                "fde_p50_m": float(np.percentile(h16_fdes, 50)) if h16_fdes else None,
            },
            "h3p2_32wp": {
                "ade_mean_m": float(np.mean(h32_ades)) if h32_ades else None,
                "ade_p50_m": float(np.percentile(h32_ades, 50)) if h32_ades else None,
                "fde_mean_m": float(np.mean(h32_fdes)) if h32_fdes else None,
                "fde_p50_m": float(np.percentile(h32_fdes, 50)) if h32_fdes else None,
            },
            "h6p4_64wp": {
                "ade_mean_m": float(np.mean(ades)) if ades else None,
                "ade_p50_m": float(np.percentile(ades, 50)) if ades else None,
                "fde_mean_m": float(np.mean(fdes)) if fdes else None,
                "fde_p50_m": float(np.percentile(fdes, 50)) if fdes else None,
            },
        },
    }
    bundle.train()
    return out


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
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        train_items = select_split_items(args, args.train_split, int(args.num_train_samples))
        val_items = select_split_items(args, args.val_split, int(args.num_val_samples))
        train_eval_items = train_items[: int(args.num_train_eval_samples)]
        summary["train_count"] = len(train_items)
        summary["val_count"] = len(val_items)
        summary["train_eval_count"] = len(train_eval_items)

        student, student_tokenizer, student_processor, base_model = ae84.load_student(args)
        summary["student_base_model"] = str(base_model)
        print(json.dumps({"event": "load_teacher_action_modules_start", "device": args.teacher_load_device}), flush=True)
        teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = ae84.load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=ae84.torch_dtype_from_name(args.ae_dtype),
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
        ae84.force_attention(teacher_model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")
        bundle, selected_layers = ae84.build_bundle(teacher_model, args)
        summary["ae28_selected_teacher_layers"] = selected_layers
        summary["trainable_params"] = int(sum(p.numel() for p in bundle.parameters() if p.requires_grad))
        if hasattr(teacher_model, "vlm"):
            delattr(teacher_model, "vlm")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        optimizer = torch.optim.AdamW(
            [
                {"params": bundle.expert.parameters(), "lr": float(args.expert_lr)},
                {"params": bundle.action_in_proj.parameters(), "lr": float(args.proj_lr)},
                {"params": bundle.action_out_proj.parameters(), "lr": float(args.proj_lr)},
            ],
            weight_decay=float(args.weight_decay),
        )
        log_handle = log_path.open("a", encoding="utf-8")
        best_val: dict[str, Any] | None = None

        started = time.perf_counter()
        opt_step = 0
        train_batches = list(ae84.iter_batches(train_items, int(args.batch_size)))
        total_batches = len(train_batches) * int(args.epochs)
        for epoch in range(int(args.epochs)):
            for batch_index, batch_items in enumerate(train_batches):
                batch_started = time.perf_counter()
                batch = build_one_batch(
                    args=args,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    batch_items=batch_items,
                )
                cache_build_sec = time.perf_counter() - batch_started
                last_loss = None
                last_stats: dict[str, float] = {}
                for _ in range(int(args.inner_updates)):
                    opt_step += 1
                    optimizer.zero_grad(set_to_none=True)
                    loss, stats = ae88.train_step_cached(
                        bundle=bundle,
                        teacher_model=teacher_model,
                        batch=batch,
                        num_time_samples=int(args.num_time_samples),
                        train_timestep_sampler=str(args.train_timestep_sampler),
                        velocity_scale_loss_weight=float(args.velocity_scale_loss_weight),
                        action_recon_loss_weight=float(args.action_recon_loss_weight),
                        traj_horizon_loss_weight=float(args.traj_horizon_loss_weight),
                        traj_final_loss_weight=float(args.traj_final_loss_weight),
                        traj_horizon_weights=str(args.traj_horizon_weights),
                        device=device,
                    )
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(bundle.parameters(), float(args.grad_clip_norm))
                    optimizer.step()
                    last_loss = float(loss.detach().cpu())
                    last_stats = stats | {
                        "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    }
                    if opt_step == 1 or opt_step % int(args.log_every) == 0:
                        row = {
                            "event": "train_step",
                            "opt_step": int(opt_step),
                            "epoch": int(epoch),
                            "batch_index": int(batch_index),
                            "global_batch_index": int(epoch * len(train_batches) + batch_index),
                            "total_batches": int(total_batches),
                            "loss": last_loss,
                            "elapsed_sec": round(time.perf_counter() - started, 3),
                            "cache_build_sec": round(cache_build_sec, 3),
                            "traj_start_hit_rate": batch["traj_start_hit_rate"],
                            **last_stats,
                        }
                        print(json.dumps(row), flush=True)
                        log_handle.write(json.dumps(row) + "\n")
                        log_handle.flush()
                    del loss

                    if opt_step % int(args.eval_every_updates) == 0:
                        ev_train = evaluate_streaming(
                            name="train",
                            args=args,
                            bundle=bundle,
                            student=student,
                            student_processor=student_processor,
                            student_tokenizer=student_tokenizer,
                            teacher_model=teacher_model,
                            items=train_eval_items,
                            step=opt_step,
                        )
                        ev_val = evaluate_streaming(
                            name="val",
                            args=args,
                            bundle=bundle,
                            student=student,
                            student_processor=student_processor,
                            student_tokenizer=student_tokenizer,
                            teacher_model=teacher_model,
                            items=val_items,
                            step=opt_step,
                        )
                        for ev in (ev_train, ev_val):
                            print(json.dumps(ev), flush=True)
                            log_handle.write(json.dumps(ev) + "\n")
                        log_handle.flush()
                        if best_val is None or float(ev_val["ade_mean_m"]) < float(best_val["ade_mean_m"]):
                            best_val = ev_val
                            ae88.save_checkpoint(
                                args.output_dir / "best.pt",
                                bundle=bundle,
                                payload={"opt_step": opt_step, "eval_val": ev_val, "args": vars(args)},
                            )
                if opt_step % int(args.eval_every_updates) != 0:
                    row = {
                        "event": "train_batch_done",
                        "opt_step": int(opt_step),
                        "epoch": int(epoch),
                        "batch_index": int(batch_index),
                        "loss": last_loss,
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                        "cache_build_sec": round(cache_build_sec, 3),
                        "traj_start_hit_rate": batch["traj_start_hit_rate"],
                        **last_stats,
                    }
                    if batch_index == 0 or (batch_index + 1) % 25 == 0:
                        print(json.dumps(row), flush=True)
                        log_handle.write(json.dumps(row) + "\n")
                        log_handle.flush()
                del batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        final_train = evaluate_streaming(
            name="train",
            args=args,
            bundle=bundle,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            items=train_eval_items,
            step=opt_step,
        )
        final_val = evaluate_streaming(
            name="val",
            args=args,
            bundle=bundle,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            items=val_items,
            step=opt_step,
        )
        for ev in (final_train, final_val):
            print(json.dumps(ev), flush=True)
            log_handle.write(json.dumps(ev) + "\n")
        ae88.save_checkpoint(args.output_dir / "final.pt", bundle=bundle, payload={"opt_step": opt_step, "args": vars(args)})
        if best_val is None or float(final_val["ade_mean_m"]) < float(best_val["ade_mean_m"]):
            best_val = final_val
            ae88.save_checkpoint(
                args.output_dir / "best.pt",
                bundle=bundle,
                payload={"opt_step": opt_step, "eval_val": final_val, "args": vars(args)},
            )
        summary.update(
            {
                "status": "ok",
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "opt_steps": int(opt_step),
                "final_train_eval": final_train,
                "final_val_eval": final_val,
                "best_val_eval": best_val,
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
