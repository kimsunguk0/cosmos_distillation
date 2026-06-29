#!/usr/bin/env python3
"""X1/X2 inference-only sampling diagnostics for AE28 Stage 0 checkpoints."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_84_PATH = Path(__file__).resolve().with_name("84_train_student_ae28_official.py")
DEFAULT_W2_DIR = (
    REPO_ROOT
    / "outputs/action_expert/student_ae28/"
    / "stage0_overfit_32_s3000_seed42_recipe_draw16_full444k_retry_20260531"
)


def load_script_84():
    spec = importlib.util.spec_from_file_location("script_84_train_student_ae28_official", SCRIPT_84_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT_84_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script_84 = load_script_84()


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_W2_DIR / "best.pt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/action_expert/x1_x2_sampling_diagnostics_best_step2750",
    )
    parser.add_argument("--inference-steps", default="10,20,40,80")
    parser.add_argument("--seed-offsets", default="0,1,2,3,4")
    parser.add_argument("--best-of-n", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    return parser.parse_args()


def pathify_args(args: argparse.Namespace) -> argparse.Namespace:
    for key in ("corpus_jsonl", "student_checkpoint_dir", "teacher_checkpoint_path", "output_dir"):
        if hasattr(args, key) and getattr(args, key) is not None:
            setattr(args, key, Path(getattr(args, key)))
    return args


def load_checkpoint(path: Path) -> dict[str, Any]:
    # This checkpoint is produced locally by script 84 and contains pathlib objects in payload args.
    return torch.load(path, map_location="cpu", weights_only=False)


def eval_args_from_checkpoint(cli: argparse.Namespace, checkpoint: dict[str, Any]) -> argparse.Namespace:
    payload = checkpoint.get("payload") or {}
    base_args = dict(payload.get("args") or {})
    base_args.update(
        {
            "device": str(cli.device),
            "teacher_load_device": str(cli.teacher_load_device),
            "attn_implementation": str(cli.attn_implementation),
            "student_dtype": str(cli.student_dtype),
            "ae_dtype": str(cli.ae_dtype),
            "max_length": int(cli.max_length),
            "eval_samples": int(cli.eval_samples),
            "eval_batch_size": int(cli.eval_batch_size),
            "eval_num_paths": 1,
            "eval_seed_mode": "fixed",
            "output_dir": cli.output_dir,
        }
    )
    args = argparse.Namespace(**base_args)
    return pathify_args(args)


def horizon_summary(values: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            "ade_mean_m": float(np.mean(v["ade"])) if v["ade"] else None,
            "ade_p50_m": float(np.percentile(v["ade"], 50)) if v["ade"] else None,
            "fde_mean_m": float(np.mean(v["fde"])) if v["fde"] else None,
            "fde_p50_m": float(np.percentile(v["fde"], 50)) if v["fde"] else None,
        }
        for name, v in values.items()
    }


def evaluate_cached(
    *,
    bundle: Any,
    teacher_model: Any,
    batches: list[dict[str, Any]],
    inference_steps: int,
    eval_seed_base: int,
    num_paths: int,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    original_steps = int(getattr(teacher_model.diffusion, "num_inference_steps", inference_steps))
    teacher_model.diffusion.num_inference_steps = int(inference_steps)
    bundle.eval()
    horizon_specs = (("h1p6_16wp", 16), ("h3p2_32wp", 32), ("h6p4_64wp", 64))
    horizon_names = tuple(name for name, _ in horizon_specs)
    horizon_values = {name: {"ade": [], "fde": []} for name in horizon_names}
    horizon_best_values = {name: {"ade": [], "fde": []} for name in horizon_names}
    rows: list[dict[str, Any]] = []
    best_ades: list[float] = []
    best_fdes: list[float] = []
    mean_paths: list[float] = []
    std_paths: list[float] = []
    try:
        for batch_index, batch in enumerate(batches):
            target_xyz = batch["target_xyz"].detach().cpu().numpy()
            sample_ids = list(batch["sample_ids"])
            per_sample_ades: list[list[float]] = [[] for _ in sample_ids]
            per_sample_fdes: list[list[float]] = [[] for _ in sample_ids]
            per_sample_h_ades = [{name: [] for name in horizon_names} for _ in sample_ids]
            per_sample_h_fdes = [{name: [] for name in horizon_names} for _ in sample_ids]
            first_path_pred_xyz: list[np.ndarray | None] = [None for _ in sample_ids]
            for path_idx in range(int(num_paths)):
                path_seed = int(eval_seed_base) + batch_index * int(num_paths) + path_idx
                pred = script_84.sample_paths(
                    bundle=bundle,
                    teacher_model=teacher_model,
                    batch=batch,
                    seed=path_seed,
                    device=device,
                )
                for row_index, _sample_id in enumerate(sample_ids):
                    pred_xyz = pred["pred_xyz"][row_index]
                    target_xyz_row = target_xyz[row_index]
                    ade, fde = script_84.ade_fde(pred_xyz, target_xyz_row)
                    per_sample_ades[row_index].append(float(ade))
                    per_sample_fdes[row_index].append(float(fde))
                    for name, horizon in horizon_specs:
                        n = min(horizon, int(pred_xyz.shape[0]), int(target_xyz_row.shape[0]))
                        h_ade, h_fde = script_84.ade_fde(pred_xyz[:n], target_xyz_row[:n])
                        per_sample_h_ades[row_index][name].append(float(h_ade))
                        per_sample_h_fdes[row_index][name].append(float(h_fde))
                    if path_idx == 0:
                        first_path_pred_xyz[row_index] = pred_xyz.copy()
                del pred
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for row_index, sample_id in enumerate(sample_ids):
                ades_n = per_sample_ades[row_index]
                fdes_n = per_sample_fdes[row_index]
                row = {
                    "sample_id": sample_id,
                    "ade_m": ades_n[0],
                    "fde_m": fdes_n[0],
                    "pred_path_length_m": script_84.path_len(first_path_pred_xyz[row_index]),
                    "target_path_length_m": script_84.path_len(target_xyz[row_index]),
                }
                for name in horizon_names:
                    h0_ade = per_sample_h_ades[row_index][name][0]
                    h0_fde = per_sample_h_fdes[row_index][name][0]
                    horizon_values[name]["ade"].append(h0_ade)
                    horizon_values[name]["fde"].append(h0_fde)
                    row[f"{name}_ade_m"] = h0_ade
                    row[f"{name}_fde_m"] = h0_fde
                if int(num_paths) > 1:
                    best_idx = int(np.argmin(ades_n))
                    best_ades.append(float(ades_n[best_idx]))
                    best_fdes.append(float(fdes_n[best_idx]))
                    mean_paths.append(float(np.mean(ades_n)))
                    std_paths.append(float(np.std(ades_n)))
                    row.update(
                        {
                            "ade_best_of_n_m": float(ades_n[best_idx]),
                            "fde_best_of_n_m": float(fdes_n[best_idx]),
                            "ade_mean_over_paths_m": float(np.mean(ades_n)),
                            "ade_std_over_paths_m": float(np.std(ades_n)),
                            "best_path_idx": best_idx,
                            "ade_all_paths_m": [float(v) for v in ades_n],
                        }
                    )
                    for name in horizon_names:
                        h_ades = per_sample_h_ades[row_index][name]
                        h_fdes = per_sample_h_fdes[row_index][name]
                        h_best_idx = int(np.argmin(h_ades))
                        horizon_best_values[name]["ade"].append(float(h_ades[h_best_idx]))
                        horizon_best_values[name]["fde"].append(float(h_fdes[h_best_idx]))
                        row[f"{name}_ade_best_of_n_m"] = float(h_ades[h_best_idx])
                        row[f"{name}_fde_best_of_n_m"] = float(h_fdes[h_best_idx])
                rows.append(row)
    finally:
        teacher_model.diffusion.num_inference_steps = original_steps
    ades = [row["ade_m"] for row in rows]
    fdes = [row["fde_m"] for row in rows]
    out: dict[str, Any] = {
        "inference_steps": int(inference_steps),
        "eval_seed_base": int(eval_seed_base),
        "eval_num_paths": int(num_paths),
        "eval_count": len(rows),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "ade_mean_m": float(np.mean(ades)) if ades else None,
        "ade_p50_m": float(np.percentile(ades, 50)) if ades else None,
        "fde_mean_m": float(np.mean(fdes)) if fdes else None,
        "fde_p50_m": float(np.percentile(fdes, 50)) if fdes else None,
        "horizon": horizon_summary(horizon_values),
        "rows": rows,
    }
    if int(num_paths) > 1:
        out.update(
            {
                "ade_best_of_n_mean_m": float(np.mean(best_ades)) if best_ades else None,
                "ade_best_of_n_p50_m": float(np.percentile(best_ades, 50)) if best_ades else None,
                "fde_best_of_n_mean_m": float(np.mean(best_fdes)) if best_fdes else None,
                "fde_best_of_n_p50_m": float(np.percentile(best_fdes, 50)) if best_fdes else None,
                "ade_mean_over_paths_mean_m": float(np.mean(mean_paths)) if mean_paths else None,
                "ade_std_over_paths_mean_m": float(np.mean(std_paths)) if std_paths else None,
                "horizon_best_of_n": horizon_summary(horizon_best_values),
            }
        )
    return out


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cli.output_dir / "run.log"
    summary_path = cli.output_dir / "summary.json"
    checkpoint = load_checkpoint(cli.checkpoint)
    ckpt_payload = checkpoint.get("payload") or {}
    ckpt_step = int(ckpt_payload.get("step") or -1)
    ckpt_eval = ckpt_payload.get("eval") or {}
    base_seed = int(ckpt_eval.get("eval_seed_base") or (int((ckpt_payload.get("args") or {}).get("seed", 42)) + 1000 + ckpt_step))
    args = eval_args_from_checkpoint(cli, checkpoint)
    args.num_samples = max(int(getattr(args, "num_samples", 0) or 0), int(cli.eval_samples))
    args.eval_samples = int(cli.eval_samples)
    args.eval_batch_size = int(cli.eval_batch_size)
    args.output_dir = cli.output_dir
    args.device = str(cli.device)

    with log_path.open("w", encoding="utf-8") as log_handle:
        def emit(row: dict[str, Any]) -> None:
            print(json.dumps(row), flush=True)
            log_handle.write(json.dumps(row) + "\n")
            log_handle.flush()

        emit(
            {
                "event": "x_sampling_start",
                "checkpoint": str(cli.checkpoint),
                "checkpoint_step": ckpt_step,
                "checkpoint_eval_seed_base": base_seed,
                "inference_steps": parse_csv_ints(cli.inference_steps),
                "seed_offsets": parse_csv_ints(cli.seed_offsets),
                "best_of_n": int(cli.best_of_n),
            }
        )
        items = script_84.select_items(args)
        student, student_tokenizer, student_processor, base_model = script_84.load_student(args)
        emit({"event": "load_teacher_action_modules_start", "device": str(args.teacher_load_device)})
        teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = script_84.load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=script_84.torch_dtype_from_name(args.ae_dtype),
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
        bundle, selected_layers = script_84.build_bundle(teacher_model, args, student=student)
        missing, unexpected = bundle.load_state_dict(checkpoint["bundle_state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"bundle.load_state_dict mismatch missing={missing} unexpected={unexpected}")
        bundle.eval()
        if hasattr(teacher_model, "vlm"):
            delattr(teacher_model, "vlm")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        original_diffusion_steps = int(getattr(teacher_model.diffusion, "num_inference_steps", -1))
        emit(
            {
                "event": "bundle_loaded",
                "student_base_model": str(base_model),
                "selected_layers": selected_layers,
                "diffusion_class": f"{type(teacher_model.diffusion).__module__}.{type(teacher_model.diffusion).__name__}",
                "diffusion_int_method": str(getattr(teacher_model.diffusion, "int_method", None)),
                "diffusion_num_inference_steps": original_diffusion_steps,
            }
        )

        batches = []
        for batch_items in script_84.iter_batches(items[: int(args.eval_samples)], int(args.eval_batch_size)):
            batches.append(
                script_84.build_batch(
                    args=args,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    batch_items=batch_items,
                )
            )
        emit({"event": "batches_built", "batch_count": len(batches), "eval_samples": int(args.eval_samples)})

        device = torch.device(args.device)
        x1_results = []
        for inference_steps in parse_csv_ints(cli.inference_steps):
            result = evaluate_cached(
                bundle=bundle,
                teacher_model=teacher_model,
                batches=batches,
                inference_steps=inference_steps,
                eval_seed_base=base_seed,
                num_paths=1,
                device=device,
            )
            result["event"] = "x1_inference_step_sweep"
            emit(result)
            x1_results.append(result)

        current_steps = original_diffusion_steps if original_diffusion_steps > 0 else parse_csv_ints(cli.inference_steps)[0]
        x2_seed_results = []
        for offset in parse_csv_ints(cli.seed_offsets):
            result = evaluate_cached(
                bundle=bundle,
                teacher_model=teacher_model,
                batches=batches,
                inference_steps=current_steps,
                eval_seed_base=base_seed + int(offset),
                num_paths=1,
                device=device,
            )
            result["event"] = "x2_seed_variance"
            result["seed_offset"] = int(offset)
            emit(result)
            x2_seed_results.append(result)

        x2_best_of_n = evaluate_cached(
            bundle=bundle,
            teacher_model=teacher_model,
            batches=batches,
            inference_steps=current_steps,
            eval_seed_base=base_seed,
            num_paths=int(cli.best_of_n),
            device=device,
        )
        x2_best_of_n["event"] = "x2_best_of_n"
        emit(x2_best_of_n)

        seed_ades = [float(r["ade_mean_m"]) for r in x2_seed_results]
        summary = {
            "event": "x_sampling_summary",
            "checkpoint": str(cli.checkpoint),
            "checkpoint_step": ckpt_step,
            "checkpoint_eval": ckpt_eval,
            "checkpoint_eval_seed_base": base_seed,
            "diffusion_source": {
                "sample_call": "scripts/84_train_student_ae28_official.py sample_paths -> teacher_model.diffusion.sample(...)",
                "class": f"{type(teacher_model.diffusion).__module__}.{type(teacher_model.diffusion).__name__}",
                "int_method": str(getattr(teacher_model.diffusion, "int_method", None)),
                "num_inference_steps": original_diffusion_steps,
            },
            "x1": x1_results,
            "x2_seed_variance": {
                "results": x2_seed_results,
                "ade_mean_m_mean": float(np.mean(seed_ades)) if seed_ades else None,
                "ade_mean_m_std": float(np.std(seed_ades)) if seed_ades else None,
                "ade_mean_m_min": float(np.min(seed_ades)) if seed_ades else None,
                "ade_mean_m_max": float(np.max(seed_ades)) if seed_ades else None,
            },
            "x2_best_of_n": x2_best_of_n,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        emit({"event": "done", "summary_json": str(summary_path)})


if __name__ == "__main__":
    main()
