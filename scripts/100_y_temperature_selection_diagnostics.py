#!/usr/bin/env python3
"""Y1/Y2 inference-only temperature and deployable selection diagnostics."""

from __future__ import annotations

import argparse
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


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_W2_DIR / "best.pt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/action_expert/y_temperature_selection_best_step2750",
    )
    parser.add_argument("--temperatures", default="1.0,0.85,0.7,0.5,0.3")
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--num-paths", type=int, default=1)
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
    return pathify_args(argparse.Namespace(**base_args))


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


def metric_summary(ades: list[float], fdes: list[float]) -> dict[str, float | None]:
    return {
        "ade_mean_m": float(np.mean(ades)) if ades else None,
        "ade_p50_m": float(np.percentile(ades, 50)) if ades else None,
        "fde_mean_m": float(np.mean(fdes)) if fdes else None,
        "fde_p50_m": float(np.percentile(fdes, 50)) if fdes else None,
    }


def sample_paths_temperature(
    *,
    bundle: Any,
    teacher_model: Any,
    batch: dict[str, Any],
    seed: int,
    device: torch.device,
    inference_steps: int,
    temperature: float,
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
        action = teacher_model.diffusion.sample(
            batch_size=batch_size,
            step_fn=step_fn,
            device=device,
            inference_step=int(inference_steps),
            temperature=float(temperature),
        )
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


def medoid_index(paths: np.ndarray) -> int:
    # paths: [N, T, 3]. Use mean pointwise L2 distance to all other predictions.
    diff = paths[:, None, :, :] - paths[None, :, :, :]
    dist = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return int(np.argmin(dist.sum(axis=1)))


def evaluate_temperature(
    *,
    bundle: Any,
    teacher_model: Any,
    batches: list[dict[str, Any]],
    inference_steps: int,
    eval_seed_base: int,
    num_paths: int,
    temperature: float,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bundle.eval()
    horizon_specs = (("h1p6_16wp", 16), ("h3p2_32wp", 32), ("h6p4_64wp", 64))
    horizon_names = tuple(name for name, _ in horizon_specs)
    methods = ["single"]
    if int(num_paths) > 1:
        methods.extend(["oracle_best", "medoid", "mean_traj"])
    method_ades = {name: [] for name in methods}
    method_fdes = {name: [] for name in methods}
    method_horizons = {
        method: {name: {"ade": [], "fde": []} for name in horizon_names}
        for method in methods
    }
    rows: list[dict[str, Any]] = []
    std_paths_all: list[float] = []
    mean_paths_all: list[float] = []

    for batch_index, batch in enumerate(batches):
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        sample_ids = list(batch["sample_ids"])
        n_samples = len(sample_ids)
        preds_by_sample: list[list[np.ndarray]] = [[] for _ in range(n_samples)]

        for path_idx in range(int(num_paths)):
            path_seed = int(eval_seed_base) + batch_index * int(num_paths) + path_idx
            pred = sample_paths_temperature(
                bundle=bundle,
                teacher_model=teacher_model,
                batch=batch,
                seed=path_seed,
                device=device,
                inference_steps=int(inference_steps),
                temperature=float(temperature),
            )
            for row_index in range(n_samples):
                preds_by_sample[row_index].append(pred["pred_xyz"][row_index].copy())
            del pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for row_index, sample_id in enumerate(sample_ids):
            target = target_xyz[row_index]
            paths = np.stack(preds_by_sample[row_index], axis=0)
            path_ades = []
            path_fdes = []
            for path in paths:
                ade, fde = script_84.ade_fde(path, target)
                path_ades.append(float(ade))
                path_fdes.append(float(fde))

            selected: dict[str, np.ndarray] = {"single": paths[0]}
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "target_path_length_m": script_84.path_len(target),
                "single_path_length_m": script_84.path_len(paths[0]),
                "single_ade_m": path_ades[0],
                "single_fde_m": path_fdes[0],
            }
            if int(num_paths) > 1:
                oracle_idx = int(np.argmin(path_ades))
                medoid_idx = medoid_index(paths)
                selected["oracle_best"] = paths[oracle_idx]
                selected["medoid"] = paths[medoid_idx]
                selected["mean_traj"] = paths.mean(axis=0)
                std_paths_all.append(float(np.std(path_ades)))
                mean_paths_all.append(float(np.mean(path_ades)))
                row.update(
                    {
                        "ade_all_paths_m": [float(v) for v in path_ades],
                        "ade_mean_over_paths_m": float(np.mean(path_ades)),
                        "ade_std_over_paths_m": float(np.std(path_ades)),
                        "oracle_best_idx": oracle_idx,
                        "medoid_idx": medoid_idx,
                    }
                )

            for method, path in selected.items():
                ade, fde = script_84.ade_fde(path, target)
                method_ades[method].append(float(ade))
                method_fdes[method].append(float(fde))
                row[f"{method}_ade_m"] = float(ade)
                row[f"{method}_fde_m"] = float(fde)
                for name, horizon in horizon_specs:
                    n = min(horizon, int(path.shape[0]), int(target.shape[0]))
                    h_ade, h_fde = script_84.ade_fde(path[:n], target[:n])
                    method_horizons[method][name]["ade"].append(float(h_ade))
                    method_horizons[method][name]["fde"].append(float(h_fde))
                    row[f"{method}_{name}_ade_m"] = float(h_ade)
                    row[f"{method}_{name}_fde_m"] = float(h_fde)
            rows.append(row)

    result: dict[str, Any] = {
        "event": "temperature_eval",
        "temperature": float(temperature),
        "inference_steps": int(inference_steps),
        "eval_seed_base": int(eval_seed_base),
        "eval_num_paths": int(num_paths),
        "eval_count": len(rows),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "methods": {
            method: {
                **metric_summary(method_ades[method], method_fdes[method]),
                "horizon": horizon_summary(method_horizons[method]),
            }
            for method in methods
        },
        "rows": rows,
    }
    if int(num_paths) > 1:
        result["ade_mean_over_paths_mean_m"] = float(np.mean(mean_paths_all)) if mean_paths_all else None
        result["ade_std_over_paths_mean_m"] = float(np.mean(std_paths_all)) if std_paths_all else None
    return result


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
                "event": "y_sampling_start",
                "checkpoint": str(cli.checkpoint),
                "checkpoint_step": ckpt_step,
                "checkpoint_eval_seed_base": base_seed,
                "temperatures": parse_csv_floats(cli.temperatures),
                "inference_steps": int(cli.inference_steps),
                "num_paths": int(cli.num_paths),
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
        emit(
            {
                "event": "bundle_loaded",
                "student_base_model": str(base_model),
                "selected_layers": selected_layers,
                "diffusion_class": f"{type(teacher_model.diffusion).__module__}.{type(teacher_model.diffusion).__name__}",
                "diffusion_int_method": str(getattr(teacher_model.diffusion, "int_method", None)),
                "diffusion_num_inference_steps": int(getattr(teacher_model.diffusion, "num_inference_steps", -1)),
                "temperature_source": "teacher_model.diffusion.sample(..., temperature=temperature)",
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
        results = []
        for temperature in parse_csv_floats(cli.temperatures):
            result = evaluate_temperature(
                bundle=bundle,
                teacher_model=teacher_model,
                batches=batches,
                inference_steps=int(cli.inference_steps),
                eval_seed_base=base_seed,
                num_paths=int(cli.num_paths),
                temperature=float(temperature),
                device=device,
            )
            emit(result)
            results.append(result)

        summary = {
            "event": "y_sampling_summary",
            "checkpoint": str(cli.checkpoint),
            "checkpoint_step": ckpt_step,
            "checkpoint_eval": ckpt_eval,
            "checkpoint_eval_seed_base": base_seed,
            "diffusion_source": {
                "sample_call": "scripts/100_y_temperature_selection_diagnostics.py -> teacher_model.diffusion.sample(..., temperature=...)",
                "class": f"{type(teacher_model.diffusion).__module__}.{type(teacher_model.diffusion).__name__}",
                "int_method": str(getattr(teacher_model.diffusion, "int_method", None)),
                "num_inference_steps": int(getattr(teacher_model.diffusion, "num_inference_steps", -1)),
            },
            "results": results,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        emit({"event": "done", "summary_json": str(summary_path)})


if __name__ == "__main__":
    main()
