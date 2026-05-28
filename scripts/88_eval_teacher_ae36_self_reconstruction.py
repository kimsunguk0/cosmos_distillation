#!/usr/bin/env python3
"""Teacher KV36 + original AE36 self-reconstruction sanity.

This script checks the floor of the official Alpamayo action expert path before
we blame the student KV adapter. It builds the normal teacher VLM prefix cache,
runs the original AE36 diffusion sampler, and compares sampled trajectories to
the cached teacher raw trajectory. It also reports the action-space roundtrip
floor: raw teacher traj -> traj_to_action -> action_to_traj.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_SCRIPT = PROJECT_ROOT / "scripts" / "51_train_stage1_ae28_teacher_kv_scale.py"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_TEACHER = Path("/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "action_expert" / "teacher_ae36_self_reconstruction"


def load_stage1_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage1_ae28_teacher_kv_scale", STAGE1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {STAGE1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seeds", default="1097,97,0,1,2024")
    parser.add_argument("--vram-cap-gb", type=float, default=0.0)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "eval_log.jsonl"
    summary_path = args.output_dir / "summary.json"
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]

    stage1 = load_stage1_module()
    if float(args.vram_cap_gb or 0.0) > 0:
        stage1.configure_vram_cap(args.device, args.vram_cap_gb)

    stage_args = SimpleNamespace(
        corpus_jsonl=args.corpus_jsonl,
        split=args.split,
        num_samples=args.num_samples,
        io_workers=args.io_workers,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        stage1_mode="teacher_velocity",
        prefix_mode="teacher_forced",
    )

    started = time.perf_counter()
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
    items = stage1.select_items(stage_args)

    print(json.dumps({"event": "load_teacher_start", "checkpoint": str(args.teacher_checkpoint_path)}), flush=True)
    model, processor, _config, _config_path, _runtime = stage1.load_model_and_processor(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=stage1.torch_dtype_from_name(args.dtype),
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
    stage1.force_attention(model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")

    rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    roundtrip_ades: list[float] = []
    roundtrip_fdes: list[float] = []
    best_ades: list[float] = []
    best_fdes: list[float] = []
    seed_metrics: dict[int, dict[str, list[float]]] = {seed: {"ade": [], "fde": []} for seed in seeds}

    batches = stage1.iter_batches(items, int(args.batch_size))
    for batch_index, batch_items in enumerate(batches):
        batch = stage1.build_batch(
            model=model,
            processor=processor,
            batch_items=batch_items,
            selected_old_indices=[],
            args=stage_args,
        )
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=stage1.torch_dtype_from_name(args.dtype),
            enabled=str(args.device).startswith("cuda") and torch.cuda.is_available(),
        ):
            rt_xyz, _rt_rot = model.action_space.action_to_traj(
                batch["target_action"],
                batch["ego_history_xyz"][:, -1].to(args.device),
                batch["ego_history_rot"][:, -1].to(args.device),
            )
        rt_xyz_np = rt_xyz.detach().float().cpu().numpy()
        for row_index, sample_id in enumerate(batch["sample_ids"]):
            ade, fde = stage1.ade_fde(rt_xyz_np[row_index], target_xyz[row_index])
            roundtrip_ades.append(ade)
            roundtrip_fdes.append(fde)
            rows.append(
                {
                    "sample_id": sample_id,
                    "batch_index": batch_index,
                    "metric_type": "target_action_roundtrip",
                    "ade_m": ade,
                    "fde_m": fde,
                    "pred_path_length_m": stage1.path_len(rt_xyz_np[row_index]),
                    "target_path_length_m": stage1.path_len(target_xyz[row_index]),
                }
            )

        per_seed_preds: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            pred = stage1.sample_modules_paths_batch(
                expert=model.expert,
                action_in_proj=model.action_in_proj,
                action_out_proj=model.action_out_proj,
                model=model,
                prompt_cache=batch["cache"],
                context=batch["context"],
                ego_history_xyz=batch["ego_history_xyz"],
                ego_history_rot=batch["ego_history_rot"],
                seed=int(seed) + batch_index,
                device=torch.device(args.device),
            )
            per_seed_preds[seed] = pred
            for row_index, sample_id in enumerate(batch["sample_ids"]):
                ade, fde = stage1.ade_fde(pred["pred_xyz"][row_index], target_xyz[row_index])
                seed_metrics[seed]["ade"].append(ade)
                seed_metrics[seed]["fde"].append(fde)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "batch_index": batch_index,
                        "metric_type": "teacher_ae36_sample_vs_raw",
                        "seed": int(seed) + batch_index,
                        "ade_m": ade,
                        "fde_m": fde,
                        "pred_path_length_m": stage1.path_len(pred["pred_xyz"][row_index]),
                        "target_path_length_m": stage1.path_len(target_xyz[row_index]),
                    }
                )

        if seeds:
            first_seed = seeds[0]
            repeat_pred = stage1.sample_modules_paths_batch(
                expert=model.expert,
                action_in_proj=model.action_in_proj,
                action_out_proj=model.action_out_proj,
                model=model,
                prompt_cache=batch["cache"],
                context=batch["context"],
                ego_history_xyz=batch["ego_history_xyz"],
                ego_history_rot=batch["ego_history_rot"],
                seed=int(first_seed) + batch_index,
                device=torch.device(args.device),
            )
            first_pred = per_seed_preds[first_seed]
            for row_index, sample_id in enumerate(batch["sample_ids"]):
                ade, fde = stage1.ade_fde(first_pred["pred_xyz"][row_index], repeat_pred["pred_xyz"][row_index])
                repeat_rows.append({"sample_id": sample_id, "ade_m": ade, "fde_m": fde})

        for row_index, sample_id in enumerate(batch["sample_ids"]):
            seed_ade_fde = [
                stage1.ade_fde(per_seed_preds[seed]["pred_xyz"][row_index], target_xyz[row_index]) for seed in seeds
            ]
            if seed_ade_fde:
                best_ade, best_fde = min(seed_ade_fde, key=lambda pair: pair[0])
                best_ades.append(best_ade)
                best_fdes.append(best_fde)

        print(
            json.dumps(
                {
                    "event": "batch_done",
                    "batch_index": batch_index,
                    "batch_size": len(batch_items),
                    "cache_layer_count": batch["meta"].get("cache_layer_count"),
                    "cache_seq_len": batch["meta"].get("cache_seq_len"),
                    "generated_len_mean": batch["meta"].get("generated_len_mean"),
                    "generated_text_preview": batch["meta"].get("generated_text_preview"),
                }
            ),
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            rows.clear()

    repeat_ades = [row["ade_m"] for row in repeat_rows]
    repeat_fdes = [row["fde_m"] for row in repeat_rows]
    summary = {
        "status": "ok",
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "args": {
            "corpus_jsonl": str(args.corpus_jsonl),
            "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
            "output_dir": str(args.output_dir),
            "split": args.split,
            "num_samples": int(args.num_samples),
            "batch_size": int(args.batch_size),
            "seeds": seeds,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "selected_count": len(items),
        "target_action_roundtrip": {
            "ade_m": summarize(roundtrip_ades),
            "fde_m": summarize(roundtrip_fdes),
            "interpretation": "Floor from raw teacher xyz/rot -> action_space.traj_to_action -> action_space.action_to_traj.",
        },
        "teacher_ae36_sample_vs_raw_by_seed": {
            str(seed): {"ade_m": summarize(vals["ade"]), "fde_m": summarize(vals["fde"])}
            for seed, vals in seed_metrics.items()
        },
        "teacher_ae36_sample_vs_raw_best_of_seeds": {
            "ade_m": summarize(best_ades),
            "fde_m": summarize(best_fdes),
        },
        "same_cache_same_seed_repeat": {
            "ade_m": summarize(repeat_ades),
            "fde_m": summarize(repeat_fdes),
            "interpretation": "Should be near zero; checks deterministic sampler/cache handling.",
        },
        "repeat_rows_head": repeat_rows[:16],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path), "status": "ok"}), flush=True)


if __name__ == "__main__":
    main()
