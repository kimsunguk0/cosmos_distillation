#!/usr/bin/env python3
"""Sweep official Alpamayo-1.5-10B flow-matching inference steps.

This keeps the input samples, VLM decoding setup, trajectory sample count, and
seed policy fixed, then overrides only diffusion_kwargs["inference_step"].
It is meant as a sanity check against student AE step sweeps.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo/alpamayo1.5/src"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(str(PROJECT_ROOT))

import benchmark_4models as bench4  # noqa: E402
from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (  # noqa: E402
    build_model_inputs_batch,
    enforce_generation_mode,
    load_materialized_samples,
    load_model_and_processor,
    torch_dtype_from_name,
)


DEFAULT_OUT = PROJECT_ROOT / "outputs/benchmarks/teacher10b_fm_step_sweep_20260706"
DEFAULT_TEACHER = bench4.DEFAULT_TEACHER
DEFAULT_CORPUS = bench4.DEFAULT_CORPUS


def parse_steps(text: str) -> list[int]:
    values: list[int] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"inference step must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("no inference steps provided")
    return sorted(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=64, help="0 uses all selected rows.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="flash_attention_2")
    parser.add_argument("--eval-num-paths", type=int, default=6)
    parser.add_argument("--eval-temperature", type=float, default=0.85)
    parser.add_argument("--eval-selection-method", choices=("single", "oracle_best", "medoid", "mean_traj"), default="mean_traj")
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-top-k", type=int, default=0)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", default="1,2,3,4,5,6,8,10")
    parser.add_argument("--reference-step", type=int, default=10)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": float(clean.mean()),
        "p50": float(np.percentile(clean, 50)),
        "p95": float(np.percentile(clean, 95)),
    }


def path_drift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = bench4.squeeze_path(a)
    b = bench4.squeeze_path(b)
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(a[:n, :2] - b[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flatten_texts(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, np.ndarray):
        return flatten_texts(value.tolist())
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(flatten_texts(item))
    return out


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    steps = parse_steps(args.steps)
    if int(args.reference_step) not in steps:
        steps.append(int(args.reference_step))
        steps = sorted(dict.fromkeys(steps))

    out_dir = Path(args.output_dir)
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path.unlink(missing_ok=True)

    rows = bench4.read_rows(Path(args.corpus_jsonl), str(args.split), int(args.num_samples))
    settings = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "output_dir": str(out_dir),
        "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
        "split": str(args.split),
        "selected_count": len(rows),
        "category_counts": dict(sorted(Counter(bench4.category(row) for row in rows).items())),
        "batch_size": int(args.batch_size),
        "eval_num_paths": int(args.eval_num_paths),
        "eval_temperature": float(args.eval_temperature),
        "eval_selection_method": str(args.eval_selection_method),
        "steps": steps,
        "reference_step": int(args.reference_step),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation),
    }
    write_json(out_dir / "settings.json", settings)
    print(json.dumps({"event": "teacher10b_sweep_start", **settings}, ensure_ascii=False), flush=True)

    model, processor, _, config_path, _ = load_model_and_processor(
        checkpoint_path=Path(args.teacher_checkpoint_path),
        dtype=torch_dtype_from_name(args.dtype),
        device=str(args.device),
        config_json=None,
        runtime_support=None,
        attn_implementation=str(args.attn_implementation),
        min_pixels=163840,
        max_pixels=196608,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(json.dumps({"event": "teacher10b_load_done", "config": str(config_path)}), flush=True)

    metric_rows_by_step: dict[int, list[dict[str, Any]]] = {step: [] for step in steps}
    selected_paths_by_step: dict[int, dict[str, np.ndarray]] = {step: {} for step in steps}
    started = time.perf_counter()
    top_k_value = None if int(args.teacher_top_k) <= 0 else int(args.teacher_top_k)
    model_dtype = next(model.parameters()).dtype
    autocast_context = (
        torch.autocast("cuda", dtype=model_dtype)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else torch.no_grad()
    )

    for batch_start in range(0, len(rows), int(args.batch_size)):
        batch_rows = rows[batch_start : batch_start + int(args.batch_size)]
        sample_dirs = [Path(str((row.get("input") or {}).get("materialized_sample_path"))) for row in batch_rows]
        samples = load_materialized_samples(sample_dirs, int(args.io_workers))
        model_inputs = build_model_inputs_batch(processor=processor, samples=samples, device=str(args.device))
        targets = [bench4.load_gt_xyz(row) for row in batch_rows]
        for step in steps:
            seed = int(args.seed) + int(batch_start)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            step_started = time.perf_counter()
            with torch.inference_mode(), autocast_context, enforce_generation_mode(model, "sample"):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=float(args.teacher_top_p),
                    top_k=top_k_value,
                    temperature=float(args.eval_temperature),
                    num_traj_samples=int(args.eval_num_paths),
                    return_extra=True,
                    max_generation_length=int(args.teacher_max_new_tokens),
                    diffusion_kwargs={"inference_step": int(step)},
                )
            elapsed_ms = (time.perf_counter() - step_started) * 1000.0 / max(len(batch_rows), 1)
            pred_xyz_np = pred_xyz.detach().cpu().numpy()
            cot_texts = flatten_texts((extra or {}).get("cot") if isinstance(extra, dict) else None)
            for i, row in enumerate(batch_rows):
                paths = bench4.squeeze_paths(pred_xyz_np[i])
                target = bench4.squeeze_path(targets[i])
                path_ades = [bench4.ade_fde(path, target)[0] for path in paths]
                path_fdes = [bench4.ade_fde(path, target)[1] for path in paths]
                chosen, chosen_idx = bench4.select_path(paths, path_ades, str(args.eval_selection_method))
                best_idx = int(np.nanargmin(np.asarray(path_ades, dtype=np.float64)))
                ade, fde = bench4.ade_fde(chosen, target)
                sample_id = str(row.get("sample_id"))
                rec = {
                    "sample_id": sample_id,
                    "category": bench4.category(row),
                    "inference_steps": int(step),
                    "ade_gt_m": float(ade),
                    "fde_gt_m": float(fde),
                    "minade6_gt_m": float(path_ades[best_idx]),
                    "minfde6_gt_m": float(path_fdes[best_idx]),
                    "best_path_idx_gt": int(best_idx),
                    "selected_path_idx": chosen_idx,
                    "path_ade_gt_m": [float(v) for v in path_ades],
                    "elapsed_ms": float(elapsed_ms),
                    "cot_preview": cot_texts[i][:240] if i < len(cot_texts) else "",
                }
                append_jsonl(rows_path, rec)
                metric_rows_by_step[int(step)].append(rec)
                selected_paths_by_step[int(step)][sample_id] = bench4.squeeze_path(chosen).astype(np.float32, copy=True)
            del pred_xyz, pred_rot, extra
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        done = min(batch_start + int(args.batch_size), len(rows))
        progress: dict[str, Any] = {
            "event": "teacher10b_sweep_progress",
            "done": done,
            "total": len(rows),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        for step in steps:
            recs = metric_rows_by_step[int(step)]
            progress[f"ade_s{step}"] = summarize([r["ade_gt_m"] for r in recs])["mean"]
            progress[f"minade6_s{step}"] = summarize([r["minade6_gt_m"] for r in recs])["mean"]
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        del model_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ref_step = int(args.reference_step)
    ref_paths = selected_paths_by_step[ref_step]
    table: list[dict[str, Any]] = []
    step_summaries: dict[str, Any] = {}
    ref_recs = metric_rows_by_step[ref_step]
    ref_ade = summarize([r["ade_gt_m"] for r in ref_recs])["mean"]
    ref_minade = summarize([r["minade6_gt_m"] for r in ref_recs])["mean"]
    for step in steps:
        recs = metric_rows_by_step[int(step)]
        drift_ade: list[float] = []
        drift_fde: list[float] = []
        for rec in recs:
            sample_id = str(rec["sample_id"])
            if sample_id in ref_paths and sample_id in selected_paths_by_step[int(step)]:
                a, f = path_drift(selected_paths_by_step[int(step)][sample_id], ref_paths[sample_id])
                drift_ade.append(a)
                drift_fde.append(f)
        metrics = {
            "ade_gt_m": summarize([r["ade_gt_m"] for r in recs]),
            "fde_gt_m": summarize([r["fde_gt_m"] for r in recs]),
            "minade6_gt_m": summarize([r["minade6_gt_m"] for r in recs]),
            "minfde6_gt_m": summarize([r["minfde6_gt_m"] for r in recs]),
            "elapsed_ms": summarize([r["elapsed_ms"] for r in recs]),
            "selected_path_drift_vs_ref_m": summarize(drift_ade),
            "selected_final_drift_vs_ref_m": summarize(drift_fde),
        }
        step_summaries[str(step)] = {"count": len(recs), "metrics": metrics}
        ade = metrics["ade_gt_m"]["mean"]
        minade = metrics["minade6_gt_m"]["mean"]
        table.append(
            {
                "step": int(step),
                "ade": ade,
                "fde": metrics["fde_gt_m"]["mean"],
                "minade6": minade,
                "minfde6": metrics["minfde6_gt_m"]["mean"],
                "ade_delta_vs_ref": None if ade is None or ref_ade is None else float(ade - ref_ade),
                "minade6_delta_vs_ref": None if minade is None or ref_minade is None else float(minade - ref_minade),
                "drift_vs_ref": metrics["selected_path_drift_vs_ref_m"]["mean"],
                "final_drift_vs_ref": metrics["selected_final_drift_vs_ref_m"]["mean"],
                "latency_ms": metrics["elapsed_ms"]["mean"],
            }
        )

    summary = {
        "event": "teacher10b_step_sweep_done",
        "settings": settings,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "step_summaries": step_summaries,
        "table": table,
        "rows_jsonl": str(rows_path),
        "report_md": str(report_path),
    }
    write_json(summary_path, summary)

    lines = [
        "# Alpamayo-1.5-10B Flow-Matching Step Sweep",
        "",
        f"- samples: `{len(rows)}`",
        f"- paths/sample: `{int(args.eval_num_paths)}`",
        f"- temperature: `{float(args.eval_temperature)}`",
        f"- selection: `{args.eval_selection_method}`",
        f"- reference step: `{ref_step}`",
        "",
        "| steps | ADE | FDE | minADE6 | minFDE6 | dADE vs ref | dminADE6 vs ref | path drift vs ref | final drift vs ref | ms/sample |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        def fmt(value: Any, digits: int = 4) -> str:
            return "-" if value is None else f"{float(value):.{digits}f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["step"]),
                    fmt(row["ade"]),
                    fmt(row["fde"]),
                    fmt(row["minade6"]),
                    fmt(row["minfde6"]),
                    fmt(row["ade_delta_vs_ref"]),
                    fmt(row["minade6_delta_vs_ref"]),
                    fmt(row["drift_vs_ref"]),
                    fmt(row["final_drift_vs_ref"]),
                    fmt(row["latency_ms"], 2),
                ]
            )
            + " |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "teacher10b_sweep_done", "summary": str(summary_path), "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
