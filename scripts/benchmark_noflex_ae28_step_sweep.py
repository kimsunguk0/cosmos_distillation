#!/usr/bin/env python3
"""Sweep AE28 flow-matching inference steps for the no-FLEX 2B student.

The benchmark keeps the student prefix, AE checkpoint, sampled path count,
temperature, and seed policy fixed, then changes only the Euler denoising step
count. It reports normal trajectory metrics and path drift versus a reference
step count, usually 10.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
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


DEFAULT_OUT = PROJECT_ROOT / "outputs/benchmarks/noflex_ae28_fm_step_sweep_20260706"
DEFAULT_STUDENT = bench4.DEFAULT_STUDENT_NOFLEX
DEFAULT_AE28 = bench4.DEFAULT_AE28_NOFLEX
DEFAULT_TEACHER = bench4.DEFAULT_TEACHER
DEFAULT_CORPUS = bench4.DEFAULT_CORPUS


def parse_steps(text: str) -> list[int]:
    steps = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"inference step must be positive: {value}")
        steps.append(value)
    if not steps:
        raise ValueError("no inference steps provided")
    return sorted(dict.fromkeys(steps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--ae-checkpoint", type=Path, default=DEFAULT_AE28)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=0, help="0 uses all selected rows.")
    parser.add_argument("--student-batch-size", type=int, default=8)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="flash_attention_2")
    parser.add_argument("--eval-num-paths", type=int, default=6)
    parser.add_argument("--eval-temperature", type=float, default=0.85)
    parser.add_argument("--eval-selection-method", choices=("single", "oracle_best", "medoid", "mean_traj"), default="mean_traj")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-max-new-tokens", type=int, default=160)
    parser.add_argument("--steps", default="1,2,3,4,5,6,8,10,12,16,20")
    parser.add_argument("--reference-step", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summarize(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": float(clean.mean()),
        "p50": float(np.percentile(clean, 50)),
        "p95": float(np.percentile(clean, 95)),
    }


def make_ae_args(args: argparse.Namespace, selected_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        student_checkpoint_dir=Path(args.student_checkpoint_dir),
        corpus_jsonl=Path(args.corpus_jsonl),
        teacher_checkpoint_path=Path(args.teacher_checkpoint_path),
        student_dtype=str(args.dtype),
        ae_dtype=str(args.dtype),
        device=str(args.device),
        student_model=str(SUKIM_ROOT / "base_weights/cosmos-reason-2b"),
        ae_init_mode="student_backbone_init",
        init_ae_source_checkpoint="",
        attn_implementation=str(args.attn_implementation),
        disable_student_deepstack=False,
        qat_quantization="",
        qat_calib_samples=256,
        num_samples=selected_count,
        val_samples=0,
        val_fraction=0.0,
        split_seed=None,
        split_cache_json=None,
        split=str(args.split),
        split_scan_all=True,
        compressed_layers=28,
        mapping="linspace_round",
        prefix_mode="student_free",
        preserve_flex_positions=False,
        flex_selection_strategy="first",
        flex_scene_deepstack=False,
        target_source="gt",
        max_new_tokens=int(args.student_max_new_tokens),
        max_length=4096,
        stage2_attention_mode="official_none",
        seed=int(args.seed),
        teacher_load_device="cpu",
        eval_num_paths=int(args.eval_num_paths),
        eval_temperature=float(args.eval_temperature),
        eval_selection_method=str(args.eval_selection_method),
        eval_seed_mode="fixed",
        eval_vectorize_paths=True,
        eval_path_batch_size=int(args.eval_num_paths),
        eval_batch_size=int(args.student_batch_size),
        eval_cleanup_every=1,
        eval_log_rows=-1,
        io_workers=int(args.io_workers),
    )


def selected_path(paths: np.ndarray, target_gt: np.ndarray, method: str) -> tuple[np.ndarray, int | None, list[float], list[float]]:
    paths = bench4.squeeze_paths(paths)
    target_gt = bench4.squeeze_path(target_gt)
    ades: list[float] = []
    fdes: list[float] = []
    for path in paths:
        ade, fde = bench4.ade_fde(path, target_gt)
        ades.append(float(ade))
        fdes.append(float(fde))
    path, idx = bench4.select_path(paths, ades, method)
    return bench4.squeeze_path(path), idx, ades, fdes


def path_drift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = bench4.squeeze_path(a)
    b = bench4.squeeze_path(b)
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(a[:n, :2] - b[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    steps = parse_steps(args.steps)
    if int(args.reference_step) not in steps:
        steps.append(int(args.reference_step))
        steps = sorted(dict.fromkeys(steps))
    out_dir = Path(args.output_dir)
    summary_path = out_dir / "summary.json"
    rows_path = out_dir / "rows.jsonl"
    report_path = out_dir / "report.md"
    if args.skip_existing and summary_path.exists():
        print(json.dumps({"event": "skip_existing", "summary": str(summary_path)}), flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path.unlink(missing_ok=True)

    rows = bench4.read_rows(Path(args.corpus_jsonl), str(args.split), int(args.num_samples))
    category_counts = dict(sorted(Counter(bench4.category(row) for row in rows).items()))
    settings = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "output_dir": str(out_dir),
        "student_checkpoint_dir": str(args.student_checkpoint_dir),
        "ae_checkpoint": str(args.ae_checkpoint),
        "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
        "split": str(args.split),
        "selected_count": len(rows),
        "category_counts": category_counts,
        "student_batch_size": int(args.student_batch_size),
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
    print(json.dumps({"event": "sweep_start", **settings}, ensure_ascii=False), flush=True)

    ae = load_module(PROJECT_ROOT / "scripts/84_train_student_ae28_official.py", "ae84_noflex_step_sweep")
    ae_args = make_ae_args(args, len(rows))

    print(json.dumps({"event": "student_load_start", "checkpoint": str(args.student_checkpoint_dir)}), flush=True)
    student, tokenizer, processor, _base = ae.load_student(ae_args)
    teacher_model, _, _, _, _ = ae.load_model_and_processor(
        checkpoint_path=Path(args.teacher_checkpoint_path),
        dtype=ae.torch_dtype_from_name(ae_args.ae_dtype),
        device=ae_args.teacher_load_device,
        config_json=None,
        runtime_support=None,
        attn_implementation=ae_args.attn_implementation,
        min_pixels=163840,
        max_pixels=196608,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    ae.force_attention(teacher_model.expert, "sdpa" if ae_args.attn_implementation != "eager" else "eager")
    bundle, selected_layers = ae.build_bundle(teacher_model, ae_args, student=student)
    payload = ae.load_bundle_checkpoint(Path(args.ae_checkpoint), bundle=bundle)
    bundle = bundle.to(device=ae_args.device, dtype=ae.torch_dtype_from_name(ae_args.ae_dtype)).eval()
    print(
        json.dumps(
            {
                "event": "student_load_done",
                "payload_step": payload.get("step"),
                "selected_layers": selected_layers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    device = torch.device(args.device)
    items = [{"sample_id": str(row["sample_id"]), "row": row} for row in rows]
    metric_rows_by_step: dict[int, list[dict[str, Any]]] = {step: [] for step in steps}
    selected_paths_by_step: dict[int, dict[str, np.ndarray]] = {step: {} for step in steps}
    started = time.perf_counter()
    batch_size = int(args.student_batch_size)
    eval_seed_base = int(args.seed) + 1000
    num_paths = int(args.eval_num_paths)

    for batch_start in range(0, len(items), batch_size):
        batch_items = items[batch_start : batch_start + batch_size]
        batch = ae.build_batch(
            args=ae_args,
            student=student,
            student_processor=processor,
            student_tokenizer=tokenizer,
            teacher_model=teacher_model,
            batch_items=batch_items,
        )
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        sample_ids = list(batch["sample_ids"])
        n_batch = len(sample_ids)
        path_seed = eval_seed_base + batch_start
        generated_texts = list(batch.get("generated_texts") or [])
        generated_preview = str(batch.get("generated_text_preview") or "")
        for step in steps:
            step_started = time.perf_counter()
            repeated = ae.repeat_eval_batch_for_paths(batch, num_paths)
            pred = ae.sample_paths(
                bundle=bundle,
                teacher_model=teacher_model,
                batch=repeated,
                seed=path_seed,
                device=device,
                inference_steps=int(step),
                temperature=float(args.eval_temperature),
                kv_layer_indices=None,
            )
            pred_xyz = np.asarray(pred["pred_xyz"], dtype=np.float32).reshape(
                n_batch, num_paths, *np.asarray(pred["pred_xyz"]).shape[1:]
            )
            elapsed_ms = (time.perf_counter() - step_started) * 1000.0 / max(n_batch, 1)
            for row_index, sample_id in enumerate(sample_ids):
                row = batch_items[row_index]["row"]
                paths = bench4.squeeze_paths(pred_xyz[row_index])
                target_gt = bench4.squeeze_path(target_xyz[row_index])
                chosen, chosen_idx, ades, fdes = selected_path(paths, target_gt, str(args.eval_selection_method))
                best_idx = int(np.nanargmin(np.asarray(ades, dtype=np.float64)))
                ade, fde = bench4.ade_fde(chosen, target_gt)
                rec = {
                    "sample_id": sample_id,
                    "category": bench4.category(row),
                    "inference_steps": int(step),
                    "ade_gt_m": float(ade),
                    "fde_gt_m": float(fde),
                    "minade6_gt_m": float(ades[best_idx]),
                    "minfde6_gt_m": float(fdes[best_idx]),
                    "best_path_idx_gt": int(best_idx),
                    "selected_path_idx": chosen_idx,
                    "path_ade_gt_m": [float(v) for v in ades],
                    "elapsed_ms": float(elapsed_ms),
                    "traj_start_hit_rate_batch": batch.get("traj_start_hit_rate"),
                    "generated_text_preview_batch0": generated_preview,
                    "generated_text": str(generated_texts[row_index])[:2000] if row_index < len(generated_texts) else "",
                }
                append_jsonl(rows_path, rec)
                metric_rows_by_step[int(step)].append(rec)
                selected_paths_by_step[int(step)][sample_id] = chosen.astype(np.float32, copy=True)
            del pred, repeated
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del batch
        done = min(batch_start + batch_size, len(items))
        progress = {
            "event": "sweep_progress",
            "done": done,
            "total": len(items),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        for step in steps:
            recs = metric_rows_by_step[int(step)]
            progress[f"ade_s{step}"] = summarize([r["ade_gt_m"] for r in recs])["mean"]
            progress[f"minade6_s{step}"] = summarize([r["minade6_gt_m"] for r in recs])["mean"]
        print(json.dumps(progress, ensure_ascii=False), flush=True)

    ref_step = int(args.reference_step)
    ref_paths = selected_paths_by_step[ref_step]
    step_summaries: dict[str, Any] = {}
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
                rec["selected_path_drift_vs_ref_m"] = a
                rec["selected_final_drift_vs_ref_m"] = f
        step_summaries[str(step)] = {
            "count": len(recs),
            "metrics": {
                "ade_gt_m": summarize([r["ade_gt_m"] for r in recs]),
                "fde_gt_m": summarize([r["fde_gt_m"] for r in recs]),
                "minade6_gt_m": summarize([r["minade6_gt_m"] for r in recs]),
                "minfde6_gt_m": summarize([r["minfde6_gt_m"] for r in recs]),
                "elapsed_ms": summarize([r["elapsed_ms"] for r in recs]),
                "selected_path_drift_vs_ref_m": summarize(drift_ade),
                "selected_final_drift_vs_ref_m": summarize(drift_fde),
                "severe_ade_gt_gt5_rate": float(sum(1 for r in recs if float(r["ade_gt_m"]) > 5.0) / max(len(recs), 1)),
            },
            "category_counts": dict(sorted(Counter(r["category"] for r in recs).items())),
        }

    ref_ade = step_summaries[str(ref_step)]["metrics"]["ade_gt_m"]["mean"]
    ref_minade = step_summaries[str(ref_step)]["metrics"]["minade6_gt_m"]["mean"]
    rows_table = []
    stable_candidates = []
    for step in steps:
        data = step_summaries[str(step)]["metrics"]
        ade = data["ade_gt_m"]["mean"]
        minade = data["minade6_gt_m"]["mean"]
        drift = data["selected_path_drift_vs_ref_m"]["mean"]
        final_drift = data["selected_final_drift_vs_ref_m"]["mean"]
        ade_delta = None if ade is None or ref_ade is None else float(ade - ref_ade)
        minade_delta = None if minade is None or ref_minade is None else float(minade - ref_minade)
        stable = (
            ade_delta is not None
            and minade_delta is not None
            and drift is not None
            and ade_delta <= 0.05
            and minade_delta <= 0.05
            and drift <= 0.25
        )
        if stable:
            stable_candidates.append(int(step))
        rows_table.append(
            {
                "step": int(step),
                "ade": ade,
                "fde": data["fde_gt_m"]["mean"],
                "minade6": minade,
                "minfde6": data["minfde6_gt_m"]["mean"],
                "ade_delta_vs_ref": ade_delta,
                "minade6_delta_vs_ref": minade_delta,
                "drift_vs_ref": drift,
                "final_drift_vs_ref": final_drift,
                "latency_ms": data["elapsed_ms"]["mean"],
                "severe_rate": data["severe_ade_gt_gt5_rate"],
                "stable_by_rule": stable,
            }
        )
    recommended_step = min(stable_candidates) if stable_candidates else ref_step

    summary = {
        "event": "noflex_ae28_step_sweep_done",
        "settings": settings,
        "checkpoint": {
            "student_checkpoint": str(args.student_checkpoint_dir),
            "ae_checkpoint": str(args.ae_checkpoint),
            "ae_payload_step": payload.get("step"),
            "compressed_layers": 28,
            "selected_layers": selected_layers,
        },
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "step_summaries": step_summaries,
        "table": rows_table,
        "stability_rule": {
            "reference_step": ref_step,
            "max_ade_delta_m": 0.05,
            "max_minade6_delta_m": 0.05,
            "max_selected_path_drift_m": 0.25,
        },
        "recommended_min_stable_step_by_rule": recommended_step,
        "rows_jsonl": str(rows_path),
        "report_md": str(report_path),
    }
    write_json(summary_path, summary)

    lines = [
        "# No-FLEX 2B AE28 Flow-Matching Step Sweep",
        "",
        f"- samples: `{len(rows)}`",
        f"- paths/sample: `{num_paths}`",
        f"- temperature: `{float(args.eval_temperature)}`",
        f"- selection: `{args.eval_selection_method}`",
        f"- reference step: `{ref_step}`",
        f"- recommended min stable step by rule: `{recommended_step}`",
        "",
        "| steps | ADE | FDE | minADE6 | minFDE6 | dADE vs ref | dminADE6 vs ref | path drift vs ref | final drift vs ref | ms/sample | ADE>5m | stable |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows_table:
        def fmt(value: Any, digits: int = 4) -> str:
            if value is None:
                return "-"
            return f"{float(value):.{digits}f}"
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
                    fmt(row["severe_rate"], 3),
                    "Y" if row["stable_by_rule"] else "N",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Stability rule: dADE <= 0.05 m, dminADE6 <= 0.05 m, and selected-path mean drift vs reference <= 0.25 m.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "sweep_done", "summary": str(summary_path), "report": str(report_path)}), flush=True)

    bundle.cpu()
    student.backbone.cpu()
    teacher_model.cpu()
    del bundle, student, teacher_model, tokenizer, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
