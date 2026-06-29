#!/usr/bin/env python3
"""Run the 4-model planning benchmark requested for FLEX deployment.

Models:
  1. public Alpamayo 1.5 10B
  2. no-FLEX student backbone + AE28
  3. FLEX K512 student backbone + AE28
  4. FLEX K512 student backbone + AE14

For every model this writes per-sample prediction NPZs and JSONL rows with:
  - ADE/FDE of the deployable selected path vs GT
  - minADE6/minFDE6 vs GT
  - student models: selected/minADE6 vs the 10B selected path when available
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
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


DEFAULT_CORPUS = PROJECT_ROOT / "data/corpus/benchmark_semantic_val_cap50_seed42.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs/benchmarks/semantic_val806_4models_20260612"
DEFAULT_TEACHER = SUKIM_ROOT / "base_weights/Alpamayo-1.5-10B"
DEFAULT_STUDENT_NOFLEX = (
    PROJECT_ROOT
    / "outputs/checkpoints/no_nav_camera_labeled_official_full444k"
    / "no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
)
DEFAULT_STUDENT_FLEX = PROJECT_ROOT / "outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final"
DEFAULT_AE28_NOFLEX = PROJECT_ROOT / "outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/best.pt"
DEFAULT_AE28_FLEX = PROJECT_ROOT / "outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt"
DEFAULT_AE14_FLEX = PROJECT_ROOT / "outputs/action_expert/ae14_from_ae28_10step/best.pt"
DEFAULT_AE14_SOURCE = PROJECT_ROOT / "outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt"


MODEL_CONFIGS = {
    "teacher10b": {
        "label": "Alpamayo-1.5-10B",
        "kind": "teacher",
        "checkpoint": DEFAULT_TEACHER,
        "requires_10b_ref": False,
    },
    "student_noflex_ae28": {
        "label": "Student-2B-NoFLEX-AE28",
        "kind": "student",
        "student_checkpoint": DEFAULT_STUDENT_NOFLEX,
        "ae_checkpoint": DEFAULT_AE28_NOFLEX,
        "compressed_layers": 28,
        "ae_init_mode": "student_backbone_init",
        "init_ae_source_checkpoint": "",
        "has_flex": False,
        "inference_steps_key": "default",
        "requires_10b_ref": True,
    },
    "student_flex_ae28": {
        "label": "Student-2B-FLEXK512-AE28",
        "kind": "student",
        "student_checkpoint": DEFAULT_STUDENT_FLEX,
        "ae_checkpoint": DEFAULT_AE28_FLEX,
        "compressed_layers": 28,
        "ae_init_mode": "student_backbone_init",
        "init_ae_source_checkpoint": "",
        "has_flex": True,
        "inference_steps_key": "default",
        "requires_10b_ref": True,
    },
    "student_flex_ae14": {
        "label": "Student-2B-FLEXK512-AE14",
        "kind": "student",
        "student_checkpoint": DEFAULT_STUDENT_FLEX,
        "ae_checkpoint": DEFAULT_AE14_FLEX,
        "compressed_layers": 14,
        "ae_init_mode": "ae_checkpoint_compressed",
        "init_ae_source_checkpoint": DEFAULT_AE14_SOURCE,
        "has_flex": True,
        "inference_steps_key": "ae14",
        "requires_10b_ref": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--model",
        choices=("all", *MODEL_CONFIGS.keys()),
        default="all",
        help="Run one model or the full sequence. Full sequence runs teacher10b first.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=0, help="0 uses all rows in the benchmark JSONL.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--student-batch-size", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="flash_attention_2")
    parser.add_argument("--eval-num-paths", type=int, default=6)
    parser.add_argument("--eval-temperature", type=float, default=0.85)
    parser.add_argument("--eval-selection-method", choices=("single", "oracle_best", "medoid", "mean_traj"), default="mean_traj")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=192)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-top-k", type=int, default=0)
    parser.add_argument("--student-max-new-tokens", type=int, default=160)
    parser.add_argument("--default-inference-steps", type=int, default=10)
    parser.add_argument("--ae14-inference-steps", type=int, default=4)
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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def category(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("semantic_scene_category") or "unknown")


def safe_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "_").replace("\\", "_")


def squeeze_path(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :3]


def squeeze_paths(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    return arr[:, :, :3]


def load_gt_xyz(row: dict[str, Any]) -> np.ndarray:
    sample_dir = Path(str((row.get("input") or {}).get("materialized_sample_path")))
    candidates = [
        sample_dir / "ego/ego_future_xyz.npy",
        sample_dir / "ego/future_xyz.npy",
    ]
    for path in candidates:
        if path.exists():
            return squeeze_path(np.load(path))
    raw = row.get("hard_target") or {}
    path_raw = raw.get("ego_future_xyz_path") or raw.get("future_xyz_path")
    if path_raw and Path(path_raw).exists():
        return squeeze_path(np.load(path_raw))
    raise FileNotFoundError(f"Cannot find GT future xyz for {row.get('sample_id')}")


def ade_fde(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = squeeze_path(pred)
    target = squeeze_path(target)
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def path_len(path: np.ndarray) -> float:
    path = squeeze_path(path)
    if int(path.shape[0]) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=-1).sum())


def medoid_index(paths: np.ndarray) -> int:
    diff = paths[:, None, :, :2] - paths[None, :, :, :2]
    dist = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return int(np.argmin(dist.sum(axis=1)))


def select_path(paths: np.ndarray, ades: list[float], method: str) -> tuple[np.ndarray, int | None]:
    if method == "single":
        return paths[0], 0
    if method == "oracle_best":
        idx = int(np.nanargmin(np.asarray(ades, dtype=np.float64)))
        return paths[idx], idx
    if method == "medoid":
        idx = medoid_index(paths)
        return paths[idx], idx
    if method == "mean_traj":
        return paths.mean(axis=0), None
    raise ValueError(f"Unknown selection method: {method}")


def summarize(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": float(clean.mean()),
        "p50": float(np.percentile(clean, 50)),
        "p95": float(np.percentile(clean, 95)),
    }


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


def read_rows(path: Path, split: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if split and str(row.get("split")) != str(split):
            continue
        rows.append(row)
        if int(limit) > 0 and len(rows) >= int(limit):
            break
    if not rows:
        raise RuntimeError(f"No rows selected from {path}")
    return rows


def npz_path(output_dir: Path, model_key: str, sample_id: str) -> Path:
    return output_dir / "predictions" / model_key / f"{safe_id(sample_id)}.npz"


def write_prediction_npz(
    *,
    output_dir: Path,
    model_key: str,
    sample_id: str,
    paths: np.ndarray,
    selected_path: np.ndarray,
    target_gt: np.ndarray,
    selected_path_idx: int | None,
) -> str:
    path = npz_path(output_dir, model_key, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=np.asarray(paths, dtype=np.float32),
        selected_path=np.asarray(selected_path, dtype=np.float32),
        target_gt=np.asarray(target_gt, dtype=np.float32),
        selected_path_idx=np.asarray([-1 if selected_path_idx is None else selected_path_idx], dtype=np.int32),
    )
    return str(path)


def load_10b_selected(output_dir: Path, sample_id: str) -> np.ndarray | None:
    path = npz_path(output_dir, "teacher10b", sample_id)
    if not path.exists():
        return None
    with np.load(path) as data:
        return np.asarray(data["selected_path"], dtype=np.float32)


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def common_record(
    *,
    model_key: str,
    model_label: str,
    row: dict[str, Any],
    paths: np.ndarray,
    selected_path: np.ndarray,
    selected_path_idx: int | None,
    target_gt: np.ndarray,
    output_dir: Path,
    tenb_target: np.ndarray | None,
    elapsed_ms: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = squeeze_paths(paths)
    target_gt = squeeze_path(target_gt)
    selected_path = squeeze_path(selected_path)
    ade_all: list[float] = []
    fde_all: list[float] = []
    for path in paths:
        ade, fde = ade_fde(path, target_gt)
        ade_all.append(ade)
        fde_all.append(fde)
    best_idx = int(np.nanargmin(np.asarray(ade_all, dtype=np.float64)))
    ade, fde = ade_fde(selected_path, target_gt)
    record: dict[str, Any] = {
        "model_key": model_key,
        "model_label": model_label,
        "sample_id": str(row.get("sample_id")),
        "category": category(row),
        "ade_gt_m": float(ade),
        "fde_gt_m": float(fde),
        "minade6_gt_m": float(ade_all[best_idx]),
        "minfde6_gt_m": float(fde_all[best_idx]),
        "best_path_idx_gt": int(best_idx),
        "selected_path_idx": selected_path_idx,
        "path_ade_gt_m": [float(v) for v in ade_all],
        "selected_path_length_m": path_len(selected_path),
        "target_gt_path_length_m": path_len(target_gt),
        "elapsed_ms": elapsed_ms,
        "prediction_npz": write_prediction_npz(
            output_dir=output_dir,
            model_key=model_key,
            sample_id=str(row.get("sample_id")),
            paths=paths,
            selected_path=selected_path,
            target_gt=target_gt,
            selected_path_idx=selected_path_idx,
        ),
    }
    if tenb_target is not None:
        t_ade, t_fde = ade_fde(selected_path, tenb_target)
        t_ades: list[float] = []
        t_fdes: list[float] = []
        for path in paths:
            a, f = ade_fde(path, tenb_target)
            t_ades.append(a)
            t_fdes.append(f)
        t_best = int(np.nanargmin(np.asarray(t_ades, dtype=np.float64)))
        record.update(
            {
                "ade_10b_m": float(t_ade),
                "fde_10b_m": float(t_fde),
                "minade6_10b_m": float(t_ades[t_best]),
                "minfde6_10b_m": float(t_fdes[t_best]),
                "best_path_idx_10b": int(t_best),
            }
        )
    if extra:
        record.update(extra)
    return record


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "ade_gt_m",
        "fde_gt_m",
        "minade6_gt_m",
        "minfde6_gt_m",
        "ade_10b_m",
        "fde_10b_m",
        "minade6_10b_m",
        "minfde6_10b_m",
        "elapsed_ms",
    )
    return {
        "count": len(rows),
        "category_counts": dict(sorted(Counter(row.get("category", "unknown") for row in rows).items())),
        "metrics": {key: summarize([float(row[key]) for row in rows if row.get(key) is not None]) for key in keys},
    }


def run_teacher10b(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (
        load_materialized_samples,
        load_model_and_processor,
        run_request_batch,
        torch_dtype_from_name,
    )

    cfg = MODEL_CONFIGS["teacher10b"]
    model_key = "teacher10b"
    out_dir = args.output_dir / model_key
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    rows_path.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "teacher_load_start", "checkpoint": str(cfg["checkpoint"])}), flush=True)
    model, processor, model_config, config_path, runtime_support_path = load_model_and_processor(
        checkpoint_path=Path(cfg["checkpoint"]),
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
    print(json.dumps({"event": "teacher_load_done", "config": str(config_path)}), flush=True)

    records: list[dict[str, Any]] = []
    started_model = time.perf_counter()
    for batch_index in range(0, len(rows), int(args.batch_size)):
        batch_rows = rows[batch_index : batch_index + int(args.batch_size)]
        sample_dirs = [Path(str((row.get("input") or {}).get("materialized_sample_path"))) for row in batch_rows]
        samples = load_materialized_samples(sample_dirs, int(args.io_workers))
        outputs = run_request_batch(
            model=model,
            processor=processor,
            samples=samples,
            device=str(args.device),
            decoding_mode="sample",
            top_p=float(args.teacher_top_p),
            top_k=int(args.teacher_top_k),
            temperature=float(args.eval_temperature),
            num_traj_samples=int(args.eval_num_paths),
            max_generation_length=int(args.teacher_max_new_tokens),
            seed=int(args.seed) + int(batch_index),
            write_text_artifacts=False,
            text_top_k=0,
        )
        for row, out in zip(batch_rows, outputs, strict=True):
            target_gt = load_gt_xyz(row)
            paths = squeeze_paths(out["pred_xyz"])
            path_ades = [ade_fde(path, target_gt)[0] for path in paths]
            selected_path, selected_idx = select_path(paths, path_ades, str(args.eval_selection_method))
            cot_texts = flatten_texts((out.get("extra") or {}).get("cot"))
            extra = {
                "cot_preview": cot_texts[0][:240] if cot_texts else "",
                "cot_texts": cot_texts[: int(args.eval_num_paths)],
                "batch_elapsed_sec": out.get("batch_elapsed_sec"),
            }
            record = common_record(
                model_key=model_key,
                model_label=str(cfg["label"]),
                row=row,
                paths=paths,
                selected_path=selected_path,
                selected_path_idx=selected_idx,
                target_gt=target_gt,
                output_dir=args.output_dir,
                tenb_target=None,
                elapsed_ms=float(out.get("elapsed_sec") or 0.0) * 1000.0,
                extra=extra,
            )
            append_row(rows_path, record)
            records.append(record)
        print(
            json.dumps(
                {
                    "event": "model_progress",
                    "model": model_key,
                    "done": min(batch_index + int(args.batch_size), len(rows)),
                    "total": len(rows),
                    "ade_gt_so_far": summarize([r["ade_gt_m"] for r in records])["mean"],
                    "minade6_gt_so_far": summarize([r["minade6_gt_m"] for r in records])["mean"],
                }
            ),
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "event": "model_done",
        "model_key": model_key,
        "model_label": str(cfg["label"]),
        "rows_jsonl": str(rows_path),
        "elapsed_sec": round(time.perf_counter() - started_model, 3),
        "settings": settings_dict(args),
        **summarize_model(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "model_done", "model": model_key, "summary": str(summary_path)}), flush=True)
    model.cpu()
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def student_args(args: argparse.Namespace, cfg: dict[str, Any]) -> SimpleNamespace:
    has_flex = bool(cfg["has_flex"])
    return SimpleNamespace(
        student_checkpoint_dir=Path(cfg["student_checkpoint"]),
        corpus_jsonl=Path(args.corpus_jsonl),
        teacher_checkpoint_path=DEFAULT_TEACHER,
        student_dtype=str(args.dtype),
        ae_dtype=str(args.dtype),
        device=str(args.device),
        student_model=str(SUKIM_ROOT / "base_weights/cosmos-reason-2b"),
        ae_init_mode=str(cfg["ae_init_mode"]),
        init_ae_source_checkpoint=str(cfg.get("init_ae_source_checkpoint") or ""),
        attn_implementation=str(args.attn_implementation),
        disable_student_deepstack=False,
        qat_quantization="",
        qat_calib_samples=256,
        num_samples=len(list(iter_jsonl(args.corpus_jsonl))),
        val_samples=0,
        val_fraction=0.0,
        split_seed=None,
        split_cache_json=None,
        split=str(args.split),
        split_scan_all=True,
        compressed_layers=int(cfg["compressed_layers"]),
        mapping="linspace_round",
        prefix_mode="student_free",
        preserve_flex_positions=has_flex,
        flex_selection_strategy="uniform" if has_flex else "first",
        flex_scene_deepstack=has_flex,
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
    )


def run_student_model(args: argparse.Namespace, rows: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    ae = load_module(PROJECT_ROOT / "scripts/84_train_student_ae28_official.py", f"ae84_{model_key}")
    cfg = MODEL_CONFIGS[model_key]
    out_dir = args.output_dir / model_key
    rows_path = out_dir / "rows.jsonl"
    summary_path = out_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    rows_path.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ae_args = student_args(args, cfg)
    print(
        json.dumps(
            {
                "event": "student_load_start",
                "model": model_key,
                "student_checkpoint": str(ae_args.student_checkpoint_dir),
                "ae_checkpoint": str(cfg["ae_checkpoint"]),
                "ae_init_mode": str(ae_args.ae_init_mode),
                "compressed_layers": int(ae_args.compressed_layers),
            }
        ),
        flush=True,
    )
    student, tokenizer, processor, _base = ae.load_student(ae_args)
    teacher_model, _, _, _, _ = ae.load_model_and_processor(
        checkpoint_path=ae_args.teacher_checkpoint_path,
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
    payload = ae.load_bundle_checkpoint(Path(cfg["ae_checkpoint"]), bundle=bundle)
    bundle = bundle.to(device=ae_args.device, dtype=ae.torch_dtype_from_name(ae_args.ae_dtype)).eval()
    expert_n = int(bundle.expert.config.num_hidden_layers)
    kv_layer_indices = selected_layers if expert_n < 28 else None
    inference_steps = int(args.ae14_inference_steps if cfg["inference_steps_key"] == "ae14" else args.default_inference_steps)
    print(
        json.dumps(
            {
                "event": "student_load_done",
                "model": model_key,
                "selected_layers": selected_layers,
                "payload_step": payload.get("step"),
                "inference_steps": inference_steps,
            }
        ),
        flush=True,
    )

    items = [{"sample_id": str(row["sample_id"]), "row": row} for row in rows]
    records: list[dict[str, Any]] = []
    started_model = time.perf_counter()
    batch_size = int(args.student_batch_size)
    eval_seed_base = int(args.seed) + 1000
    for batch_start in range(0, len(items), batch_size):
        batch_items = items[batch_start : batch_start + batch_size]
        batch_started = time.perf_counter()
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
        repeated = ae.repeat_eval_batch_for_paths(batch, int(args.eval_num_paths))
        pred = ae.sample_paths(
            bundle=bundle,
            teacher_model=teacher_model,
            batch=repeated,
            seed=eval_seed_base + batch_start,
            device=torch.device(args.device),
            inference_steps=inference_steps,
            temperature=float(args.eval_temperature),
            kv_layer_indices=kv_layer_indices,
        )
        pred_xyz = np.asarray(pred["pred_xyz"], dtype=np.float32).reshape(
            n_batch, int(args.eval_num_paths), *np.asarray(pred["pred_xyz"]).shape[1:]
        )
        batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000.0 / max(n_batch, 1)
        for row_index, sample_id in enumerate(sample_ids):
            row = batch_items[row_index]["row"]
            paths = squeeze_paths(pred_xyz[row_index])
            target_gt = squeeze_path(target_xyz[row_index])
            path_ades = [ade_fde(path, target_gt)[0] for path in paths]
            selected_path, selected_idx = select_path(paths, path_ades, str(args.eval_selection_method))
            tenb_target = load_10b_selected(args.output_dir, sample_id)
            generated_texts = list(batch.get("generated_texts") or [])
            generated_text = str(generated_texts[row_index]) if row_index < len(generated_texts) else ""
            record = common_record(
                model_key=model_key,
                model_label=str(cfg["label"]),
                row=row,
                paths=paths,
                selected_path=selected_path,
                selected_path_idx=selected_idx,
                target_gt=target_gt,
                output_dir=args.output_dir,
                tenb_target=tenb_target,
                elapsed_ms=batch_elapsed_ms,
                extra={
                    "inference_steps": inference_steps,
                    "kv_selected_layers": selected_layers if kv_layer_indices is not None else None,
                    "traj_start_hit_rate_batch": batch.get("traj_start_hit_rate"),
                    "generated_text_preview_batch0": batch.get("generated_text_preview"),
                    "generated_text": generated_text[:2000],
                    "cot_preview": generated_text.split("<|cot_end|>", 1)[0][:240],
                },
            )
            append_row(rows_path, record)
            records.append(record)
        print(
            json.dumps(
                {
                    "event": "model_progress",
                    "model": model_key,
                    "done": min(batch_start + batch_size, len(items)),
                    "total": len(items),
                    "ade_gt_so_far": summarize([r["ade_gt_m"] for r in records])["mean"],
                    "minade6_gt_so_far": summarize([r["minade6_gt_m"] for r in records])["mean"],
                    "ade_10b_so_far": summarize([r["ade_10b_m"] for r in records if r.get("ade_10b_m") is not None])["mean"],
                }
            ),
            flush=True,
        )
        del batch, repeated, pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "event": "model_done",
        "model_key": model_key,
        "model_label": str(cfg["label"]),
        "rows_jsonl": str(rows_path),
        "elapsed_sec": round(time.perf_counter() - started_model, 3),
        "settings": {**settings_dict(args), "student_inference_steps": inference_steps},
        "checkpoint": {
            "student_checkpoint": str(cfg["student_checkpoint"]),
            "ae_checkpoint": str(cfg["ae_checkpoint"]),
            "ae_payload_step": payload.get("step"),
            "compressed_layers": int(cfg["compressed_layers"]),
            "selected_layers": selected_layers,
        },
        **summarize_model(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "model_done", "model": model_key, "summary": str(summary_path)}), flush=True)

    bundle.cpu()
    student.backbone.cpu()
    teacher_model.cpu()
    del bundle, student, teacher_model, tokenizer, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def settings_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": str(args.split),
        "num_samples_arg": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "student_batch_size": int(args.student_batch_size),
        "eval_num_paths": int(args.eval_num_paths),
        "eval_temperature": float(args.eval_temperature),
        "eval_selection_method": str(args.eval_selection_method),
        "teacher_top_p": float(args.teacher_top_p),
        "teacher_top_k": int(args.teacher_top_k),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation),
        "default_inference_steps": int(args.default_inference_steps),
        "ae14_inference_steps": int(args.ae14_inference_steps),
    }


def write_combined_summary(args: argparse.Namespace, summaries: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> None:
    out = {
        "event": "benchmark_done",
        "output_dir": str(args.output_dir),
        "settings": settings_dict(args),
        "selected_count": len(selected_rows),
        "category_counts": dict(sorted(Counter(category(row) for row in selected_rows).items())),
        "models": summaries,
    }
    path = args.output_dir / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "combined_summary_written", "path": str(path)}), flush=True)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.corpus_jsonl, str(args.split), int(args.num_samples))
    print(
        json.dumps(
            {
                "event": "benchmark_start",
                "output_dir": str(args.output_dir),
                "selected_count": len(rows),
                "category_counts": dict(sorted(Counter(category(row) for row in rows).items())),
                "model": str(args.model),
                "settings": settings_dict(args),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.model == "all":
        model_order = ["teacher10b", "student_noflex_ae28", "student_flex_ae28", "student_flex_ae14"]
    else:
        model_order = [str(args.model)]
        if MODEL_CONFIGS[str(args.model)].get("requires_10b_ref") and not (args.output_dir / "teacher10b/summary.json").exists():
            print(
                json.dumps(
                    {
                        "event": "warning",
                        "message": "10B reference summary is missing; student rows will omit 10B-comparison metrics.",
                        "expected": str(args.output_dir / "teacher10b/summary.json"),
                    }
                ),
                flush=True,
            )

    summaries: list[dict[str, Any]] = []
    for model_key in model_order:
        if MODEL_CONFIGS[model_key]["kind"] == "teacher":
            summaries.append(run_teacher10b(args, rows))
        else:
            summaries.append(run_student_model(args, rows, model_key))
    write_combined_summary(args, summaries, rows)


if __name__ == "__main__":
    main()
