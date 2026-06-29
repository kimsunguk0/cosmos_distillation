#!/usr/bin/env python3
"""Evaluate Alpamayo-1.5 public 10B DeepStack on/off.

Reports two paths on the same corpus rows:
  1) direct VLM 128-discrete-token trajectory decoding
  2) official VLM-rollout -> action expert trajectory sampling
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpamayo1_5 import helper  # noqa: E402
from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (  # noqa: E402
    enforce_generation_mode,
    load_model_and_processor,
    run_request_batch,
    torch_dtype_from_name,
)
from distillation.dataset_prep.scripts.probe_alpamayo15_discrete_traj import (  # noqa: E402
    compute_ade_fde,
    load_materialized_sample,
)
from src.inference.decoding import (  # noqa: E402
    StopOnTrajOnlyEndCriteria,
    TrajOnlyDecodingContract,
    TrajOnlyLogitsProcessor,
)
from src.utils.runtime_paths import remap_external_path  # noqa: E402


DEFAULT_TEACHER = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "flex_heldout256_stage2val_seed42.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "reports" / "public10b_deepstack_ablation_256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--max-new-tokens", type=int, default=129)
    parser.add_argument("--expert-max-generation-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--modes", default="on,off", help="Comma-separated: on,off")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_existing(raw: str | Path | None) -> Path | None:
    value = remap_external_path(raw)
    if value in (None, ""):
        return None
    path = Path(value)
    return path if path.exists() else None


def select_rows(rows: list[dict[str, Any]], *, split: str, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if split and row.get("split") != split:
            continue
        sample_dir = resolve_existing((row.get("input") or {}).get("materialized_sample_path"))
        raw_json = resolve_existing((row.get("teacher_cache") or {}).get("text_raw_json_path"))
        if sample_dir is None or raw_json is None:
            continue
        selected.append(row)
        if len(selected) >= int(limit):
            break
    return selected


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    width = max(int(batch_size), 1)
    for index in range(0, len(items), width):
        yield items[index : index + width]


def load_samples_for_rows(rows: list[dict[str, Any]], io_workers: int) -> list[dict[str, Any]]:
    sample_dirs = [Path(str((row.get("input") or {}).get("materialized_sample_path"))) for row in rows]

    def load_one(item: tuple[dict[str, Any], Path]) -> dict[str, Any]:
        row, sample_dir = item
        sample = load_materialized_sample(sample_dir)
        if bool((row.get("input") or {}).get("nav_available")):
            sample["nav_text"] = (row.get("input") or {}).get("nav_text")
        else:
            sample["nav_text"] = None
        return sample

    if int(io_workers) <= 1 or len(rows) <= 1:
        return [load_one(item) for item in zip(rows, sample_dirs, strict=True)]
    with ThreadPoolExecutor(max_workers=min(int(io_workers), len(rows))) as pool:
        return list(pool.map(load_one, zip(rows, sample_dirs, strict=True)))


def raw_teacher_xyz(row: dict[str, Any]) -> np.ndarray | None:
    path = resolve_existing((row.get("teacher_cache") or {}).get("text_raw_json_path"))
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = (payload.get("results") or [None])[0]
        xyz = np.asarray((result or {}).get("pred_xyz"), dtype=np.float32).reshape(-1, 64, 3)[0]
        return xyz
    except Exception:
        return None


def ade_fde_np(pred: np.ndarray, target: np.ndarray | None) -> tuple[float | None, float | None]:
    if target is None:
        return None, None
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return None, None
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def summarize(values: list[float | None]) -> dict[str, float | None]:
    clean = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": float(clean.mean()),
        "p50": float(np.percentile(clean, 50)),
        "p95": float(np.percentile(clean, 95)),
    }


def deepstack_targets(model: Any) -> list[tuple[str, Any, str]]:
    out: list[tuple[str, Any, str]] = []
    seen: set[tuple[int, str]] = set()

    def add(name: str, obj: Any) -> None:
        if obj is None:
            return
        for attr in ("deepstack_visual_indexes",):
            if hasattr(obj, attr) and (id(obj), attr) not in seen:
                seen.add((id(obj), attr))
                out.append((name, obj, attr))
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            add(f"{name}.config", cfg)
            vision_cfg = getattr(cfg, "vision_config", None)
            if vision_cfg is not None:
                add(f"{name}.config.vision_config", vision_cfg)

    add("model", model)
    add("model.vlm", getattr(model, "vlm", None))
    add("model.vlm.model", getattr(getattr(model, "vlm", None), "model", None))
    add("model.vlm.model.visual", getattr(getattr(getattr(model, "vlm", None), "model", None), "visual", None))
    add(
        "model.vlm.model.model.visual",
        getattr(getattr(getattr(getattr(model, "vlm", None), "model", None), "model", None), "visual", None),
    )
    return out


def disable_deepstack(model: Any) -> list[dict[str, Any]]:
    disabled: list[dict[str, Any]] = []
    for name, obj, attr in deepstack_targets(model):
        old = list(getattr(obj, attr) or [])
        setattr(obj, attr, [])
        disabled.append({"target": f"{name}.{attr}", "old_indexes": old})
    return disabled


def build_traj_only_inputs(model: Any, processor: Any, samples: list[dict[str, Any]], device: str) -> tuple[dict[str, Any], int]:
    messages_batch = []
    hist_xyz = []
    hist_rot = []
    num_traj_token = 48
    hist_placeholder = f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    user_text = f"{hist_placeholder}output the future trajectory."
    for sample in samples:
        frames = sample["image_frames"].flatten(0, 1)
        image_content = helper._build_image_content(  # noqa: SLF001
            frames,
            sample["camera_indices"],
            int(sample.get("metadata", {}).get("config", {}).get("num_frames_per_camera", 4)),
        )
        messages_batch.append(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a driving assistant that generates safe and accurate actions.",
                        }
                    ],
                },
                {"role": "user", "content": image_content + [{"type": "text", "text": user_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": "<|traj_future_start|>"}]},
            ]
        )
        hist_xyz.append(sample["ego_history_xyz"])
        hist_rot.append(sample["ego_history_rot"])

    tokenized = processor.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    input_ids = model.fuse_traj_tokens(
        tokenized["input_ids"],
        {
            "ego_history_xyz": torch.cat(hist_xyz, dim=0),
            "ego_history_rot": torch.cat(hist_rot, dim=0),
        },
    )
    tokenized["input_ids"] = input_ids
    return helper.to_device({"tokenized_data": tokenized}, device)["tokenized_data"], int(input_ids.shape[1])


def run_discrete_batch(
    *,
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    device: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tokenized = build_traj_only_inputs(model, processor, samples, device)[0]
    input_ids = tokenized.pop("input_ids")
    prompt_len = int(input_ids.shape[1])
    token_count = int(model.config.tokens_per_future_traj)
    contract = TrajOnlyDecodingContract.from_tokenizer(
        model.tokenizer,
        prompt_lengths=[prompt_len] * int(input_ids.shape[0]),
        traj_token_count=token_count,
    )
    logits_processor = LogitsProcessorList([TrajOnlyLogitsProcessor(contract)])
    stopping_criteria = StoppingCriteriaList([StopOnTrajOnlyEndCriteria(contract)])
    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.num_return_sequences = 1
    generation_config.top_p = 1.0
    generation_config.top_k = None
    generation_config.temperature = 1.0
    generation_config.max_new_tokens = int(max_new_tokens)
    generation_config.return_dict_in_generate = False
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=next(model.parameters()).dtype,
        enabled=str(device).startswith("cuda") and torch.cuda.is_available(),
    ):
        generated = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            **tokenized,
        )
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    elapsed = time.perf_counter() - started
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    new_ids = sequences[:, prompt_len:]
    body_ids = new_ids[:, :token_count]
    traj_tokens = torch.clamp(
        body_ids.to(torch.long) - int(model.future_token_start_idx),
        min=0,
        max=int(model.traj_tokenizer.vocab_size) - 1,
    )
    hist_xyz = torch.cat([sample["ego_history_xyz"][:, -1] for sample in samples], dim=0).to(device)
    hist_rot = torch.cat([sample["ego_history_rot"][:, -1] for sample in samples], dim=0).to(device)
    with torch.inference_mode():
        pred_xyz, pred_rot, _ = model.traj_tokenizer.decode(hist_xyz=hist_xyz, hist_rot=hist_rot, tokens=traj_tokens)
    per_sample_elapsed = float(elapsed / max(len(samples), 1))
    out = []
    for index, sample in enumerate(samples):
        gt_xyz = sample["ego_future_xyz"][0, 0].to(pred_xyz.device)
        ade_gt, fde_gt = compute_ade_fde(pred_xyz[index], gt_xyz)
        out.append(
            {
                "elapsed_sec": per_sample_elapsed,
                "ade_gt_m": float(ade_gt),
                "fde_gt_m": float(fde_gt),
                "pred_xyz": pred_xyz[index].detach().float().cpu().numpy(),
                "token_count": int(token_count),
                "unique_tokens": int(torch.unique(traj_tokens[index]).numel()),
                "generated_len": int(new_ids.shape[1]),
            }
        )
    return out


def squeeze_path(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :3]


def run_variant(
    *,
    variant: str,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows_path = args.output_dir / f"{variant}_rows.jsonl"
    rows_path.unlink(missing_ok=True)
    metrics: dict[str, list[float | None]] = {
        "discrete_ade_gt_m": [],
        "discrete_fde_gt_m": [],
        "discrete_ade_teacher_m": [],
        "discrete_fde_teacher_m": [],
        "expert_ade_gt_m": [],
        "expert_fde_gt_m": [],
        "expert_ade_teacher_m": [],
        "expert_fde_teacher_m": [],
    }
    started = time.perf_counter()
    processed = 0
    for batch_index, batch_rows in enumerate(batched(rows, int(args.batch_size))):
        samples = load_samples_for_rows(batch_rows, int(args.io_workers))
        discrete = run_discrete_batch(
            model=model,
            processor=processor,
            samples=samples,
            device=str(args.device),
            max_new_tokens=int(args.max_new_tokens),
        )
        expert = run_request_batch(
            model=model,
            processor=processor,
            samples=samples,
            device=str(args.device),
            decoding_mode="greedy",
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            num_traj_samples=1,
            max_generation_length=int(args.expert_max_generation_length),
            seed=int(args.seed) + batch_index,
            write_text_artifacts=False,
            text_top_k=0,
        )
        with rows_path.open("a", encoding="utf-8") as handle:
            for row, sample, drow, erow in zip(batch_rows, samples, discrete, expert, strict=True):
                teacher_xyz = raw_teacher_xyz(row)
                gt_xyz = sample["ego_future_xyz"][0, 0].detach().cpu().numpy()
                discrete_xyz = np.asarray(drow.pop("pred_xyz"), dtype=np.float32)
                expert_xyz = squeeze_path(erow.get("pred_xyz"))
                d_ade_teacher, d_fde_teacher = ade_fde_np(discrete_xyz, teacher_xyz)
                e_ade_gt, e_fde_gt = ade_fde_np(expert_xyz, gt_xyz)
                e_ade_teacher, e_fde_teacher = ade_fde_np(expert_xyz, teacher_xyz)
                rec = {
                    "variant": variant,
                    "sample_id": row.get("sample_id"),
                    "discrete": {
                        **drow,
                        "ade_teacher_m": d_ade_teacher,
                        "fde_teacher_m": d_fde_teacher,
                    },
                    "expert": {
                        "elapsed_sec": erow.get("elapsed_sec"),
                        "ade_gt_m": e_ade_gt,
                        "fde_gt_m": e_fde_gt,
                        "ade_teacher_m": e_ade_teacher,
                        "fde_teacher_m": e_fde_teacher,
                    },
                }
                metrics["discrete_ade_gt_m"].append(float(drow["ade_gt_m"]))
                metrics["discrete_fde_gt_m"].append(float(drow["fde_gt_m"]))
                metrics["discrete_ade_teacher_m"].append(d_ade_teacher)
                metrics["discrete_fde_teacher_m"].append(d_fde_teacher)
                metrics["expert_ade_gt_m"].append(e_ade_gt)
                metrics["expert_fde_gt_m"].append(e_fde_gt)
                metrics["expert_ade_teacher_m"].append(e_ade_teacher)
                metrics["expert_fde_teacher_m"].append(e_fde_teacher)
                handle.write(json.dumps(rec, ensure_ascii=True) + "\n")
                processed += 1
        print(
            json.dumps(
                {
                    "event": "public10b_batch_done",
                    "variant": variant,
                    "batch_index": batch_index,
                    "processed": processed,
                    "total": len(rows),
                }
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "variant": variant,
        "rows_jsonl": str(rows_path),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "count": int(processed),
        "metrics": {key: summarize(value) for key, value in metrics.items()},
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(load_jsonl(args.corpus_jsonl), split=str(args.split), limit=int(args.num_samples))
    if not rows:
        raise SystemExit(f"No usable rows selected from {args.corpus_jsonl}")
    print(json.dumps({"event": "selected_rows", "count": len(rows), "corpus": str(args.corpus_jsonl)}), flush=True)
    model, processor, model_config, config_path, runtime_support_path = load_model_and_processor(
        checkpoint_path=args.checkpoint_path,
        dtype=torch_dtype_from_name(args.dtype),
        device=str(args.device),
        config_json=None,
        runtime_support=None,
        attn_implementation=str(args.attn_implementation),
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
    )
    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    summaries = []
    disabled = None
    for mode in modes:
        if mode == "off" and disabled is None:
            disabled = disable_deepstack(model)
            print(json.dumps({"event": "public10b_deepstack_disabled", "targets": disabled}), flush=True)
        elif mode not in {"on", "off"}:
            raise ValueError(f"Unsupported mode: {mode}")
        summaries.append(run_variant(variant=f"deepstack_{mode}", model=model, processor=processor, rows=rows, args=args))
    summary = {
        "status": "ok",
        "checkpoint_path": str(args.checkpoint_path),
        "config_path": str(config_path),
        "runtime_support": str(runtime_support_path) if runtime_support_path else None,
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": str(args.split),
        "selected_count": len(rows),
        "batch_size": int(args.batch_size),
        "dtype": str(args.dtype),
        "attn_implementation": str(args.attn_implementation),
        "min_pixels": int(model_config.min_pixels or args.min_pixels),
        "max_pixels": int(model_config.max_pixels or args.max_pixels),
        "disabled_deepstack_targets": disabled,
        "variants": summaries,
    }
    out_path = args.output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "summary_written", "path": str(out_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
