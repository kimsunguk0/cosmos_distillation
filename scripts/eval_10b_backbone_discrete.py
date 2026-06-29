#!/usr/bin/env python3
"""Evaluate Alpamayo-1.5-10B VLM discrete trajectory generation without AE.

This is the diagnostic "10B backbone-only" path: run only the VLM autoregressive
decode, force a valid future-trajectory token span after CoT, then decode those
discrete tokens with Alpamayo's trajectory tokenizer. It intentionally does not
call the Alpamayo action expert / diffusion sampler.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, LogitsProcessorList, StoppingCriteriaList


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
PROBE_PATH = PROJECT_ROOT.parent / "dataset_prep" / "scripts" / "probe_alpamayo15_discrete_traj.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location("alpamayo15_probe", PROBE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load probe helpers from {PROBE_PATH}")
probe = importlib.util.module_from_spec(spec)
sys.modules["alpamayo15_probe"] = probe
spec.loader.exec_module(probe)

from alpamayo1_5.models.token_utils import extract_text_tokens, extract_traj_tokens  # noqa: E402
from src.inference.decoding import (  # noqa: E402
    StopOnTrajEndCriteria,
    TrajDecodingContract,
    TrajSpanLogitsProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=PROJECT_ROOT / "data/corpus/benchmark_semantic_val_cap50_seed42.jsonl")
    parser.add_argument("--checkpoint-path", type=Path, default=SUKIM_ROOT / "base_weights/Alpamayo-1.5-10B")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=0, help="0 uses all rows in the JSONL split.")
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], split: str, num_samples: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("split") or "") == split]
    if int(num_samples) > 0:
        selected = selected[: int(num_samples)]
    return selected


def squeeze_path(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :3]


def load_gt_xyz(row: dict[str, Any]) -> np.ndarray:
    sample_dir = Path(str((row.get("input") or {}).get("materialized_sample_path")))
    for rel in ("ego/ego_future_xyz.npy", "ego/future_xyz.npy"):
        path = sample_dir / rel
        if path.exists():
            return squeeze_path(np.load(path))
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


def summarize(values: list[float]) -> dict[str, float | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "p50": None, "p95": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def metric_mean(values: list[float]) -> float | None:
    return summarize(values)["mean"]


def _sample_dir(row: dict[str, Any]) -> Path:
    raw = (row.get("input") or {}).get("materialized_sample_path")
    if not raw:
        raise FileNotFoundError(f"Row is missing materialized_sample_path: {row.get('sample_id')}")
    return Path(str(raw))


def run_one(
    *,
    model: Any,
    model_config: Any,
    processor: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    row_index: int,
) -> dict[str, Any]:
    sample = probe.load_materialized_sample(_sample_dir(row))
    device = str(next(model.parameters()).device)
    model_inputs, prompt_len = probe.build_model_inputs(model, model_config, sample, device, processor=processor)
    tokenized_data = dict(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    samples_per_row = max(int(args.samples_per_row), 1)

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.do_sample = samples_per_row > 1
    generation_config.num_return_sequences = samples_per_row
    generation_config.max_new_tokens = int(args.max_new_tokens)
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    generation_config.temperature = float(args.temperature)
    generation_config.top_p = float(args.top_p)
    generation_config.top_k = None if int(args.top_k) <= 0 else int(args.top_k)

    prompt_lengths = [int(prompt_len)] * samples_per_row
    contract = TrajDecodingContract.from_tokenizer(
        model.tokenizer,
        prompt_lengths=prompt_lengths,
        traj_token_count=int(model.config.tokens_per_future_traj),
    )
    logits_processor = LogitsProcessorList([TrajSpanLogitsProcessor(contract)])
    stopping_criteria = StoppingCriteriaList([StopOnTrajEndCriteria(contract)])

    seed_value = int(args.seed) + int(row_index)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=next(model.parameters()).dtype,
        enabled=device.startswith("cuda"),
    ):
        generated = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            **tokenized_data,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    generated_tokens = generated.sequences[:, prompt_len:]
    traj_token_ids = extract_traj_tokens(
        generated_tokens,
        model.special_token_ids,
        int(model.config.tokens_per_future_traj),
        int(model.future_token_start_idx),
        int(model.traj_tokenizer.vocab_size),
    )
    text_extra = extract_text_tokens(model.tokenizer, generated_tokens)

    hist_xyz = sample["ego_history_xyz"][:, -1].to(next(model.parameters()).device)
    hist_rot = sample["ego_history_rot"][:, -1].to(next(model.parameters()).device)
    hist_xyz_rep = hist_xyz.repeat(samples_per_row, 1, 1)
    hist_rot_rep = hist_rot.repeat(samples_per_row, 1, 1, 1)
    with torch.inference_mode():
        pred_xyz, pred_rot, _ = model.traj_tokenizer.decode(
            hist_xyz=hist_xyz_rep,
            hist_rot=hist_rot_rep,
            tokens=traj_token_ids,
        )

    target_gt = load_gt_xyz(row)
    paths = pred_xyz.detach().cpu().numpy().astype(np.float32)
    rots = pred_rot.detach().cpu().numpy().astype(np.float32)
    path_ades: list[float] = []
    path_fdes: list[float] = []
    for path in paths:
        ade, fde = ade_fde(path, target_gt)
        path_ades.append(ade)
        path_fdes.append(fde)
    best_idx = int(np.nanargmin(np.asarray(path_ades, dtype=np.float64)))
    first_idx = 0
    selected_idx = first_idx if samples_per_row == 1 else best_idx
    selected_path = paths[selected_idx]
    selected_ade = path_ades[selected_idx]
    selected_fde = path_fdes[selected_idx]

    sample_id = str(row.get("sample_id"))
    npz_path = args.output_dir / "predictions" / f"{sample_id}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        pred_xyz=paths,
        pred_rot=rots,
        selected_xyz=selected_path,
        traj_token_ids=traj_token_ids.detach().cpu().numpy().astype(np.int32),
    )

    tokens_for_selected = traj_token_ids[selected_idx].detach().cpu().numpy().astype(np.int32).tolist()
    cot_values = text_extra.get("cot") or []
    return {
        "sample_id": sample_id,
        "category": str((row.get("metadata") or {}).get("semantic_scene_category") or (row.get("weights") or {}).get("semantic_scene_balance_category") or "unknown"),
        "ade_gt_m": float(selected_ade),
        "fde_gt_m": float(selected_fde),
        "minade6_gt_m": float(path_ades[best_idx]),
        "minfde6_gt_m": float(path_fdes[best_idx]),
        "first_ade_gt_m": float(path_ades[first_idx]),
        "first_fde_gt_m": float(path_fdes[first_idx]),
        "best_path_idx_gt": int(best_idx),
        "selected_path_idx": int(selected_idx),
        "path_ade_gt_m": [float(value) for value in path_ades],
        "path_fde_gt_m": [float(value) for value in path_fdes],
        "selected_path_length_m": path_len(selected_path),
        "target_gt_path_length_m": path_len(target_gt),
        "elapsed_ms": float(elapsed_ms),
        "prediction_npz": str(npz_path),
        "cot_preview": str(cot_values[selected_idx] if selected_idx < len(cot_values) else "")[:240],
        "selected_traj_tokens": tokens_for_selected,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    rows = select_rows(iter_jsonl(args.corpus_jsonl), args.split, args.num_samples)
    if not rows:
        raise SystemExit(f"No rows selected from {args.corpus_jsonl} split={args.split!r}")

    dtype = probe.torch_dtype_from_name(args.dtype)
    model, model_config = probe.load_model(args.checkpoint_path, dtype=dtype, device=args.device)
    processor_kwargs: dict[str, Any] = {}
    if model_config.min_pixels is not None:
        processor_kwargs["min_pixels"] = model_config.min_pixels
    if model_config.max_pixels is not None:
        processor_kwargs["max_pixels"] = model_config.max_pixels
    processor = AutoProcessor.from_pretrained(model_config.vlm_name_or_path, **processor_kwargs)
    processor.tokenizer = model.tokenizer
    processor.tokenizer.padding_side = "left"
    model.tokenizer.padding_side = "left"

    rows_path = args.output_dir / "rows.jsonl"
    rows_path.unlink(missing_ok=True)
    records: list[dict[str, Any]] = []
    token_hist: Counter[int] = Counter()
    started_all = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        record = run_one(model=model, model_config=model_config, processor=processor, row=row, args=args, row_index=index)
        records.append(record)
        token_hist.update(int(token) for token in record["selected_traj_tokens"])
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "event": "sample_done",
                    "index": index,
                    "num_samples": len(rows),
                    "sample_id": record["sample_id"],
                    "ade_m": record["ade_gt_m"],
                    "minade6_m": record["minade6_gt_m"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    samples_per_row = max(int(args.samples_per_row), 1)
    total_tokens = max(sum(token_hist.values()), 1)
    summary = {
        "model_key": "teacher10b_backbone_discrete",
        "model_label": "Alpamayo-1.5-10B VLM discrete only, no Action Expert",
        "checkpoint_path": str(args.checkpoint_path),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": str(args.split),
        "num_samples": len(records),
        "samples_per_row": samples_per_row,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "elapsed_sec": round(time.perf_counter() - started_all, 3),
        "rows_jsonl": str(rows_path),
        "category_counts": dict(sorted(Counter(record["category"] for record in records).items())),
        "metrics": {
            "ade_gt_m": summarize([record["ade_gt_m"] for record in records]),
            "fde_gt_m": summarize([record["fde_gt_m"] for record in records]),
            "minade6_gt_m": summarize([record["minade6_gt_m"] for record in records]),
            "minfde6_gt_m": summarize([record["minfde6_gt_m"] for record in records]),
            "first_ade_gt_m": summarize([record["first_ade_gt_m"] for record in records]),
            "first_fde_gt_m": summarize([record["first_fde_gt_m"] for record in records]),
            "elapsed_ms": summarize([record["elapsed_ms"] for record in records]),
        },
        "avg_ade_m": metric_mean([record["ade_gt_m"] for record in records]),
        "avg_fde_m": metric_mean([record["fde_gt_m"] for record in records]),
        "ade@6.4s_m": metric_mean([record["ade_gt_m"] for record in records]) if samples_per_row == 1 else None,
        "minADE6@6.4s_m": metric_mean([record["minade6_gt_m"] for record in records]) if samples_per_row == 6 else None,
        "top_token_histogram": [
            {"token": int(token), "count": int(count), "mass": float(count / total_tokens)}
            for token, count in token_hist.most_common(30)
        ],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(args.summary_json)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
