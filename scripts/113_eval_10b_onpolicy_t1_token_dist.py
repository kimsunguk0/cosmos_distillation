#!/usr/bin/env python3
"""Measure Alpamayo-1.5-10B on-policy T=1 trajectory-token distributions.

This runs the 10B VLM backbone freely with temperature=1.0/top_p=1.0 sampling,
then measures the raw pre-warper output distribution at each generated
trajectory-body token position. It intentionally reads generate().logits rather
than generate().scores, and it normalizes probabilities only over the
trajectory-bin token span defined by TrajDecodingContract.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
PROBE_PATH = PROJECT_ROOT.parent / "dataset_prep" / "scripts" / "probe_alpamayo15_discrete_traj.py"

DEFAULT_CORPUS_JSONL = PROJECT_ROOT / "data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
DEFAULT_CHECKPOINT_PATH = SUKIM_ROOT / "base_weights/Alpamayo-1.5-10B"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs/reports/10b_onpolicy_t1_token_dist_val512.json"
DEFAULT_CUMULATIVE_CUTOFFS = (1, 3, 5, 10, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS_JSONL)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument(
        "--samples-per-row",
        type=int,
        default=1,
        help="Must remain 1 for the on-policy T=1 protocol.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-traj-tokens", type=int, default=128)
    parser.add_argument("--top-n", type=int, default=64, help="Top probabilities retained per step before aggregation.")
    parser.add_argument("--rank-curve-n", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--output-json", "--summary-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic aggregation checks without loading the 10B model.",
    )
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


def sample_dir(row: dict[str, Any]) -> Path:
    raw = (row.get("input") or {}).get("materialized_sample_path")
    if not raw:
        raise FileNotFoundError(f"Row is missing materialized_sample_path: {row.get('sample_id')}")
    return Path(str(raw))


def seed_everything(seed: int, *, include_cuda: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quantiles(values: Iterable[float], qs: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)) -> dict[str, float | None]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return {f"p{int(q * 100):02d}": None for q in qs}
    return {f"p{int(q * 100):02d}": float(np.quantile(arr, q)) for q in qs}


@dataclass
class DistributionAccumulator:
    rank_curve_n: int = 32
    cumulative_cutoffs: tuple[int, ...] = DEFAULT_CUMULATIVE_CUTOFFS
    positions: int = 0
    top_prob_sums: np.ndarray = field(init=False)
    cumulative_sums: dict[int, float] = field(init=False)
    entropy_sum: float = 0.0
    self_agree_count: int = 0
    top1_probs: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rank_curve_n = int(self.rank_curve_n)
        if self.rank_curve_n <= 0:
            raise ValueError("rank_curve_n must be positive")
        self.cumulative_cutoffs = tuple(int(value) for value in self.cumulative_cutoffs)
        if max(self.cumulative_cutoffs) > self.rank_curve_n:
            raise ValueError("cumulative cutoffs cannot exceed rank_curve_n")
        self.top_prob_sums = np.zeros((self.rank_curve_n,), dtype=np.float64)
        self.cumulative_sums = {cutoff: 0.0 for cutoff in self.cumulative_cutoffs}

    def add_step(self, *, top_probs: np.ndarray, entropy_nats: float, sampled_is_argmax: bool) -> None:
        top_probs = np.asarray(top_probs, dtype=np.float64).reshape(-1)
        if top_probs.shape[0] < self.rank_curve_n:
            raise ValueError(f"Need at least {self.rank_curve_n} top probabilities, got {top_probs.shape[0]}")
        used = top_probs[: self.rank_curve_n]
        if np.any(used < -1.0e-8) or np.any(used > 1.0 + 1.0e-8):
            raise ValueError("top probabilities must be in [0, 1]")
        self.positions += 1
        self.top_prob_sums += used
        for cutoff in self.cumulative_cutoffs:
            self.cumulative_sums[cutoff] += float(np.sum(used[:cutoff], dtype=np.float64))
        self.entropy_sum += float(entropy_nats)
        self.self_agree_count += int(bool(sampled_is_argmax))
        self.top1_probs.append(float(used[0]))

    def summary(self) -> dict[str, Any]:
        if self.positions <= 0:
            rank_curve = {f"top_{rank}": None for rank in range(1, self.rank_curve_n + 1)}
            cumulative = {f"top_{cutoff}": None for cutoff in self.cumulative_cutoffs}
            return {
                "rank_curve_top1_to_top32_mean_prob": rank_curve,
                "top1_prob_mean": None,
                "top2_prob_mean": None,
                "top3_prob_mean": None,
                "cumulative_mass_mean": cumulative,
                "entropy_nats_mean": None,
                "self_agreement_acc": None,
                "top1_prob_percentiles": quantiles([]),
                "top1_prob_gt_0_9_rate": None,
                "top1_prob_lt_0_3_rate": None,
            }

        means = self.top_prob_sums / float(self.positions)
        rank_curve = {f"top_{rank}": float(means[rank - 1]) for rank in range(1, self.rank_curve_n + 1)}
        cumulative = {
            f"top_{cutoff}": float(self.cumulative_sums[cutoff] / float(self.positions))
            for cutoff in self.cumulative_cutoffs
        }
        top1_arr = np.asarray(self.top1_probs, dtype=np.float64)
        return {
            "rank_curve_top1_to_top32_mean_prob": rank_curve,
            "top1_prob_mean": float(means[0]),
            "top2_prob_mean": float(means[1]) if self.rank_curve_n >= 2 else None,
            "top3_prob_mean": float(means[2]) if self.rank_curve_n >= 3 else None,
            "cumulative_mass_mean": cumulative,
            "entropy_nats_mean": float(self.entropy_sum / float(self.positions)),
            "self_agreement_acc": float(self.self_agree_count / float(self.positions)),
            "top1_prob_percentiles": quantiles(top1_arr),
            "top1_prob_gt_0_9_rate": float(np.mean(top1_arr > 0.9)),
            "top1_prob_lt_0_3_rate": float(np.mean(top1_arr < 0.3)),
        }


def step_distribution_stats(
    *,
    step_logits: torch.Tensor,
    traj_token_ids: torch.Tensor,
    sampled_token_id: int,
    top_n: int,
) -> dict[str, Any]:
    if step_logits.ndim == 2:
        if int(step_logits.shape[0]) != 1:
            raise ValueError(f"Expected batch size 1 step logits, got shape {tuple(step_logits.shape)}")
        step_logits = step_logits[0]
    if step_logits.ndim != 1:
        raise ValueError(f"Expected rank-1 or rank-2 logits for a step, got shape {tuple(step_logits.shape)}")

    traj_token_ids = traj_token_ids.to(device=step_logits.device, dtype=torch.long)
    traj_logits = step_logits.index_select(0, traj_token_ids).float()
    if int(traj_logits.numel()) <= 0:
        raise ValueError("Empty trajectory token span")
    top_n = min(int(top_n), int(traj_logits.numel()))
    log_probs = torch.log_softmax(traj_logits, dim=-1)
    probs = torch.exp(log_probs)
    top_log_probs, top_local = torch.topk(log_probs, k=top_n, dim=-1)
    top_probs = torch.exp(top_log_probs)
    entropy = -(probs * log_probs).sum()
    argmax_abs = int(traj_token_ids[int(top_local[0].detach().cpu().item())].detach().cpu().item())
    return {
        "top_probs": top_probs.detach().cpu().numpy().astype(np.float64),
        "entropy_nats": float(entropy.detach().cpu().item()),
        "sampled_is_argmax": int(sampled_token_id) == argmax_abs,
        "argmax_token_id": argmax_abs,
    }


def generated_step_logits(generated: Any) -> list[torch.Tensor]:
    logits = getattr(generated, "logits", None)
    if logits is None:
        raise RuntimeError("generate(..., output_logits=True) did not return a .logits field")
    if isinstance(logits, torch.Tensor):
        if logits.ndim == 3:
            return [logits[:, step_index, :] for step_index in range(int(logits.shape[1]))]
        if logits.ndim == 2:
            return [logits]
        raise RuntimeError(f"Unsupported generated.logits tensor shape: {tuple(logits.shape)}")
    return list(logits)


def trajectory_body_steps(
    *,
    generated_new_tokens: torch.Tensor,
    contract: Any,
    max_traj_tokens: int,
) -> list[tuple[int, int, int]]:
    token_ids = [int(value) for value in generated_new_tokens.reshape(-1).detach().cpu().tolist()]
    traj_token_set = set(int(value) for value in contract.traj_token_ids)
    body_steps: list[tuple[int, int, int]] = []
    cot_end_seen = False
    traj_started = False
    for step_index, token_id in enumerate(token_ids):
        if not cot_end_seen:
            if token_id == int(contract.cot_end_id):
                cot_end_seen = True
            continue
        if not traj_started:
            if token_id == int(contract.traj_start_id):
                traj_started = True
            continue
        if token_id == int(contract.traj_end_id):
            break
        if token_id in traj_token_set:
            body_steps.append((step_index, int(token_id), len(body_steps)))
            if len(body_steps) >= int(max_traj_tokens):
                break
    return body_steps


def generation_seed_kwargs(generate_fn: Any, seed: int, device: str) -> dict[str, Any]:
    try:
        signature = inspect.signature(generate_fn)
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, Any] = {}
    if "seed" in signature.parameters:
        kwargs["seed"] = int(seed)
    if "generator" in signature.parameters:
        gen_device = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        kwargs["generator"] = torch.Generator(device=gen_device).manual_seed(int(seed))
    return kwargs


def load_probe_module() -> Any:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("alpamayo15_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load probe helpers from {PROBE_PATH}")
    probe = importlib.util.module_from_spec(spec)
    sys.modules["alpamayo15_probe"] = probe
    spec.loader.exec_module(probe)
    return probe


def build_processor(model: Any, model_config: Any) -> Any:
    from transformers import AutoProcessor

    processor_kwargs: dict[str, Any] = {}
    if model_config.min_pixels is not None:
        processor_kwargs["min_pixels"] = model_config.min_pixels
    if model_config.max_pixels is not None:
        processor_kwargs["max_pixels"] = model_config.max_pixels
    processor = AutoProcessor.from_pretrained(model_config.vlm_name_or_path, **processor_kwargs)
    processor.tokenizer = model.tokenizer
    processor.tokenizer.padding_side = "left"
    model.tokenizer.padding_side = "left"
    return processor


def run_one(
    *,
    model: Any,
    model_config: Any,
    processor: Any,
    probe: Any,
    row: dict[str, Any],
    row_index: int,
    args: argparse.Namespace,
    accumulator: DistributionAccumulator,
    traj_span_meta: dict[str, Any],
) -> dict[str, Any]:
    from transformers import LogitsProcessorList, StoppingCriteriaList

    from src.inference.decoding import StopOnTrajEndCriteria, TrajDecodingContract, TrajSpanLogitsProcessor

    sample = probe.load_materialized_sample(sample_dir(row))
    device = str(next(model.parameters()).device)
    model_inputs, prompt_len = probe.build_model_inputs(model, model_config, sample, device, processor=processor)
    tokenized_data = dict(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = int(args.max_new_tokens)
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    generation_config.temperature = float(args.temperature)
    generation_config.top_p = float(args.top_p)
    generation_config.top_k = int(args.top_k)
    if hasattr(generation_config, "seed"):
        generation_config.seed = int(args.seed) + int(row_index)

    contract = TrajDecodingContract.from_tokenizer(
        model.tokenizer,
        prompt_lengths=[int(prompt_len)],
        traj_token_count=int(model.config.tokens_per_future_traj),
    )
    if not traj_span_meta:
        traj_span_meta.update(
            {
                "contract_class": "src.inference.decoding.TrajDecodingContract",
                "logits_processor": "src.inference.decoding.TrajSpanLogitsProcessor",
                "traj_token_count_from_model_config": int(model.config.tokens_per_future_traj),
                "max_traj_tokens_used_per_sample": int(args.max_traj_tokens),
                "traj_vocab_size": len(contract.traj_token_ids),
                "traj_token_start_id": int(contract.traj_token_ids[0]),
                "traj_token_end_id": int(contract.traj_token_ids[-1]),
                "cot_end_id": int(contract.cot_end_id),
                "traj_start_id": int(contract.traj_start_id),
                "traj_end_id": int(contract.traj_end_id),
                "body_step_selection": (
                    "Generated new tokens after <|cot_end|> and <|traj_future_start|>; "
                    "only ids in contract.traj_token_ids are scored, capped at max_traj_tokens."
                ),
            }
        )
    logits_processor = LogitsProcessorList([TrajSpanLogitsProcessor(contract)])
    stopping_criteria = StoppingCriteriaList([StopOnTrajEndCriteria(contract)])
    traj_token_tensor = torch.as_tensor(contract.traj_token_ids, dtype=torch.long, device=input_ids.device)

    seed_value = int(args.seed) + int(row_index)
    seed_everything(seed_value)
    seed_kwargs = generation_seed_kwargs(model.vlm.generate, seed_value, device)

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
            return_dict_in_generate=True,
            output_logits=True,
            **seed_kwargs,
            **tokenized_data,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not hasattr(generated, "sequences"):
        raise RuntimeError("generate(..., return_dict_in_generate=True) did not return .sequences")
    sequences = generated.sequences
    if int(sequences.shape[0]) != 1:
        raise RuntimeError(f"Expected exactly one generated sequence, got {tuple(sequences.shape)}")
    generated_new = sequences[:, int(prompt_len) :]
    body_steps = trajectory_body_steps(
        generated_new_tokens=generated_new[0],
        contract=contract,
        max_traj_tokens=int(args.max_traj_tokens),
    )
    step_logits_list = generated_step_logits(generated)
    if len(step_logits_list) < int(generated_new.shape[1]):
        raise RuntimeError(
            f"generate returned fewer logits steps ({len(step_logits_list)}) than generated tokens ({generated_new.shape[1]})"
        )

    for step_index, sampled_token_id, _traj_pos in body_steps:
        stats = step_distribution_stats(
            step_logits=step_logits_list[step_index],
            traj_token_ids=traj_token_tensor,
            sampled_token_id=sampled_token_id,
            top_n=int(args.top_n),
        )
        accumulator.add_step(
            top_probs=stats["top_probs"],
            entropy_nats=float(stats["entropy_nats"]),
            sampled_is_argmax=bool(stats["sampled_is_argmax"]),
        )

    return {
        "sample_id": str(row.get("sample_id")),
        "traj_positions_used": len(body_steps),
        "generated_new_tokens": int(generated_new.shape[1]),
        "elapsed_ms": float(elapsed_ms),
        "seed": seed_value,
    }


def validate_args(args: argparse.Namespace) -> None:
    if int(args.samples_per_row) != 1:
        raise SystemExit("--samples-per-row must be 1 for this on-policy T=1 distribution audit.")
    if int(args.num_samples) < 0:
        raise SystemExit("--num-samples must be >= 0")
    if int(args.max_new_tokens) <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if int(args.max_traj_tokens) <= 0:
        raise SystemExit("--max-traj-tokens must be positive")
    if int(args.rank_curve_n) <= 0:
        raise SystemExit("--rank-curve-n must be positive")
    if int(args.top_n) < int(args.rank_curve_n):
        raise SystemExit("--top-n must be >= --rank-curve-n")
    if max(DEFAULT_CUMULATIVE_CUTOFFS) > int(args.rank_curve_n):
        raise SystemExit("--rank-curve-n must be >= 32 for the requested cumulative cutoffs")


def run_eval(args: argparse.Namespace) -> int:
    validate_args(args)
    seed_everything(int(args.seed))

    rows = select_rows(iter_jsonl(args.corpus_jsonl), args.split, args.num_samples)
    if not rows:
        raise SystemExit(f"No rows selected from {args.corpus_jsonl} split={args.split!r}")

    probe = load_probe_module()
    dtype = probe.torch_dtype_from_name(args.dtype)
    model, model_config = probe.load_model(args.checkpoint_path, dtype=dtype, device=args.device)
    processor = build_processor(model, model_config)
    accumulator = DistributionAccumulator(rank_curve_n=int(args.rank_curve_n))
    traj_span_meta: dict[str, Any] = {}
    per_sample_counts: list[int] = []
    elapsed_values: list[float] = []

    started_all = time.perf_counter()
    for row_index, row in enumerate(rows):
        record = run_one(
            model=model,
            model_config=model_config,
            processor=processor,
            probe=probe,
            row=row,
            row_index=row_index,
            args=args,
            accumulator=accumulator,
            traj_span_meta=traj_span_meta,
        )
        per_sample_counts.append(int(record["traj_positions_used"]))
        elapsed_values.append(float(record["elapsed_ms"]))
        if int(args.log_every) > 0 and ((row_index + 1) % int(args.log_every) == 0 or row_index == 0):
            print(
                json.dumps(
                    {
                        "event": "sample_done",
                        "index": row_index + 1,
                        "num_samples": len(rows),
                        "sample_id": record["sample_id"],
                        "traj_positions_used": record["traj_positions_used"],
                        "elapsed_ms": record["elapsed_ms"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = accumulator.summary()
    summary = {
        "schema": "10b_onpolicy_t1_token_dist_v1",
        "model_key": "teacher10b_onpolicy_t1",
        "model_label": "Alpamayo-1.5-10B VLM discrete trajectory, on-policy T=1",
        "checkpoint_path": str(args.checkpoint_path),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": str(args.split),
        "policy": {
            "on_policy": True,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "do_sample": True,
            "samples_per_row": 1,
            "seed": int(args.seed),
            "seed_policy": "python/numpy/torch seeded once globally and per row as seed + zero_based_row_index",
            "max_new_tokens": int(args.max_new_tokens),
        },
        "logit_source": {
            "field": "generate(..., return_dict_in_generate=True, output_logits=True).logits",
            "uses_scores": False,
            "normalization": "softmax over contract.traj_token_ids only, not full vocab",
            "per_step_retention": f"top-{int(args.top_n)} probabilities retained only long enough to aggregate",
        },
        "trajectory_span_contract": traj_span_meta,
        "counts": {
            "num_samples": len(rows),
            "num_samples_with_traj_positions": int(sum(1 for value in per_sample_counts if value > 0)),
            "num_trajectory_positions_used": int(accumulator.positions),
            "traj_positions_per_sample": {
                "min": int(min(per_sample_counts)) if per_sample_counts else None,
                "max": int(max(per_sample_counts)) if per_sample_counts else None,
                "mean": float(np.mean(per_sample_counts)) if per_sample_counts else None,
            },
        },
        "metrics": metrics,
        "runtime": {
            "elapsed_sec": round(time.perf_counter() - started_all, 3),
            "sample_elapsed_ms_mean": float(np.mean(elapsed_values)) if elapsed_values else None,
        },
        "cache_conditioned_t0_6_reference_schema_parity": {
            "note": "Reference values are cache-conditioned/off-policy and are included only to document schema/unit parity.",
            "top1_prob_mean": 0.437,
            "top2_prob_mean": 0.201,
            "top3_prob_mean": 0.114,
            "entropy_nats_mean": 1.55,
            "self_agreement_acc": 0.538,
            "top1_prob_percentiles": {"p50": 0.394},
            "top1_prob_gt_0_9_rate": 0.059,
            "top1_prob_lt_0_3_rate": 0.41,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "output_json": str(args.output_json)}, ensure_ascii=False), flush=True)
    return 0


def synthetic_self_test() -> int:
    seed_everything(123, include_cuda=False)
    vocab_size = 256
    traj_token_ids = torch.arange(100, 160, dtype=torch.long)
    accumulator = DistributionAccumulator(rank_curve_n=32)
    sampled_tokens: list[int] = []
    for step_index in range(24):
        logits = torch.randn(vocab_size, dtype=torch.float32)
        logits[traj_token_ids] += torch.linspace(2.0, -1.0, steps=int(traj_token_ids.numel()))
        if step_index % 3 == 0:
            sampled = int(traj_token_ids[0].item())
        else:
            sampled = int(traj_token_ids[(step_index * 7) % int(traj_token_ids.numel())].item())
        sampled_tokens.append(sampled)
        stats = step_distribution_stats(
            step_logits=logits,
            traj_token_ids=traj_token_ids,
            sampled_token_id=sampled,
            top_n=64,
        )
        accumulator.add_step(
            top_probs=stats["top_probs"],
            entropy_nats=float(stats["entropy_nats"]),
            sampled_is_argmax=bool(stats["sampled_is_argmax"]),
        )

    summary = accumulator.summary()
    curve = summary["rank_curve_top1_to_top32_mean_prob"]
    cumulative = summary["cumulative_mass_mean"]
    percentiles = summary["top1_prob_percentiles"]

    assert accumulator.positions == 24
    assert len(sampled_tokens) == 24
    assert all(0.0 <= float(value) <= 1.0 for value in curve.values() if value is not None)
    cumulative_values = [float(cumulative[f"top_{cutoff}"]) for cutoff in DEFAULT_CUMULATIVE_CUTOFFS]
    assert cumulative_values == sorted(cumulative_values)
    assert all(0.0 <= value <= 1.0 for value in cumulative_values)
    assert float(summary["entropy_nats_mean"]) >= 0.0
    ordered_percentiles = [float(percentiles[key]) for key in ("p10", "p25", "p50", "p75", "p90")]
    assert ordered_percentiles == sorted(ordered_percentiles)
    assert 0.0 <= float(summary["self_agreement_acc"]) <= 1.0
    assert 0.0 <= float(summary["top1_prob_gt_0_9_rate"]) <= 1.0
    assert 0.0 <= float(summary["top1_prob_lt_0_3_rate"]) <= 1.0

    print(
        json.dumps(
            {
                "event": "synthetic_self_test_passed",
                "positions": accumulator.positions,
                "assertions": [
                    "probabilities in [0,1]",
                    "cumulative mass monotonic non-decreasing",
                    "entropy >= 0",
                    "percentiles ordered p10<=p25<=p50<=p75<=p90",
                    "rates in [0,1]",
                ],
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return synthetic_self_test()
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
