#!/usr/bin/env python3
"""Run AE28 eval across checkpoints and diffusion seeds on one fixed val split.

This is intentionally eval-only. It loads the student/teacher/bundle once, then
loads each checkpoint into the same bundle and evaluates multiple sampling seeds.
Use a split cache to keep the eval set identical to training/Q2/Q3.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


_84_PATH = Path(__file__).resolve().parent / "84_train_student_ae28_official.py"
if not _84_PATH.exists():
    raise FileNotFoundError(f"Cannot locate sibling 84 script at {_84_PATH}")
_spec = importlib.util.spec_from_file_location("script_84", _84_PATH)
script_84 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(script_84)


def parse_seed_sweep_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="Checkpoint spec, either LABEL=PATH or PATH. Can be repeated.",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated base seeds. eval_seed_base becomes seed+1000 in fixed mode.",
    )
    parser.add_argument("--seed-sweep-output-dir", type=Path, required=True)
    return parser.parse_known_args(argv)


def parse_ckpt_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in specs:
        if "=" in raw:
            label, path_s = raw.split("=", 1)
        else:
            path_s = raw
            label = Path(path_s).parent.name
        path = Path(path_s)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        out.append((label, path))
    return out


def compact_eval(label: str, ckpt_path: Path, payload_step: int | None, seed: int, ev: dict) -> dict:
    horizon = ev.get("horizon") or {}
    h_best = ev.get("horizon_best_of_n") or {}
    return {
        "event": "seed_sweep_eval",
        "label": label,
        "ckpt_path": str(ckpt_path),
        "payload_step": payload_step,
        "seed": int(seed),
        "eval_seed_base": ev.get("eval_seed_base"),
        "eval_count": ev.get("eval_count"),
        "eval_num_paths": ev.get("eval_num_paths"),
        "eval_temperature": ev.get("eval_temperature"),
        "eval_selection_method": ev.get("eval_selection_method"),
        "ade_mean_m": ev.get("ade_mean_m"),
        "ade_p50_m": ev.get("ade_p50_m"),
        "fde_mean_m": ev.get("fde_mean_m"),
        "fde_p50_m": ev.get("fde_p50_m"),
        "ade_best_of_n_mean_m": ev.get("ade_best_of_n_mean_m"),
        "ade_best_of_n_p50_m": ev.get("ade_best_of_n_p50_m"),
        "fde_best_of_n_mean_m": ev.get("fde_best_of_n_mean_m"),
        "fde_best_of_n_p50_m": ev.get("fde_best_of_n_p50_m"),
        "minade_at_n": ev.get("minade_at_n"),
        "minade_at_n_mean_m": ev.get("minade_at_n_mean_m"),
        "minade_at_n_p50_m": ev.get("minade_at_n_p50_m"),
        "minfde_at_n_mean_m": ev.get("minfde_at_n_mean_m"),
        "minfde_at_n_p50_m": ev.get("minfde_at_n_p50_m"),
        "minade_at_6_mean_m": ev.get("minade_at_6_mean_m"),
        "minade_at_6_p50_m": ev.get("minade_at_6_p50_m"),
        "minfde_at_6_mean_m": ev.get("minfde_at_6_mean_m"),
        "minfde_at_6_p50_m": ev.get("minfde_at_6_p50_m"),
        "ade_mean_over_paths_mean_m": ev.get("ade_mean_over_paths_mean_m"),
        "ade_std_over_paths_mean_m": ev.get("ade_std_over_paths_mean_m"),
        "h1p6_ade_mean_m": (horizon.get("h1p6_16wp") or {}).get("ade_mean_m"),
        "h3p2_ade_mean_m": (horizon.get("h3p2_32wp") or {}).get("ade_mean_m"),
        "h6p4_ade_mean_m": (horizon.get("h6p4_64wp") or {}).get("ade_mean_m"),
        "h1p6_best_of_n_ade_mean_m": (h_best.get("h1p6_16wp") or {}).get("ade_mean_m"),
        "h3p2_best_of_n_ade_mean_m": (h_best.get("h3p2_32wp") or {}).get("ade_mean_m"),
        "h6p4_best_of_n_ade_mean_m": (h_best.get("h6p4_64wp") or {}).get("ade_mean_m"),
    }


def main() -> None:
    sweep_args, remaining = parse_seed_sweep_args(sys.argv[1:])
    ckpts = parse_ckpt_specs(sweep_args.ckpt)
    seeds = [int(x.strip()) for x in str(sweep_args.seeds).split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds produced an empty list")

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + list(remaining)
        args = script_84.parse_args()
    finally:
        sys.argv = saved_argv

    args.eval_seed_mode = "fixed"
    args.eval_selection_method = str(getattr(args, "eval_selection_method", "mean_traj"))

    out_dir = sweep_args.seed_sweep_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "seed_sweep.jsonl"
    summary_path = out_dir / "summary.json"

    print(
        json.dumps(
            {
                "event": "seed_sweep_start",
                "output_dir": str(out_dir),
                "ckpts": [{"label": label, "path": str(path)} for label, path in ckpts],
                "seeds": seeds,
                "eval_samples": int(args.eval_samples),
                "val_samples": int(getattr(args, "val_samples", 0)),
                "split_cache_json": str(getattr(args, "split_cache_json", "")),
                "eval_num_paths": int(getattr(args, "eval_num_paths", 1)),
                "eval_temperature": float(getattr(args, "eval_temperature", 1.0)),
                "eval_selection_method": str(getattr(args, "eval_selection_method", "single")),
            }
        ),
        flush=True,
    )

    train_items, val_items, split_summary = script_84.select_train_val_items(args)
    print(
        json.dumps(
            {
                "event": "seed_sweep_split_loaded",
                "train_count": len(train_items),
                "val_count": len(val_items),
                "eval_count_used": min(int(args.eval_samples), len(val_items)),
                "sample_id_overlap_count": split_summary.get("sample_id_overlap_count"),
                "split_group_overlap_count": split_summary.get("split_group_overlap_count"),
            }
        ),
        flush=True,
    )

    student, student_tokenizer, student_processor, base_model = script_84.load_student(args)
    print(json.dumps({"event": "seed_sweep_student_loaded", "base_model": str(base_model)}), flush=True)

    teacher_model, _, _, _, _ = script_84.load_model_and_processor(
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
    script_84.force_attention(
        teacher_model.expert,
        "sdpa" if args.attn_implementation != "eager" else "eager",
    )

    bundle, selected_layers = script_84.build_bundle(teacher_model, args, student=student)
    print(json.dumps({"event": "seed_sweep_bundle_built", "selected_layers": selected_layers}), flush=True)
    bundle = bundle.to(device=args.device, dtype=script_84.torch_dtype_from_name(args.ae_dtype))

    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: list[dict] = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for label, ckpt_path in ckpts:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if not isinstance(state, dict) or "bundle_state_dict" not in state:
                raise ValueError(f"Checkpoint {ckpt_path} missing 'bundle_state_dict'")
            missing, unexpected = bundle.load_state_dict(state["bundle_state_dict"], strict=False)
            payload = state.get("payload", {})
            payload_step = payload.get("step")
            print(
                json.dumps(
                    {
                        "event": "seed_sweep_ckpt_loaded",
                        "label": label,
                        "ckpt_path": str(ckpt_path),
                        "payload_step": payload_step,
                        "missing_keys_count": len(missing),
                        "unexpected_keys_count": len(unexpected),
                    }
                ),
                flush=True,
            )
            del state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for seed in seeds:
                args.seed = int(seed)
                ev = script_84.evaluate(
                    args=args,
                    bundle=bundle,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    items=val_items,
                    step=int(payload_step or 0),
                )
                row = compact_eval(label, ckpt_path, payload_step, seed, ev)
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)

    grouped: dict[str, list[dict]] = {}
    for row in results:
        grouped.setdefault(str(row["label"]), []).append(row)

    def mean_metric(rows: list[dict], key: str) -> float | None:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        "event": "seed_sweep_done",
        "output_dir": str(out_dir),
        "jsonl": str(jsonl_path),
        "split_summary": split_summary,
        "seeds": seeds,
        "results": results,
        "by_label": {
            label: {
                "n": len(rows),
                "ade_mean_m_mean": mean_metric(rows, "ade_mean_m"),
                "ade_best_of_n_mean_m_mean": mean_metric(rows, "ade_best_of_n_mean_m"),
                "minade_at_n_mean_m_mean": mean_metric(rows, "minade_at_n_mean_m"),
                "minade_at_6_mean_m_mean": mean_metric(rows, "minade_at_6_mean_m"),
                "ade_mean_over_paths_mean_m_mean": mean_metric(rows, "ade_mean_over_paths_mean_m"),
                "ade_std_over_paths_mean_m_mean": mean_metric(rows, "ade_std_over_paths_mean_m"),
            }
            for label, rows in grouped.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "seed_sweep_done", "summary_json": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
