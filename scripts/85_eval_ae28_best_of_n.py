#!/usr/bin/env python3
"""Eval-only best-of-N runner for the AE28 setup trained by 84_train_student_ae28_official.py.

Loads a checkpoint (``best.pt`` / ``final.pt``) saved by the training script and runs
``evaluate()`` once with the requested ``--eval-num-paths``. Does NOT touch training; the
model, bundle structure, sample_paths(), and the diffusion sampler are unchanged.

Usage:
    python scripts/85_eval_ae28_best_of_n.py \
        --ckpt-path outputs/action_expert/.../best.pt \
        --eval-num-paths 8 \
        [any other 84-script args ...]

All other CLI flags are passed through to ``script_84.parse_args()`` so the eval input
distribution (corpus, samples, batch_size, prefix-mode, ae-init-mode, target-source, etc.)
matches the training run that produced the checkpoint.
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


def _extract_ckpt_path(argv: list[str]) -> tuple[Path, Path | None, list[str]]:
    """Pull --ckpt-path out of argv before handing the rest to 84's parser."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt-path", type=Path, required=True)
    pre.add_argument("--eval-summary-json", type=Path, default=None)
    parsed, remaining = pre.parse_known_args(argv)
    return parsed.ckpt_path, parsed.eval_summary_json, remaining


def main() -> None:
    ckpt_path, eval_summary_json, remaining = _extract_ckpt_path(sys.argv[1:])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Hand the remaining argv to 84's argparse so all training/eval flags stay consistent.
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + list(remaining)
        args = script_84.parse_args()
    finally:
        sys.argv = saved_argv

    print(json.dumps({
        "event": "eval_only_start",
        "ckpt_path": str(ckpt_path),
        "eval_num_paths": int(getattr(args, "eval_num_paths", 1)),
        "eval_samples": int(args.eval_samples),
        "eval_batch_size": int(args.eval_batch_size),
        "prefix_mode": str(args.prefix_mode),
        "ae_init_mode": str(args.ae_init_mode),
        "target_source": str(getattr(args, "target_source", "teacher")),
        "seed": int(args.seed),
    }), flush=True)

    # 1) Load student (same path as training entrypoint).
    items = script_84.select_items(args)
    print(json.dumps({"event": "select_items_done", "selected_count": len(items)}), flush=True)

    student, student_tokenizer, student_processor, base_model = script_84.load_student(args)
    print(json.dumps({"event": "load_student_done", "base_model": str(base_model)}), flush=True)

    # 2) Load teacher model (action_space + flow-matching helpers).
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

    # 3) Build the bundle with the same init policy as training (then overwrite weights).
    bundle, selected_layers = script_84.build_bundle(teacher_model, args, student=student)
    print(json.dumps({"event": "bundle_built", "selected_layers": selected_layers}), flush=True)

    # 4) Load saved bundle weights.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "bundle_state_dict" not in state:
        raise ValueError(f"Checkpoint {ckpt_path} missing 'bundle_state_dict'")
    missing, unexpected = bundle.load_state_dict(state["bundle_state_dict"], strict=False)
    print(json.dumps({
        "event": "bundle_state_loaded",
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "ckpt_payload_step": state.get("payload", {}).get("step"),
    }), flush=True)
    bundle = bundle.to(device=args.device, dtype=script_84.torch_dtype_from_name(args.ae_dtype))

    # Free teacher VLM weights from memory (training does the same after build_bundle).
    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5) Run evaluate() exactly once.
    ev = script_84.evaluate(
        args=args,
        bundle=bundle,
        student=student,
        student_processor=student_processor,
        student_tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        items=items,
        step=int(state.get("payload", {}).get("step", 0)),
    )
    if eval_summary_json is not None:
        eval_summary_json.parent.mkdir(parents=True, exist_ok=True)
        eval_summary_json.write_text(json.dumps(ev, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    # Drop rows from the printed summary (still in the json for log files); keep top-level keys.
    print(json.dumps(ev), flush=True)


if __name__ == "__main__":
    main()
