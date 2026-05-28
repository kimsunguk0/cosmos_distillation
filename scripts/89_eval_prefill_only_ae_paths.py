#!/usr/bin/env python3
"""Evaluate no-CoT prefill KV caches through the action expert.

Variants:
  teacher_prefill:
    teacher 8B VLM prompt/image/ego prefill KV only -> original AE36
  student_prefill:
    student 2B VLM prompt/image/ego prefill KV only -> 28->36 adapter -> AE36

Both variants intentionally stop before any generated CoT body or
<|traj_future_start|> token. The assistant prefix <|cot_start|> is still part of
the normal Alpamayo chat template, matching the pre-generation prefix.
"""

from __future__ import annotations

import argparse
import copy
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
SCRIPT84 = PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"
SCRIPT87 = PROJECT_ROOT / "scripts" / "87_train_student_kv_adapter_ae36.py"
STAGE1_SCRIPT = PROJECT_ROOT / "scripts" / "51_train_stage1_ae28_teacher_kv_scale.py"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_TEACHER = Path("/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B")
DEFAULT_STUDENT_CKPT = (
    PROJECT_ROOT
    / "outputs/checkpoints/no_nav_camera_labeled_official_200k/"
    / "no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509/best_decode"
)
DEFAULT_ADAPTER_CKPT = (
    PROJECT_ROOT
    / "outputs/action_expert/student_kv_adapter_ae36/"
    / "student_kv_adapter_ae36_headproj_stageb_teacherforced_16_overfit_beta_20260519_clean/best.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "action_expert" / "prefill_only_ae_paths"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--adapter-checkpoint", type=Path, default=DEFAULT_ADAPTER_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seeds", default="1097")
    parser.add_argument("--run-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-student", action=argparse.BooleanOptionalAction, default=True)
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


def make_context_from_prefill(
    *,
    teacher_model: Any,
    cache: Any,
    rope_deltas: torch.Tensor,
    attention_mask: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    kv_cache_seq_len = int(cache.get_seq_length())
    n_diffusion_tokens = int(teacher_model.action_space.get_action_space_dims()[0])
    if attention_mask is not None:
        offset = attention_mask.sum(dim=1).to(device=device, dtype=torch.long)
    else:
        offset = torch.full((batch_size,), kv_cache_seq_len, device=device, dtype=torch.long)
    position_ids, expert_attention_mask = teacher_model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas.to(device),
        kv_cache_seq_len=kv_cache_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=batch_size,
        device=device,
        prefix_mask=attention_mask,
    )
    return {
        "kv_cache_seq_len": kv_cache_seq_len,
        "n_diffusion_tokens": n_diffusion_tokens,
        "position_ids": position_ids,
        "attention_mask": expert_attention_mask,
        "offset": offset.detach().cpu().tolist(),
    }


def build_teacher_prefill_batch(stage1: Any, items: list[dict[str, Any]], model: Any, processor: Any, args: argparse.Namespace):
    sample_dirs = [Path(item["sample_dir"]) for item in items]
    samples = stage1.load_materialized_samples(sample_dirs, int(args.io_workers))
    model_inputs = stage1.build_model_inputs_batch(processor=processor, samples=samples, device=args.device)
    tokenized_data = dict(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"],
        },
    )
    dtype = next(model.parameters()).dtype
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=str(args.device).startswith("cuda")):
        outputs = model.vlm(input_ids=input_ids, **tokenized_data, use_cache=True, return_dict=True, logits_to_keep=1)
    rope_deltas = getattr(outputs, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = getattr(model.vlm.model, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long)
    target_xyz_np, target_rot_np = [], []
    for item in items:
        xyz, rot = stage1.raw_teacher_pred(Path(item["raw_json"]))
        target_xyz_np.append(xyz)
        target_rot_np.append(rot)
    target_xyz = torch.from_numpy(np.stack(target_xyz_np, axis=0)).to(args.device, dtype=torch.float32)
    target_rot = torch.from_numpy(np.stack(target_rot_np, axis=0)).to(args.device, dtype=torch.float32)
    ego_history_xyz = model_inputs["ego_history_xyz"].detach()
    ego_history_rot = model_inputs["ego_history_rot"].detach()
    context = make_context_from_prefill(
        teacher_model=model,
        cache=outputs.past_key_values,
        rope_deltas=rope_deltas,
        attention_mask=tokenized_data.get("attention_mask"),
        batch_size=len(items),
        device=torch.device(args.device),
    )
    return {
        "sample_ids": [item["sample_id"] for item in items],
        "cache": outputs.past_key_values,
        "context": context,
        "target_xyz": target_xyz.detach(),
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "meta": {"cache_layer_count": len(getattr(outputs.past_key_values, "layers", [])), "cache_seq_len": outputs.past_key_values.get_seq_length()},
    }


def build_student_prefill_batch(ae84: Any, items: list[dict[str, Any]], student: Any, processor: Any, tokenizer: Any, teacher_model: Any, args: argparse.Namespace):
    rows = [item["row"] for item in items]
    image_batch = [ae84.load_sample_images(row, PROJECT_ROOT) for row in rows]
    histories_xyz = [ae84.load_ego_history_xyz(row, PROJECT_ROOT).astype(np.float32) for row in rows]
    histories_rot = [ae84.normalize_history_rot(ae84.load_ego_history_rot(row, PROJECT_ROOT)) for row in rows]
    prompt_messages = []
    for row, images, hist_xyz in zip(rows, image_batch, histories_xyz):
        camera_indices = ae84.resolve_camera_indices(row, PROJECT_ROOT, image_count=len(images))
        frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
        prompt_text = ae84.build_user_prompt(
            row,
            PROJECT_ROOT,
            ego_history_xyz=hist_xyz,
            prompt_text_style="official_alpamayo",
        )
        prompt_messages.append(
            ae84.build_messages(
                prompt_text,
                len(images),
                completion_text=None,
                assistant_prefix="<|cot_start|>",
                image_prompt_style="camera_labeled",
                camera_indices=camera_indices,
                num_frames_per_camera=frames_per_camera,
            )
        )
    encoded = ae84._encode_messages(
        processor,
        prompt_messages,
        image_batch,
        args.max_length,
        continue_final_message=True,
    )
    encoded["input_ids"] = ae84.fuse_history_tokens_in_input_ids(
        encoded["input_ids"],
        tokenizer,
        histories_xyz,
    )
    encoded = ae84._to_device_batch(encoded, torch.device(args.device))
    model_kwargs = dict(encoded)
    input_ids = model_kwargs.pop("input_ids")
    dtype = ae84.torch_dtype_from_name(args.student_dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=str(args.device).startswith("cuda")):
        try:
            outputs = student.backbone(
                input_ids=input_ids,
                **model_kwargs,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        except TypeError:
            outputs = student.backbone(input_ids=input_ids, **model_kwargs, use_cache=True, return_dict=True)
    rope_deltas = getattr(outputs, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = ae84.get_rope_deltas(student.backbone)
    target_xyz_np, target_rot_np = [], []
    for item in items:
        xyz, rot = ae84.raw_teacher_pred(Path(item["raw_json"]))
        target_xyz_np.append(xyz)
        target_rot_np.append(rot)
    target_xyz = torch.from_numpy(np.stack(target_xyz_np, axis=0)).to(args.device, dtype=torch.float32)
    ego_history_xyz = torch.from_numpy(np.stack(histories_xyz, axis=0)).to(args.device, dtype=torch.float32)
    ego_history_rot = torch.from_numpy(np.stack(histories_rot, axis=0)).to(args.device, dtype=torch.float32)
    context = make_context_from_prefill(
        teacher_model=teacher_model,
        cache=outputs.past_key_values,
        rope_deltas=rope_deltas,
        attention_mask=encoded.get("attention_mask"),
        batch_size=len(items),
        device=torch.device(args.device),
    )
    return {
        "sample_ids": [item["sample_id"] for item in items],
        "cache": outputs.past_key_values,
        "context": context,
        "target_xyz": target_xyz.detach(),
        "ego_history_xyz": ego_history_xyz.detach(),
        "ego_history_rot": ego_history_rot.detach(),
        "meta": {"cache_layer_count": len(getattr(outputs.past_key_values, "layers", [])), "cache_seq_len": outputs.past_key_values.get_seq_length()},
    }


def evaluate_variant(
    *,
    name: str,
    batches: list[dict[str, Any]],
    seeds: list[int],
    sample_fn: Any,
    stage1: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_seed = {seed: {"ade": [], "fde": []} for seed in seeds}
    best_ades, best_fdes = [], []
    for batch_index, batch in enumerate(batches):
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        seed_preds = {}
        for seed in seeds:
            pred = sample_fn(batch=batch, seed=int(seed) + batch_index)
            seed_preds[seed] = pred["pred_xyz"]
            for row_index, sample_id in enumerate(batch["sample_ids"]):
                ade, fde = stage1.ade_fde(pred["pred_xyz"][row_index], target_xyz[row_index])
                by_seed[seed]["ade"].append(ade)
                by_seed[seed]["fde"].append(fde)
                rows.append({"variant": name, "sample_id": sample_id, "seed": int(seed) + batch_index, "ade_m": ade, "fde_m": fde})
        for row_index in range(len(batch["sample_ids"])):
            pairs = [stage1.ade_fde(seed_preds[seed][row_index], target_xyz[row_index]) for seed in seeds]
            best_ade, best_fde = min(pairs, key=lambda x: x[0])
            best_ades.append(best_ade)
            best_fdes.append(best_fde)
    return {
        "variant": name,
        "by_seed": {str(seed): {"ade_m": summarize(vals["ade"]), "fde_m": summarize(vals["fde"])} for seed, vals in by_seed.items()},
        "best_of_seeds": {"ade_m": summarize(best_ades), "fde_m": summarize(best_fdes)},
        "rows_head": rows[:32],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    started = time.perf_counter()
    seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    stage1 = load_module(STAGE1_SCRIPT, "stage1_for_prefill_eval")
    ae84 = load_module(SCRIPT84, "ae84_for_prefill_eval")
    ae87 = load_module(SCRIPT87, "ae87_for_prefill_eval")
    item_args = SimpleNamespace(corpus_jsonl=args.corpus_jsonl, split=args.split, num_samples=args.num_samples)
    items = ae84.select_items(item_args)
    item_batches = [items[i : i + int(args.batch_size)] for i in range(0, len(items), int(args.batch_size))]
    results: dict[str, Any] = {
        "status": "running",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "selected_count": len(items),
    }

    teacher_model = None
    if args.run_teacher or args.run_student:
        print(json.dumps({"event": "load_teacher_start", "checkpoint": str(args.teacher_checkpoint_path)}), flush=True)
        teacher_model, teacher_processor, *_ = stage1.load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=stage1.torch_dtype_from_name(args.teacher_dtype),
            device=args.device,
            config_json=None,
            runtime_support=None,
            attn_implementation=args.attn_implementation,
            min_pixels=163840,
            max_pixels=196608,
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        stage1.force_attention(teacher_model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")

    if args.run_teacher:
        teacher_batches = []
        for batch_items in item_batches:
            teacher_batches.append(build_teacher_prefill_batch(stage1, batch_items, teacher_model, teacher_processor, args))
        results["teacher_prefill_only_original_ae36"] = evaluate_variant(
            name="teacher_prefill_only_original_ae36",
            batches=teacher_batches,
            seeds=seeds,
            stage1=stage1,
            sample_fn=lambda batch, seed: stage1.sample_modules_paths_batch(
                expert=teacher_model.expert,
                action_in_proj=teacher_model.action_in_proj,
                action_out_proj=teacher_model.action_out_proj,
                model=teacher_model,
                prompt_cache=batch["cache"],
                context=batch["context"],
                ego_history_xyz=batch["ego_history_xyz"],
                ego_history_rot=batch["ego_history_rot"],
                seed=seed,
                device=torch.device(args.device),
            ),
        )

    if args.run_student:
        # Free teacher VLM memory before loading student; action modules stay.
        if hasattr(teacher_model, "vlm"):
            delattr(teacher_model, "vlm")
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        student_args = argparse.Namespace(**vars(args))
        student_args.student_model = ae84.resolve_student_model_path()
        student, student_tokenizer, student_processor, base_model = ae84.load_student(student_args)
        ae87._move_action_modules(
            teacher_model,
            device=torch.device(args.device),
            dtype=stage1.torch_dtype_from_name(args.teacher_dtype),
            attn_implementation=args.attn_implementation,
        )
        old_layers = int(getattr(getattr(student.backbone.config, "text_config", None), "num_hidden_layers", 28))
        new_layers = int(teacher_model.expert.config.num_hidden_layers)
        adapter = ae87.StudentKVToAE36Adapter(
            old_layers=old_layers,
            new_layers=new_layers,
            kv_heads=8,
            head_dim=128,
            init_alpha=0.5,
            use_affine=True,
            use_head_proj=True,
        ).to(args.device)
        state = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
        adapter.load_state_dict(state["adapter_state_dict"], strict=True)
        if "expert_state_dict" in state:
            teacher_model.expert.load_state_dict(state["expert_state_dict"], strict=True)
        if "action_in_proj_state_dict" in state:
            teacher_model.action_in_proj.load_state_dict(state["action_in_proj_state_dict"], strict=True)
        if "action_out_proj_state_dict" in state:
            teacher_model.action_out_proj.load_state_dict(state["action_out_proj_state_dict"], strict=True)
        adapter.eval()
        teacher_model.expert.eval()
        teacher_model.action_in_proj.eval()
        teacher_model.action_out_proj.eval()
        student_batches = []
        for batch_items in item_batches:
            student_batches.append(
                build_student_prefill_batch(ae84, batch_items, student, student_processor, student_tokenizer, teacher_model, args)
            )
        results["student_prefill_only_adapter_ae36"] = evaluate_variant(
            name="student_prefill_only_adapter_ae36",
            batches=student_batches,
            seeds=seeds,
            stage1=stage1,
            sample_fn=lambda batch, seed: ae87.sample_paths(
                adapter=adapter,
                model=teacher_model,
                batch=batch,
                seed=seed,
                device=torch.device(args.device),
            ),
        )
        results["student_base_model"] = str(base_model)
        results["adapter_checkpoint_loaded"] = str(args.adapter_checkpoint)

    results["status"] = "ok"
    results["elapsed_sec"] = round(time.perf_counter() - started, 3)
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(summary_path), "elapsed_sec": results["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
