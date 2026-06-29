#!/usr/bin/env python3
"""Benchmark AE28 inference latency: total and per-step."""
from __future__ import annotations
import argparse, importlib, json, os, sys, time
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

def _import_ae():
    spec = importlib.util.spec_from_file_location(
        "ae_train", str(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint-dir", type=str, required=True)
    parser.add_argument("--ae28-checkpoint", type=str, required=True)
    parser.add_argument("--corpus-jsonl", type=str, required=True)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-bench", type=int, default=20)
    parser.add_argument("--inference-steps", type=int, nargs="+", default=[1, 2, 4, 10])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--prefix-mode", type=str, default="teacher_forced")
    parser.add_argument("--preserve-flex-positions", action="store_true", default=True)
    parser.add_argument("--flex-selection-strategy", type=str, default="uniform")
    parser.add_argument("--flex-scene-deepstack", action="store_true", default=True)
    parser.add_argument("--qat-quantization", type=str, default="")
    parser.add_argument("--qat-calib-samples", type=int, default=256)
    parser.add_argument("--target-source", type=str, default="teacher")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--split-scan-all", action="store_true", default=True)
    parser.add_argument("--stage2-attention-mode", type=str, default="official_none")
    parser.add_argument("--student-model", type=str, default="")
    parser.add_argument("--teacher-checkpoint-path", type=str,
                        default=str(PROJECT_ROOT.parent.parent / "base_weights" / "Alpamayo-1.5-10B"))
    parser.add_argument("--ae-init-mode", type=str, default="student_backbone_init")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--disable-student-deepstack", action="store_true", default=False)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--split-cache-json", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument("--mapping", type=str, default="linspace_round")
    parser.add_argument("--ae-dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.student_checkpoint_dir = Path(args.student_checkpoint_dir)
    args.corpus_jsonl = Path(args.corpus_jsonl)
    args.teacher_checkpoint_path = Path(args.teacher_checkpoint_path)

    ae = _import_ae()

    # Load models
    student, student_tokenizer, student_processor, base_model = ae.load_student(args)

    def _torch_dtype(name):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.bfloat16)
    args.teacher_load_device = "cpu"
    _load_fn = getattr(ae, "load_model_and_processor", None)
    if _load_fn is None:
        from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor as _load_fn
    teacher_model, _, _, _, _ = _load_fn(
        checkpoint_path=args.teacher_checkpoint_path, dtype=_torch_dtype(args.student_dtype),
        device=args.teacher_load_device, config_json=None, runtime_support=None,
        attn_implementation=args.attn_implementation, min_pixels=163840, max_pixels=196608)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_model.to(device)

    bundle, selected = ae.build_bundle(teacher_model, args, student=student)
    ae.load_bundle_checkpoint(Path(args.ae28_checkpoint), bundle=bundle)
    bundle.eval()
    for p in bundle.parameters():
        p.requires_grad_(False)
    bundle.to(device)

    # Build one sample batch
    train_items, _, _ = ae.select_train_val_items(args)
    batch_items = train_items[:1]
    batch = ae.build_batch(
        args=args, student=student, student_processor=student_processor,
        student_tokenizer=student_tokenizer, teacher_model=teacher_model,
        batch_items=batch_items)

    print(json.dumps({"event": "models_loaded"}), flush=True)

    # Benchmark for each step count
    for n_steps in args.inference_steps:
        times = []
        per_step_times = []

        for trial in range(args.num_warmup + args.num_bench):
            prompt_cache = batch["cache"]
            context = batch["context"]
            dtype = next(bundle.parameters()).dtype
            prefill_seq_len = int(context["kv_cache_seq_len"])
            n_diff = int(context["n_diffusion_tokens"])
            action_dims = teacher_model.action_space.get_action_space_dims()
            kwargs = {}
            if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
                kwargs["is_causal"] = False

            step_times_trial = []

            def step_fn(*, x, t):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fut = bundle.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
                if fut.dim() == 2:
                    fut = fut.view(x.shape[0], n_diff, -1)
                attn = context.get("attention_mask")
                if attn is not None:
                    attn = attn.to(dtype=fut.dtype)
                out = bundle.expert(
                    inputs_embeds=fut, position_ids=context["position_ids"],
                    past_key_values=prompt_cache, attention_mask=attn,
                    use_cache=True, **kwargs)
                prompt_cache.crop(prefill_seq_len)
                hidden = out.last_hidden_state[:, -n_diff:]
                v = bundle.action_out_proj(hidden).view(-1, *action_dims)
                e.record()
                torch.cuda.synchronize()
                step_times_trial.append(s.elapsed_time(e))
                return v

            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                start_ev.record()
                teacher_model.diffusion.sample(
                    batch_size=1, step_fn=step_fn, device=device,
                    inference_step=n_steps, temperature=1.0)
                end_ev.record()
            torch.cuda.synchronize()

            if trial >= args.num_warmup:
                total_ms = start_ev.elapsed_time(end_ev)
                times.append(total_ms)
                per_step_times.append(np.mean(step_times_trial))

        result = {
            "inference_steps": n_steps,
            "total_ms_mean": round(np.mean(times), 2),
            "total_ms_std": round(np.std(times), 2),
            "total_ms_min": round(np.min(times), 2),
            "total_ms_max": round(np.max(times), 2),
            "per_step_ms_mean": round(np.mean(per_step_times), 2),
            "per_step_ms_std": round(np.std(per_step_times), 2),
        }
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
