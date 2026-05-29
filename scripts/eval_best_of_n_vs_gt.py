#!/usr/bin/env python3
"""Compute best-of-N ADE/FDE vs GT for step_006250 on vis_4per_category_val corpus."""
from __future__ import annotations
import json, math, re, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList

DISTILL_ROOT = Path("/home/pm97/workspace/sukim/distillation/cosmos_distillation")
CKPT_DIR = (
    DISTILL_ROOT / "outputs/checkpoints/no_nav_camera_labeled_official_full444k"
    / "no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
)
CORPUS_JSONL = DISTILL_ROOT / "data/corpus/vis_4per_category_val.jsonl"
N_CANDIDATES = 4
TEMPERATURE  = 1.0
TOP_P        = 0.95

if str(DISTILL_ROOT) not in sys.path:
    sys.path.insert(0, str(DISTILL_ROOT))

from src.inference.checkpoint_eval import TrajectoryTokenDecoder, load_ego_history_rot, resolve_traj_tokenizer_config_path
from src.inference.decoding import TrajDecodingContract, TrajSpanLogitsProcessor, StopOnTrajEndCriteria
from src.model.student_wrapper import StudentWrapperConfig, build_student_model
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.tokenizer_ext import distill_trainable_token_ids
from src.training.collator import (build_messages, build_user_prompt,
    fuse_history_tokens_in_input_ids, load_ego_future_xyz, load_ego_history_xyz,
    load_sample_images, load_traj_future_token_ids, resolve_camera_indices)


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]

def load_gt_tokens(row):
    ht = row.get("hard_target") or {}
    inline = ht.get("traj_future_token_ids")
    if inline: return [int(t) for t in inline]
    npy = ht.get("traj_future_token_ids_path")
    if npy and Path(npy).exists(): return np.load(npy).astype(int).tolist()
    return []

def ade_fde(a, b):
    if a is None or b is None or a.size==0 or b.size==0: return float("nan"), float("nan")
    n = min(a.shape[0], b.shape[0])
    d = np.linalg.norm(a[:n,:2]-b[:n,:2], axis=-1)
    return float(d.mean()), float(d[-1])

def load_model(device):
    ckpt = CKPT_DIR
    train_cfg = json.loads((ckpt/"train_config.json").read_text()) if (ckpt/"train_config.json").exists() else {}
    manifest  = json.loads((ckpt/"checkpoint_manifest.json").read_text()) if (ckpt/"checkpoint_manifest.json").exists() else {}
    base_model = str((train_cfg.get("args") or {}).get("student_model") or "/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
    use_lora   = not bool((train_cfg.get("args") or {}).get("disable_lora", False))
    data_view  = train_cfg.get("data_view") or {}
    tok_dir, proc_dir = ckpt/"tokenizer", ckpt/"processor"
    tokenizer = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else base_model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(proc_dir if proc_dir.exists() else base_model, local_files_only=True)
    processor.tokenizer = tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"): processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else None
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_cfg.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=dtype, local_files_only=Path(base_model).expanduser().exists(),
        traj_teacher_hidden_size=int(data_view["teacher_traj_hidden_size"]) if data_view.get("teacher_traj_hidden_size") not in (None,"",0) else None,
        traj_hidden_bridge_size=int(manifest["traj_hidden_bridge_size"]) if manifest.get("traj_hidden_bridge_size") not in (None,"",0) else None,
    )
    model = build_student_model(wrapper_cfg, tokenizer)
    fmt = detect_checkpoint_format(ckpt)
    if fmt == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(model.backbone, LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))), enabled=True)
    load_student_checkpoint(ckpt, model, use_lora=use_lora)
    return model.to(device).eval(), tokenizer, processor, base_model

def generate_candidates(model, tokenizer, processor, sample, *, device, history_xyz):
    images = load_sample_images(sample, DISTILL_ROOT)
    cam_idx = resolve_camera_indices(sample, DISTILL_ROOT, image_count=len(images))
    fpc = max(len(images) // max(len(cam_idx), 1), 1)
    prompt_text = build_user_prompt(sample, DISTILL_ROOT, ego_history_xyz=history_xyz, prompt_text_style="official_alpamayo")
    messages = build_messages(prompt_text, len(images), assistant_prefix="<|cot_start|>",
                               image_prompt_style="camera_labeled", camera_indices=cam_idx, num_frames_per_camera=fpc)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    batch = processor(text=[text], images=[images], return_tensors="pt", padding=True, truncation=True)
    batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, [history_xyz])
    target_tokens = load_traj_future_token_ids(sample.get("hard_target") or {}, DISTILL_ROOT)
    vdtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    batch = {k: (v.to(device=device, dtype=vdtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                 else v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    prompt_ids = batch["input_ids"]
    contract = TrajDecodingContract.from_tokenizer(tokenizer, prompt_lengths=[int(prompt_ids.shape[1])], traj_token_count=len(target_tokens))
    with torch.autocast("cuda", dtype=vdtype) if device.type == "cuda" else torch.no_grad():
        with torch.no_grad():
            generated = model.backbone.generate(
                **batch, max_new_tokens=256, do_sample=True,
                num_return_sequences=N_CANDIDATES, temperature=TEMPERATURE, top_p=TOP_P,
                use_cache=True,
                logits_processor=LogitsProcessorList([TrajSpanLogitsProcessor(contract)]),
                stopping_criteria=StoppingCriteriaList([StopOnTrajEndCriteria(contract)]),
            )
    cands = []
    for row in range(N_CANDIDATES):
        text = tokenizer.decode(generated[row, int(prompt_ids.shape[1]):].tolist(), skip_special_tokens=False)
        cands.append([int(m.group(1)) for m in re.finditer(r"<i(\d+)>", text)])
    return cands


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("loading model...")
    model, tokenizer, processor, base_model = load_model(device)
    decoder = TrajectoryTokenDecoder(config_path=resolve_traj_tokenizer_config_path(base_model))
    corpus = {r["sample_id"]: r for r in load_jsonl(CORPUS_JSONL)}

    greedy_vs_gt, greedy_fde_vs_gt = [], []
    best4_vs_gt,  best4_fde_vs_gt  = [], []
    teacher_vs_gt, teacher_fde_vs_gt = [], []

    # also load greedy summary for greedy tokens
    summary_by_id = {str(s["sample_id"]): s for s in json.loads(
        (DISTILL_ROOT / "outputs/reports/no_nav_distill/full_free_run_eval_step006250_20260527_batched/step_006250_val_full_4760_b16_summary.json").read_text()
    )["samples"]}

    print(f"evaluating {len(corpus)} samples...")
    for i, (sid, row) in enumerate(corpus.items(), 1):
        hist = load_ego_history_xyz(row, DISTILL_ROOT)
        rot  = load_ego_history_rot(row, DISTILL_ROOT)
        try:
            gt_xyz = load_ego_future_xyz(row, DISTILL_ROOT)
        except Exception:
            gt_toks = load_gt_tokens(row)
            gt_xyz = decoder.decode(hist, rot, gt_toks) if gt_toks else None
        if gt_xyz is None or gt_xyz.size == 0:
            print(f"  [{i}] {sid[:40]} skip (no GT)")
            continue

        # greedy from existing summary
        s = summary_by_id.get(sid)
        if s:
            greedy_toks   = [int(t) for t in (s.get("generated_traj_tokens") or [])]
            teacher_toks  = [int(t) for t in (s.get("target_traj_tokens") or [])]
            greedy_xyz    = decoder.decode(hist, rot, greedy_toks)  if greedy_toks  else None
            teacher_xyz   = decoder.decode(hist, rot, teacher_toks) if teacher_toks else None
            ga, gf = ade_fde(greedy_xyz,  gt_xyz)
            ta, tf = ade_fde(teacher_xyz, gt_xyz)
            if math.isfinite(ga): greedy_vs_gt.append(ga); greedy_fde_vs_gt.append(gf)
            if math.isfinite(ta): teacher_vs_gt.append(ta); teacher_fde_vs_gt.append(tf)

        # generate N candidates
        cand_tokens = generate_candidates(model, tokenizer, processor, row, device=device, history_xyz=hist)
        cand_ades, cand_fdes = [], []
        for toks in cand_tokens:
            xyz = decoder.decode(hist, rot, toks) if toks else None
            a, f = ade_fde(xyz, gt_xyz)
            if math.isfinite(a): cand_ades.append(a); cand_fdes.append(f)
        if cand_ades:
            best4_vs_gt.append(min(cand_ades))
            best4_fde_vs_gt.append(min(cand_fdes))

        print(f"  [{i}/{len(corpus)}] {sid[:44]}  greedy={ga:.3f}  best4={min(cand_ades) if cand_ades else float('nan'):.3f}  (vs GT)")

    n = len(greedy_vs_gt)
    n4 = len(best4_vs_gt)
    print(f"\n=== RESULTS (vs GT) ===")
    print(f"n={n}")
    print(f"greedy student:  ADE={sum(greedy_vs_gt)/n:.4f}m  FDE={sum(greedy_fde_vs_gt)/n:.4f}m")
    print(f"best-of-{N_CANDIDATES} student: ADE={sum(best4_vs_gt)/n4:.4f}m  FDE={sum(best4_fde_vs_gt)/n4:.4f}m")
    print(f"teacher:         ADE={sum(teacher_vs_gt)/n:.4f}m  FDE={sum(teacher_fde_vs_gt)/n:.4f}m")

if __name__ == "__main__":
    main()
