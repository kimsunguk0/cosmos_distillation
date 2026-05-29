#!/usr/bin/env python3
"""Compute best-of-N ADE/FDE vs GT for ALL 4760 val samples.
Saves incremental results so it can be resumed if interrupted.
"""
from __future__ import annotations
import json, math, re, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList

DISTILL_ROOT = Path("/home/pm97/workspace/sukim/distillation/cosmos_distillation")
CKPT_DIR = (
    DISTILL_ROOT / "outputs/checkpoints/no_nav_camera_labeled_official_full444k"
    / "no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
)
CORPUS_JSONL  = DISTILL_ROOT / "data/corpus/no_nav_teacher_pair_300chunks.jsonl"
SUMMARY_JSON  = (
    DISTILL_ROOT
    / "outputs/reports/no_nav_distill/full_free_run_eval_step006250_20260527_batched"
    / "step_006250_val_full_4760_b16_summary.json"
)
BASE_MODEL    = "/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b"
SAVE_PATH     = DISTILL_ROOT / "outputs/reports/no_nav_distill/best_of_4_full4760_results.jsonl"
N_CANDIDATES  = 4
TEMPERATURE   = 1.0
TOP_P         = 0.95

if str(DISTILL_ROOT) not in sys.path:
    sys.path.insert(0, str(DISTILL_ROOT))

from src.inference.checkpoint_eval import (
    TrajectoryTokenDecoder, load_ego_history_rot, resolve_traj_tokenizer_config_path,
)
from src.inference.decoding import TrajDecodingContract, TrajSpanLogitsProcessor, StopOnTrajEndCriteria
from src.model.student_wrapper import StudentWrapperConfig, build_student_model
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.tokenizer_ext import distill_trainable_token_ids
from src.training.collator import (
    build_messages, build_user_prompt, fuse_history_tokens_in_input_ids,
    load_ego_future_xyz, load_ego_history_xyz, load_sample_images,
    load_traj_future_token_ids, resolve_camera_indices,
)


def ade_fde(a, b):
    if a is None or b is None or a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    n = min(a.shape[0], b.shape[0])
    d = np.linalg.norm(a[:n, :2] - b[:n, :2], axis=-1)
    return float(d.mean()), float(d[-1])


def load_model(device):
    ckpt = CKPT_DIR
    train_cfg = json.loads((ckpt / "train_config.json").read_text()) if (ckpt / "train_config.json").exists() else {}
    manifest  = json.loads((ckpt / "checkpoint_manifest.json").read_text()) if (ckpt / "checkpoint_manifest.json").exists() else {}
    base_model = str((train_cfg.get("args") or {}).get("student_model") or BASE_MODEL)
    use_lora   = not bool((train_cfg.get("args") or {}).get("disable_lora", False))
    data_view  = train_cfg.get("data_view") or {}
    tok_dir, proc_dir = ckpt / "tokenizer", ckpt / "processor"
    tokenizer = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else base_model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(proc_dir if proc_dir.exists() else base_model, local_files_only=True)
    processor.tokenizer = tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else None
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_cfg.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=dtype, local_files_only=Path(base_model).expanduser().exists(),
        traj_teacher_hidden_size=int(data_view["teacher_traj_hidden_size"]) if data_view.get("teacher_traj_hidden_size") not in (None, "", 0) else None,
        traj_hidden_bridge_size=int(manifest["traj_hidden_bridge_size"]) if manifest.get("traj_hidden_bridge_size") not in (None, "", 0) else None,
    )
    model = build_student_model(wrapper_cfg, tokenizer)
    fmt = detect_checkpoint_format(ckpt)
    if fmt == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_student_checkpoint(ckpt, model, use_lora=use_lora)
    return model.to(device).eval(), tokenizer, processor


def generate_candidates(model, tokenizer, processor, sample, *, device, history_xyz):
    images = load_sample_images(sample, DISTILL_ROOT)
    cam_idx = resolve_camera_indices(sample, DISTILL_ROOT, image_count=len(images))
    fpc = max(len(images) // max(len(cam_idx), 1), 1)
    prompt_text = build_user_prompt(sample, DISTILL_ROOT, ego_history_xyz=history_xyz, prompt_text_style="official_alpamayo")
    messages = build_messages(
        prompt_text, len(images), assistant_prefix="<|cot_start|>",
        image_prompt_style="camera_labeled", camera_indices=cam_idx, num_frames_per_camera=fpc,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
    batch = processor(text=[text], images=[images], return_tensors="pt", padding=True, truncation=True)
    batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, [history_xyz])
    target_tokens = load_traj_future_token_ids(sample.get("hard_target") or {}, DISTILL_ROOT)
    vdtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    batch = {
        k: (v.to(device=device, dtype=vdtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
            else v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    prompt_ids = batch["input_ids"]
    contract = TrajDecodingContract.from_tokenizer(
        tokenizer,
        prompt_lengths=[int(prompt_ids.shape[1])],
        traj_token_count=len(target_tokens),
    )
    ctx = torch.autocast("cuda", dtype=vdtype) if device.type == "cuda" else torch.no_grad()
    with ctx:
        with torch.no_grad():
            generated = model.backbone.generate(
                **batch, max_new_tokens=256, do_sample=True,
                num_return_sequences=N_CANDIDATES, temperature=TEMPERATURE, top_p=TOP_P,
                use_cache=True,
                logits_processor=LogitsProcessorList([TrajSpanLogitsProcessor(contract)]),
                stopping_criteria=StoppingCriteriaList([StopOnTrajEndCriteria(contract)]),
            )
    cands = []
    for row_idx in range(N_CANDIDATES):
        decoded_text = tokenizer.decode(
            generated[row_idx, int(prompt_ids.shape[1]):].tolist(), skip_special_tokens=False
        )
        cands.append([int(m.group(1)) for m in re.finditer(r"<i(\d+)>", decoded_text)])
    return cands


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load already-processed sample ids for resume
    done_ids: set[str] = set()
    if SAVE_PATH.exists():
        for line in SAVE_PATH.open():
            if line.strip():
                done_ids.add(json.loads(line)["sample_id"])
    print(f"Resuming — {len(done_ids)} samples already done", flush=True)

    print("Loading corpus (val split)...", flush=True)
    corpus = []
    with CORPUS_JSONL.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "val" and row["sample_id"] not in done_ids:
                corpus.append(row)
    print(f"  {len(corpus)} samples remaining", flush=True)

    if not corpus:
        print("All done!", flush=True)
        _print_summary()
        return

    print("Loading summary for greedy tokens...", flush=True)
    summary = json.loads(SUMMARY_JSON.read_text())
    by_id = {s["sample_id"]: s for s in summary["samples"]}

    print("Loading model...", flush=True)
    model, tokenizer, processor = load_model(device)
    decoder = TrajectoryTokenDecoder(config_path=resolve_traj_tokenizer_config_path(BASE_MODEL))

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = SAVE_PATH.open("a")

    t0 = time.time()
    for i, row in enumerate(corpus, 1):
        sid = row["sample_id"]
        try:
            hist = load_ego_history_xyz(row, DISTILL_ROOT)
            rot  = load_ego_history_rot(row, DISTILL_ROOT)
            gt_xyz = load_ego_future_xyz(row, DISTILL_ROOT)
        except Exception as e:
            print(f"  [{i}] {sid[:44]} SKIP load error: {e}", flush=True)
            continue
        if gt_xyz is None or gt_xyz.size == 0:
            print(f"  [{i}] {sid[:44]} SKIP no GT", flush=True)
            continue

        # greedy from summary
        s = by_id.get(sid)
        greedy_ade_val = greedy_fde_val = float("nan")
        teacher_ade_val = teacher_fde_val = float("nan")
        if s:
            g_xyz = decoder.decode(hist, rot, [int(t) for t in (s.get("generated_traj_tokens") or [])])
            t_xyz = decoder.decode(hist, rot, [int(t) for t in (s.get("target_traj_tokens") or [])])
            greedy_ade_val, greedy_fde_val = ade_fde(g_xyz, gt_xyz)
            teacher_ade_val, teacher_fde_val = ade_fde(t_xyz, gt_xyz)

        # sample N candidates
        try:
            cand_tokens = generate_candidates(model, tokenizer, processor, row, device=device, history_xyz=hist)
        except Exception as e:
            print(f"  [{i}] {sid[:44]} SKIP gen error: {e}", flush=True)
            continue

        cand_ades, cand_fdes = [], []
        for toks in cand_tokens:
            xyz = decoder.decode(hist, rot, toks) if toks else None
            a, f = ade_fde(xyz, gt_xyz)
            if math.isfinite(a):
                cand_ades.append(a)
                cand_fdes.append(f)

        result = {
            "sample_id": sid,
            "greedy_ade": greedy_ade_val,
            "greedy_fde": greedy_fde_val,
            "teacher_ade": teacher_ade_val,
            "teacher_fde": teacher_fde_val,
            "best4_ade": min(cand_ades) if cand_ades else float("nan"),
            "best4_fde": min(cand_fdes) if cand_fdes else float("nan"),
            "n_valid_cands": len(cand_ades),
        }
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

        elapsed = time.time() - t0
        speed = i / elapsed
        eta = (len(corpus) - i) / speed if speed > 0 else 0
        print(
            f"  [{i+len(done_ids)}/{len(corpus)+len(done_ids)}] {sid[:40]}"
            f"  greedy={greedy_ade_val:.3f}  best4={result['best4_ade']:.3f}"
            f"  speed={speed:.2f}/s  ETA={eta/60:.0f}min",
            flush=True,
        )

    out_f.close()
    _print_summary()


def _print_summary():
    if not SAVE_PATH.exists():
        print("No results file found.", flush=True)
        return
    rows = [json.loads(l) for l in SAVE_PATH.open() if l.strip()]
    if not rows:
        print("No results.", flush=True)
        return

    def mean_finite(vals):
        v = [x for x in vals if math.isfinite(x)]
        return sum(v) / len(v) if v else float("nan"), len(v)

    g_ade, ng = mean_finite([r["greedy_ade"] for r in rows])
    g_fde, _  = mean_finite([r["greedy_fde"] for r in rows])
    t_ade, nt = mean_finite([r["teacher_ade"] for r in rows])
    t_fde, _  = mean_finite([r["teacher_fde"] for r in rows])
    b_ade, nb = mean_finite([r["best4_ade"] for r in rows])
    b_fde, _  = mean_finite([r["best4_fde"] for r in rows])

    print(f"\n=== RESULTS (vs true GT, n_total={len(rows)}) ===", flush=True)
    print(f"Greedy student  (n={ng}): ADE={g_ade:.4f}m  FDE={g_fde:.4f}m", flush=True)
    print(f"Teacher         (n={nt}): ADE={t_ade:.4f}m  FDE={t_fde:.4f}m", flush=True)
    print(f"Best-of-{N_CANDIDATES} student (n={nb}): ADE={b_ade:.4f}m  FDE={b_fde:.4f}m", flush=True)


if __name__ == "__main__":
    main()
