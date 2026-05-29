#!/usr/bin/env python3
"""Best-of-N ADE/FDE vs TEACHER for all 4760 val samples.
Batched generation: BATCH_SIZE samples × N_CANDIDATES per batch call.
Resumable via incremental JSONL output.
"""
from __future__ import annotations
import json, math, re, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList

DISTILL_ROOT  = Path("/home/pm97/workspace/sukim/distillation/cosmos_distillation")
CKPT_DIR      = (
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
SAVE_PATH     = DISTILL_ROOT / "outputs/reports/no_nav_distill/best_of_4_vs_teacher_full4760.jsonl"
N_CANDIDATES  = 4
BATCH_SIZE    = 16   # samples per generate() call
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
    load_ego_history_xyz, load_sample_images, load_traj_future_token_ids,
    resolve_camera_indices,
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
        torch_dtype=dtype,
        local_files_only=Path(base_model).expanduser().exists(),
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


def prepare_sample_input(sample, tokenizer, processor, device, history_xyz):
    """Return (text_str, images, actual_prompt_len, target_token_count)."""
    images  = load_sample_images(sample, DISTILL_ROOT)
    cam_idx = resolve_camera_indices(sample, DISTILL_ROOT, image_count=len(images))
    fpc     = max(len(images) // max(len(cam_idx), 1), 1)
    prompt_text = build_user_prompt(
        sample, DISTILL_ROOT, ego_history_xyz=history_xyz, prompt_text_style="official_alpamayo"
    )
    messages = build_messages(
        prompt_text, len(images), assistant_prefix="<|cot_start|>",
        image_prompt_style="camera_labeled", camera_indices=cam_idx, num_frames_per_camera=fpc,
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
    )
    # Encode solo to learn actual length before batching
    solo = processor(text=[text], images=[images], return_tensors="pt", padding=False, truncation=True)
    solo_ids = fuse_history_tokens_in_input_ids(solo["input_ids"], tokenizer, [history_xyz])
    actual_len = int(solo_ids.shape[1])
    target_tokens = load_traj_future_token_ids(sample.get("hard_target") or {}, DISTILL_ROOT)
    return text, images, history_xyz, actual_len, len(target_tokens)


def run_batch(model, tokenizer, processor, batch_items, device):
    """
    batch_items: list of (text, images, history_xyz, actual_len, traj_token_count)
    Returns: list of lists of candidate token sequences (one list per sample).
    """
    if len(batch_items) == 1:
        text, images, history_xyz, actual_len, ttc = batch_items[0]
        batch = processor(text=[text], images=[images], return_tensors="pt", padding=True, truncation=True)
        batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, [history_xyz])
        padded_len = int(batch["input_ids"].shape[1])
        contract = TrajDecodingContract.from_tokenizer(
            tokenizer, prompt_lengths=[padded_len], traj_token_count=ttc
        )
        vdtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        batch = {
            k: (v.to(device=device, dtype=vdtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                else v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.autocast("cuda", dtype=vdtype) if device.type == "cuda" else torch.no_grad():
            with torch.no_grad():
                generated = model.backbone.generate(
                    **batch, max_new_tokens=256, do_sample=True,
                    num_return_sequences=N_CANDIDATES,
                    temperature=TEMPERATURE, top_p=TOP_P, use_cache=True,
                    logits_processor=LogitsProcessorList([TrajSpanLogitsProcessor(contract)]),
                    stopping_criteria=StoppingCriteriaList([StopOnTrajEndCriteria(contract)]),
                )
        results = [[]]
        for c in range(N_CANDIDATES):
            decoded = tokenizer.decode(generated[c, padded_len:].tolist(), skip_special_tokens=False)
            results[0].append([int(m.group(1)) for m in re.finditer(r"<i(\d+)>", decoded)])
        return results

    # Multi-sample batch
    texts  = [it[0] for it in batch_items]
    images_list = [it[1] for it in batch_items]
    hists  = [it[2] for it in batch_items]
    ttcs   = [it[4] for it in batch_items]
    traj_token_count = ttcs[0]  # assume same across batch (same tokenizer)

    vdtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        batch = processor(text=texts, images=images_list, return_tensors="pt", padding=True, truncation=True)
        batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, hists)
        padded_len = int(batch["input_ids"].shape[1])
        # prompt_lengths: same padded length for all (left-padded)
        contract = TrajDecodingContract.from_tokenizer(
            tokenizer,
            prompt_lengths=[padded_len] * len(batch_items),
            traj_token_count=traj_token_count,
        )
        batch = {
            k: (v.to(device=device, dtype=vdtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                else v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.autocast("cuda", dtype=vdtype) if device.type == "cuda" else torch.no_grad():
            with torch.no_grad():
                generated = model.backbone.generate(
                    **batch, max_new_tokens=256, do_sample=True,
                    num_return_sequences=N_CANDIDATES,
                    temperature=TEMPERATURE, top_p=TOP_P, use_cache=True,
                    logits_processor=LogitsProcessorList([TrajSpanLogitsProcessor(contract)]),
                    stopping_criteria=StoppingCriteriaList([StopOnTrajEndCriteria(contract)]),
                )
        # generated shape: [B * N_CANDIDATES, seq_len]
        # output[i*N:(i+1)*N] = candidates for sample i
        results = [[] for _ in range(len(batch_items))]
        for i in range(len(batch_items)):
            for c in range(N_CANDIDATES):
                seq_idx = i * N_CANDIDATES + c
                decoded = tokenizer.decode(generated[seq_idx, padded_len:].tolist(), skip_special_tokens=False)
                results[i].append([int(m.group(1)) for m in re.finditer(r"<i(\d+)>", decoded)])
        return results

    except Exception as e:
        print(f"  Batch of {len(batch_items)} failed ({e}), falling back to single-sample", flush=True)
        results = []
        for item in batch_items:
            results.extend(run_batch(model, tokenizer, processor, [item], device))
        return results


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    done_ids: set[str] = set()
    if SAVE_PATH.exists():
        for line in SAVE_PATH.open():
            if line.strip():
                done_ids.add(json.loads(line)["sample_id"])
    print(f"Resuming — {len(done_ids)} samples already done", flush=True)

    print("Loading corpus (val split)...", flush=True)
    corpus = [
        row for row in (json.loads(l) for l in CORPUS_JSONL.open() if l.strip())
        if row.get("split") == "val" and row["sample_id"] not in done_ids
    ]
    print(f"  {len(corpus)} samples remaining", flush=True)

    if not corpus:
        _print_summary()
        return

    print("Loading summary for greedy/teacher tokens...", flush=True)
    summary_by_id = {s["sample_id"]: s for s in json.loads(SUMMARY_JSON.read_text())["samples"]}

    print("Loading model...", flush=True)
    model, tokenizer, processor = load_model(device)
    decoder = TrajectoryTokenDecoder(config_path=resolve_traj_tokenizer_config_path(BASE_MODEL))

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = SAVE_PATH.open("a")

    t0 = time.time()
    total_done = len(done_ids)
    total_all  = total_done + len(corpus)
    i = 0

    while i < len(corpus):
        chunk = corpus[i:i + BATCH_SIZE]

        # Prepare inputs
        items = []
        valid_rows = []
        for row in chunk:
            try:
                hist = load_ego_history_xyz(row, DISTILL_ROOT)
                rot  = load_ego_history_rot(row, DISTILL_ROOT)
                text, images, hxyz, actual_len, ttc = prepare_sample_input(
                    row, tokenizer, processor, device, hist
                )
                items.append((text, images, hxyz, actual_len, ttc))
                valid_rows.append((row, hist, rot))
            except Exception as e:
                print(f"  SKIP {row['sample_id'][:44]}: {e}", flush=True)

        if not items:
            i += len(chunk)
            continue

        try:
            cand_lists = run_batch(model, tokenizer, processor, items, device)
        except Exception as e:
            print(f"  Batch error: {e}", flush=True)
            i += len(chunk)
            continue

        for j, ((row, hist, rot), cand_tokens) in enumerate(zip(valid_rows, cand_lists)):
            sid = row["sample_id"]
            s   = summary_by_id.get(sid)

            greedy_ade = greedy_fde = float("nan")
            teacher_ade = teacher_fde = float("nan")
            teacher_xyz = None

            if s:
                g_toks = [int(t) for t in (s.get("generated_traj_tokens") or [])]
                t_toks = [int(t) for t in (s.get("target_traj_tokens") or [])]
                teacher_xyz = decoder.decode(hist, rot, t_toks) if t_toks else None
                g_xyz = decoder.decode(hist, rot, g_toks) if g_toks else None
                greedy_ade,  greedy_fde  = ade_fde(g_xyz,       teacher_xyz)
                teacher_ade, teacher_fde = ade_fde(teacher_xyz, teacher_xyz)  # sanity: should be 0

            cand_ades, cand_fdes = [], []
            for toks in cand_tokens:
                xyz = decoder.decode(hist, rot, toks) if toks else None
                a, f = ade_fde(xyz, teacher_xyz)
                if math.isfinite(a):
                    cand_ades.append(a)
                    cand_fdes.append(f)

            result = {
                "sample_id":  sid,
                "greedy_ade": greedy_ade,
                "greedy_fde": greedy_fde,
                "best4_ade":  min(cand_ades) if cand_ades else float("nan"),
                "best4_fde":  min(cand_fdes) if cand_fdes else float("nan"),
                "n_valid_cands": len(cand_ades),
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

        total_done += len(valid_rows)
        elapsed = time.time() - t0
        speed   = total_done / elapsed
        eta     = (total_all - total_done) / speed if speed > 0 else 0
        ids_str = " ".join(r["sample_id"][:20] for r, _, _ in valid_rows[:2])
        best4s  = [min(r["best4_ade"] for r in [json.loads(l) for l in SAVE_PATH.open()][-len(valid_rows):] if math.isfinite(r["best4_ade"]))] if valid_rows else []
        print(
            f"  [{total_done}/{total_all}] speed={speed:.2f}/s  ETA={eta/60:.0f}min  {ids_str}",
            flush=True,
        )

        i += len(chunk)

    out_f.close()
    _print_summary()


def _print_summary():
    if not SAVE_PATH.exists():
        print("No results.", flush=True)
        return
    rows = [json.loads(l) for l in SAVE_PATH.open() if l.strip()]
    if not rows:
        return
    def mf(vals):
        v = [x for x in vals if math.isfinite(x)]
        return (sum(v)/len(v) if v else float("nan")), len(v)

    g_ade, ng = mf([r["greedy_ade"] for r in rows])
    g_fde, _  = mf([r["greedy_fde"] for r in rows])
    b_ade, nb = mf([r["best4_ade"]  for r in rows])
    b_fde, _  = mf([r["best4_fde"]  for r in rows])
    print(f"\n=== RESULTS (vs TEACHER, n_total={len(rows)}) ===", flush=True)
    print(f"Greedy   (n={ng}): ADE={g_ade:.4f}m  FDE={g_fde:.4f}m", flush=True)
    print(f"Best-of-{N_CANDIDATES} (n={nb}): ADE={b_ade:.4f}m  FDE={b_fde:.4f}m", flush=True)


if __name__ == "__main__":
    main()
