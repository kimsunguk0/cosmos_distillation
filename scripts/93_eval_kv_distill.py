#!/usr/bin/env python3
"""Evaluation script for KV-cache distillation.

Computes on a val split:
  1. CE loss  (language modelling quality)
  2. KV distillation loss  (same objective as training, on held-out data)
  3. Layer-wise cosine similarity  K and V each, before vs after training

Usage
-----
python scripts/93_eval_kv_distill.py \
    --checkpoint outputs/kv_distill_pipeline/run_20260527/final.pt \
    [--baseline]          # also evaluate untrained student for comparison
    [--num-samples 500]   # subset of val for speed
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
for p in (PROJECT_ROOT, SUKIM_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.training.losses import kv_cache_distillation_loss               # noqa: E402
from src.utils.runtime_paths import remap_external_path                  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────

DEFAULT_CORPUS   = PROJECT_ROOT / "data" / "corpus" / "kv_distill_7k_balanced.jsonl"
DEFAULT_STUDENT  = str(SUKIM_ROOT / "base_weights" / "cosmos-reason-2b")
DEFAULT_TEACHER  = str(SUKIM_ROOT / "base_weights" / "alpamayo15_vlm_weights")
DEFAULT_CKPT     = PROJECT_ROOT / "outputs" / "kv_distill_pipeline" / "run_20260527" / "final.pt"
IGNORE_INDEX     = -100

LAYER_MAPPING: list[tuple[int, int]] = [
    (0, 0), (1, 1), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 8), (7, 9), (8, 10), (9, 12), (10, 13), (11, 14),
    (12, 16), (13, 17), (14, 18), (15, 19), (16, 21), (17, 22),
    (18, 23), (19, 25), (20, 26), (21, 27), (22, 29), (23, 30),
    (24, 31), (25, 32), (26, 34), (27, 35),
]

# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",      type=Path, default=DEFAULT_CKPT)
    p.add_argument("--corpus-jsonl",    type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--student-model",   default=DEFAULT_STUDENT)
    p.add_argument("--teacher-vlm",     default=DEFAULT_TEACHER)
    p.add_argument("--split",           default="val")
    p.add_argument("--num-samples",     type=int, default=500,
                   help="Max val samples to eval (default 500 for speed).")
    p.add_argument("--batch-size",      type=int, default=4)
    p.add_argument("--max-length",      type=int, default=2048)
    p.add_argument("--kv-huber-delta",  type=float, default=1.0)
    p.add_argument("--device",          default="cuda:0")
    p.add_argument("--dtype",           choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--baseline",        action="store_true",
                   help="Also evaluate the untrained student for comparison.")
    p.add_argument("--output-json",     type=Path, default=None,
                   help="Where to save results (default: next to checkpoint).")
    p.add_argument("--attn-implementation",
                   choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    p.add_argument("--seed",            type=int, default=97)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


# ── data ───────────────────────────────────────────────────────────────────

def _resolve(raw: Any) -> Path | None:
    r = remap_external_path(raw)
    if r is None:
        return None
    p = Path(r)
    return p if p.exists() else None


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_items(corpus: list[dict], split: str, n: int | None) -> list[dict]:
    items = []
    for row in corpus:
        if row.get("split") != split:
            continue
        sp = _resolve((row.get("input") or {}).get("materialized_sample_path"))
        if sp is None:
            continue
        items.append({"row": row, "sample_dir": sp})
        if n is not None and len(items) >= n:
            break
    return items


def _teacher_cot(row: dict) -> str:
    for key in ("teacher_target", "hard_target"):
        t = (row.get(key) or {}).get("cot_text") or ""
        if t:
            return str(t)
    return ""


def load_images(item: dict) -> list[Any]:
    from PIL import Image
    imgs = []
    sp: Path = item["sample_dir"]
    for name in sorted(p.name for p in sp.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
        try:
            imgs.append(Image.open(sp / name).convert("RGB"))
        except Exception:
            pass
    return imgs


def encode_batch(processor, tokenizer, items, max_length, device):
    ids_list, lab_list, pv_list, thw_list = [], [], [], []
    for item in items:
        images = load_images(item)
        cot    = _teacher_cot(item["row"])
        user_content = [{"type": "image"} for _ in images] + \
                       [{"type": "text", "text": "Describe the driving scene and decide the action."}]
        messages = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": cot or "The vehicle should proceed safely."},
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if images:
                enc = processor(text=[text], images=images, return_tensors="pt",
                                truncation=True, max_length=max_length, padding=False)
            else:
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=max_length)
        except Exception:
            enc = tokenizer(f"Describe the driving scene.\n{cot}", return_tensors="pt",
                            truncation=True, max_length=max_length)
        ids_list.append(enc["input_ids"][0])
        lab_list.append(enc["input_ids"][0].clone())
        if "pixel_values"    in enc: pv_list.append(enc["pixel_values"])
        if "image_grid_thw"  in enc: thw_list.append(enc["image_grid_thw"])

    maxs = max(t.shape[0] for t in ids_list)
    pad  = tokenizer.pad_token_id or 0
    pad_ids  = torch.full((len(items), maxs), pad,          dtype=torch.long)
    pad_labs = torch.full((len(items), maxs), IGNORE_INDEX, dtype=torch.long)
    attn     = torch.zeros((len(items), maxs),               dtype=torch.long)
    for i, (ids, labs) in enumerate(zip(ids_list, lab_list)):
        n = ids.shape[0]
        pad_ids[i, :n] = ids
        pad_labs[i, :n] = labs
        attn[i, :n] = 1

    out = {"input_ids": pad_ids.to(device), "attention_mask": attn.to(device),
           "labels": pad_labs.to(device)}
    if pv_list:  out["pixel_values"]   = torch.cat(pv_list,  dim=0).to(device)
    if thw_list: out["image_grid_thw"] = torch.cat(thw_list, dim=0).to(device)
    return out


# ── teacher forward ────────────────────────────────────────────────────────

def teacher_forward(teacher_vlm, batch, device, dtype):
    """Returns (teacher_kvs, teacher_image_embeds)."""
    visual_feats: dict[str, torch.Tensor] = {}
    hook = None
    pv = batch.get("pixel_values")
    if pv is not None:
        pv = pv.to(device=device, dtype=dtype)
        vis = None
        for path in ("visual", "model.visual"):
            obj = teacher_vlm
            try:
                for part in path.split("."): obj = getattr(obj, part)
                vis = obj; break
            except AttributeError: pass
        if vis is not None and hasattr(vis, "merger"):
            def _h(m, i, o): visual_feats["embeds"] = o.detach()
            hook = vis.merger.register_forward_hook(_h)

    with torch.no_grad():
        kw: dict[str, Any] = {"input_ids": batch["input_ids"].to(device),
                               "attention_mask": batch["attention_mask"].to(device),
                               "use_cache": True, "return_dict": True,
                               "output_hidden_states": False}
        if pv is not None:
            kw["pixel_values"]   = pv
            kw["image_grid_thw"] = batch["image_grid_thw"].to(device)
        try:
            out = teacher_vlm(**kw)
        except Exception:
            out = teacher_vlm.model(**kw)
        finally:
            if hook: hook.remove()

    pkv = getattr(out, "past_key_values", None)
    kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if pkv is not None:
        if hasattr(pkv, "key_cache"):
            for k, v in zip(pkv.key_cache, pkv.value_cache):
                kvs.append((k.detach().cpu(), v.detach().cpu()))
        else:
            for lkv in pkv:
                if isinstance(lkv, (tuple, list)) and len(lkv) >= 2:
                    kvs.append((lkv[0].detach().cpu(), lkv[1].detach().cpu()))
    return kvs, visual_feats.get("embeds")


# ── student forward ────────────────────────────────────────────────────────

def student_forward(student, batch, teacher_image_embeds, device, dtype):
    """Returns (logits, student_kvs)."""
    img_tok_id = getattr(student, "image_token_id", None)
    use_tv = (teacher_image_embeds is not None
              and student.vit_projection is not None
              and img_tok_id is not None)

    fwd = {"attention_mask": batch["attention_mask"],
           "use_cache": True, "return_dict": True, "output_hidden_states": False}

    if use_tv:
        t_emb = teacher_image_embeds.to(device=device, dtype=dtype)
        try:
            ie = student.embed_with_teacher_vit_features(batch["input_ids"], t_emb, img_tok_id)
            out = student.backbone(input_ids=None, inputs_embeds=ie, **fwd)
        except Exception as e:
            print(f"WARNING: teacher ViT inject failed ({e}), fallback to student ViT.", flush=True)
            use_tv = False

    if not use_tv:
        kw = dict(fwd)
        if batch.get("pixel_values")   is not None: kw["pixel_values"]   = batch["pixel_values"].to(dtype=dtype)
        if batch.get("image_grid_thw") is not None: kw["image_grid_thw"] = batch["image_grid_thw"]
        out = student.backbone(input_ids=batch["input_ids"], **kw)

    logits = getattr(out, "logits", None)
    if logits is None: raise ValueError("No logits from student backbone.")

    pkv = getattr(out, "past_key_values", None)
    kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if pkv is not None:
        if hasattr(pkv, "key_cache"):
            for k, v in zip(pkv.key_cache, pkv.value_cache): kvs.append((k, v))
        else:
            for lkv in pkv:
                if isinstance(lkv, (tuple, list)) and len(lkv) >= 2: kvs.append((lkv[0], lkv[1]))
    return logits, kvs


# ── metrics ────────────────────────────────────────────────────────────────

def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> float:
    sl = logits[..., :-1, :].contiguous()
    lb = labels[..., 1:].contiguous()
    return float(F.cross_entropy(sl.view(-1, sl.shape[-1]), lb.view(-1),
                                 ignore_index=IGNORE_INDEX).detach().cpu())


def layer_cosine(
    student_kvs: list[tuple[torch.Tensor, torch.Tensor]],
    teacher_kvs: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, list[float]]:
    """Per-layer mean cosine similarity for K and V."""
    k_sims, v_sims = [], []
    for s_idx, t_idx in LAYER_MAPPING:
        if s_idx >= len(student_kvs) or t_idx >= len(teacher_kvs):
            k_sims.append(float("nan")); v_sims.append(float("nan")); continue
        sK, sV = student_kvs[s_idx]
        tK, tV = teacher_kvs[t_idx]
        # flatten to [N, dim] then cosine
        sK = sK.reshape(-1, sK.shape[-1]).float()
        tK = tK.reshape(-1, tK.shape[-1]).to(device=sK.device).float()
        sV = sV.reshape(-1, sV.shape[-1]).float()
        tV = tV.reshape(-1, tV.shape[-1]).to(device=sV.device).float()
        k_sims.append(float(F.cosine_similarity(sK, tK, dim=-1).mean().cpu()))
        v_sims.append(float(F.cosine_similarity(sV, tV, dim=-1).mean().cpu()))
    return {"k_cosine": k_sims, "v_cosine": v_sims}


# ── main eval loop ─────────────────────────────────────────────────────────

def run_eval(
    *,
    label: str,
    student,
    teacher_vlm,
    items: list[dict],
    processor,
    tokenizer,
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    student.eval()
    n_batches = 0
    ce_acc = 0.0
    kv_loss_acc = 0.0
    k_cos_acc  = [0.0] * len(LAYER_MAPPING)
    v_cos_acc  = [0.0] * len(LAYER_MAPPING)

    batches = [items[i: i + args.batch_size] for i in range(0, len(items), args.batch_size)]
    t0 = time.perf_counter()

    for bi, batch_items in enumerate(batches):
        batch = encode_batch(processor, tokenizer, batch_items, args.max_length, device)
        labels = batch.pop("labels")

        # teacher
        teacher_kvs, teacher_img_embs = teacher_forward(teacher_vlm, batch, device, dtype)
        t_kvs_dev = [(k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype))
                     for k, v in teacher_kvs]

        # student
        with torch.no_grad():
            logits, student_kvs = student_forward(student, batch, teacher_img_embs, device, dtype)

        # CE
        ce_acc += ce_loss(logits, labels)

        # KV loss
        if student_kvs and t_kvs_dev and len(t_kvs_dev) > max(t for _, t in LAYER_MAPPING):
            kv, _ = kv_cache_distillation_loss(
                student_kvs, t_kvs_dev, LAYER_MAPPING,
                huber_delta=args.kv_huber_delta)
            kv_loss_acc += float(kv.detach().cpu())

        # cosine similarity
        if student_kvs and teacher_kvs:
            cos = layer_cosine(student_kvs, teacher_kvs)
            for i, (kc, vc) in enumerate(zip(cos["k_cosine"], cos["v_cosine"])):
                if not (kc != kc):  # nan check
                    k_cos_acc[i] += kc
                    v_cos_acc[i] += vc

        n_batches += 1
        del batch, labels, logits, student_kvs, teacher_kvs, t_kvs_dev
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if (bi + 1) % 20 == 0:
            print(json.dumps({"eval": label, "batch": bi + 1, "of": len(batches),
                              "elapsed_sec": round(time.perf_counter() - t0, 1)}), flush=True)

    if n_batches == 0:
        return {}

    layer_results = []
    for i, (s_idx, t_idx) in enumerate(LAYER_MAPPING):
        layer_results.append({
            "student_layer": s_idx,
            "teacher_layer": t_idx,
            "k_cosine": round(k_cos_acc[i] / n_batches, 6),
            "v_cosine": round(v_cos_acc[i] / n_batches, 6),
        })

    k_mean = float(np.mean([r["k_cosine"] for r in layer_results]))
    v_mean = float(np.mean([r["v_cosine"] for r in layer_results]))

    return {
        "label": label,
        "n_batches": n_batches,
        "n_samples": len(items),
        "ce_loss":      round(ce_acc      / n_batches, 6),
        "kv_loss":      round(kv_loss_acc / n_batches, 6),
        "k_cosine_mean": round(k_mean, 6),
        "v_cosine_mean": round(v_mean, 6),
        "layer_results": layer_results,
        "elapsed_sec":  round(time.perf_counter() - t0, 1),
    }


# ── entry point ────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype  = _dtype(args.dtype)

    out_json = args.output_json
    if out_json is None:
        out_json = args.checkpoint.parent / f"eval_{args.split}_{args.num_samples}samples.json"

    # ── teacher ──────────────────────────────────────────────────────────
    print(json.dumps({"event": "loading_teacher"}), flush=True)
    teacher_vlm = AutoModelForVision2Seq.from_pretrained(
        args.teacher_vlm, dtype=dtype, trust_remote_code=True).to(device)
    teacher_vlm.eval()
    teacher_vlm.requires_grad_(False)
    print(json.dumps({"event": "teacher_loaded"}), flush=True)

    # ── tokenizer / processor ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.student_model, trust_remote_code=True)
    processor.tokenizer = tokenizer

    cfg = StudentWrapperConfig(
        student_model_name=args.student_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=Path(args.student_model).exists(),
        attn_implementation=args.attn_implementation,
        vit_in_dim=4096,
        use_vit_projection=True,
    )

    # ── data ─────────────────────────────────────────────────────────────
    corpus = read_jsonl(args.corpus_jsonl)
    items  = select_items(corpus, args.split, args.num_samples)
    if not items:
        raise RuntimeError(f"No val items found in {args.corpus_jsonl}.")
    print(json.dumps({"event": "data_loaded", "n_items": len(items)}), flush=True)

    all_results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": len(items),
    }

    # ── eval: trained student ─────────────────────────────────────────────
    print(json.dumps({"event": "building_trained_student"}), flush=True)
    student = build_student_model(cfg, tokenizer).to(device=device, dtype=dtype)
    if args.checkpoint.exists():
        ckpt  = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("student_state_dict", ckpt)
        missing, unexpected = student.load_state_dict(state, strict=False)
        print(json.dumps({"event": "checkpoint_loaded",
                          "missing": len(missing), "unexpected": len(unexpected)}), flush=True)
    else:
        print(f"WARNING: checkpoint {args.checkpoint} not found; evaluating untrained student.", flush=True)

    result_trained = run_eval(
        label="trained",
        student=student, teacher_vlm=teacher_vlm,
        items=items, processor=processor, tokenizer=tokenizer,
        args=args, device=device, dtype=dtype,
    )
    all_results["trained"] = result_trained
    print(json.dumps({"event": "trained_eval_done", **{k: v for k, v in result_trained.items()
                                                        if k != "layer_results"}}), flush=True)

    # ── eval: baseline (untrained) ────────────────────────────────────────
    if args.baseline:
        print(json.dumps({"event": "building_baseline_student"}), flush=True)
        del student
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        student_base = build_student_model(cfg, tokenizer).to(device=device, dtype=dtype)
        result_base  = run_eval(
            label="baseline_untrained",
            student=student_base, teacher_vlm=teacher_vlm,
            items=items, processor=processor, tokenizer=tokenizer,
            args=args, device=device, dtype=dtype,
        )
        all_results["baseline"] = result_base
        print(json.dumps({"event": "baseline_eval_done", **{k: v for k, v in result_base.items()
                                                             if k != "layer_results"}}), flush=True)

        # delta summary
        if result_trained and result_base:
            delta = {
                "ce_loss_delta":      round(result_trained["ce_loss"]      - result_base["ce_loss"],      6),
                "kv_loss_delta":      round(result_trained["kv_loss"]      - result_base["kv_loss"],      6),
                "k_cosine_delta":     round(result_trained["k_cosine_mean"]- result_base["k_cosine_mean"],6),
                "v_cosine_delta":     round(result_trained["v_cosine_mean"]- result_base["v_cosine_mean"],6),
            }
            all_results["delta_trained_vs_baseline"] = delta
            print(json.dumps({"event": "delta", **delta}), flush=True)

    # ── save ──────────────────────────────────────────────────────────────
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "saved", "path": str(out_json)}), flush=True)

    # ── pretty summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EVAL SUMMARY  ({args.split}, {len(items)} samples)")
    print("=" * 60)
    for label, res in [("trained", result_trained)] + \
                      ([("baseline", all_results.get("baseline", {}))] if args.baseline else []):
        if not res: continue
        print(f"\n  [{label}]")
        print(f"    CE loss        : {res.get('ce_loss', 'N/A'):.4f}")
        print(f"    KV loss        : {res.get('kv_loss', 'N/A'):.4f}")
        print(f"    K cosine mean  : {res.get('k_cosine_mean', 'N/A'):.4f}")
        print(f"    V cosine mean  : {res.get('v_cosine_mean', 'N/A'):.4f}")
        lr = res.get("layer_results", [])
        if lr:
            print(f"    Layer cosines (K | V):")
            for r in lr[::4]:   # every 4th layer for brevity
                print(f"      s{r['student_layer']:>2d}→t{r['teacher_layer']:>2d}  "
                      f"K={r['k_cosine']:.4f}  V={r['v_cosine']:.4f}")
    if args.baseline and "delta_trained_vs_baseline" in all_results:
        d = all_results["delta_trained_vs_baseline"]
        print(f"\n  [delta: trained − baseline]")
        print(f"    ΔCE  : {d['ce_loss_delta']:+.4f}")
        print(f"    ΔKV  : {d['kv_loss_delta']:+.4f}")
        print(f"    ΔK cos: {d['k_cosine_delta']:+.4f}")
        print(f"    ΔV cos: {d['v_cosine_delta']:+.4f}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
