#!/usr/bin/env python3
"""Compute greedy vs GT ADE/FDE for all 4760 val samples using existing summary tokens.
No model inference needed — just decode stored tokens and load GT xyz.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

DISTILL_ROOT = Path("/home/pm97/workspace/sukim/distillation/cosmos_distillation")
CORPUS_JSONL = DISTILL_ROOT / "data/corpus/no_nav_teacher_pair_300chunks.jsonl"
SUMMARY_JSON = (
    DISTILL_ROOT
    / "outputs/reports/no_nav_distill/full_free_run_eval_step006250_20260527_batched"
    / "step_006250_val_full_4760_b16_summary.json"
)
BASE_MODEL = "/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b"

if str(DISTILL_ROOT) not in sys.path:
    sys.path.insert(0, str(DISTILL_ROOT))

from src.inference.checkpoint_eval import (
    TrajectoryTokenDecoder,
    load_ego_history_rot,
    resolve_traj_tokenizer_config_path,
)
from src.training.collator import load_ego_future_xyz, load_ego_history_xyz


def ade_fde(a, b):
    if a is None or b is None or a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    n = min(a.shape[0], b.shape[0])
    d = np.linalg.norm(a[:n, :2] - b[:n, :2], axis=-1)
    return float(d.mean()), float(d[-1])


def main():
    print("Loading summary...", flush=True)
    summary = json.loads(SUMMARY_JSON.read_text())
    by_id = {s["sample_id"]: s for s in summary["samples"]}
    print(f"  {len(by_id)} samples in summary", flush=True)

    print("Loading full val corpus (filtering split=val)...", flush=True)
    corpus = []
    with CORPUS_JSONL.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "val":
                corpus.append(row)
    print(f"  {len(corpus)} val samples in corpus", flush=True)

    decoder = TrajectoryTokenDecoder(
        config_path=resolve_traj_tokenizer_config_path(BASE_MODEL)
    )

    greedy_ade, greedy_fde_list = [], []
    teacher_ade, teacher_fde_list = [], []
    skipped = 0
    t0 = time.time()

    for i, row in enumerate(corpus, 1):
        sid = row["sample_id"]
        s = by_id.get(sid)
        if s is None:
            skipped += 1
            continue

        try:
            hist = load_ego_history_xyz(row, DISTILL_ROOT)
            rot = load_ego_history_rot(row, DISTILL_ROOT)
            gt_xyz = load_ego_future_xyz(row, DISTILL_ROOT)
        except Exception as e:
            skipped += 1
            continue

        if gt_xyz is None or gt_xyz.size == 0:
            skipped += 1
            continue

        greedy_toks = [int(t) for t in (s.get("generated_traj_tokens") or [])]
        teacher_toks = [int(t) for t in (s.get("target_traj_tokens") or [])]

        if greedy_toks:
            greedy_xyz = decoder.decode(hist, rot, greedy_toks)
            ga, gf = ade_fde(greedy_xyz, gt_xyz)
            if math.isfinite(ga):
                greedy_ade.append(ga)
                greedy_fde_list.append(gf)

        if teacher_toks:
            teacher_xyz = decoder.decode(hist, rot, teacher_toks)
            ta, tf = ade_fde(teacher_xyz, gt_xyz)
            if math.isfinite(ta):
                teacher_ade.append(ta)
                teacher_fde_list.append(tf)

        if i % 500 == 0:
            elapsed = time.time() - t0
            ng = len(greedy_ade)
            nt = len(teacher_ade)
            g_ade_str = f"{sum(greedy_ade)/ng:.4f}" if ng else "n/a"
            t_ade_str = f"{sum(teacher_ade)/nt:.4f}" if nt else "n/a"
            print(
                f"  [{i}/{len(corpus)}] {elapsed:.0f}s  greedy_ADE={g_ade_str}  teacher_ADE={t_ade_str}",
                flush=True,
            )

    print(f"\n=== RESULTS (vs true GT ego_future_xyz) ===", flush=True)
    print(f"Total corpus val samples : {len(corpus)}", flush=True)
    print(f"Skipped (no match/GT)    : {skipped}", flush=True)

    ng = len(greedy_ade)
    nt = len(teacher_ade)

    if ng:
        print(
            f"Greedy student  (n={ng}):  ADE={sum(greedy_ade)/ng:.4f}m  FDE={sum(greedy_fde_list)/ng:.4f}m",
            flush=True,
        )
    if nt:
        print(
            f"Teacher         (n={nt}):  ADE={sum(teacher_ade)/nt:.4f}m  FDE={sum(teacher_fde_list)/nt:.4f}m",
            flush=True,
        )

    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
