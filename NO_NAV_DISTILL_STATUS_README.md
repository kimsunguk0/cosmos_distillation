# No-Nav Alpamayo 1.5 Distillation Status

Last updated: 2026-05-06

This note summarizes the current non-human OOD no-nav distillation dataset/cache work.

## Goal

Build an Alpamayo 1.5 teacher-pair distillation cache from the non-human OOD dataset:

- Input: 4 cameras x 4 temporal images
- Ego history: 1.6 sec
- Ego future target horizon: 6.4 sec
- Sample anchors: 8 samples per clip, roughly 1.6s to 12.8s at 1.6s intervals
- Initial teacher mode: no navigation
- Later plan: add selected nav-conditioned samples and merge with no-nav cache

## Main Paths

- Source OOD dataset:
  `/home/pm97/workspace/dataset/physical_ai_av_ood_dataset`
- Reference human CoC materialized layout:
  `/home/pm97/workspace/dataset/human_coc_dataset/materialized`
- Working distill dataset root:
  `/home/pm97/workspace/dataset/distill_dataset`
- Distillation repo:
  `/home/pm97/workspace/sukim/distillation/cosmos_distillation`
- Alpamayo 1.5 repo:
  `/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5`
- Alpamayo 1.5 weights:
  `/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B`
- Student base:
  `/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b`

## Dataset Directory Policy

`/home/pm97/workspace/dataset/distill_dataset` should stay clean:

```text
distill_dataset/
  materialized/
  requests/
  manifests/
  reports/
  logs/
  teacher_cache/
```

Teacher cache layout:

```text
teacher_cache/
  no_nav/
    text/
      raw_outputs/
      text_tokens/
      text_topk/
      boundary_hidden/
      text_hidden/
      text_outputs/
      action_expert/
      manifest/
    traj/
      tokens/
      topk/
      hidden/
      decoded/
      outputs/
  nav/
    text/
    traj/
```

Temporary/test outputs should go under dataset `not_used/`, not under the clean distill root.

## Materialized Dataset

Materialized samples are stored under:

```text
/home/pm97/workspace/dataset/distill_dataset/materialized
```

Each sample contains:

- `images/cam{0..3}_f{0..3}.png`
- `ego/ego_history_xyz.npy`
- `ego/ego_history_rot.npy`
- future/metadata files used for QC and teacher-cache joins

The expected full no-nav sample count is about 444k samples.

## Teacher Cache Progress

Completed no-nav ranges:

| chunk range | samples |
|---|---:|
| 0-249 | 194,290 |
| 250-299 | 39,366 |
| 300-399 | 76,865 |
| 400-449 | 37,230 |

Current completed total:

```text
347,751 samples
```

Remaining estimate:

```text
chunks 450-573 ~= 96,688 samples
```

Latest completed report:

```text
/home/pm97/workspace/dataset/distill_dataset/reports/no_nav/next50_after_400
```

Latest dashboard/log symlink:

```text
/home/pm97/workspace/dataset/distill_dataset/logs/no_nav_next50_latest
```

Dashboard server script:

```text
/home/pm97/workspace/sukim/distillation/dataset_prep/scripts/serve_next50_progress_dashboard.py
```

Usual SSH tunnel:

```bash
ssh -N -L 18767:127.0.0.1:8767 pm97@58.227.59.75
```

Then open:

```text
http://127.0.0.1:18767
```

## Teacher Cache Schema Direction

Important teacher fields/artifacts now planned or written:

- Teacher CoT text
- Teacher CoT token ids
- Teacher text top-k ids/logprobs/entropy/top1 margin
- Text boundary hidden:
  - `cot_end`
  - `traj_future_start`
  - `action_expert_pre`
- Teacher action expert trajectory output
- Teacher discrete future token ids, 128 tokens
- Teacher trajectory top-k, k=32
- Teacher trajectory hidden, `[128, 4096]`
- Discrete decoded trajectory
- Prompt/request/output hashes
- Coordinate metadata:
  - frame: `ego_at_sample_time`
  - axis: `x_forward_y_left_z_up`
  - units: meters
  - horizon: 6.4 sec

Current policy:

- Store big arrays as path artifacts, not giant inline manifest blobs.
- Use text top-k from generation logits during raw teacher inference for new ranges.
- Avoid separate teacher-forced replay unless recovering older/incomplete cache.

## Nav Label Work

Nav labels were inspected under:

```text
/home/pm97/workspace/dataset/nav_labels
```

Observed category distribution included:

- `straight`
- `lane_change_right`
- `lane_change_left`
- `straight_empty`
- `turn_left_now`
- `turn_right_now`
- `sharp_left/right`
- `curve_left/right`
- `exit_left/right`

Visualization outputs were made under:

```text
/home/pm97/workspace/sukim/visualization
```

Nav strategy discussed:

- Keep the full no-nav teacher cache.
- Add nav only to selected representative samples later.
- Avoid adding near-duplicate nav text to all 8 anchors in a clip.
- Likely choose 2-3 decisive nav samples per clip when nav is useful.

## Backbone Distillation Status

Current main no-nav corpus:

```text
/home/pm97/workspace/sukim/distillation/cosmos_distillation/data/corpus/no_nav_teacher_pair_300chunks.jsonl
```

Current student checkpoint:

```text
/home/pm97/workspace/sukim/distillation/cosmos_distillation/outputs/checkpoints/no_nav_bp3_h200fast_b4/no_nav_bp3_h200fast_b4_from_step2288_20260504_053208/final
```

LoRA setup:

- rank: 64
- alpha: 128
- target modules:
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

Training signal used so far:

- CoT CE
- CoT top-k KD
- Trajectory token CE
- Trajectory top-k KD
- Trajectory hidden / bridge-related losses in the current BP3 setup

Important interpretation:

- Teacher-forced eval measures whether the student can follow teacher tokens under teacher prefix.
- Free-run decode and ADE/FDE still need to be treated as the actual behavior check.
- Action expert integration should be checked after free-run boundary state quality and hidden-interface quality are reasonable.

## Speed Profile

New profiler:

```text
/home/pm97/workspace/sukim/distillation/cosmos_distillation/scripts/17_profile_action_pre_state.py
```

This profiles the deployment-style VLM path:

- LoRA merged
- FA2 requested and verified in config
- Stops at `<|traj_future_start|> + 1 token`
- Does not generate the fixed 128 discrete future tokens
- Excludes action expert execution
- Measures ViT, prefill, decode, total-to-action-pre

Important fix:

- Student batch generation must use left-padding.
- Right-padding caused some batch samples to miss `<|traj_future_start|>` and run to 256 tokens.

Batch 1 report:

```text
/home/pm97/workspace/sukim/distillation/cosmos_distillation/outputs/reports/no_nav_distill/action_pre_profile_lora_merged_fa2_b1.json
```

Batch 1 mean result:

| metric | teacher | student | speedup |
|---|---:|---:|---:|
| ViT | 0.0577s | 0.0463s | 1.25x |
| text/prefill only | 0.1715s | 0.0737s | 2.33x |
| total prefill | 0.2291s | 0.1200s | 1.91x |
| decode to action-pre | 0.3375s | 0.2582s | 1.31x |
| total to action-pre | 0.5716s | 0.3888s | 1.47x |

Batch 8 report:

```text
/home/pm97/workspace/sukim/distillation/cosmos_distillation/outputs/reports/no_nav_distill/action_pre_profile_lora_merged_fa2_b8.json
```

Batch 8 mean result:

| metric | teacher | student | speedup |
|---|---:|---:|---:|
| ViT | 0.4388s | 0.3425s | 1.28x |
| text/prefill only | 1.2536s | 0.5069s | 2.47x |
| total prefill | 1.6924s | 0.8494s | 1.99x |
| decode to action-pre | 0.4222s | 0.5005s | 0.84x |
| total to action-pre | 2.1224s | 1.3694s | 1.55x |

## Useful Commands

Run batch-1 action-pre profile:

```bash
/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python \
  distillation/cosmos_distillation/scripts/17_profile_action_pre_state.py \
  --batch-size 1 \
  --num-batches 8 \
  --warmup-runs 1 \
  --repeats 3 \
  --max-new-tokens 256 \
  --summary-json distillation/cosmos_distillation/outputs/reports/no_nav_distill/action_pre_profile_lora_merged_fa2_b1.json
```

Run batch-8 action-pre profile:

```bash
/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python \
  distillation/cosmos_distillation/scripts/17_profile_action_pre_state.py \
  --batch-size 8 \
  --num-batches 2 \
  --warmup-runs 1 \
  --repeats 3 \
  --max-new-tokens 256 \
  --summary-json distillation/cosmos_distillation/outputs/reports/no_nav_distill/action_pre_profile_lora_merged_fa2_b8.json
```

## Next Steps

1. Finish remaining no-nav teacher cache chunks 450-573.
2. Rebuild/refresh full no-nav corpus once all chunks are ready.
3. Run free-run decode eval, not only teacher-forced eval.
4. Add ADE/FDE report for decoded student trajectory or action-expert output.
5. Decide whether to continue current BP3 setup or add schedule sampling / hidden-interface tuning.
6. Select representative nav samples and run nav teacher inference separately.
7. Merge no-nav and nav cache for final distillation experiments.

