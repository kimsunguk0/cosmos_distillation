# 132 - B0 AE vs FLEX K512 AE on Same Val512 Split

**Date:** 2026-06-10  
**Status:** Completed  
**Question:** On the same validation samples, how far is the current FLEX K512 backbone + retrained AE from the existing B0 backbone + AE baseline?

## Setup

This report compares Action Expert decode quality on the same `512` validation samples.

Corpus and split:

```text
corpus: data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl
train selected: 18,000
val selected: 1,900
eval samples: first 512 val samples from the same split construction
seed: 42
```

Common eval settings:

```text
prefix_mode: student_free
target_source: teacher
eval_num_paths: 6
eval_temperature: 1.0
eval_selection_method: single
eval_seed_mode: fixed
eval_seed_base: 1042
eval_vectorize_paths: true
eval_path_batch_size: 6
eval_batch_size: 8
stage2_attention_mode: official_none
attn_implementation: flash_attention_2
metric reference: GT future geometry
horizon: 6.4s / 64 waypoints
```

Important interpretation boundary:

```text
This is not a backbone-only comparison.
This is the deployment-relevant AE path comparison after FLEX K512 AE retraining.
```

## Models

### B0 AE baseline

```text
student backbone:
outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250

AE checkpoint:
outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt

fresh same-split eval output:
outputs/action_expert/b0_q3_best_on_flex_val512_n6_20260610/summary.json
```

This uses the existing B0 student backbone and the Q3 best AE checkpoint.

### FLEX K512 AE

```text
student backbone:
outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final

AE training output:
outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16

best AE checkpoint:
outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt

final AE checkpoint:
outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/final.pt
```

FLEX flags:

```text
preserve_flex_positions: true
flex_selection_strategy: uniform
flex_scene_deepstack: true
```

FLEX AE training settings:

```text
steps: 10,000
batch_size: 16
num_time_samples: 16
expert_lr: 1e-4
proj_lr: 1e-4
lr_warmup_steps: 0
ae_init_mode: student_backbone_init
prefix_mode: student_free
target_source: teacher
train_timestep_sampler: beta
```

## Results

All rows use the same `512` validation samples and `6` sampled AE paths.

| Model | AE step | ADE@6.4s | ADE p50 | FDE@6.4s | FDE p50 | minADE6@6.4s | minADE6 p50 | minFDE6 | minFDE6 p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Q3 best AE | 2,000 | **3.0542** | **2.2714** | **8.9647** | **6.7151** | **1.6450** | **1.0481** | **4.9588** | **3.0797** |
| FLEX K512 AE best | 7,500 | 3.1811 | 2.4210 | 9.4844 | 7.1256 | 1.7566 | 1.1219 | 5.2598 | 3.3359 |
| FLEX K512 AE final | 10,000 | 3.1809 | 2.4235 | 9.5092 | 7.2222 | 1.8650 | 1.1779 | 5.6289 | 3.5351 |

## Deltas vs B0

| Model | Delta ADE@6.4s | Delta FDE@6.4s | Delta minADE6@6.4s | Delta minFDE6 |
|---|---:|---:|---:|---:|
| FLEX K512 AE best - B0 | +0.1269 m | +0.5197 m | +0.1116 m | +0.3010 m |
| FLEX K512 AE final - B0 | +0.1267 m | +0.5446 m | +0.2200 m | +0.6701 m |

## Interpretation

1. The current FLEX K512 AE is functional, but it does not match B0 AE yet on the same samples.

2. `best.pt` is the correct FLEX AE checkpoint to use for now. `final.pt` has nearly identical single-path ADE, but its `minADE6@6.4s` is worse:

```text
best minADE6:  1.7566 m
final minADE6: 1.8650 m
delta:         +0.1083 m worse
```

3. FLEX K512 is close enough to continue investigating, but not close enough to call B0-equivalent. The remaining gap is visible in both single selected path quality and best-of-6 mode quality.

4. The `minADE6` gap is smaller than the FDE gap, which suggests the sampled trajectory distribution still contains usable modes, but endpoint quality and long-horizon stability remain weaker than B0.

5. For deployment planning, this result means the current FLEX K512 path is not yet a drop-in replacement for B0. It is a valid compression path to keep training/tuning, but B0 remains the quality reference.

## Current Decision

Use this as the AE-path baseline table for FLEX K512:

```text
B0 Q3 best AE:
  ADE@6.4s      3.0542
  minADE6@6.4s  1.6450

FLEX K512 AE best:
  ADE@6.4s      3.1811
  minADE6@6.4s  1.7566

FLEX K512 gap:
  +0.1269 m ADE
  +0.1116 m minADE6
```

Next comparison should only change one variable at a time:

```text
1. Evaluate the same AE protocol after any further FLEX K512 backbone continuation.
2. If K1024 backbone is selected, train/evaluate K1024 AE with the same val512 protocol.
3. Keep B0 Q3 best AE as the same-split baseline unless a stronger B0 AE checkpoint is explicitly selected.
```
