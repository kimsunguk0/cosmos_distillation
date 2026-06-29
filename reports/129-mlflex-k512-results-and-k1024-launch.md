# 129 - ML-FLEX K512 Result and K1024 Launch

**Date:** 2026-06-08  
**Status:** K512 20k/3epoch improved but still trails B0; K1024 capacity run launched with the same 20k/3epoch recipe

## Objective

Summarize the first real ML-FLEX backbone scale run and launch the next capacity ablation.

The question is not whether the old action expert works yet. The current gate is only:

```text
Can the FLEX-compressed backbone produce 128 discrete trajectory tokens close enough to B0?
```

## K512 Training Setup

Main run:

```text
run: outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608
config: configs/train/stage_mlflex_k512_bp3_hidden_gc_20k_e3.yaml
corpus: data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl
init: outputs/checkpoints/mlflex_stagea_prealign16_s500_20260608/final
train rows: 20,000
val rows used in training: 512 / 9,007
epochs: 3.0
steps: 3,750
batch: 16
eval interval: 625 steps
save interval: 1,250 steps
base model: /home/pm97/workspace/sukim/base_weights/cosmos-reason-2b
```

FLEX architecture:

```text
architecture: multi_level
tokens_per_image: 32
images_per_sample: 16
scene tokens: 512
DeepStack levels: 3
compression mode: per_image
selection: uniform
position handling: preserve Qwen MRoPE positions during decode
DeepStack injection: on
```

Trainable groups:

```text
language LoRA: all layers, rank 64, alpha 128
FLEX scene encoder: trainable, lr scale 2.5
trajectory hidden bridge: trainable
multimodal projector: trainable, lr scale 0.15
base model dense weights: frozen
```

Loss mix:

```text
traj CE: 0.85
CoT CE: 0.08
text top-k KD: 0.08
trajectory top-k KD: 0.12
trajectory hidden alignment: 0.08
text boundary hidden alignment: 0.05
format CE: 0.20
```

Training validation loss:

| step | epoch | val total |
|---:|---:|---:|
| 625 | 0.5 | 1.3398 |
| 1250 | 1.0 | 1.3369 |
| 1875 | 1.5 | 1.3343 |
| 2500 | 2.0 | 1.3292 |
| 3125 | 2.5 | 1.3299 |
| 3750 | 3.0 | 1.3251 |

Interpretation: training loss was still improving at the end. The run did not saturate by 3 epochs.

## Decode Eval Setup

Same 512 val sample IDs for all rows:

```text
selected ids: outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json
split: val
reference: GT future geometry
prompt_mode: joint
target_mode: traj_only
horizon: 6.4s
greedy ADE: one generated trajectory
minADE6: best ADE among 6 sampled trajectories, temperature 1.0, top_p 1.0
```

Important implementation note: FLEX manual generation was patched so `samples_per_row=6` actually produces 6 candidates instead of falling back to greedy.

## K512 Result

| model | ADE@6.4s | FDE@6.4s | minADE6@6.4s | minFDE6 selected by ADE |
|---|---:|---:|---:|---:|
| FLEX final | 3.7973 | 12.0458 | 1.9947 | 6.1104 |
| FLEX step_002500 | 3.9693 | 12.4832 | 2.1304 | 6.3496 |
| B0 step_006250 | 3.2304 | 10.2695 | 1.6886 | 4.9766 |

Deltas:

```text
final - B0 ADE@6.4s: +0.5669 m
final - B0 minADE6@6.4s: +0.3062 m
final - step2500 ADE@6.4s: -0.1720 m
final - step2500 minADE6@6.4s: -0.1356 m
```

Distribution checks:

```text
final vs B0 greedy win rate: 35.0%
final vs B0 minADE6 win rate: 39.3%
final severe failures, ADE > 5m: 137/512
B0 severe failures, ADE > 5m: 108/512
final n6 severe failures, ADE > 5m: 42/512
B0 n6 severe failures, ADE > 5m: 24/512
```

Interpretation:

```text
K512 is learning, but not B0-equivalent.
The minADE6 gap is smaller than the greedy ADE gap, so useful modes exist in the sampled distribution.
Greedy/readout stability and severe-failure rate are still the main blockers.
Do not train the action expert on this backbone yet.
```

## K1024 Capacity Run

Rationale:

```text
K512 compressed 2880 image tokens to 512 scene tokens.
K1024 compresses 2880 image tokens to 1024 scene tokens.
If the K512 gap is a capacity bottleneck, K1024 should reduce ADE/minADE6 gap against B0 under the same task-loss recipe.
```

K1024 config:

```text
config: configs/train/stage_mlflex_k1024_bp3_hidden_gc_20k_e3.yaml
tokens_per_image: 64
images_per_sample: 16
scene tokens: 1024
all other main 20k/3epoch settings: same as K512
```

Launch chain:

```text
script: scripts/tmp_run_mlflex_k1024_bp3_20k_e3_chain.sh
tmux: mlflex_k1024_bp3_20k_e3_b16
log: outputs/logs/mlflex_k1024_bp3_20k_e3_b16_20260608_chain.log
```

The chain intentionally mirrors the K512 path:

```text
1. create K1024 F0 from B0 dense checkpoint
2. run K1024 forward smoke
3. run K1024 Stage A prealign16 for 500 steps
4. run K1024 forward smoke after prealign
5. run K1024 20k / 3 epoch task adaptation
```

K1024 prealign settings:

```text
max samples: 16
max steps: 500
trainable: FLEX scene encoder only
feature target tokens per image: 64
DeepStack feature target tokens per image: 64
```

Main K1024 expected checkpoints:

```text
outputs/checkpoints/mlflex_k1024_bp3_20k_e3_b16_20260608/step_001250
outputs/checkpoints/mlflex_k1024_bp3_20k_e3_b16_20260608/step_002500
outputs/checkpoints/mlflex_k1024_bp3_20k_e3_b16_20260608/step_003750
outputs/checkpoints/mlflex_k1024_bp3_20k_e3_b16_20260608/final
```

Post-run eval should reuse the exact K512 decode eval set:

```text
outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json
```

Primary decision metrics:

```text
K1024 final - B0 ADE@6.4s
K1024 final - B0 minADE6@6.4s
severe failures, ADE > 5m
win rate vs B0
```

Decision rule:

```text
If K1024 closes most of the K512 gap, continue capacity/latency tradeoff search.
If K1024 remains close to K512, the issue is not just scene-token count; next target is training objective/readout stability rather than K.
```

## K1024 Fast Restart

The initial K1024 main run was stopped at about 537/3750 steps because throughput was not acceptable:

```text
run: mlflex_k1024_bp3_20k_e3_b16_20260608
observed mean step time: about 17.25 s/step
symptom: GPU burst followed by long idle gaps
cause: K1024 longer sequence plus GPU-side FLEX batch compression / no DataLoader overlap
```

Code/runtime fix:

```text
src/training/trainer.py: add prepare_flex_batch_for_model()
scripts/09_train_distill.py: apply FLEX compression before move_batch_to_device()
DataLoader: --num-workers 8 --pin-memory --persistent-workers --prefetch-factor 2
train stdout: --log-every-steps 100
```

Fast benchmark before relaunch:

```text
bench output: outputs/checkpoints/mlflex_k1024_bp3_fast_bench20_20260609/metrics.jsonl
steps: 20
mean step time: 5.03 s/step
median step time: 4.94 s/step
```

Fast main launch:

```text
script: scripts/tmp_run_mlflex_k1024_bp3_20k_e3_fast.sh
tmux: mlflex_k1024_bp3_fast
run: mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast
log: outputs/logs/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast.log
summary: outputs/reports/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast_summary.json
checkpoint dir: outputs/checkpoints/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast
```
