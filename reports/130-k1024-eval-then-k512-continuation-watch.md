# K1024 Eval Then K512 Continuation Watch

Date: 2026-06-09

## Objective

After the fast K1024 backbone run finishes:

1. evaluate K1024 final on the same val512 selected set
2. report `ADE@6.4s` and `minADE6@6.4s`
3. immediately launch K512 continuation training from the K512 final checkpoint

## Live Watcher

```text
tmux: mlflex_k1024_eval_then_k512
script: scripts/tmp_watch_k1024_eval_then_k512_continue.sh
log: outputs/logs/watch_k1024_eval_then_k512_continue_20260609.log
```

The watcher waits for:

```text
outputs/checkpoints/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast/final
outputs/reports/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast_summary.json
```

If the K1024 training tmux dies without those artifacts, the watcher exits instead of silently waiting forever.

## K1024 Eval

Eval set:

```text
outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json
```

Eval outputs:

```text
outputs/reports/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast_val512_trajonly_gt_greedy_summary.json
outputs/reports/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast_val512_trajonly_gt_n6_summary.json
outputs/reports/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast_val512_trajonly_gt_minade6_summary.json
outputs/reports/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast_val512_trajonly_gt_minade6_summary.md
```

Settings:

```text
prompt_mode: joint
target_mode: traj_only
geometry_reference: gt
samples_per_row: 1 for ADE, 6 for minADE6
max_new_tokens: 160
batch_size: 2
temperature: 1.0
top_p: 1.0
FLEX flags: preserve positions, uniform selection, scene DeepStack
```

## K512 Continuation

Initial checkpoint:

```text
outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608/final
```

Continuation output:

```text
outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024
outputs/reports/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024_summary.json
```

Training settings:

```text
config: configs/train/stage_mlflex_k512_bp3_hidden_gc_20k_e3.yaml
corpus: data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl
extra epochs: 3.0
learning_rate: 1e-5
batch_size: 16
num_workers: 8
pin_memory: true
persistent_workers: true
prefetch_factor: 2
log_every_steps: 100
eval_every_epochs: 0.5
save_every_epochs: 1.0
```
