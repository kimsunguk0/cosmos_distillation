# 140 - AE Path Reranker Bootstrap

Date: 2026-06-12

## Question

Should reranking happen on backbone discrete trajectories or on the final AE
pipeline outputs?

Decision: deployment quality must be improved on the AE path. The candidate set
to rerank is:

```text
student self-generated CoT/prefix -> AE diffusion N paths -> selected path
```

Backbone discrete reranking remains useful only as a diagnostic.

## Code Added

```text
scripts/train_ae_path_reranker.py
```

The script trains a lightweight scorer on saved benchmark prediction NPZ files.
It consumes `rows.jsonl` plus each sample's `paths`, `selected_path`, and
`target_gt` arrays.

Two objectives were tested:

1. `oracle_ce`: classify which of 6 paths is GT-oracle best.
2. `weighted_mse`: predict soft weights over 6 paths and output a learned
   weighted trajectory.

Features are deployable path/ensemble geometry only: path length, endpoint,
speed/accel/jerk/yaw-rate summaries, distance to the ensemble mean, and pairwise
distance features. GT is used only to create training labels/losses.

## Candidate Generation

New B0 NoFLEX AE28 train candidates:

```text
outputs/benchmarks/ae_rerank_b0_train1024_20260612/
```

Command:

```bash
.venv/bin/python scripts/benchmark_4models.py \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl \
  --split train \
  --num-samples 1024 \
  --model student_noflex_ae28 \
  --output-dir outputs/benchmarks/ae_rerank_b0_train1024_20260612 \
  --student-batch-size 8 \
  --batch-size 4 \
  --eval-num-paths 6 \
  --eval-temperature 0.85 \
  --eval-selection-method mean_traj \
  --seed 42
```

Train1024 candidate metrics:

| Eval | ADE | FDE | minADE6 | minFDE6 |
|---|---:|---:|---:|---:|
| B0 AE28 train1024 mean_traj | `2.6095` | `7.9378` | `1.6383` | `5.0961` |

## Experiments

### 1. Val806 internal bootstrap

Artifact:

```text
outputs/reranker/ae_path_reranker_b0_val806_seed42/
```

Result on held-out val806 split:

| Method | ADE | FDE |
|---|---:|---:|
| first path | `3.3799` | `10.2385` |
| mean_traj | `2.6517` | `8.1448` |
| medoid | `2.7207` | `8.3961` |
| oracle best | `1.7516` | `5.3386` |
| learned hard reranker | `3.4200` | `10.3470` |

Hard argmax reranking failed. It was worse than first path and much worse than
`mean_traj`.

### 2. Val806 internal learned weighted aggregation

Artifact:

```text
outputs/reranker/ae_path_weighted_b0_val806_seed42/
```

Result on held-out val806 split:

| Method | ADE | FDE |
|---|---:|---:|
| mean_traj | `2.6517` | `8.1448` |
| oracle best | `1.7516` | `5.3386` |
| learned weighted | `2.7118` | `8.3173` |

Learned weighting also failed to beat `mean_traj`.

### 3. Train1024 -> external semantic val806

Artifact:

```text
outputs/reranker/ae_path_weighted_b0_train1024_ext_val806_seed42/
```

External test is the full semantic val806 benchmark:

```text
outputs/benchmarks/semantic_val806_4models_20260612/
```

Result:

| Method | ADE | FDE |
|---|---:|---:|
| first path | `3.1990` | `9.5897` |
| mean_traj | `2.7227` | `8.1559` |
| medoid | `2.8139` | `8.4370` |
| oracle best | `1.6835` | `5.0521` |
| learned argmax | `3.4770` | `10.3411` |
| learned weighted | `2.7529` | `8.2555` |

The learned weighted model still did not beat `mean_traj`.

## Interpretation

The AE path has real oracle headroom:

```text
semantic val806 B0 AE28:
  mean_traj ADE: 2.7227
  oracle best ADE: 1.6835
  recoverable gap: ~1.04 m
```

But this first reranker shows that the gap is not recoverable from simple
GT-free path geometry features alone. The model needs richer signals than
smoothness, path length, centrality, and ensemble distance.

The failure is useful:

1. Reranking belongs on the AE path, but naive path-only reranking is not enough.
2. `mean_traj` is a strong baseline for the current candidate distribution.
3. A useful reranker likely needs CoT/prefix confidence, student token logits,
   teacher-style text/action priors, or an additional learned value head during
   candidate generation.

## Next Steps

1. Add path-specific diffusion/probability features if available from the AE
   sampler.
2. Add CoT/prefix-level features: generated text length, boundary hit, token
   confidence, entropy/margin around `<|traj_future_start|>`.
3. Add teacher-supervised labels: train against which student path best matches
   the 10B selected/mean path, not only GT oracle.
4. Generate a larger train candidate set after feature expansion. The current
   train1024 run is enough to reject path-only reranking, not enough to close the
   oracle gap.
