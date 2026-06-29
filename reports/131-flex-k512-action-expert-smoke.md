# FLEX K512 Action Expert Smoke

Date: 2026-06-09

## Objective

Check whether the Action Expert training path can consume a FLEX K512 student backbone checkpoint before the final K512 continuation run completes.

## Code Finding

`scripts/84_train_student_ae28_official.py` already has FLEX-specific arguments:

```text
--preserve-flex-positions
--flex-selection-strategy {first,uniform}
--flex-scene-deepstack
```

Updated status after follow-up patch:

```text
FLEX checkpoints now support --prefix-mode student_free in the AE path.
The script uses a manual FLEX generation loop so student-generated CoT and
<|traj_future_start|> are included in the student KV cache before AE conditioning.
```

Patches applied:

```text
Moved FLEX placeholder compression before _to_device_batch() in build_batch().
Added FLEX student_free manual generation with returned past_key_values.
```

Reason:

```text
Avoid GPU-side Python compression/sync in the AE path, matching the K512/K1024 backbone training speed fix.
```

## Smoke 1: Scratch AE Init

Command shape:

```text
student checkpoint: outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608/final
prefix_mode: teacher_forced
ae_init_mode: scratch
steps: 2
batch_size: 1
num_time_samples: 1
FLEX flags: preserve positions, uniform selection, scene DeepStack
```

Output:

```text
outputs/action_expert/flex_k512_ae_smoke_20260609/summary.json
```

Result:

```text
status: ok
train step 1/2: ok
train step 2/2: ok
val eval: ok
traj_start_hit_rate: 1.0
```

## Smoke 2: Student Backbone Init

Command shape:

```text
student checkpoint: outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608/final
prefix_mode: teacher_forced
ae_init_mode: student_backbone_init
steps: 1
batch_size: 1
num_time_samples: 1
FLEX flags: preserve positions, uniform selection, scene DeepStack
```

Output:

```text
outputs/action_expert/flex_k512_ae_smoke_studentinit_20260609/summary.json
```

Result:

```text
status: ok
train step 1/1: ok
val eval: ok
traj_start_hit_rate: 1.0
```

## Interpretation

The FLEX K512 backbone is plug-compatible with the current AE training code when using the correct FLEX flags.

Teacher-forced prefix construction validates the direct prefill/KV path. Student-free prefix construction now validates the deployment-relevant path where the student generates CoT through `<|traj_future_start|>` before the AE consumes the student KV cache.

This smoke does not validate final AE quality. It only validates:

```text
FLEX checkpoint load
FLEX compressed prompt construction
DeepStack scene-token injection
student KV cache extraction at <|traj_future_start|>
AE forward/backward
optimizer update
AE eval sampling path
```

## Smoke 3: Student-Free FLEX Prefix

Command shape:

```text
student checkpoint: outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608/final
prefix_mode: student_free
ae_init_mode: student_backbone_init
steps: 1
batch_size: 1
num_time_samples: 1
FLEX flags: preserve positions, uniform selection, scene DeepStack
```

Output:

```text
outputs/action_expert/flex_k512_ae_smoke_studentfree_20260609/summary.json
```

Result:

```text
status: ok
train step 1/1: ok
val eval: ok
train traj_start_hit_rate: 1.0
generated preview: Stop for the stop sign since the intersection is controlled by a stop sign<|cot_end|><|traj_future_start|><i1401>
```

## Recommended Full AE Launch After Backbone Selection

Use this shape after choosing the final K512 backbone checkpoint:

```text
--student-checkpoint-dir <final-flex-k512-backbone>
--prefix-mode student_free
--preserve-flex-positions
--flex-selection-strategy uniform
--flex-scene-deepstack
--ae-init-mode student_backbone_init
--target-source teacher
--num-time-samples 16
--expert-lr 1e-4
--proj-lr 1e-4
--lr-warmup-steps 0
--stage2-attention-mode official_none
```

Before starting a long AE run, run one more preflight with the actual full `num_time_samples=16` and intended batch size after the backbone training process releases GPU memory.
