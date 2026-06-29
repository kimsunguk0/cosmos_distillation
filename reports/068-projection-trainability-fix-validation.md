# 068 Projection Trainability Fix Validation

Date: 2026-05-31

## One-line Verdict

**버그 수정으로 FM 복구됨 = N.**

`action_in_proj/action_out_proj` freeze 버그는 실제였고 수정됨. V1에서 trainable, optimizer membership, 1-step delta 모두 확인됨. 그러나 수정 후 actual bundle E3 random `(x0,t)`는 여전히 collapse했고, Stage 0 early gate도 pass 징후가 없었다. 따라서 현재 결론은:

> projection trainability fixed = Y, random FM vector-field learning restored = N.

## Code Fix

Patched: `scripts/84_train_student_ae28_official.py`

Main behavioral fix:

```python
set_module_requires_grad(action_in_proj, True)
set_module_requires_grad(action_out_proj, True)
```

This is applied once after the `ae_init_mode` branch in `build_bundle`, so it covers:

- `teacher_compressed`
- `scratch`
- `student_backbone_init`
- `student_backbone_init_teacher_q`

Also added logging/sanity helpers:

- `bundle_trainable_summary`
- `optimizer_membership_summary`
- `projection_delta_step1`

Diff stat:

```text
scripts/84_train_student_ae28_official.py | 142 ++++++++++++++++++++++++++++++
```

Compile check:

```text
.venv/bin/python -m py_compile scripts/84_train_student_ae28_official.py scripts/96_e3_actual_bundle_no_kv.py
PASS
```

## V1: Trainable / Optimizer / 1-step Delta

Run:

```text
outputs/action_expert/verify_v1_projection_trainable_seed42
```

Config note: used `--lr-warmup-steps 0` for this V1 sanity so the first optimizer step has nonzero LR. With the Stage 0 warmup schedule, step 1 LR is 0, so delta-at-step1 is expected to be 0 there.

Trainable params:

| module | trainable tensors | trainable params |
|---|---:|---:|
| expert | 309 | 1,409,410,048 |
| action_in_proj | 10 | 1,349,632 |
| action_out_proj | 2 | 4,098 |
| total | - | 1,410,763,778 |

Optimizer membership:

| module | trainable | included | missing | LR |
|---|---:|---:|---:|---:|
| action_in_proj | 1,349,632 | 1,349,632 | 0 | 3e-5 |
| action_out_proj | 4,098 | 4,098 | 0 | 3e-5 |

1-step parameter delta:

| module | changed elems | max abs delta |
|---|---:|---:|
| action_in_proj | 236,624 / 1,349,632 | 3.0518e-05 |
| action_out_proj | 1,513 / 4,098 | 3.0518e-05 |

V1 verdict: **fix is active.**

## V2: E3 Actual Bundle No-KV

Script:

```text
scripts/96_e3_actual_bundle_no_kv.py
```

This uses the real `84.build_bundle()` path after freezing the teacher, then removes KV/conditioning for the 1-sample E3 sanity.

| run | final loss | pred_v_abs | target_v_abs | cosine | alpha | verdict |
|---|---:|---:|---:|---:|---:|---|
| random reset projections | 1.2016 | 0.2324 | 0.8203 | -0.0425 | -0.0123 | FAIL |
| fixed reset projections | 0.0011 | 0.8711 | 0.8711 | 0.9996 | 0.9985 | PASS |
| random teacher `action_in_proj` graft | 1.0253 | 0.2373 | 0.8203 | 0.2408 | 0.0716 | FAIL |
| random teacher `action_in/out_proj` graft | 1.0652 | 0.2598 | 0.8203 | 0.1892 | 0.0610 | FAIL |

Output dirs:

- `outputs/action_expert/v2_e3_actual_bundle_no_kv_seed42`
- `outputs/action_expert/v2_e3_actual_bundle_no_kv_fixed_seed42`
- `outputs/action_expert/v2_e3_actual_bundle_no_kv_teacher_action_in_seed42`
- `outputs/action_expert/v2_e3_actual_bundle_no_kv_teacher_projections_seed42`

V2 verdict:

- fixed `(x0,t)` memorization works, so basic forward/backward path can learn.
- random `(x0,t)` FM remains collapsed even after projection trainability fix.
- teacher projection graft does not rescue random E3, so the remaining failure is not explained by frozen projections or reset projection scale alone.

## V3: Stage 0 Re-run

Run:

```text
outputs/action_expert/student_ae28/stage0_overfit_32_s3000_seed42_projfix_20260531
```

Command matched the previous Stage 0 repro hyperparameters:

- `student_backbone_init`
- `student_free`
- `target_source=teacher`
- `num_samples=32`
- `batch_size=2`
- `eval_samples=32`
- `steps=3000`
- `eval_every=250`
- `train_ade_every=100`
- `seed=42`
- `proj_lr=1e-4`
- `lr_warmup_steps=150`

Important V3 note: this was stopped after step 525 because V2 had already failed and the run was projected to take roughly 1.5-2 hours. The first 500 steps already reproduced the same collapse pattern.

Train/eval summary before stop:

| step | train loss | pred_v_abs | target_v_abs | train_inb_ade | eval ADE |
|---:|---:|---:|---:|---:|---:|
| 0 | - | - | - | - | 9.144 |
| 100 | 1.958 | 0.273 | 1.016 | 4.337 | - |
| 200 | 1.232 | 0.237 | 0.871 | 14.108 | - |
| 250 | 4.926 | 0.316 | 1.977 | - | 5.218 |
| 300 | 1.495 | 0.291 | 0.934 | 9.230 | - |
| 400 | 1.369 | 0.221 | 0.938 | 4.309 | - |
| 500 | 1.673 | 0.134 | 1.031 | 2.001 | 6.902 |

V3 early verdict: **FAIL / no pass trajectory.**

The same amplitude shrink is still visible in train rows. Some batches spike or momentarily improve, but the overall train loss does not settle and eval ADE is far above the Stage 0 pass criterion.

## V4: Oracle KV

Not run in this pass.

Reason: V2 is the decisive cheap gate and failed after the freeze fix. V3 early also failed with actual student KV. Running the bonus oracle-KV 32-sample job would be expensive and would not change the core conclusion until random E3 is fixed.

## Interpretation

The original root cause statement needs refinement.

Confirmed:

1. `action_in_proj/action_out_proj` were frozen because `teacher_model.parameters().requires_grad_(False)` happened before `deepcopy + reset`.
2. The fix correctly unfreezes them and puts them in optimizer groups.
3. The projections now update under nonzero LR.

Not confirmed:

1. That freeze bug was the sole cause of FM collapse.
2. That fixing it restores random `(x0,t)` vector-field learning.

Current best diagnosis:

> The projection freeze bug was a real bug and a necessary fix, but the remaining collapse is in the random-noise FM vector-field learning path. The model can memorize one fixed `(x0,t)`, but cannot learn the stochastic random `(x0,t) -> x1-x0` function under the current AE/FM parameterization and optimization setup.

## Next Action

Do not proceed to long KD training yet. The next useful fix should target the random FM path itself, not only trainability:

- Add a tiny analytical/MLP baseline for the exact same 1-sample random FM task to confirm the target formula and sampler are learnable.
- Try an AE warm-start objective that first learns direct `x1`/action reconstruction or fixed-noise denoising, then enables random FM.
- Revisit the timestep/noise parameterization for random FM, especially high/low `t` conditioning and the implicit division by `(1 - t)` needed to recover `x0` from `x_t`.
