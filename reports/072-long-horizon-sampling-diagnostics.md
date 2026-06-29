# 072 - Long-Horizon Sampling Diagnostics

## Context

W1/W2 fixed the random-FM collapse: velocity prediction now tracks target velocity and short-horizon ADE is nearly solved. The remaining Stage 0 miss is concentrated in long horizon:

- W2 best checkpoint: `step=2750`
- checkpoint: `outputs/action_expert/student_ae28/stage0_overfit_32_s3000_seed42_recipe_draw16_full444k_retry_20260531/best.pt`
- W2 best eval: full 6.4s ADE `0.7891m`, h1.6 ADE `0.0730m`, h3.2 ADE `0.2374m`

Goal: separate the remaining full-horizon gap into:

- A: Euler sampling step count too small
- B: long-horizon conditioning / sampling ambiguity
- C: action-space integration accumulation

## Sampling Code Path

Evaluation path in `scripts/84_train_student_ae28_official.py`:

```python
with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
    action = teacher_model.diffusion.sample(batch_size=batch_size, step_fn=step_fn, device=device)
```

The loaded diffusion class is `alpamayo1_5.diffusion.flow_matching.FlowMatching`.

Official diffusion implementation:

```python
def __init__(
    self,
    int_method: Literal["euler"] = "euler",
    num_inference_steps: int = 10,
    inference_guidance_weight: float = 1.0,
    *args,
    **kwargs,
):
    self.int_method = int_method
    self.num_inference_steps = num_inference_steps
```

```python
int_method = int_method or self.int_method
inference_step = inference_step or self.num_inference_steps
if int_method == "euler":
    return self._euler(...)
```

```python
x = torch.randn(batch_size, *self.x_dims, device=device) * temperature
time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
for i in range(inference_step):
    dt = time_steps[i + 1] - time_steps[i]
    t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
    v = step_fn(x=x, t=t_start)
    x = x + dt * v
```

Teacher config has no explicit `num_inference_steps`, only:

```json
{"_target_": "alpamayo1_5.diffusion.flow_matching.FlowMatching", "int_method": "euler", "x_dims": null}
```

Therefore current inference uses Euler with `num_inference_steps=10`.

Diagnostic script:

- `scripts/99_x1_x2_sampling_diagnostics.py`
- output: `outputs/action_expert/x1_x2_sampling_best_step2750_full32/summary.json`
- run log: `outputs/action_expert/x1_x2_sampling_best_step2750_full32/run.log`

## X1 - Euler Step Sweep

Same checkpoint, same 32 eval samples, same base seed `3792`; only Euler inference steps changed.

| Euler steps | full ADE mean | full ADE p50 | full FDE mean | h1.6 ADE | h3.2 ADE |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.7836 | 0.5453 | 2.1045 | 0.0711 | 0.2371 |
| 20 | 0.7788 | 0.5363 | 2.0738 | 0.0743 | 0.2408 |
| 40 | 0.7828 | 0.5829 | 2.0805 | 0.0750 | 0.2423 |
| 80 | 0.7871 | 0.5808 | 2.0903 | 0.0761 | 0.2440 |

Verdict: increasing Euler steps `10 -> 20 -> 40 -> 80` does not improve full-horizon ADE. A is rejected.

## X2 - Seed Variance

Same checkpoint, 10 Euler steps, 32 eval samples, one path per eval; only sampling seed changed.

| seed offset | seed base | full ADE mean | full ADE p50 | full FDE mean | h1.6 ADE | h3.2 ADE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3792 | 0.7836 | 0.5453 | 2.1045 | 0.0711 | 0.2371 |
| 1 | 3793 | 0.7973 | 0.6908 | 2.1689 | 0.0700 | 0.2404 |
| 2 | 3794 | 0.7646 | 0.5109 | 2.2135 | 0.0611 | 0.2060 |
| 3 | 3795 | 0.7731 | 0.5100 | 2.1729 | 0.0735 | 0.2360 |
| 4 | 3796 | 0.7456 | 0.6615 | 2.0560 | 0.0641 | 0.2098 |

Aggregate full ADE across seeds:

- mean `0.7728m`
- std `0.0175m`
- min `0.7456m`
- max `0.7973m`

Seed-level mean is stable around `0.75-0.80m`; changing one seed is not enough to pass strict Stage 0.

## X2 - Best-of-8

Same checkpoint, 10 Euler steps, 32 eval samples, 8 sampled paths per sample.

Important caveat: this is oracle best-of-N using target ADE for diagnosis, not deployable inference by itself.

| metric | value |
|---|---:|
| single-path ADE mean from this run | 0.8403 |
| single-path ADE p50 from this run | 0.5942 |
| mean ADE over 8 paths | 0.8760 |
| mean ADE std over 8 paths | 0.4628 |
| best-of-8 full ADE mean | 0.3201 |
| best-of-8 full ADE p50 | 0.2842 |
| best-of-8 full FDE mean | 0.8954 |
| best-of-8 full FDE p50 | 0.8689 |

Best-of-8 by horizon:

| horizon | ADE mean | ADE p50 | FDE mean |
|---|---:|---:|---:|
| h1.6 | 0.0279 | 0.0240 | 0.0726 |
| h3.2 | 0.0902 | 0.0765 | 0.2445 |
| h6.4 | 0.3201 | 0.2842 | 0.8954 |

Verdict: good long-horizon trajectories exist inside the sampling distribution. The strict failure is mainly single-sample selection / sampling variance, not a deterministic integration-step limit.

## Conclusion

long-horizon gap 원인 = Euler step 부족이 아니라 single-sample FM sampling variance/selection 문제. Best-of-8 oracle 선택이면 full ADE `0.320m`로 Stage 0 strict threshold `<0.5m`를 통과한다.

Implications:

- A, Euler step shortage: rejected.
- C, Euler numerical accumulation: unlikely; 8x more Euler steps does not help.
- B, conditioning/sampling ambiguity: plausible as the upstream reason the distribution is broad, but the immediate observed failure mode is sample selection variance.

Recommended next cheap test:

- Sweep inference `temperature` below 1.0 (`0.5`, `0.7`, `0.85`) because official `FlowMatching.sample()` documents lower temperature as more stable with less diversity.
- If lower temperature improves single-sample full ADE, expose temperature as an eval/inference argument.
- If temperature does not help, add a deployable reranker or consistency score to approximate best-of-N without oracle target access.
