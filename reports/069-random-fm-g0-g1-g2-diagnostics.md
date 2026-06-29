# 069 Random FM G0/G1/G2 Diagnostics

Date: 2026-05-31

## One-line Verdict

**random FM 실패 원인 = FM target 공식/부호 오류가 아니라, AE/expert parameterization이 single-random-draw `(x_t,t) -> v` 회귀를 안정적으로 학습하지 못해 low-amplitude 평균으로 수축하는 optimization/parameterization 문제.**

## G0: Independent MLP Baseline

Script:

```text
scripts/97_g0_random_fm_mlp_baseline.py
```

Task:

- Fix one target action `x1`.
- Sample `x0 ~ N(0, I)` and `t ~ 0.999 - Beta(1.5,1.0) * 0.999`.
- Build `x_t = (1 - t) * x0 + t * x1`.
- Train an independent MLP to predict `target_v = x1 - x0` from flattened `x_t` plus Fourier `t`.
- No AE, no KV, no action projections.

Sample:

```text
01d3588e-bca7-4a18-8e74-c6cfe9e996db__sg_00__t0_1600000
target_action_shape = [64, 2]
target_action_abs_mean = 0.1599
```

Results:

| run | steps | batch | loss | pred_v_abs | target_v_abs | cosine | alpha | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MLP | 1000 | 1 | 0.7778 | 0.4278 | 0.8195 | 0.5134 | 0.2743 | partial, positive |
| MLP | 5000 | 1 | 0.4781 | 0.6336 | 0.8206 | 0.7416 | 0.5797 | improving, positive |
| MLP | 1000 | 16 | 0.1250 | 0.7442 | 0.8195 | 0.9391 | 0.8623 | PASS |
| MLP | 1000 | 256 | 0.0753 | 0.7828 | 0.8205 | 0.9638 | 0.9283 | PASS |

Controls:

| control | loss | cosine | alpha |
|---|---:|---:|---:|
| zero prediction | ~1.056 | 0.0 | 0.0 |
| analytic oracle `(x1 - x_t)/(1 - t)` | 0.0 | 1.0 | 1.0 |

G0 conclusion:

**MLP가 random FM을 학습하는가 = Y.**

The exact target/sampler is learnable. With batch size 16, it crosses cosine 0.9 by 500 steps and reaches 0.94 by 1000 steps. Batch size 1 is noisy and slower, but it moves in the correct positive direction; it does not reproduce AE's negative/near-zero cosine collapse.

## G1: AE Time/Shape Path

Local training path: `scripts/84_train_student_ae28_official.py`

Relevant code:

```python
x1 = target_action.to(device=device, dtype=dtype)
x0 = torch.randn_like(x1)
t = sample_fm_timesteps(batch_size=int(x1.shape[0]), ...)
x_t = (1.0 - t) * x0 + t * x1
target_v = x1 - x0
future_token_embeds = bundle.action_in_proj(x_t, t)
pred_v = bundle.action_out_proj(last_hidden).view(-1, *action_dims)
loss = F.mse_loss(pred_v.float(), target_v.float())
```

Shape flow:

| tensor | shape in Stage0/E3 | note |
|---|---|---|
| `x1` | `[B, 64, 2]` | target action |
| `x0` | `[B, 64, 2]` | same tensor used for `x_t` and `target_v` |
| `t` | `[B, 1, 1]` | per sample, broadcasts over waypoints/action dims |
| `x_t` | `[B, 64, 2]` | `(1-t)x0 + t*x1` |
| `target_v` | `[B, 64, 2]` | `x1 - x0` |
| `action_in_proj(x_t,t)` | `[B, 64, hidden]` | one embedding per waypoint |

`num_time_samples` handling:

```python
if repeats > 1:
    prompt_cache.batch_repeat_interleave(repeats)
    context = repeat_context(context, repeats)
    target_action = target_action.repeat_interleave(repeats, dim=0)
...
t = sample_fm_timesteps(batch_size=int(x1.shape[0]), ...)
```

So repeated samples get separate `t` draws after repeat. No sample-level t mixing was found.

`action_in_proj` path:

```python
B, T, _ = x.shape
action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
timestep_feats = timestep_feats.repeat(1, T, 1)
x = torch.cat((action_feats, timestep_feats), dim=-1)
return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))
```

For `t` shape `[B,1,1]`, `timesteps[..., -1]` becomes `[B,1]`; Fourier encoding becomes `[B,1,C]`; repeat gives `[B,64,C]`. That is shape-consistent and per-sample.

G1 conclusion:

**No direct `x_t/t/target_v` tensor mismatch found.** The same `t` and same `x0` are used consistently. Fixed-vs-random failure is not explained by a simple tensor ordering or broadcasting bug.

## G2: Official FM Formula / Direction

Official training reference:

```python
t = self.beta_dist.sample((batch_size,)).to(x.device)
t = self.beta_scale_constant - t * self.beta_scale_constant
...
noise = torch.randn_like(x)
noisy_x = t * x + (1 - t) * noise
...
target = (x - noise).to(dtype=pred.dtype)
return torch.nn.functional.mse_loss(target, pred)
```

Local code equivalence:

| official | local |
|---|---|
| `x` | `x1` |
| `noise` | `x0` |
| `noisy_x = t*x + (1-t)*noise` | `x_t = t*x1 + (1-t)*x0` |
| `target = x - noise` | `target_v = x1 - x0` |

Official inference direction:

```python
x = torch.randn(batch_size, *self.x_dims, device=device)
time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
...
v = step_fn(x=x, t=t_start)
x = x + dt * v
```

Local sampling uses the same direction conceptually: start from noise at `t=0`, integrate positive `v = x1 - x0` toward data at `t=1`.

G2 conclusion:

**No sign or time-direction mismatch found.** A sign bug would make the analytic/MLP baseline fail or produce systematically negative cosine. Instead, the MLP baseline learns the positive direction.

## G3: Scale / Sampler Distribution

From G0 eval draws:

| metric | value |
|---|---:|
| `t_mean` | ~0.40 |
| `t_p50` | ~0.36-0.37 |
| `t_p95` | ~0.87 |
| `x0_abs_mean` | ~0.798 |
| `x_t_abs_mean` | ~0.494-0.496 |
| `target_v_abs_mean` | ~0.820 |

The target velocity is not tiny, and the input scale is not pathological for a small MLP. The sampler is learnable but noisy with batch size 1.

## Interpretation

V2 said:

- AE fixed `(x0,t)` reaches cosine 0.9996.
- AE random `(x0,t)` reaches cosine -0.0425, alpha -0.0123.

G0 now says:

- the same random FM target is learnable by a small MLP,
- batch size 16 is enough for cosine > 0.9,
- batch size 1 is noisy but still moves in the correct positive direction.

Therefore the negative AE cosine is not a target sign bug. The most likely failure mode is:

> The AE/expert stack is a bad parameterization for this stochastic one-draw FM regression when initialized this way. With only one random `x0/t` draw per step, gradients are high-variance; the giant expert + Fourier action projection collapses toward a low-amplitude average instead of learning the per-draw inverse relation from `(x_t,t)` to `x1-x0`.

## Immediate Next Checks

1. Re-run E3 actual bundle with `num_time_samples`/random draws per step > 1, e.g. 8 or 16, before touching hidden/KV again.
2. Try a short warm-start objective: direct `x1` reconstruction or fixed-noise denoising, then random FM.
3. Consider training `action_in_proj/action_out_proj` with a higher effective LR or temporarily freezing most expert layers, because the MLP result says the function is simple but the AE optimization is not finding it.
