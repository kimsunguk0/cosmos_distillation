# 067 - action_in_proj / t Sensitivity Diagnostics

Date: 2026-05-31

## TL;DR

Collapse 원인 = **t embedding이 빠진 것이 아니라, `student_backbone_init` 경로에서 teacher freeze 후 `action_in_proj/action_out_proj`를 deepcopy+reset하면서 `requires_grad=False`가 그대로 보존되어 projection들이 학습되지 않았고, reset된 `action_in_proj` 출력 scale도 teacher 원본 대비 약 4배 커진 frozen random feature가 되어 expert만 이를 보정하다 low-amplitude velocity로 수축한 것**.

F3로 최종 확인할 만한 다음 실험은 `action_in_proj/action_out_proj.requires_grad_(True)`를 강제로 켠 뒤 E3 random 1-sample 또는 D1 oracle 32-sample을 재시도하는 것이다.

## F1 Code Path

`action_in_proj`는 `alpamayo1_5.models.action_in_proj.PerWaypointActionInProjV2`다.

Timestep embedding:

```python
freqs = torch.logspace(0, math.log10(max_freq), steps=half)
self.register_buffer("freqs", freqs[None, :], persistent=False)
arg = x[..., None] * self.freqs * 2 * torch.pi
return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)
```

Forward 결합 방식:

```python
action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
timestep_feats = timestep_feats.repeat(1, T, 1)
x = torch.cat((action_feats, timestep_feats), dim=-1)
return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))
```

해석:

- `t`는 learned embedding이 아니라 fixed Fourier features다.
- `x_t` 각 action dimension도 fixed Fourier features를 탄다.
- 결합은 add/FiLM이 아니라 concat이다.
- 마지막에 `LayerNorm(out_dim=2048)`이 있다.
- Fourier `freqs` buffer는 teacher 원본과 reset copy가 완전히 동일했다: max_abs_diff 0.0, range 1.0-100.0.

## F1 Teacher vs Reset Output

Output: `outputs/action_expert/fm_collapse_diagnostics/f1_f2_action_in_proj_seed42/summary.json`

동일한 32-sample `(x_t, t)` 입력에서:

| module | mean | std | abs_mean | rms | min | max |
|---|---:|---:|---:|---:|---:|---:|
| teacher original | -0.00018 | 0.254 | 0.165 | 0.254 | -2.203 | 2.141 |
| student reset | 0.00000 | 1.000 | 0.799 | 1.000 | -5.250 | 5.125 |
| D1 best | 0.00000 | 1.000 | 0.799 | 1.000 | -5.250 | 5.125 |

Pairwise:

- teacher vs reset cosine: 0.0031
- teacher vs reset relative_delta_rms: 4.05
- reset vs D1 best cosine: 1.000
- reset vs D1 best delta_rms: 0.0

해석:

- reset output은 teacher original보다 std/RMS가 약 3.9배 크다.
- D1 best의 `action_in_proj`는 reset과 bitwise 동일한 출력이다. 즉 D1 학습 동안 `action_in_proj`가 전혀 바뀌지 않았다.

## Frozen Projection Bug

원인은 train script 순서다.

`scripts/84_train_student_ae28_official.py`에서 teacher를 먼저 freeze한다:

```python
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad_(False)
...
bundle, selected_layers = build_bundle(teacher_model, args, student=student)
```

그 다음 `student_backbone_init`에서 frozen teacher module을 deepcopy하고 reset한다:

```python
action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(...).train()
action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(...).train()
reset_module_parameters(action_in_proj)
reset_module_parameters(action_out_proj)
```

하지만 `reset_module_parameters()`는 값을 reset할 뿐 `requires_grad=True`로 돌리지 않는다:

```python
for child in module.modules():
    reset = getattr(child, "reset_parameters", None)
    if callable(reset):
        reset()
```

optimizer param split도 `requires_grad=False`를 skip한다:

```python
for pname, p in mod.named_parameters():
    if not p.requires_grad:
        continue
```

Checkpoint 확인:

- D1 summary `trainable_params`: 1,409,410,048
- 이는 expert param만 해당한다. `action_in_proj` 1,349,632 + `action_out_proj` 4,098는 빠져 있다.
- D1 `best.pt -> final.pt` 비교:
  - `action_in_proj`: changed 0 / 1,349,632, max_abs_diff 0.0
  - `action_out_proj`: changed 0 / 4,098, max_abs_diff 0.0
  - `expert`: changed 15,780,632 / 1,409,410,048

## F2 t / x_t Sensitivity

F2는 D1 `best.pt`를 로드하고 oracle KV batch에서 probe했다.

### Same x_t, t Only Sweep

Baseline `t=0.1`.

| t | emb relative delta | emb cosine | pred_v abs_mean | pred relative delta | pred cosine |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.000 | 1.000 | 0.378 | 0.000 | 1.000 |
| 0.3 | 0.887 | 0.607 | 0.318 | 0.671 | 0.741 |
| 0.5 | 0.838 | 0.649 | 0.353 | 0.727 | 0.692 |
| 0.7 | 0.754 | 0.716 | 0.328 | 0.731 | 0.689 |
| 0.9 | 0.837 | 0.650 | 0.353 | 0.709 | 0.713 |

### Same t, x_t / Noise Scale Sweep

Fixed `t=0.5`, baseline noise_scale 0.0.

| noise_scale | emb relative delta | emb cosine | pred_v abs_mean | pred relative delta | pred cosine |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.000 | 1.000 | 0.315 | 0.000 | 1.000 |
| 0.5 | 1.085 | 0.411 | 0.342 | 0.725 | 0.767 |
| 1.0 | 1.102 | 0.393 | 0.353 | 0.722 | 0.766 |
| 1.5 | 1.107 | 0.387 | 0.379 | 0.765 | 0.798 |

해석:

- t를 바꾸면 action_in_proj embedding도 크게 바뀌고 pred_v도 바뀐다.
- x_t/noise를 바꿔도 embedding과 pred_v가 바뀐다.
- 따라서 "t embedding이 빠졌다" 또는 "model이 t를 완전히 무시한다"는 판정은 아니다.
- 다만 이 반응은 학습된 projection이 아니라 frozen reset random projection을 통해 나온 반응이다.
- pred_v abs_mean은 0.31-0.38 수준으로 계속 낮다. 이전 E2의 target_v abs_mean 1.07 대비 여전히 약 30% scale이다.

## F4 FM Direction Check

Training code:

```python
x_t = (1.0 - t) * x0 + t * x1
target_v = x1 - x0
future_token_embeds = bundle.action_in_proj(x_t, t)
...
pred_v = bundle.action_out_proj(last_hidden).view(-1, *action_dims)
loss = F.mse_loss(pred_v.float(), target_v.float())
```

Inference Euler:

```python
x = torch.randn(batch_size, *self.x_dims, device=device) * temperature
time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
...
v = step_fn(x=x, t=t_start)
x = x + dt * v
```

해석:

- 방향은 정합한다. `t=0` noise에서 시작해서 `t=1` data로 가고, training target도 `x1 - x0`다.
- sampling 방향 mismatch는 이번 collapse의 주 원인이 아니다.

## Conclusion

F1/F2 기준으로는 `t`가 빠진 것이 아니다. 더 직접적인 문제는 **reset된 action projections가 freeze된 채 학습에서 빠졌고, 그 결과 expert만 teacher 원본과 scale/방향이 전혀 다른 frozen random action embedding을 받아 FM velocity를 맞추려다 amplitude-shrink 해로 수축한 것**이다.

즉 이번 진단의 한 줄:

**collapse 원인 = frozen reset `action_in_proj/action_out_proj` + teacher 대비 4x 큰 action embedding scale mismatch.**

## Next Fix

가장 먼저 고칠 부분:

```python
reset_module_parameters(action_in_proj)
reset_module_parameters(action_out_proj)
for p in action_in_proj.parameters():
    p.requires_grad_(True)
for p in action_out_proj.parameters():
    p.requires_grad_(True)
```

그리고 재검증 순서:

1. build 직후 trainable param count가 expert + 1,353,730 증가했는지 확인.
2. E3 random 1-sample 500 step 재실행.
3. D1 oracle KV 32-sample overfit 재실행.
4. 그래도 collapse하면 그때 teacher original action_in_proj graft(F3)와 x1-pred objective를 비교한다.

## Artifacts

- Probe script: `scripts/95_probe_action_in_proj_t_sensitivity.py`
- Summary: `outputs/action_expert/fm_collapse_diagnostics/f1_f2_action_in_proj_seed42/summary.json`
