# 069 G0 MLP Baseline: Random FM Task Itself Is Learnable

- status: closed
- owner: sukim
- date: 2026-05-31
- context: 068에서 발견된 "fixed (x0,t)는 외우지만 random (x0,t) FM은 cosine -0.04로 collapse" 의 원인을 가르기. AE 학습 경로 문제냐 vs FM 정의/sampler 문제냐.

## TL;DR

- **G0 PASS**: 작은 독립 MLP (3-layer, 549K params)가 1500 step으로 같은 random FM task를 학습. **final cosine 0.91, alpha 0.81**.
- → FM target formula + Beta(1.5, 1.0) sampler **자체는 학습 가능**. 068의 random FM collapse는 **AE 경로 (action_in_proj + expert + action_out_proj)** 문제.
- G1, G2 코드 추적도 정합 OK 확인.

## G1 (x_t/t 시간 정합 — 코드 추적)

`scripts/84_train_student_ae28_official.py::train_step` (L824-879):

```python
x1 = target_action.to(device=device, dtype=dtype)
x0 = torch.randn_like(x1)
t = sample_fm_timesteps(...)               # shape (B, 1, 1)
x_t = (1.0 - t) * x0 + t * x1              # broadcast (B, T, D)
target_v = x1 - x0
future_token_embeds = bundle.action_in_proj(x_t, t)
```

- (1) x_t 만들기에 쓴 t와 action_in_proj에 넣는 t **동일 텐서**.
- (2) target_v의 x0와 x_t의 x0 **동일 텐서**.
- (3) batch 내 각 sample이 자기 t (shape (B, 1, 1)). num_time_samples > 1일 때 repeat_interleave로 정합.
- (4) action_in_proj.forward (PerWaypointActionInProjV2 L148-169):
  - `timesteps[..., -1]` (B,1,1) → (B,1), `FourierEncoderV2(timesteps[..., -1])` → (B, 1, dim), `.repeat(1, T, 1)` → (B, T, dim). 모든 T token에 동일 t embed. 정상.

→ **G1 통과**.

## G2 (FM 수식 대조 — 코드 인용)

### Teacher official (`alpamayo_base/src/alpamayo_r1/diffusion/flow_matching.py` L140-164)
```python
noise = torch.randn_like(x)
noisy_x = t * x + (1 - t) * noise          # teacher
target = (x - noise).to(dtype=pred.dtype)  # teacher
```

### Ours (84 L835-844)
```python
x_t = (1.0 - t) * x0 + t * x1              # ours, x=x1, noise=x0
target_v = x1 - x0                          # ours
```

대응 (x=x1, noise=x0):
- noisy_x = t·x + (1-t)·noise = t·x1 + (1-t)·x0 = x_t (commutative)
- target = x - noise = x1 - x0 = target_v

→ **G2 byte-level 정합. 통과**.

## G0 (Analytical MLP baseline)

### Setup (matches V2 random FM E3)
- T = 64 waypoints, D = 3 action dim, **1 fixed target** x1 (random init, abs_mean ≈ 0.30).
- Random x0 ~ N(0, 1), t ~ Beta(1.5, 1.0) → `t = 0.999 - sample * 0.999`.
- Same x_t formula, same target_v formula.
- MLP: 3-layer (Linear+SiLU)*3 + Linear, hidden=512. Fourier encoder for t (num_freqs=20, max_freq=100). 549K params.
- AdamW LR=1e-3, weight_decay=0.01, batch=4.

### Results

| step | loss | cosine | alpha | pred_abs_mean | target_abs_mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.15 | 0.016 | 0.0006 | 0.03 | 0.86 |
| 100 | 0.42 | 0.797 | 0.603 | 0.63 | 0.86 |
| 500 | 0.30 | 0.838 | 0.701 | 0.68 | 0.82 |
| 1000 | 0.48 | 0.667 | 0.573 | 0.58 | 0.85 |
| 1500 | **0.19** | **0.914** | **0.812** | 0.75 | 0.85 |

**Verdict**: PASS (final cosine > 0.9).

## Comparison vs V2 AE random FM

| | V2 random FM (AE) | G0 random FM (MLP) |
|---|---|---|
| final loss | 1.20 | 0.19 |
| final cosine | **-0.04** (negative!) | **0.91** |
| final pred_abs | 0.23 | 0.75 |
| final target_abs | 0.82 | 0.85 |

**같은 task, 같은 sampler, 같은 target 공식**인데 결과는 완전히 다름. MLP는 학습, AE는 collapse.

## 범인 좁힘

FM 자체는 정상. AE 경로 (action_in_proj → expert → action_out_proj)의 어떤 요소가 random FM 학습을 막음. 가장 의심:

### 1. `action_in_proj`의 마지막 `LayerNorm` (PerWaypointActionInProjV2 L146, L169)
```python
self.norm = nn.LayerNorm(out_dim)
...
return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))
```
LayerNorm은 hidden 분포를 unit variance로 normalize. random FM의 large-magnitude input (특히 t≈1일 때 x_t는 거의 x1이지만 target_v = (x1-x_t)/(1-t)가 매우 큰 magnitude) 정보 손실 가능.

### 2. `action_out_proj` = 단순 `nn.Linear`
kaiming init scale 작아서 large-magnitude output 만들기 어려움. Reset 후 학습으로 키워야 하는데 1 sample × 500 step에서 못 키움 (G0 MLP는 4-layer 더 깊고 weight decay에 덜 민감).

### 3. expert layers (Qwen3VL 28-layer) + LayerNorm normalize
각 layer 출력이 RMSNorm → 다음 layer. magnitude 정보가 layer-wise normalize. Output의 absolute magnitude가 마지막 Linear 의존.

## 다음 진단 후보

### G3a (가장 빠른 단일 ablation)
`action_in_proj`의 마지막 `LayerNorm`을 임시로 Identity로 대체 후 V2 random FM E3 재실행.
- cosine > 0.9 → LayerNorm이 magnitude 정보 죽이는 게 root cause 확정.
- 여전히 collapse → 다음 component (expert layers 또는 action_out_proj).

### G3b (AE-MLP 직접 대조)
G0의 MLP와 동일 hyper로 AE bundle만 사용. AE 학습 가능한지 확인.
- 학습됨 → train_step 외부 setup 문제.
- Collapse → AE 구조 직접 원인.

### G3c (action_in_proj/out_proj 분포 측정)
random FM 학습 중 layer-by-layer hidden state magnitude 측정. 어디서 magnitude 정보 사라지는지 시각화.

추천 순서: **G3a (LayerNorm ablation, 5분) → 결과에 따라 G3b/G3c**.

## Code / Scripts

- `scripts/g0_mlp_random_fm.py` (신규, standalone): tiny TFMlp 모듈 + random FM 학습 루프 + cosine/alpha 로깅. 약 90 lines.
- 84 / probe scripts 무변경.

## Reproduce

```bash
cd $DISTILL_ROOT
.venv/bin/python3 scripts/g0_mlp_random_fm.py
```
