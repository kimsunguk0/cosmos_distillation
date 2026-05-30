# 063 Stage 0 Overfit Sanity FAIL: AE Cannot Memorize 32 Samples

- status: closed
- owner: sukim
- date: 2026-05-31
- context: 060/061의 3.6m 천장이 데이터 부족 때문인지, recipe/pipeline 버그인지, 또는 backbone hidden 분포 문제인지 게이트 판정. 32 sample × 3000 step overfit attempt.

## TL;DR

- **❌ FAIL. 32 sample을 sample당 ~187 epoch 봐도 train loss가 0.9~1.4에 진동, train_inb_ade 평균 3-4m, 가장 낮은 batch 1m, 최고 11m.**
- 같은 batch가 학습 중 여러 번 다시 와도 train_inb_ade가 매 측정마다 1m~11m 진동. **AE가 input sample을 distinguish 못 함**.
- 데이터 부족 아님. **Student backbone hidden이 32 sample조차 distinguish할 수 있는 representation을 제공 안 함** (060 hidden-distribution-mismatch 가설 확정).
- 062의 "AE는 멀쩡, 1.7m best-of-8" 결론은 lucky 16-sample artifact. eval_samples=64+로 재측정한 best-of-8 = 3.81m (별도 검증). Stage 0 FAIL은 그 천장의 진짜 원인 확정.

## Setup

- `--num-samples 32 --eval-samples 32`: 학습/평가 동일 32 sample (의도적 in-distribution overfit).
- `--steps 3000 --batch-size 2`: 16 batch × 3000 step ≈ **187 epoch** per sample.
- `--eval-every 250 --log-every 25 --train-ade-every 100` (신규).
- 직전 3.63m baseline과 동일 nondestructive args:
  - `--ae-init-mode student_backbone_init --prefix-mode student_free --target-source teacher`
  - `--expert-lr 1e-5 --proj-lr 1e-4 --weight-decay 0.01 --grad-clip-norm 5.0`
  - `--lr-warmup-steps 150 --min-lr 1e-6 --no-norm-bias-decay`
  - `--train-timestep-sampler beta --num-time-samples 1`
  - `--ae-dtype bfloat16 --attn-implementation sdpa --stage2-attention-mode official_none`
  - `--mapping linspace_round --compressed-layers 28`
  - `--seed 42 --eval-seed-mode step`

### Code change (84 script)

`--train-ade-every` 신규 인자만 추가. Training loop에 in-batch ADE 측정 분기 추가 (gradient 영향 X, RNG state 보존). 학습/loss/모델 무변경. Default 0 = 기존 동작 100% 보존.

```python
if int(getattr(args, "train_ade_every", 0)) > 0 and step % int(args.train_ade_every) == 0:
    _torch_rng = torch.get_rng_state()
    _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    bundle.eval()
    try:
        with torch.no_grad():
            pred_tr = sample_paths(bundle=..., teacher_model=..., batch=batch, seed=args.seed + 2_000_000 + step, device=device)
    finally:
        bundle.train()
        torch.set_rng_state(_torch_rng)
        if _cuda_rng is not None:
            torch.cuda.set_rng_state_all(_cuda_rng)
    # compute ade vs batch["target_xyz"] → log "train_inb_ade_m"
```

## Results

### Train loss (sampled every 200 steps)

| step | loss | grad_norm | pred_v | target_v | target_act |
|---|---|---|---|---|---|
| 1 | 1.36 | 29.8 | 0.535 | 0.80 | 0.19 |
| 200 | 1.22 | 10.6 | 0.247 | 0.87 | 0.19 |
| 400 | 1.38 | 19.3 | 0.181 | 0.94 | 0.43 |
| 600 | 0.86 | 9.4 | 0.210 | 0.74 | 0.19 |
| 800 | 1.13 | 10.1 | 0.385 | 0.98 | 0.43 |
| 1000 | 0.96 | 9.8 | 0.201 | 0.78 | 0.19 |
| 1500 | 1.18 | 14.5 | 0.214 | 0.88 | 0.19 |
| 2000 | 1.18 | 10.4 | 0.484 | 0.97 | 0.43 |
| 2500 | 1.13 | 7.5 | 0.469 | 0.92 | 0.43 |
| 3000 | 1.01 | 9.1 | 0.307 | 0.82 | 0.19 |

- 0.86~1.38 사이 진동. **수렴 안 함**.
- `pred_v` 0.18~0.54 vs `target_v` 0.74~0.98 → 학습 진행하지만 underestimate 패턴 그대로.

### train_inb_ade (in-batch ADE, every 100 steps)

| step | inb_ade | 16wp | 32wp | batch_loss | target_act |
|---|---|---|---|---|---|
| 100 | 4.38 | 0.18 | 1.06 | 1.97 | 0.19 |
| 200 | 11.48 | 1.04 | 3.31 | 1.22 | 0.19 |
| 400 | 3.15 | 0.10 | 0.55 | 1.38 | 0.43 |
| 500 | 2.24 | 0.07 | 0.51 | 1.67 | 0.19 |
| 800 | 1.03 | 0.10 | 0.25 | 1.13 | 0.43 |
| 1000 | 10.38 | 0.52 | 2.79 | 0.96 | 0.19 |
| 1500 | 3.23 | 0.33 | 0.73 | 1.14 | 0.19 |
| 1800 | 7.21 | 0.64 | 2.46 | 1.11 | 0.19 |
| 2000 | 1.74 | 0.31 | 0.70 | 1.18 | 0.43 |
| 2500 | 1.97 | 0.16 | 0.55 | 1.63 | 0.19 |
| 3000 | 7.29 | 0.92 | 1.98 | 1.01 | 0.19 |

- 같은 batch가 다시 와도 inb_ade가 1m~11m으로 진동.
- **정적 sample (target_act=0.19, 거의 정지) batch에서 inb_ade가 가장 큼** (200, 1000, 1400, 1800, 3000에서 7~11m). 학습 187 epoch 봤는데도 "정지" pattern을 못 외움.
- 같은 batch (16-sample, target_act=0.19) 가 step 200/1000/1400/1800/3000에서 inb_ade=11.5/10.4/7.6/7.2/7.3m. **외워야 할 32 sample 중 절반 (정적 sample)에 대해 학습 자체가 안 일어남**.

### Eval ADE (32 sample = train set)

| step | ADE | 16wp |
|---|---|---|
| 0 | 9.14 | 0.85 |
| 750 | 3.77 | 0.59 |
| 1250 | 3.71 | 0.42 |
| 2250 | 3.44 | 0.43 |
| 3000 | **3.32** | 0.51 |

3.3~4.3m에 머무름. 1K baseline의 3.63m과 사실상 동일 = sample 수 줄여도 천장 같음.

## 판정: ❌ FAIL

**32 sample도 외우지 못함.** Train loss 0.9~1.4 진동, train_inb_ade 평균 3-4m (정적 sample batch에 대해 7-11m). 데이터 부족이 천장의 원인이 아니다.

## 가장 결정적 신호

**정적 sample (target_act=0.19) batch에 학습이 일어나지 않는다.**

같은 16-sample batch (정적 sample 모음)가 step 200, 1000, 1400, 1800, 3000에 다시 와도 inb_ade가 11.5/10.4/7.6/7.2/7.3m. 7번을 187 epoch 동안 봤지만 외워지지 않음.

해석: AE가 input sample을 distinguish하지 못함. 같은 backbone KV cache에 대해 매번 다른 (random) trajectory output 생성. 즉 **backbone hidden이 32 sample을 distinguish하기 위한 정보를 인코딩 못함**. 학생 backbone hidden은 (in our setup) 거의 input-invariant.

비교: target_act=0.43 (동적 sample) batch에서는 inb_ade 1-3m으로 학습 가능. 동적 sample은 input에 더 강한 trajectory signal이 있어서 backbone hidden도 distinguish할 정보 일부 가짐. **정적 sample은 trajectory가 거의 0이라 weakly distinguishable**.

## 진단 점검 (코드 사실, 추측 없음)

### (a) Action normalization
- `teacher_model.action_space.traj_to_action(history_xyz, history_rot, future_xyz, future_rot)` 호출 (84 L680-685).
- `action_space` 모듈 내부에서 normalize. 84 script 직접 normalize 안 함.
- 학습 / 평가 모두 같은 함수, **모든 mode에서 baseline과 일관**.

### (b) Timestep sampling
- `--train-timestep-sampler beta` (default).
- `src/alpamayo_r1/diffusion/flow_matching.py` L54-57: `Beta(1.5, 1.0)`, `t = 0.999 - sample * 0.999`.
- Beta(1.5, 1.0) sample mean = 1.5/2.5 = 0.6 → `t` 평균 ≈ 0.4. high-t (target 가까운 영역) cover.
- **공식 설정. 변경 이유 없음**.

### (c) KV 전달
- `--stage2-attention-mode official_none`. position_ids = `arange(n_diff).view(1,1,-1).repeat(3,B,1) + rope_deltas + kv_cache_seq_len` (84 L750-756).
- `sample_paths` 내부 `prompt_cache.crop(prefill_seq_len)` (L960). attention_mask = None.
- 공식 SFT Stage 2와 일치. **정상**.

### (d) action_in_proj
- `student_backbone_init` 모드 분기 (84 L587-588): `action_in_proj = copy.deepcopy(teacher_model.action_in_proj)` 후 `reset_module_parameters(action_in_proj)`.
- Fourier `freqs`는 `register_buffer`로 등록 → reset_parameters() 영향 X (유지).
- Linear weights + LayerNorm + RMSNorm은 reset.
- 의도된 reset. teacher_compressed 모드와는 다름. **scratch projection이 학습 가능한 capacity 보유**.

→ 4가지 점검 모두 baseline과 일관, recipe 자체에 명백한 버그 없음. **천장은 backbone hidden의 capability 부족**.

## Implications

### 062의 "AE는 1.7m까지 가능" 결론 정정
- 062: best-of-8 = 1.68m @ eval_samples=16. 매우 lucky 16-sample artifact.
- eval_samples=64+: best-of-8 = 3.81m (별도 stability sweep).
- Stage 0 FAIL은 그 천장이 진짜라는 증거: 32 sample조차 외울 수 없는 backbone에서 best-of-N으로 3.8m 짜내는 것이 한계.

### 060/061 hidden-distribution-mismatch 가설 확정
- 다른 가설 다 검증해서 기각: hyperparameter (LR/scheduler/weight_decay), prefix mode (student_free vs teacher_forced), AE init (teacher AE / Q swap / scratch projections), data scale (1K→5K), target source (teacher pred vs GT), backbone joint LoRA, best-of-N reranking.
- 모든 시도가 4m 천장 근처에서 막힘.
- Stage 0가 마지막 sanity: **32개도 못 외움 = backbone hidden이 distinguish 정보 못 줌 = distillation 단계 결함**.

### 다음 단계 (강제 정공법)

**060 Plan Phase 1-4 진행 외에 다른 path 없음.**

Phase 1 (즉시, 1-2일):
- `student_wrapper.py`에서 `traj_hidden_bridge_teacher.requires_grad = False` (teacher bridge freeze, 학생만 teacher 분포로 끌어옴).
- `loss_weights.teacher_traj_hidden_align_loss: 0.08 → 0.85` (traj_ce 동급).
- `traj_hidden_bridge.cosine_weight: 0.85 → 0.15`, `mse_weight: 0.15 → 0.85` (scale도 강제 매칭).
- 200k semantic-balanced corpus 그대로 재학습 (~1.5일).

Phase 2 (1주): token-wise teacher hidden cache (현재 pooled 1-vector only).
Phase 3 (1-2주): layer-wise teacher K/V cache for AE cross-attention alignment.
Phase 4 (1-2주): vision feature distillation (teacher visual.merger → student projection MSE).

Phase 1 후 다시 Stage 0 overfit sanity 돌려서 통과하는지 확인. 통과하면 본 학습 진행.

## Stage 0 Repro

```bash
python scripts/84_train_student_ae28_official.py \
  --ae-init-mode student_backbone_init --prefix-mode student_free --target-source teacher \
  --train-timestep-sampler beta --num-time-samples 1 \
  --ae-dtype bfloat16 --attn-implementation sdpa --stage2-attention-mode official_none \
  --mapping linspace_round --compressed-layers 28 \
  --student-checkpoint-dir <step_006250> \
  --num-samples 32 --eval-samples 32 --batch-size 2 --eval-batch-size 2 \
  --steps 3000 --eval-every 250 --log-every 25 --train-ade-every 100 \
  --eval-seed-mode step --seed 42 \
  --expert-lr 1e-5 --proj-lr 1e-4 --weight-decay 0.01 --grad-clip-norm 5.0 \
  --lr-warmup-steps 150 --min-lr 1e-6 --no-norm-bias-decay \
  --output-dir outputs/action_expert/student_ae28/stage0_overfit_32_s3000_seed42
```

## Stage 0 통과 기준 (Phase 1 후 재실행 시)

- ✅ train loss가 step 진행에 따라 단조 감소, 끝 0.2~0.3 이하 수렴
- ✅ train_inb_ade < 0.5m (per-batch)
- ✅ 정적 batch (target_act=0.19)에서도 inb_ade < 1m
- 위 3가지 모두 만족하면 → 32 sample memorize 능력 입증 → 본 학습 진행
- 하나라도 실패 → 추가 hidden alignment 변경 필요 (Phase 2~4)
