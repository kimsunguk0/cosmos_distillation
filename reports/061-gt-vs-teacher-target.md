# 061 GT vs Teacher Trajectory as Flow-Matching Target

- status: closed
- owner: sukim
- date: 2026-05-31
- context: 4m 천장이 teacher 샘플링 노이즈 때문인지 검증 위해 flow-matching target을 GT로 교체하고 비교.

## TL;DR

- Hypothesis: Teacher pred는 flow-matching 샘플링 결과라 ±1.6m 흔들리는데, 이 노이즈가 deterministic regression target으로 안 좋을 수 있다. GT로 바꾸면 4m 천장이 깨질지 검증.
- Result: **GT target이 오히려 0.5m 나쁨** (best 3.67m → 4.14m). 가설 기각.
- 4m 천장은 target source가 아니라 backbone hidden 분포 문제임이 재확인됨. Report 060의 distillation 재학습 plan 유지.

## Setup

- `scripts/84_train_student_ae28_official.py` 에 `--target-source {teacher,gt}` 플래그 추가.
  - `teacher`: 기존 `raw_teacher_pred(raw_json)` 사용 (default, 행동 보존).
  - `gt`: `load_ego_future_xyz` (collator) + `load_ego_future_rot` (checkpoint_eval, 신규) 로 canonicalized ego-local at t0 좌표 GT 로드. 두 경로 모두 동일한 `action_space.traj_to_action(...)` 통과.
- Disk shape 처리: `ego_future_rot.npy` 는 `(1, 1, 64, 3, 3)` 으로 leading singleton dims 있음 — `load_ego_future_rot` 에서 `reshape(-1, 3, 3)` 으로 flatten (collator의 `load_ego_future_xyz` 와 동일 방식).
- Shape assert: `target_action.shape[1:] == action_space.get_action_space_dims()` 두 모드 모두 통과.

### Sanity check (4 samples, batch=2, 1 step, teacher_forced)

| Mode | target_action_abs_mean | target_v_abs_mean | pred_v_abs_mean | loss |
|------|---|---|---|---|
| teacher | 0.190 | 0.816 | 0.531 | 1.311 |
| gt | 0.383 | 0.949 | 0.586 | 1.678 |

- 두 모드 모두 합리적 범위 (`target_v ≈ 1.0`).
- GT 모드에서 `target_action` 이 2배 큼 — teacher pred가 평균 근처로 smoothed 됐기 때문.

## Main experiment

Same hyperparameters except `--target-source`:

- `--ae-init-mode student_backbone_init --prefix-mode student_free`
- `--num-samples 1000 --steps 1000 --eval-every 200 --batch-size 2 --seed 42`
- official cfg: `--expert-lr 1e-5 --proj-lr 1e-4 --weight-decay 0.01 --grad-clip-norm 5.0 --lr-warmup-steps 100 --min-lr 1e-6 --no-norm-bias-decay`

### Eval curve (ADE m)

| step | Teacher | GT |
|------|---------|----|
| 0 | 10.77 | 10.91 |
| 200 | 7.81 | 8.84 |
| 400 | 4.12 | 5.01 |
| 600 | 6.61 | 6.63 |
| **800** | **3.67** (best) | **4.14** (best) |
| 1000 | 6.00 | 6.54 |

Teacher run은 060 baseline (3.63m) 거의 재현. GT 모드가 모든 step에서 동등하거나 나쁨.

### 16wp (1.6s horizon)

| step | Teacher | GT |
|------|---------|----|
| 400 | 0.420m | 0.470m |
| 800 | 0.553m | 0.556m |

16wp short-horizon에서도 GT가 살짝 나쁨. 차이는 long horizon에서 더 큼 (4m 천장 근처).

### pred_v vs target_v progression

| step | Teacher pred_v | Teacher target_v | GT pred_v | GT target_v |
|------|---|---|---|---|
| 1 | 0.535 | 0.762 | 0.531 | 0.922 |
| 250 | 0.160 | 0.992 | 0.120 | 0.914 |
| 500 | 0.158 | 1.305 | 0.348 | 1.719 |
| 1000 | 0.260 | 1.367 | 0.316 | 1.781 |

두 모드 모두 `pred_v` collapse (0.1~0.3 vs target 1.0~1.7). 5배 magnitude 불일치 동일.

## Why GT was worse

1. **GT 가 target magnitude 더 큼**: step 500/1000 에서 GT `target_action_abs_mean = 1.56` vs teacher `0.99`. GT는 dataset의 raw variance 그대로 (큰 가속/감속, sharp turn). Teacher pred는 flow-matching 샘플링 결과라 평균 근처로 smoothed.
2. **MSE loss는 target magnitude의 제곱에 비례**: step 500 loss = GT 4.82 vs teacher 2.92. Target이 큰 만큼 gradient도 큼, but 학습 안정성은 더 나쁨 (grad_norm 75.5 vs 34.0 at step 1000).
3. **모델 capacity 한계**: 060 분석대로 backbone hidden 분포가 trajectory-relevant feature 가지지 않음. Target source 변경으로는 풀 수 없는 한계.

## Implications

- **Teacher pred의 ±1.6m 샘플링 노이즈가 4m 천장의 원인이 아니다.** Noise를 제거(GT)해도 천장 동일 (오히려 나쁨).
- **Smoothed teacher target이 학생에게 더 쉬운 supervision**: 학생 모델의 표현력 한계 내에서 학습 가능한 trajectory 분포가 teacher pred 쪽에 더 가까움.
- Report 060의 결론 강화: 4m 천장은 backbone hidden distribution mismatch 문제. Distillation 재학습 plan (Phase 1-4) 유지.

## Code changes (committed)

- `src/inference/checkpoint_eval.py`: `load_ego_future_rot()` 추가 (leading singleton dim flatten 포함).
- `scripts/84_train_student_ae28_official.py`:
  - `load_ego_future_xyz`, `load_ego_future_rot` import 추가
  - `--target-source {teacher,gt}` 플래그 (default `teacher`, 기존 행동 보존)
  - `build_batch` 내 GT 분기: `(1,1,64,3)`/`(1,1,64,3,3)` 처리 + 64 wp 절단
  - 학습 시작 시 `{"event": "target_source", "mode": ...}` 로그
  - `target_action.shape[1:] == action_space.get_action_space_dims()` shape assert

## Reproduce

```bash
# Teacher target (baseline)
python scripts/84_train_student_ae28_official.py \
  --ae-init-mode student_backbone_init --prefix-mode student_free \
  --target-source teacher \
  --num-samples 1000 --steps 1000 --eval-every 200 --batch-size 2 --seed 42 \
  --expert-lr 1e-5 --proj-lr 1e-4 --weight-decay 0.01 --grad-clip-norm 5.0 \
  --lr-warmup-steps 100 --min-lr 1e-6 --no-norm-bias-decay \
  --student-checkpoint-dir <step_006250> \
  --output-dir outputs/action_expert/student_ae28/target_teacher_1k_s1000_seed42

# GT target
python scripts/84_train_student_ae28_official.py \
  --ae-init-mode student_backbone_init --prefix-mode student_free \
  --target-source gt \
  --num-samples 1000 --steps 1000 --eval-every 200 --batch-size 2 --seed 42 \
  --expert-lr 1e-5 --proj-lr 1e-4 --weight-decay 0.01 --grad-clip-norm 5.0 \
  --lr-warmup-steps 100 --min-lr 1e-6 --no-norm-bias-decay \
  --student-checkpoint-dir <step_006250> \
  --output-dir outputs/action_expert/student_ae28/target_gt_1k_s1000_seed42
```
