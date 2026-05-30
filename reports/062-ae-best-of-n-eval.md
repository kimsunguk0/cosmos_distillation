# 062 AE Best-of-N Eval: 4m Ceiling Was a Single-Sample Artifact

- status: closed
- owner: sukim
- date: 2026-05-31
- context: 060/061에서 발견한 student AE의 ~3.6m 천장이 진짜 모델 한계인지, 아니면 단일 샘플 sampling 노이즈인지 진단.

## TL;DR

- Best-of-8 ADE = **1.68m** (teacher AE + teacher KV의 1.5m에 근접).
- N=8/16 short-horizon ADE: 16wp = **0.17m**, 32wp = **0.55m**.
- Per-sample std over paths = **2.66~3.10m** → FM은 noise를 정상적으로 다양한 mode로 변환 중 (collapse 아님).
- **3.6m 천장은 "AE 능력 부족"이 아니라 "단일 샘플 vs oracle 아티팩트"**. 060/061의 backbone-hidden-distribution 가설은 부분적으로 틀렸다.
- Implication: 학습보다 **inference-time sampling+reranker**가 ROI 큰 우선 path.

## Setup

`scripts/85_eval_ae28_best_of_n.py` (신규, eval-only)이 saved `best.pt`를 load하고 `scripts/84_train_student_ae28_official.py::evaluate()`를 한 번 호출. 학습/모델 구조 무손상.

- Checkpoint: `officialcfg_studentbb_studentfree_1k_s1000_seed42/best.pt`
  - step 800, args.seed=42, eval_seed_mode=step → eval_seed_base = 1842
  - 저장된 best_eval ADE = **3.6332m** (060/061 baseline)
- Eval config: `--eval-samples 16 --eval-batch-size 2 --prefix-mode student_free --ae-init-mode student_backbone_init`
- Per-batch seed scheme: `seed = eval_seed_base + batch_index * num_paths + path_idx`
  - N=1일 때 `eval_seed_base + batch_index` (기존 공식과 동일 → 회귀 통과)

## Regression check (N=1)

| | saved best_eval | new eval N=1 |
|---|---|---|
| `ade_mean_m` | 3.6332 | **3.6332** |
| `h1p6_16wp.ade_mean_m` | 0.5472 | 0.5472 |
| `h3p2_32wp.ade_mean_m` | 1.3732 | 1.3732 |

소수점 4자리까지 일치. 기존 출력 행동 100% 보존 확인.

## Best-of-N sweep (same checkpoint, eval-only)

| N | single ADE (path 0) | best-of-N ADE | std over paths | 16wp single→best | 32wp single→best | 64wp single→best |
|---|---|---|---|---|---|---|
| 1 | 3.633 | — | — | 0.547 | 1.373 | 3.633 |
| 4 | 5.909 | 2.654 | 2.94 | 0.598 → 0.314 | 1.812 → 0.821 | 5.909 → 2.654 |
| 8 | 4.215 | **1.681** | 2.66 | 0.665 → **0.170** | 1.800 → **0.554** | 4.215 → **1.681** |
| 16 | 5.676 | 1.753 | 3.10 | 0.440 → **0.164** | 1.493 → **0.541** | 5.676 → 1.753 |

p50 (median) best-of-N: N=4 → 2.18m, N=8 → 1.48m, N=16 → 1.38m. Mean보다 median이 약간 더 좋음 (long tail 있음).

### Single ADE의 변동성

각 N에서 path 0이 받는 seed는 다름 (N=4/8/16의 path 0 seed = `1842 + batch_index * N`). 같은 모델/입력인데 single ADE가 3.6 / 5.9 / 4.2 / 5.7m로 ±1m 진동. **N=1의 3.63m은 운 좋은 단일 sample**이었을 가능성. 진짜 모델 sampling 평균은 4.2~5.9m 수준.

### Best-of-N 포화

N=8 → N=16: 1.681 → 1.753m (오히려 약간 상승, 또는 sampling 분산). **N=8이면 oracle 한계에 근접**. 더 큰 N으로 짜낼 여지 적음.

### Std over paths

평균 std ~2.7m. 즉 같은 input에 sampling seed만 바꿔도 ADE가 평균 2.7m씩 다름. FM 노이즈가 trajectory output에 정상적으로 반영되고 있다는 증거. **검증 B 통과** (std ≈ 0 시나리오 — FM이 noise 무시 — 가 아님).

## Why the prior "4m ceiling = backbone hidden mismatch" story was incomplete

060/061의 핵심 주장은 "학생 backbone hidden이 trajectory-relevant feature를 못 만들어서 AE가 4m에서 막혔다"였다. 이 best-of-N 결과는 그 일부를 반박한다:

1. **AE는 1.7m까지 도달하는 trajectory들을 생성할 수 있다** (best-of-8). Backbone KV가 완전히 부적합하면 어떤 path도 1.7m에 못 갔어야 한다.
2. **AE 출력 분포가 풍부하고 다양**. backbone hidden이 망가졌다면 모든 path가 비슷한 평균 궤적으로 collapse했을 것. std 2.7m은 정반대 신호.
3. **여전히 single-sample mean은 4~6m**. 즉 좋은 path는 존재하지만 단일 샘플로 못 골라낸다.

수정된 해석:
- Backbone KV는 "충분히 좋은" trajectory를 만들 수 있는 정보를 담고 있다.
- AE는 그 정보 위에서 다양한 plausible trajectory를 sampling 할 수 있다.
- 진짜 bottleneck은 **"여러 plausible 후보 중 best를 식별하는 mechanism이 없다"**.

## Comparison vs token-level best-of-N

- 학생 단독 token-decode ADE (eval/checkpoint_eval): 2.5m (단일 greedy)
- 학생 token best-of-N (보고된 1.7m, 이전 세션)
- **AE best-of-8 = 1.68m**: token best-of-N과 거의 동등

AE pipeline이 token pipeline 대비 추가 손실 없다. AE의 inference-time multi-sample이 token best-of-N과 동급 성능. 즉 **AE 자체는 production-ready, sampling+reranker만 추가하면 됨**.

## Updated path forward

### 새로운 1순위: Inference-time sampling + reranker (1~2주)
**이게 가장 비용 대비 효과 큼**. GT 없이 unsupervised reranking 필요:
- Comfort/smoothness score: longitudinal/lateral acceleration, jerk, yaw rate. PAI Cosmos-RL의 `comfort_reward.py` 활용.
- Path length consistency: target과 prediction path_length 차이.
- Multi-path ensemble: 평균 / weighted average (다양한 path의 weighted blend).
- Token-pipeline cross-check: AE output과 token-decode output의 일관성.

### 2순위: DPO/RL on AE (2~4주)
Best-of-N으로 만든 pairwise data (best vs worst trajectory)로 AE를 fine-tune. 학습 후 single-sample이 best path에 collapse하도록 유도. RL이 더 자연스러움 (continuous reward).

### 3순위 (보류): Distillation 재학습 (060 Plan)
원래 정공법이라고 결론냈으나 best-of-N 결과로 **우선순위 하락**. Backbone이 완전히 망가진 게 아니라는 증거가 강함. 단, 천장을 1.7m → 1.2m로 더 깎으려면 결국 distillation 단계 손봐야 할 수도. 1순위/2순위로 1.7m 안정화 후 결정.

## Code changes

- `scripts/84_train_student_ae28_official.py`:
  - `--eval-num-paths` 플래그 (default 1, N=1 회귀 보존).
  - `evaluate()` 재작성: per-sample N path 수집 → best/single/mean/std + horizon별 best-of-N 보고.
  - `sample_paths()` 본문 무변경. Seed 공식이 N=1일 때 기존과 동일.
- `scripts/85_eval_ae28_best_of_n.py` (신규):
  - 84를 `importlib`로 load (filename이 숫자 시작이라).
  - `--ckpt-path` 추출 후 나머지 argv를 84의 `parse_args`로 위임.
  - load_student → load_teacher → build_bundle → `bundle.load_state_dict(...)` → `evaluate()` 한 번.

## Reproduce

```bash
# Best.pt가 step 800 시점 가중치인 baseline:
CKPT=outputs/action_expert/student_ae28/officialcfg_studentbb_studentfree_1k_s1000_seed42/best.pt
STUDENT=outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250

for N in 1 4 8 16; do
  python scripts/85_eval_ae28_best_of_n.py \
    --ckpt-path $CKPT \
    --eval-num-paths $N \
    --num-samples 16 --eval-samples 16 --eval-batch-size 2 \
    --eval-seed-mode step --batch-size 2 --seed 42 \
    --ae-init-mode student_backbone_init --prefix-mode student_free \
    --student-checkpoint-dir $STUDENT \
    --output-dir outputs/action_expert/student_ae28/eval_only_bon_N$N
done
```
