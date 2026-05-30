# 060 Student AE-Pairing 4m Ceiling: Root Cause and Re-distillation Plan

- status: open
- owner: sukim
- date: 2026-05-30
- context: student backbone (step_006250, val ADE 2.5m) + 28-layer Action Expert로 trajectory 생성 시 학생 KV에서 4m 천장에서 막힌 원인 분석 및 재학습 계획.

## TL;DR

- Teacher AE + Teacher KV = **1.5m**, Student AE + Student KV (최선) = **3.63m**, Teacher AE + Student KV = **8.8m**.
- AE side 모든 hyperparameter / init / data-scale / joint-LoRA trick으로도 **4m 천장이 깨지지 않음**.
- Root cause는 distillation 단계의 hidden alignment 설계 결함 + vision-side distillation 부재. 짧은 AE fine-tune으로는 못 푼다.
- Distillation 재학습이 정공법. 4-phase plan 첨부.

## 1. 문제 정의

학생 모델 (Cosmos-Reason-2B base, LoRA fine-tuned via distillation) 위에 학생 전용 Action Expert를 학습시켜야 함. Teacher (Alpamayo-1.5-10B + teacher AE)의 ADE 1.5m을 학생이 따라가야 production화 가능.

학생 backbone checkpoint: `step_006250`
- token-level val ADE 2.5m (학생 단독 traj-token 디코드 vs teacher traj-token 디코드)
- Distillation run: `no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838`

## 2. 실험 결과 (1K 데이터, seed=42)

### 2.1 AE init mode 비교 (모든 학생 backbone frozen)

| Config | Best ADE | Notes |
|---|---|---|
| Teacher AE + Teacher KV | **1.5m** | baseline (목표) |
| Teacher AE (full) + Student KV | 8.82m | step 0=13.17m, 학습 중 더 나빠짐 |
| Teacher Q only + Student backbone rest | 5.48m | Q swap, K/V는 student |
| Student backbone init + scratch projs (학생 free) | 3.90m | step 400 best |
| Student backbone init + scratch projs (teacher_forced) | 4.27m | step 400 best |

### 2.2 Hyperparameter sweeps (Student backbone init, 1K)

| Setting | Best ADE |
|---|---|
| Default LR, clip 1.0 (student_free) | 3.90m |
| proj_lr 3e-5 → 1e-3, clip 1.0 → 5.0 | 4.21m |
| **alpamayo_base SFT spec (proj_lr 1e-4, cosine warmup=100, no-decay grouping, clip 5.0)** | **3.63m** (step 800) |
| Joint LoRA (backbone unfreeze + AE, teacher_forced) | 4.47m (역회귀) |

### 2.3 Data scale (Student backbone init, student_free)

| Data | Best ADE |
|---|---|
| 1K | 3.90m |
| 5K (steps=5000) | 4.13m (step 2000) |

데이터 5배 늘려도 천장 동일. Hyperparameter도 한계 명확.

### 2.4 pred_v collapse 패턴

모든 setup에서 `pred_v_abs_mean ≈ 0.2 vs target_v_abs_mean ≈ 1.0~1.3` (5배 작음). LR↑ + clip↑ + cosine schedule로 0.7까지 잠깐 자랐다가 다시 collapse. 모델이 사실상 "0에 가까운 속도"를 출력하는 데서 막힘.

## 3. Root Cause Analysis

### 3.1 Backbone hidden distribution mismatch

가장 결정적 증거: **Teacher AE + Student KV = 8.8m**. 만약 distillation이 hidden-level matching을 잘 했다면 1.5m에 가까워야 함. 6배 차이.

#### Distillation loss 설계 결함 (구체적)

**(a) Bridge가 양방향 trainable** (student_wrapper.py L242-268):
```
Teacher (4096) ──→ teacher_bridge ──→ 512 ─┐
                   (trainable!)            │
                                           ├─ cosine + MSE
Student (2048) ──→ student_bridge ──→ 512 ─┘
                   (trainable)
```
Teacher bridge가 학습 가능 → teacher 분포가 학생 쪽으로 끌어내려져서 만남. 결과적으로 학생 2048-D hidden은 teacher의 **원본 4096-D 분포와 닮을 필요 없음**. AE는 원본 분포 입력을 받으므로 학생 KV에서 깨짐.

**(b) Loss weight 10:1 불균형** (config L33-50):
```
traj_loss: 0.85       (token output CE)
teacher_traj_topk_kd_loss: 0.12
gt_cot_loss: 0.08
teacher_topk_kd_loss: 0.08
teacher_traj_hidden_align_loss: 0.08   ← hidden은 부산물
teacher_text_boundary_hidden_align_loss: 0.05
```
학습 신호 97% vs 13% — 토큰 출력 분포 맞추기가 압도적.

**(c) Cosine 비중 0.85, MSE 0.15** (config L54-55):
```python
cosine_weight * (1 - cosine) + mse_weight * mse_loss
```
방향만 맞추고 norm/scale은 자유. AE는 norm-sensitive.

**(d) 정렬 대상이 traj 토큰 + boundary 뿐**: CoT 텍스트, 이미지, history token hidden은 unsupervised. AE prefill 시 전체 sequence KV를 보는데 그 중 traj 토큰만 정렬된 상태 → mismatch 누적.

**(e) 마지막 layer만 정렬** (`outputs.hidden_states[-1]` in 05_extract_teacher_signal_cache.py L198): AE는 layer-wise KV cache 전체를 cross-attend. 중간 layer 분포는 무방비.

**(f) Teacher cache가 pooled hidden 1 vector뿐** (05 L208: `pooled_hidden = target_hidden.mean(dim=0)`): 샘플당 단일 4096-D 벡터만 저장. Token-level alignment 자체가 불가능한 정보 부족 상태.

**(g) Eval 2.5m이 token-level ADE** (`src/inference/checkpoint_eval.py::evaluate_decode_subset` L598-603): 학생이 만든 traj-token 시퀀스를 같은 discrete decoder로 디코드 → teacher token 시퀀스 디코드와 비교. **AE 안 씀**. Token 분포 학습은 잘 됐으니 2.5m이지만 AE-호환 hidden 만들지는 않음.

### 3.2 Vision-side issue (보조)

**Vision encoder가 본질적으로 다름**:
- Teacher: Qwen3VL 8B vision (depth 27, hidden 1152, deepstack [8,16,24], merger→4096)
- Student: Qwen3VL 2B vision (depth 24, hidden 1024, deepstack [5,11,17], merger→2048)

**Vision distillation supervision 사실상 0**:
- LoRA가 vision tower 안 건드림 (peft_setup.py L18-25에 visual.* 없음)
- visual.merger만 lr_scale 0.15로 학습 (config L66-67)
- `feature_align_loss: 0.0` — pooled features 정렬 loss인데 dim mismatch로 어차피 dead code

**해상도는 무관**: 입력 이미지가 320×576 = 184320 px이라 distillation의 `min_pixels=49152`와 AE의 `min_pixels=163840` 모두 통과 (resize 없음). 초기 의심이었지만 실제 데이터 검증 후 기각.

### 3.3 Joint LoRA 실험으로 확인

Student backbone LoRA를 unfreeze해서 AE supervision으로 같이 학습 시도. 결과 **4.47m로 후퇴**. 이유:
- 1K step의 짧은 fine-tune으로는 backbone을 trajectory-aware hidden 만들도록 fix 불가
- AE loss로 들어오는 gradient가 backbone LoRA를 흔들지만 token output 능력만 깎임 (catastrophic interference)

즉 **현재 backbone은 token output에 optimize됐고, 가벼운 fine-tune으로는 hidden-level fix 불가능**. 정공법 = distillation 재학습.

## 4. Path Decision

| Path | Approach | 예상 결과 | 시간 |
|---|---|---|---|
| A | 3.63m 수용, AE production 안정화 (EMA, best ckpt, longer data) | ~3.5m 안정 production | 1주 |
| B | KV adapter (학생 KV → teacher-like KV 변환 어댑터 학습) | 미지수, 8.8m → 3-5m 추정 | 1주 |
| C | Backbone joint LoRA + AE | 4.47m (실패 확인) | 완료 |
| **D** | **Distillation 재학습 (정공법)** | **AE-pairing < 2m 기대** | **3-5주** |

**선택**: Path D. Path A는 production만 다듬고 본질 못 풀음. Path C는 검증 완료 (실패). Path B는 우회로지만 4m 천장의 다음 step에서 비슷한 한계 부딪힐 가능성.

## 5. Re-distillation Plan (Phase 1-4)

### Phase 1 — 최소 변경 검증 (2-3일)

기존 teacher cache 그대로 사용. 코드/config만 수정.

**변경 사항**:
1. **Teacher bridge freeze**: `src/model/student_wrapper.py`에서 `traj_hidden_bridge_teacher.requires_grad = False`
2. **Loss weight 재조정** (새 config):
   - `teacher_traj_hidden_align_loss`: 0.08 → **0.85** (traj_ce 동급)
   - 나머지 weight 유지
3. **Cosine/MSE 비율 뒤집기**:
   - `cosine_weight`: 0.85 → 0.15
   - `mse_weight`: 0.15 → 0.85
4. **재학습**: 200k 데이터, batch 16, 12500 step. ~1.5일.

**기대**: Pooled hidden alignment가 강해지면 teacher 평균 hidden distribution을 학생이 따라감. AE-pairing 8.8m → 4-5m 추정 (큰 폭은 아니지만 sanity check).

**리스크**: Pooled 하나뿐이라 효과 제한적일 수 있음. Phase 1이 약한 결과면 곧바로 Phase 2 필요.

### Phase 2 — Teacher cache token-wise 확장 (1주)

**작업**:
1. `scripts/05_extract_teacher_signal_cache.py` 수정: `outputs.hidden_states[-1]` 전체 토큰 저장 (현재 pooled.mean)
2. Storage 계산:
   - Full sequence: 200k × ~3000 tokens × 4096-D × 2 byte (fp16) ≈ **5TB** (비현실적)
   - **Traj + boundary span만**: 200k × 200 tokens × 4096 × 2 ≈ **330GB** (현실적)
3. Token-wise hidden alignment로 진정한 token-level matching 활성화
4. Re-train, 또 1.5일

### Phase 3 — Layer-wise alignment (1-2주)

**문제**: AE는 layer-wise KV cache를 사용 (each backbone layer의 K, V 따로). 마지막 layer만 매칭으로는 부족.

**옵션 A: 모든 layer hidden states** — Storage 5TB × 28 layers = **140TB** (불가능)

**옵션 B: K, V projection output만 cache** (head_dim × num_kv_heads × seq_len). Hidden 전체보다 훨씬 작음. AE가 실제로 cross-attend하는 게 K, V니까 이게 가장 직접적 supervision.
- Storage: 200k × 200 tokens × (8 heads × 128 head_dim × 2 (K,V) × 28 layers) × 2 byte ≈ **45TB** 여전히 큼
- 또는 `<traj_future_start>` 토큰의 KV만: 200k × 1 × ... ≈ **6GB** 작지만 너무 단일 토큰
- 절충: traj + boundary token의 K, V → **2-3TB**

**구현**:
1. Teacher KV cache 추출 script 작성
2. 새 loss: `layerwise_kv_alignment_loss` (per-layer K, V MSE)
3. Loss aggregation

### Phase 4 — Vision feature distillation (1-2주)

**작업**:
1. Teacher `visual.merger` 출력 cache (4096-D per vision token)
   - Storage: 200k × 180 vision tokens (per image) × 4 images × 4096 × 2 = **1.2TB**
2. Student visual.merger 출력 (2048-D) → projection (Linear 2048→4096) → MSE with teacher
3. 또는 LoRA target에 `visual.merger.*`, `visual.layers.*.attn.q/k/v_proj` 추가
4. 새 loss weight: `vision_feature_align_loss: 0.5`

### 진행 순서

**Phase 1 먼저 돌려서 효과 측정.**
- 효과 의미 있음 (예: 4m 미만 또는 best 천장 명확히 깨짐) → Phase 2로 진행
- 효과 없음 → Phase 2 곧장 (token-level 정보 자체가 필요)

각 phase는 독립적으로 평가 가능 (AE-pairing ADE).

## 6. Open Questions

- Phase 3의 K/V layer alignment loss가 실제로 hidden alignment 위에 추가 효과 있는지 (먼저 hidden만 강화해도 충분할 수도)
- Phase 4 vision distillation을 LoRA on visual로 할지 (학습 효율) vs MSE projection으로 할지 (안정성)
- Inference time multi-camera 입력에서 vision encoder LoRA의 효과 검증

## 7. Reference Files (변경 예정)

**Phase 1**:
- `src/model/student_wrapper.py` L256-267 — teacher bridge freeze
- `configs/train/stage_bp3_no_nav_camera_labeled_full444k_balanced_hidden_gc_v2.yaml` — 새 config (Phase 1)
- `scripts/09_train_distill.py` — 학습 entry (변경 없음)

**Phase 2**:
- `scripts/05_extract_teacher_signal_cache.py` L185-218 — token-wise hidden 저장
- `src/training/collator.py` L900-910 — token-wise hidden load
- `src/training/losses.py::token_hidden_alignment_bridge_loss` — token-wise input shape 대응

**Phase 3**:
- 새 script: `06_extract_teacher_kv_cache.py`
- 새 loss: `src/training/losses.py::layerwise_kv_alignment_loss`

**Phase 4**:
- 새 script: `07_extract_teacher_vision_features.py`
- `src/model/peft_setup.py` L18-25 — visual.* target 추가 (옵션)
- 새 loss: `src/training/losses.py::vision_feature_alignment_loss`

## 8. Decision Required

Phase 1 코드 변경 + 학습 시작 OK 여부 확인. 1.5일 GPU 점유. Joint LoRA 실험은 종료됨, GPU 비어있음.
