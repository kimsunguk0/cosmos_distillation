# 065 Hypotheses A/B/C Isolation: Conditioning Break Root Cause

- status: closed
- owner: sukim
- date: 2026-05-31
- context: 064 "AE conditioning break (attn_to_prefill_frac 0.55)"의 정확한 원인을 A(init/reset), B(RoPE), C(causal mask) 세 가설로 분리 검증.

## TL;DR

- **B (RoPE 오정렬) 기각**: build_batch가 HF 표준 공식과 정확히 일치. override 가 prefill_frac을 안 올림 (delta -0.027).
- **A (action_in_proj reset) 부분 기여**: init mode가 raw attention 분포 결정 (teacher_compressed 0.97 vs student_backbone_init 0.54). 하지만 teacher_compressed로도 32-sample overfit fail (eval 6.47m).
- **C (causal mask) 기각**: masked mode가 prefill_frac을 더 낮춤 (0.47), eval 더 나쁨 (5.46m).
- **진짜 본질**: 학생 backbone KV 분포 자체가 어떤 init/mask 조합으로도 호환 안 됨. Distillation 재학습 외 해결 불가.

## Experiment Setup

각 가설에 대해 두 가지 측정:
1. **Raw probe** (학습 X): `--skip-ckpt-load` 옵션으로 학습되지 않은 bundle inference. attn_to_prefill_frac per layer.
2. **Stage 0 overfit** (32 sample × 1000 step): train_inb_ade, eval ADE가 외우기 능력 게이트.

모든 hyperparameter는 직전 Stage 0 (063)와 동일. ae-init-mode / stage2-attention-mode만 가설별 변경.

## Diagnostic 1 (코드 사실)

### get_rope_index 의미 (RoPE 추적)
`transformers/models/qwen3_vl/modeling_qwen3_vl.py:1006-1007`:
```python
mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
```
→ **`rope_deltas = max(mrope_positions) + 1 - sequence_length`**.

L1212 (forward):
```python
(cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
```
→ 새 token mrope position = `cache_position + rope_deltas`. 첫 query position = `kv_cache_seq_len + rope_deltas`.

우리 84 script L750-756: `arange(n_diff).view(1,1,-1).repeat(3,B,1) + rope_deltas + kv_cache_seq_len`. **공식 일치**.

prefill 마지막 mrope position (text token):
- `max(positions) = rope_deltas + kv_cache_seq_len - 1`
- query first position = `max(positions) + 1` (자연 연속). ✓

## Experiment B1 (RoPE 검증)

### B1 Diagnostic
| | Value |
|---|---|
| prefill_seq_len | 3096 |
| n_diffusion_tokens | 64 |
| rope_deltas (inferred) | -2592 (vision token compression) |
| first_query_pos_3d | [504, 504, 504] |
| `first_query_pos == prefill + rope_deltas` | ✓ True |
| matches_hf_convention | True |

### B1 Override probe
| | overall_attn_to_prefill_frac |
|---|---|
| Baseline (build_batch) | **0.551** |
| Override `arange(prefill, prefill+64)` | 0.524 |
| Delta | **-0.027** (오히려 하락) |

**verdict**: `RoPE_ALIGNMENT_OK`. **B 기각.**

## Experiment A1 (Init mode 검증)

### A1a Raw probe (학습 X)

| init mode | overall_attn_to_prefill_frac | per-layer 분포 |
|---|---|---|
| student_backbone_init **RAW** | **0.544** | layer 17-18: 0.30-0.35 (saturate) |
| student_backbone_init TRAINED (3.63m ckpt) | 0.551 | 거의 동일 |
| **teacher_compressed RAW** | **0.973** | 거의 균일 (≈ uniform_baseline 0.98) |

**Init mode가 attention 분포를 결정**. student_backbone_init은 raw에서도 self saturate (학습 무관).

### A1b Overfit (teacher_compressed Stage 0 1000 step)

| step | train_loss | inb_ade | pred_v | target_v |
|---|---|---|---|---|
| 1 | 3.32 | - | **1.23 (overshoot!)** | 0.80 |
| 200 | 1.43 | 24.35m | 0.44 | 0.87 |
| 500 | 1.57 | 2.33m | 0.30 | 0.96 |
| 800 | 1.85 | 7.22m | 0.31 | 0.98 |
| 1000 | 0.94 | 13.70m | 0.23 | 0.78 |
| **eval@1000** | - | **6.47m** | - | - |

- Train loss 0.87-1.85 진동 (수렴 X), train_inb_ade 2-24m 변동.
- 흥미: **pred_v overshoot at step 1** (student_backbone_init는 underestimate). 양쪽 init 모두 다른 양상으로 broken.
- **외우기 fail** (eval 6.47m, baseline 3.32m보다 나쁨).

### A 종합
- ✅ Init mode가 attention pattern 결정 (raw 0.97 vs 0.54).
- ❌ teacher_compressed로도 외우기 회복 안 됨.
- **A는 부분 기여 (표면 attention 분포), 단독 범인 아님**.

## Experiment C1 (Causal mask 검증)

### C1 Raw probe (학습 X)
- student_backbone_init + **masked mode** raw: overall = **0.475**
- per-layer: layer 18 = 0.13, layer 19 = 0.12 (더 심한 saturate)
- official_none (0.544) 대비 **더 낮음**.

`masked` mode가 prefill_frac을 **악화**시킴 — `_build_expert_pos_ids_and_attn_mask`가 만든 attention_mask가 query 128 사이 non-causal로 풀어주면서 self attention 더 강화한 것으로 추정.

### C1 Overfit (1000 step)

| step | train_loss | inb_ade | eval ADE |
|---|---|---|---|
| 1 | 1.25 | - | 7.83 |
| 500 | - | 3.11 | 5.19 |
| **750** | - | - | **4.81** (best) |
| 1000 | 0.95 | 10.23 | 5.46 |

- Train loss 0.83-1.48 진동, train_inb_ade 2.5-11m 변동.
- **외우기 fail**. eval best 4.81m, baseline 3.32m보다 나쁨.

### C 종합
- ❌ masked mode가 attention 분포 악화 + 학습 결과 더 나쁨.
- **C 기각**.

## 종합 표

| 실험 | init mode | mask mode | Raw attn_frac | Stage 0 eval | Stage 0 통과? |
|---|---|---|---|---|---|
| Stage 0 (063) baseline | student_backbone_init | official_none | 0.544 | **3.32m** | ❌ |
| **A1b** | **teacher_compressed** | official_none | **0.973** | 6.47m | ❌ |
| **C1** | student_backbone_init | **masked** | **0.475** | 5.46m | ❌ |

Raw attention 분포는 init mode가 거의 결정 (uniform 또는 collapse). 둘 다 학습으로 외우기 회복 못 함.

## 주범 확정 (한 줄)

> **주범 = A/B/C 어느 것도 단독 아님. 학생 backbone hidden distribution이 trajectory-relevant 정보를 충분히 인코딩하지 못해, 어떤 expert init/mask 조합으로도 32 sample을 외울 수 없음.** Distillation 단계의 hidden alignment 결함 (060 분석)이 본질적 root cause.

### 근거
1. **B 기각**: RoPE alignment는 HF convention과 정확히 일치 (B1).
2. **A 부분**: init mode가 attention 표면 분포 결정 (teacher_compressed 0.97 vs student_backbone_init 0.54), 그러나 teacher_compressed도 trained 후 외우기 fail. 즉 attention 분포가 정상이어도 정보 추출 안 됨.
3. **C 기각**: masked mode가 오히려 악화.
4. **공통**: 모든 setting에서 pred_v와 target_v magnitude mismatch (underestimate or overshoot), train_inb_ade 2-24m 진동, 동일 batch가 학습 후 다시 와도 다른 결과 → 학생 backbone KV가 sample-specific 정보를 안정적으로 전달 못 함.

## Forward Plan

이전 보고서들 (060/061/063/064) 의 결론 강화:

**정공법 (Distillation 재학습) 외 다른 path 없음**:
1. **Phase 1** (즉시): teacher bridge freeze, hidden align weight ↑ (0.08→0.85), cosine→MSE 위주.
2. **Phase 2** (1주): token-wise teacher hidden cache (현재 pooled 1-vector only).
3. **Phase 3** (1-2주): layer-wise teacher K/V cache for AE cross-attention alignment.
4. **Phase 4** (1-2주): vision feature distillation.

각 phase 후 **Stage 0 sanity test (32 sample × 1000 step) 통과 여부**가 게이트:
- train_inb_ade < 0.5m (정적/동적 batch 모두)
- pred_v ≈ target_v
- train loss 0.2~0.3 이하 수렴

## Code / Scripts

- `scripts/88_probe_rope_alignment.py` (이미 064에 추가): `--skip-ckpt-load` flag로 raw bundle probe 가능. Position diagnostic + force-override.
- `scripts/84_train_student_ae28_official.py`: 변경 없음. ae-init-mode / stage2-attention-mode argument만 활용.

## Reproduce

```bash
DISTILL=/path/to/cosmos_distillation
CKPT=$DISTILL/outputs/.../officialcfg_studentbb_studentfree_1k_s1000_seed42/best.pt
STUDENT=$DISTILL/outputs/.../step_006250

# B1
python scripts/88_probe_rope_alignment.py --ckpt-path $CKPT \
  --ae-init-mode student_backbone_init --stage2-attention-mode official_none \
  [stage0 args] --num-samples 32 --batch-size 1 ...

# A1a (raw teacher_compressed)
python scripts/88_probe_rope_alignment.py --skip-ckpt-load \
  --ae-init-mode teacher_compressed --stage2-attention-mode official_none ...

# A1b (overfit teacher_compressed 1000 step)
python scripts/84_train_student_ae28_official.py \
  --ae-init-mode teacher_compressed --stage2-attention-mode official_none \
  --num-samples 32 --steps 1000 --train-ade-every 100 ...

# C1 (masked mode probe + overfit)
python scripts/88_probe_rope_alignment.py --skip-ckpt-load \
  --ae-init-mode student_backbone_init --stage2-attention-mode masked ...

python scripts/84_train_student_ae28_official.py \
  --ae-init-mode student_backbone_init --stage2-attention-mode masked \
  --num-samples 32 --steps 1000 --train-ade-every 100 ...
```
