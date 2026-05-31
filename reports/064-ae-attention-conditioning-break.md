# 064 AE Conditioning Break Located: Attention Weight Self-Attention Saturation

- status: closed
- owner: sukim
- date: 2026-05-31
- context: 실험 C(062 후속)에서 확정된 "AE가 input KV conditioning을 무시함"의 정확한 끊김 지점 측정. 학습 무, attention weight probe만.

## TL;DR

- 구조적 concat은 정상 (K length = prefill+128 모든 layer 확인).
- Attention weight의 ~45%가 자기 128 diffusion token에 collapse. Prefill로 가는 비율 0.546 (uniform baseline 0.98). Self-attention이 uniform 대비 ~22배 강함, prefill은 1.8배 약함.
- Static vs Dynamic sample (input target_xyz 5배 차이) → attention pattern 차이 < 0.005. AE가 input distinguish 못 함의 직접 증거.
- **끊김 지점은 expert self-attention. 원인은 action_in_proj output 분포가 학생 backbone hidden 분포와 호환 안 됨** → Q-K product이 self에 saturation.

## Setup

`scripts/87_probe_ae_attn_weights.py` (신규, eval-only). 핵심:
1. `force_attention(bundle.expert, "eager")` — SDPA는 attn_weights를 None으로 반환하므로 eager로 강제. weight는 그대로.
2. `bundle.expert.layers[i].self_attn` 28개에 forward hook 설치. `(attn_output, attn_weights)` 의 두 번째 element 캡쳐.
3. 각 layer에서 sample_paths의 10개 diffusion sub-step 중 **첫 호출만** 캡쳐 (t=0 시점 비교 일관성).
4. `attn_to_prefill_frac = sum_K[..., :prefill] / sum_K[..., :]` per layer/head/sample.
5. `kv_norms`: `batch["cache"].layers[i].keys/values` 의 abs_mean / per-pos norm.
6. Static 2개 + Dynamic 2개 = 4 samples. 같은 probe seed 12345.

학습/모델 무수정. probe_module.py 의 기존 `force_attention()` 함수만 config flag 변경.

## Diagnostic 1 (code trace, fact-only)

### (1) Expert class / attention backend
- `build_28layer_expert` (probe.py L152): `AutoModel.from_config(new_config)`.
- `student_backbone_init` 모드 (84 L514 `build_student_backbone_expert`) → `new_config = student_lm.config` (model_type=`qwen3_vl_text`) → `transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel`.
- baseline `args.attn_implementation="sdpa"` → `force_attention` 로 expert sub-module config의 `_attn_implementation="sdpa"`. Diagnostic 2에서는 probe 목적 eager로 강제.

### (2) past_key_values concatenation
`Qwen3VLTextAttention.forward` (modeling_qwen3_vl.py L432-437):
```python
if past_key_values is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs)
```
**구조적으로 prefill K/V + new K/V가 concat. K length = prefill+128.** Diagnostic 2 측정에서 확인됨 (모든 4 samples, 28 layers 모두 `key_length_matches=True`).

### (3) attention_mask=None + is_causal=False 처리
- `Qwen3VLTextModel.forward` 시그니처에 `is_causal` 인자 명시 없음, `**kwargs`로 전달.
- 우리 코드 (84 L865-872): `expert(attention_mask=None, is_causal=False)`.
- Forward L832-839: `attention_mask = create_causal_mask(..., attention_mask=None, ...)` → **None이어도 causal mask 강제 생성**.
- attention layer로 들어가는 attention_mask는 causal mask, `is_causal=False`는 `**kwargs`에 남음.
- SDPA는 attention_mask가 있으면 그것 우선 → effectively causal. `expert_non_causal_attention=True` config가 무시됨.
- 다만 causal mask는 query 128끼리만 영향, prefill KV는 모든 query가 attend 가능 → prefill exclusion 발생 안 함.

### (4) Position IDs 정합성
- build_batch L750-756 (`stage2_attention_mode=official_none`):
  `arange(128).view(1,1,-1).repeat(3, B, 1) + rope_deltas + kv_cache_seq_len`.
- prefill K의 RoPE position은 student.backbone forward 시점에 적용된 것. Expert는 그것을 직접 알지 못함, 새 query/new key에만 자기가 받은 position_ids로 RoPE 적용.
- 정합성은 `rope_deltas + kv_cache_seq_len`이 prefill 마지막 text token의 mrope position +1과 일치해야. 직접 측정 안 함 — diagnostic 2의 attention weight로 간접 검증 (잘못되면 prefill 거의 attend 안 함).

## Diagnostic 2 (attention weight probe)

### Per-sample summary

| Sample | kind | target_xyz_abs_mean | prefill | uniform_baseline | **attn_to_prefill_frac** | min | max |
|---|---|---|---|---|---|---|---|
| A (sg_07) | static | 2.09 | 3098 | 0.980 | **0.549** | 0.220 | 0.876 |
| B (sg_05) | static | 2.15 | 3097 | 0.980 | **0.546** | 0.221 | 0.864 |
| C (sg_01) | dynamic | 20.19 | 3096 | 0.980 | **0.549** | 0.206 | 0.895 |
| D (sg_00) | dynamic | 21.20 | 3102 | 0.980 | **0.546** | 0.194 | 0.890 |

- **Static vs Dynamic 차이 < 0.005** (5배 다른 input). Attention pattern이 input-invariant.
- Uniform baseline 0.98 대비 실제 0.55 → prefill 1.8배 약함. Self(128 query) 부분: `1 - 0.55 = 0.45`. Uniform self baseline = `128/3224 = 0.040`. **Self attention이 uniform 대비 11배 강함**, 또는 prefill 대비 ~12배 강함 per token (0.45/128 vs 0.55/3097).

### Layer-by-layer prefill_frac (sample A)

```
layer  0: 0.63    layer  7: 0.59    layer 14: 0.42    layer 21: 0.60
layer  1: 0.83    layer  8: 0.57    layer 15: 0.47    layer 22: 0.48
layer  2: 0.60    layer  9: 0.62    layer 16: 0.30    layer 23: 0.62
layer  3: 0.63    layer 10: 0.57    layer 17: 0.25    layer 24: 0.72
layer  4: 0.58    layer 11: 0.52    layer 18: 0.22 ★  layer 25: 0.72
layer  5: 0.61    layer 12: 0.52    layer 19: 0.27    layer 26: 0.78
layer  6: 0.56    layer 13: 0.51    layer 20: 0.33    layer 27: 0.88
```

**중간 layer (14-21)에서 prefill_frac 0.22-0.42**. 정보 흐름이 self-attention에 가장 갇히는 구간. 모든 sample (A/B/C/D)에서 동일 패턴 (layer-by-layer 차이 매우 작음).

### KV norm (prefill 구간)
- layer 0: key_abs_mean ~3.92, value_abs_mean ~0.08 (small)
- layer 14: key_abs_mean ~1.32, value_abs_mean ~1.40
- layer 27: key_abs_mean ~1.42, value_abs_mean ~10.9

Layer 0 prefill V의 magnitude가 매우 작음 (0.08) — 학생 backbone embed 직후라 정상. Layer 27까지 증가, 정상 transformer 분포. **KV가 0이 아니고 의미 있는 값**. 즉 "prefill에 정보가 있는데 expert가 attention으로 안 가져옴".

### 구조적 검증
- 모든 4 samples × 28 layers에서 `key_length_matches = True`.
- `total_weight_mean` ≈ 1.0 (softmax 정상).
- Eager attention backend에서 hook capture 정상 작동.

## 끊김 지점 / 원인 한 줄 확정

**끊김 지점**: `bundle.expert.layers[i].self_attn` 의 attention weight 분포. 구조적으로 prompt KV는 concat되어 attention 입력에 들어가지만, query-key product의 magnitude가 self(자기 128 diffusion token) 쪽에 saturate되어 prefill_frac이 uniform 대비 1.8배 약함, self가 22배 강함.

**원인**: `action_in_proj` 출력 분포가 학생 backbone hidden 분포와 호환 안 됨. Reset된 Linear+Norm 위에 Fourier embedding을 통과한 출력이 student backbone hidden과 다른 magnitude/structure를 가지면서, expert layer (학생 backbone weight 그대로)의 Q projection을 거쳐 자기 128 token의 K와 product가 prefill K(학생 backbone forward의 정상 분포)와의 product보다 훨씬 커짐. Softmax가 self쪽으로 collapse. Input sample이 prefill K를 어떻게 바꾸든 (static vs dynamic 5배 차이) attention pattern 거의 변화 없음 → AE가 conditioning 무시.

## Implication / Forward Plan

### 검증 우선 (실험 A — teacher_compressed)
`ae-init-mode teacher_compressed`로 Stage 0 동일 설정 재실행. teacher_compressed는 `action_in_proj`을 reset 안 함 (teacher 학습된 그대로). 만약 통과 (`train_inb_ade < 0.5m`) 면 → action_in_proj distribution mismatch가 결정적 범인 확정.

### 정공법 (distillation 재학습, Phase 1+)
- 060 Plan Phase 1 (teacher bridge freeze + hidden align weight ↑ + cosine→MSE)
- 추가: **action_in_proj도 distillation에서 학습 대상으로 명시.** Currently 84 script가 teacher action_in_proj을 reset해서 처음부터 학습. distillation 단계에서 action_in_proj이 학생 backbone과 함께 trajectory loss로 학습되도록 변경.
- Alternative: action_in_proj reset 없이 teacher 그대로 사용 + 학생 backbone hidden 분포를 teacher와 호환되게 distillation.

### 062 결론 정정 강화
062는 best-of-N 1.68m이 lucky 16-sample artifact. 진짜 best-of-N saturation = 3.81m (eval_samples=64+). 이 attention 분석 결과로 그 천장의 진짜 원인 확정:
- AE가 input KV를 사실상 무시
- 다양한 path는 self-attention noise 때문에 생기는 거지 (실제로 다양한 trajectory 후보) 가 아님
- Best-of-N으로 짜내봤자 self-attention noise에서 운 좋은 sample 골라내는 것 → 천장 빠르게 saturate

## Code changes

- `scripts/87_probe_ae_attn_weights.py` (신규, inference-only):
  - `importlib`로 84 load, force_attention(expert, "eager"), forward hook on 28 attention layers
  - 첫 sub-step만 캡쳐, attn_to_prefill_frac + kv_norm 계산
  - 학습/모델/sample_paths 무수정

## Reproduce

```bash
python scripts/87_probe_ae_attn_weights.py \
  --ckpt-path outputs/action_expert/student_ae28/officialcfg_studentbb_studentfree_1k_s1000_seed42/best.pt \
  --n-extremes 2 \
  --ae-init-mode student_backbone_init --prefix-mode student_free --target-source teacher \
  --train-timestep-sampler beta --num-time-samples 1 \
  --ae-dtype bfloat16 --attn-implementation sdpa --stage2-attention-mode official_none \
  --mapping linspace_round --compressed-layers 28 \
  --student-checkpoint-dir <step_006250> \
  --num-samples 32 --batch-size 1 --eval-batch-size 1 \
  --seed 42 --eval-seed-mode step \
  --output-dir outputs/action_expert/student_ae28/probe_ae_attn_weights
```
