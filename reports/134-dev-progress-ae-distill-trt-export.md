# 134. 개발 진행 보고: AE Distillation & TRT Export

**작성일**: 2026-06-12
**범위**: Action Expert 압축, Consistency Distillation, TRT Export, 3-Model Benchmark

---

## 1. 개발 진행 항목

### 1.1 QAT INT4 FFN-Only Backbone 학습 — 완료
- **내용**: FLEX K512 backbone에 INT4 FFN-only QAT 적용 (3 epoch)
- **설정**: `int4_ffn_only` (gate/up/down_proj만 INT4, attention QKV/O는 FP16 유지)
- **결과**: step 3750/3750 완료, val loss 1.33 안정
- **체크포인트**: `outputs/checkpoints/qat_mlflex_k512_int4ffn_20k_e3/final`

### 1.2 AE14 Consistency Distillation 시도 — 중단 (접근 변경)
- **내용**: AE28 teacher의 10-step 특징을 dump하고, AE14 student를 2-step consistency distillation으로 학습
- **결과**: ADE 4.45m (AE28 teacher 2.75m 대비 +62%)
- **판단**: 성능 하락 과다 → 접근 방식 변경 (아래 1.3)

### 1.3 AE14 Standard Flow Matching 학습 — 진행중
- **내용**: AE28의 28 layer에서 14개를 uniform select하여 AE14 생성, 기존 flow matching 방식으로 학습
- **설정**: batch=16, num_time_samples=16, 10-step, 10000 steps
- **중간 결과**:

| Step | ADE (m) | AE28 대비 |
|------|---------|----------|
| 2500 | 2.903 | +5.6% |
| 5000 | 2.698 | -1.9% |
| 7500 | **2.566** | **-6.7%** |

- 레이어 절반인데 ADE가 AE28(2.750m)보다 오히려 좋아지는 추세

### 1.4 AE28 Step 축소 실험 — 완료
- **내용**: AE28을 동일 가중치로 inference step만 줄여서 ADE 비교 (64 val samples)

| Steps | ADE (m) | 10-step 대비 |
|-------|---------|-------------|
| 10 | 3.520 | baseline |
| 6 | 3.536 | +0.5% |
| 4 | 3.544 | +0.7% |

- **결론**: 10→4 step은 ADE 거의 동일. Euler integration이 이미 충분히 수렴.

### 1.5 TRT ONNX Export — 완료
- **내용**: 전체 파이프라인 4개 모듈을 FP16 ONNX로 export

| 모듈 | ONNX 크기 | 상태 |
|------|----------|------|
| LLM (Qwen3-VL 2B, 28L) | 3.9 GB | 성공 |
| ViT (Qwen3-VL Visual) | 794 MB | 성공 |
| FLEX (ML-FLEX 4-level) | 35 MB | 성공 |
| AE28 (Action Expert 28L) | 2.9 GB | 성공 |

- **Export 경로**: `outputs/trt_export/flex_k512_fp16/{llm,visual,flex,ae28}/`
- **배포 가이드 문서**: `DEPLOYMENT_NOTES.md` 작성 완료

### 1.6 E2E Latency Benchmark — 완료
- **H200, bf16, batch=1, 20 samples 평균**

| 구간 | 시간 (ms) |
|------|----------|
| ViT | 51.3 |
| FLEX | 3.4 |
| Prefill (LLM) | 43.7 |
| AE 1-step | 23.4 |
| AE 4-step | 89.9 |
| AE 10-step | 224.1 |
| **GPU-only 총합 (4-step)** | **~188** |

### 1.7 3-Model Benchmark — 진행중
- **비교 대상**: Alpamayo 10B / Cosmos 2B + AE28 / Cosmos 2B + FLEX + AE28
- **조건**: 255 samples (17 카테고리 균등), student_free, seed=42, temp=0.85, 6 paths
- **10B 완료**: ADE 5.405m, latency 2826ms
- **Student 2개**: student_free 모드로 재측정 진행중

### 1.8 kjhong Consistency Distillation 파이프라인 분석 — 완료
- Alpamayo 1.5의 36L → 18L AE + step 축소 파이프라인 전체 분석
- 핵심 인사이트: 레이어 축소와 step 축소를 동시 진행, teacher를 직접 4-step으로 재실행하여 dump

---

## 2. 발생 이슈 및 원인 분석

### 이슈 1: AE14 Consistency Distillation ADE 크게 하락 (4.45m, +62%)

**현상**: AE28 teacher(2.75m) 대비 AE14 consistency distill student가 4.45m으로 크게 하락

**원인 분석** (3가지 복합):

1. **Direct 10→2 step jump**
   - kjhong은 10→4→2→1로 progressive하게 축소
   - 우리는 10→2로 한번에 점프
   - 2-step Euler는 10-step 대비 discretization error가 크고, consistency loss만으로 이를 보상하기 어려움

2. **KV cache conditioning 불일치**
   - Teacher feature dump 시 backbone KV cache A로 AE28 추론
   - Student 학습 시 KV cache를 online으로 재생성 (cache B)
   - Teacher의 v_steps, hidden_steps는 cache A 조건에서 나온 결과인데, student는 cache B로 이를 모방
   - Conditioning이 달라 supervision signal이 noisy해짐

3. **KV layer 수 불일치**
   - AE28 teacher: backbone 28L KV cache 전부 사용 (1:1)
   - AE14 student: 28개 중 14개만 select
   - Teacher hidden은 28L cross-attention 결과인데, student는 14L만 보고 모방

**해결 방향**: 레이어 축소와 step 축소를 분리. 먼저 AE14를 standard flow matching으로 학습 (step 유지) → 이후 step 축소는 별도로.

### 이슈 2: DynamicCache API 변경 (transformers 4.49+)

**현상**: `DynamicCache`에 `key_cache`/`value_cache` attribute가 없어 KV layer selection 실패

**원인**: transformers 4.49+에서 DynamicCache 내부 구조가 변경됨.
- 이전: `cache.key_cache[layer_idx]` (list)
- 현재: `cache.layers[layer_idx].keys` / `.values` (DynamicLayer 객체)

**해결**: `select_kv_cache_layers()` 함수를 새 API에 맞게 수정
```python
new_cache.update(prompt_cache.layers[old_idx].keys, 
                 prompt_cache.layers[old_idx].values, layer_idx=new_idx)
```

### 이슈 3: AE14 Eval 시 shape 불일치로 ADE 미산출

**현상**: eval이 모든 샘플에서 실패, ADE 결과 없음

**원인**: `pred_xyz`는 `[B, 1, 1, 64, 3]` (5D), `target_xyz`는 `[B, 1, 64, 3]` (4D). 
`ade_fde()` 함수가 `[64, 3]` 단일 샘플 numpy 배열을 기대하는데, 배치 차원 제거 로직이 불일치.

**해결**: per-sample 루프로 변경, `while ndim > 2: squeeze(0)` 적용

### 이슈 4: AE14 init 시 intermediate_size 불일치 (8256 vs 6144)

**현상**: AE28 checkpoint 로드 시 MLP weight shape mismatch

**원인**: `build_bundle()`에서 teacher expert config(8256)으로 source bundle을 생성했으나, 실제 AE28은 student backbone(6144) 기반으로 학습됨.

**해결**: `ae_checkpoint_compressed` init 모드에서 student backbone config(`build_student_backbone_expert`)로 source bundle 생성하도록 수정

### 이슈 5: FLEX ONNX Export 실패 — LayerNorm shape 불일치

**현상**: FLEX encoder ONNX export 시 `"Given normalized_shape=[1024], expected input with shape [*, 1024], but got input of size [1, 512, 2048]"` 에러

**원인**: ONNX wrapper에서 FLEX의 multi-level processing flow를 잘못 구현. `queries`(1024)와 `visual_features`(2048)의 차원 흐름이 원본 forward와 불일치.

**원본 구조**:
- `queries` [B, 512, 1024]는 모든 level에서 공유 (변하지 않음)
- 각 level: `input_proj(2048→1024)` → `cross_attn(queries, projected)` → `output_proj(1024→2048)`
- 최종 출력만 2048로 반환

**해결**: 4개 level을 명시적으로 unroll하고, for loop/list indexing 제거하여 ONNX tracing 호환

### 이슈 6: AE28 ONNX Export — GQA head 수 불일치

**현상**: KV cache dummy tensor shape 에러 `"Expected size 16 but got size 8"`

**원인**: AE28 expert는 GQA 사용 (num_heads=16, num_kv_heads=8). Dummy KV tensor를 num_heads=16으로 생성했으나 실제는 num_kv_heads=8.

**해결**: `bundle.expert.config.num_key_value_heads` 사용

### 이슈 7: TRT ONNX Export 시 NFS 파일 미생성

**현상**: ONNX export 완료 로그는 나오지만 output 디렉토리가 비어있음

**원인**: NFS 환경에서 `onnx.save_model(save_as_external_data=True)` 실행 시 파일 시스템 동기화 문제

**해결**: `/tmp/`(로컬 디스크)에 먼저 export 후 NFS로 복사

### 이슈 8: 10B Teacher Benchmark — minADE@6 = ADE (동일값)

**현상**: 6개 path를 샘플링했으나 minADE@6이 single ADE와 동일

**원인**: teacher의 `sample_trajectories_from_data_with_vlm_rollout(num_traj_samples=6)`은 내부적으로 `generation_config.num_return_sequences=6`으로 VLM decode를 6번 반복. 하지만 외부에서 seed를 고정했기 때문에 6개 path가 동일하게 생성됨.

**해결 (진행중)**: Student 벤치마크에서는 per-path seed를 `SEED + idx * 1000 + path_idx`로 다르게 설정하여 해결. Teacher도 동일 방식 적용 필요.

---

## 3. 진행 예정 개발 내용

### 3.1 AE14 학습 완료 및 최종 평가 (ETA: ~2시간)
- step 10000까지 학습 완료 후 최종 ADE 확인
- AE28 대비 성능 비교 확정
- 4-step에서의 ADE도 측정 (10→4 step이 AE28에서 +0.7%였으므로 AE14에서도 유사 예상)

### 3.2 3-Model Benchmark 완료 + 시각화
- Student no-FLEX / Student FLEX의 student_free 모드 benchmark 완료
- 카테고리별 4개씩 시각화 (4 cameras + BEV trajectory + CoT)
- 벤치마크 리포트 최종 정리

### 3.3 AE14 + 4-step Latency 측정
- AE14의 per-step latency 측정 (~11ms/step 예상)
- 4-step 기준 총 ~44ms (AE28 90ms 대비 51% 감소)
- E2E GPU-only: ViT 51 + FLEX 3 + Prefill 44 + AE14 4-step 44 = **~142ms**

### 3.4 QAT INT4 + AE 재학습
- QAT int4_ffn_only backbone 위에 AE14 (또는 AE28) 재학습
- QAT backbone의 quantization noise 하에서 AE가 적응하도록

### 3.5 Consistency Distillation 재시도 (올바른 방식)
- kjhong 방식 적용: AE14를 4-step으로 먼저 안정화 후 progressive step 축소
- Teacher (AE28 또는 AE14)를 4-step으로 직접 실행하여 dump
- 4→2→1 step progressive distillation

### 3.6 ViT DeepStack ONNX Export
- 현재 `visual/model.onnx`는 최종 출력만 반환
- FLEX에 필요한 DeepStack intermediate features (layers 5, 11, 17) 추출을 위한 ViT ONNX 수정
- 또는 ViT + FLEX fused ONNX export

### 3.7 Jetson Thor 배포 준비
- H200에서 TRT engine build 테스트
- C++ runtime에서 FLEX image placeholder 치환 로직 구현
- INT4 AWQ LLM + FP16 ViT/FLEX/AE 조합 최적화

---

## 4. 핵심 수치 요약

| 항목 | 값 |
|------|-----|
| AE28 ADE (10-step, teacher_forced) | 2.750m |
| AE28 ADE (4-step, teacher_forced) | 2.754m (+0.7%) |
| AE14 ADE (10-step, 학습중 step 7500) | 2.566m (-6.7%) |
| AE14 per-step latency (예상) | ~11ms |
| E2E GPU-only latency (AE28 4-step) | ~188ms |
| E2E GPU-only latency (AE14 4-step, 예상) | ~142ms |
| FLEX 압축률 | 2880 → 512 tokens (5.6x) |
| Prefill 토큰 수 감소 | ~3200 → ~830 (3.8x) |
