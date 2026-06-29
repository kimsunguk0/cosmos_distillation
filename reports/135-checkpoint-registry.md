# 135. Checkpoint Registry — 가중치 목록 및 경로

**최종 업데이트**: 2026-06-12

---

## 1. Base Weights (원본 사전학습 모델)

| 이름 | 경로 | 파라미터 | 용도 |
|------|------|---------|------|
| Alpamayo 1.5 10B | `/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B/` | 10B | Teacher VLM + 36L Action Expert |
| Cosmos-Reason 2B | `/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b/` | 2B | Student backbone 베이스 (Qwen3-VL 2B) |

---

## 2. Student Backbone Checkpoints

### 2.1 No-FLEX Backbone (FLEX 미적용)

| 항목 | 내용 |
|------|------|
| **경로** | `outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250` |
| **베이스 모델** | Cosmos-Reason 2B |
| **구조** | Qwen3-VL 2B + LoRA (rank 64) |
| **FLEX** | 미적용 (이미지 2880 tokens 그대로 사용) |
| **QAT** | 미적용 (FP16) |
| **학습 데이터** | 200K samples (full444k corpus) |
| **형식** | `lora_adapter` |
| **Hidden bridge** | student/teacher bridge 512dim, teacher_hidden 4096 |
| **용도** | No-FLEX AE28과 짝으로 사용 |

### 2.2 FLEX K512 Backbone (현재 주력)

| 항목 | 내용 |
|------|------|
| **경로** | `outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final` |
| **베이스 모델** | Cosmos-Reason 2B |
| **구조** | Qwen3-VL 2B + LoRA (rank 64) + ML-FLEX Scene Encoder |
| **FLEX** | ML-FLEX K512 (tokens_per_image=32, 16 images → 512 scene tokens) |
| **FLEX 구조** | multi_level, 4 levels (DeepStack 3 + final), hidden=1024, heads=8, 1 layer/level |
| **FLEX 압축** | 2880 → 512 tokens (5.6x), per_image compression, uniform selection |
| **DeepStack** | 활성화 (levels 5, 11, 17 intermediate features 사용) |
| **QAT** | 미적용 (FP16) |
| **학습 데이터** | 20K samples, 6 epochs (K1024 3ep → K512 3ep continuation) |
| **형식** | `lora_adapter` + `flex_scene_encoder.pt` |
| **LR** | 1e-5 (backbone), FLEX scene LR scale 20x |
| **용도** | FLEX AE28/AE14와 짝으로 사용 |

### 2.3 QAT INT4 FFN-Only Backbone

| 항목 | 내용 |
|------|------|
| **경로** | `outputs/checkpoints/qat_mlflex_k512_int4ffn_20k_e3/final` |
| **베이스 모델** | Cosmos-Reason 2B |
| **구조** | Qwen3-VL 2B + LoRA (rank 64) + ML-FLEX K512 + INT4 fake-quantization |
| **FLEX** | FLEX K512 backbone (2.2)에서 이어서 학습 |
| **QAT** | INT4 FFN-only (gate/up/down_proj → INT4 AWQ, Q/K/V/O proj → FP16 유지) |
| **양자화 범위** | LLM language_model만 (ViT, FLEX encoder 미양자화) |
| **Quantizer 수** | 168개 (FFN 3개 proj × 28 layers × 2 direction) |
| **학습 데이터** | 20K samples, 3 epochs |
| **LR** | 5e-6, FLEX scene LR scale 20x |
| **용도** | 향후 Jetson 배포 시 INT4 추론용. QAT AE 재학습 필요 |

---

## 3. Action Expert Checkpoints

### 3.1 AE28 No-FLEX (이전 버전)

| 항목 | 내용 |
|------|------|
| **경로** | `outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/best.pt` |
| **Expert 구조** | 28 layers, hidden=2048, heads=16, kv_heads=8, intermediate=6144 |
| **초기화** | Student backbone layers에서 복사 (student_backbone_init) |
| **학습 방식** | Standard flow matching (MSE on velocity field) |
| **Timestep sampler** | Beta(1.5, 1.0), t = 0.999 - Beta * 0.999 |
| **Inference steps** | 10 (Euler integration) |
| **Action space** | UnicycleAccelCurvature, 64 waypoints × 2 dims (accel, curvature) |
| **짝 Backbone** | No-FLEX backbone (2.1) |
| **학습 데이터** | 200K samples, 15000 steps, batch=8, num_time_samples=16 |
| **Best ADE** | 2.221m (step 15000, teacher_forced, 512 val, mean_traj@16, temp=0.85) |
| **파일 크기** | 2.7 GB |

### 3.2 AE28 FLEX (현재 주력)

| 항목 | 내용 |
|------|------|
| **경로** | `outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt` |
| **Expert 구조** | 28 layers, hidden=2048, heads=16, kv_heads=8, intermediate=6144 |
| **초기화** | Student backbone layers에서 복사 (student_backbone_init) |
| **학습 방식** | Standard flow matching |
| **짝 Backbone** | FLEX K512 backbone (2.2) |
| **학습 데이터** | 18K samples, 10000 steps, batch=16, num_time_samples=16 |
| **Prefix mode** | teacher_forced |
| **Attention mode** | official_none (attention_mask=None) |
| **Best ADE** | 2.750m (step 7500, teacher_forced, 512 val, mean_traj@16, temp=0.85) |
| **파일 크기** | 2.7 GB |
| **KV cache** | Backbone 28L과 1:1 매핑 |

### 3.3 AE14 from AE28 (학습중)

| 항목 | 내용 |
|------|------|
| **경로** | `outputs/action_expert/ae14_from_ae28_10step/best.pt` |
| **Expert 구조** | 14 layers, hidden=2048, heads=16, kv_heads=8, intermediate=6144 |
| **초기화** | AE28 FLEX (3.2)에서 14개 layer uniform select: [0,2,4,6,8,10,12,15,17,19,21,23,25,27] |
| **학습 방식** | Standard flow matching (consistency distill 아님) |
| **짝 Backbone** | FLEX K512 backbone (2.2) |
| **KV cache 처리** | Backbone 28L cache에서 14개 layer select (ae14_selected indices) |
| **학습 데이터** | 17900 samples, 10000 steps, batch=16, num_time_samples=16 |
| **Best ADE** | 2.566m (step 7500, teacher_forced, 512 val, mean_traj@16, temp=0.85) |
| **파일 크기** | 1.4 GB |
| **Per-step latency** | ~11ms (AE28의 ~22ms 대비 50% 감소) |

### 3.4 AE14 Consistency Distill (중단)

| 항목 | 내용 |
|------|------|
| **경로** | `outputs/action_expert/ae14_consistency_2step/best.pt` |
| **Expert 구조** | 14 layers |
| **학습 방식** | Consistency distillation (reflow + multi-loss) from AE28 10-step dump |
| **Target steps** | 2-step |
| **Best ADE** | 4.453m (너무 높아서 중단) |
| **중단 사유** | 10→2 step 직접 점프 + KV cache 불일치 + layer 동시 축소 → 접근 변경 |
| **파일 크기** | 1.4 GB |

---

## 4. TRT Export (ONNX FP16)

| 모듈 | 경로 | 크기 | 출처 |
|------|------|------|------|
| LLM | `outputs/trt_export/flex_k512_fp16/llm/model.onnx` | 3.9 GB | Backbone 2.2 (LoRA merged) |
| ViT | `outputs/trt_export/flex_k512_fp16/visual/model.onnx` | 794 MB | Backbone 2.2의 visual 모듈 |
| FLEX | `outputs/trt_export/flex_k512_fp16/flex/flex_encoder.onnx` | 138.8 MB | Backbone 2.2의 flex_scene_encoder; 4-level input + DeepStack outputs |
| AE28 | `outputs/trt_export/flex_k512_fp16/ae28/ae28_single_step.onnx` | 2.9 GB | AE28 FLEX 3.2 |

- 배포 가이드: `outputs/trt_export/flex_k512_fp16/DEPLOYMENT_NOTES.md`

---

## 5. Teacher Feature Dumps

| 이름 | 경로 | 크기 | 내용 |
|------|------|------|------|
| AE28 10-step dump | `outputs/ae28_teacher_dumps/flex_k512_fp16/` | 186 GB | 19900 samples, AE28 FLEX의 10-step 추론 결과 (x_all_steps, v_steps, last_hidden_steps, reflow targets) |

---

## 6. 짝 조합 (사용 시 반드시 맞춰야 하는 것)

| 조합 | Backbone | Action Expert | 비고 |
|------|----------|---------------|------|
| **No-FLEX 파이프라인** | 2.1 No-FLEX | 3.1 AE28 No-FLEX | FLEX 없이 2880 tokens |
| **FLEX 파이프라인 (주력)** | 2.2 FLEX K512 | 3.2 AE28 FLEX | 512 tokens, ADE 2.750m |
| **FLEX + AE14 (경량)** | 2.2 FLEX K512 | 3.3 AE14 | 512 tokens, 14L expert, ADE 2.566m |
| **QAT 파이프라인** | 2.3 QAT INT4 | (미학습) | QAT backbone 위에 AE 재학습 필요 |

**주의**: Backbone과 AE는 반드시 짝이 맞아야 합니다.
- AE는 backbone의 KV cache를 cross-attention으로 참조
- No-FLEX backbone의 KV cache (2880 tokens) ≠ FLEX backbone의 KV cache (512 tokens)
- AE28은 28L KV 1:1, AE14는 28L에서 14개 select

---

## 7. 학습 Seed / Hyperparameters 공통

| 항목 | 값 |
|------|-----|
| Seed | 42 |
| FM timestep sampler | Beta(1.5, 1.0) |
| num_time_samples | 16 |
| Eval temperature | 0.85 |
| Eval num_paths | 16 (mean_traj selection) |
| Eval inference steps | 10 (Euler) |
| Action space | UnicycleAccelCurvature, 64wp × 2dims |
| Expert attention | non-causal (official_none) |
