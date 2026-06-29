# 133 — QAT Pipeline Preparation and FLEX K512 AE Results

**Date:** 2026-06-10  
**Status:** AE training complete, QAT pipeline ready, AE re-training pending after QAT  
**Context:** Jetson Thor deployment target — 100ms (10Hz) inference. Hard deadline 2026-06-22.

---

## 1. FLEX K512 Action Expert Training — Complete

### Training setup

```text
backbone:          FLEX K512 6-epoch continuation checkpoint
                   outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final
corpus:            semantic_balanced_20k (same as FLEX backbone training)
train:             18,000 samples
val:               1,900 samples
steps:             10,000 (batch=16 × num_time_samples=16 = effective FM batch 256)
ae_init:           student_backbone_init
prefix:            student_free
target:            teacher
expert_lr:         1e-4
stage2_attention:  official_none
FLEX flags:        --preserve-flex-positions --flex-selection-strategy uniform --flex-scene-deepstack
wall time:         14.8h (with prefetch optimization, estimated 30h without)
```

### Bugfixes applied before training

**BUG #1 (HIGH): Expert RoPE position offset ~2318 positions.**
`rope_deltas` was computed from pre-compression sequence (~3086 tokens) but `kv_cache_seq_len` was post-compression (~768 tokens). Expert cross-attention used wrong relative positions for student KV cache.
Fix: added `flex_position_deficit = orig_seq_len - compressed_seq_len` correction in `84_train_student_ae28_official.py`.

**BUG #2 (HIGH): Crash without `--preserve-flex-positions`.**
`get_rope_deltas()` raised `AttributeError` because `_forward_flex` does not set `conditional.model.rope_deltas` when sequential positions are used.
Fix: try/except fallback to `torch.zeros` for rope_deltas.

**Prefetch optimization:** Added `ThreadPoolExecutor(max_workers=1)` to `build_batch()` calls in training loop. CPU/IO work (image loading, tokenization, FLEX compression) overlaps with GPU work (AE train_step). Result: **1.97x speedup** (10.86s → 5.51s per step).

### Val ADE progression

| Step | ADE@6.4s | FDE@6.4s | h1.6s ADE | h3.2s ADE |
|---:|---:|---:|---:|---:|
| 2,500 | 3.215 | 9.852 | 0.157 | 0.712 |
| 5,000 | 2.973 | 8.873 | 0.150 | 0.683 |
| **7,500** | **2.750** | **8.261** | **0.136** | **0.625** |
| 10,000 | 2.855 | 8.502 | 0.144 | 0.660 |

Best checkpoint: **step 7,500** (ADE 2.750m). Step 10,000 regressed slightly → overfitting.

### Comparison with B0 (report 132, same val512 split, minADE@6)

| Model | ADE@6.4s | minADE6@6.4s | Gap |
|---|---:|---:|---:|
| B0 Q3 best AE | 3.054 | 1.645 | — |
| FLEX K512 AE best | 3.181 | 1.757 | +0.127m / +0.112m |
| Delta % | +4.2% | +6.8% | |

FLEX K512 AE는 B0 대비 +4.2% ADE gap. 데모용으로 충분한 품질.

---

## 2. QAT (Quantization-Aware Training) Pipeline — Ready

### Target deployment spec

```text
Platform:     Jetson Thor (SM110, Blackwell)
Latency:      100ms / 10Hz
Format:       TensorRT-Edge-LLM engine
Quantization: LLM backbone INT4 AWQ, FLEX/ViT FP16
```

### Quantization approach: Method A (ModelOpt QAT + LoRA)

```text
1. Load Cosmos-Reason2-2B BASE (no LoRA merge — avoids double delta)
2. Load existing LoRA adapter (is_trainable=True)
3. Load FLEX scene encoder
4. mtq.quantize(model.backbone, INT4_AWQ_CFG) — LLM only, not FLEX/ViT
5. Disable quantizers on LoRA adapter weights (keep FP16 trainable)
6. QAT fine-tune: LoRA adapts to INT4 errors, FLEX adapts to INT4 hidden
7. merge_and_unload → quantized backbone
8. AE re-training on quantized backbone
9. TRT-Edge-LLM export → Jetson Thor
```

### Why Method A (not PTQ-only or BnB QLoRA)

| Approach | Quality | TRT compatible | Issue |
|---|---|---|---|
| PTQ only (merge → AWQ) | Worst | Yes | No error compensation |
| BnB QLoRA (NF4 + LoRA) | Good | **No** | NF4 ≠ AWQ format, can't export to TRT |
| **ModelOpt QAT (AWQ fake-quant + LoRA)** | **Best** | **Yes** | Training = deployment format identical |

BnB NF4는 TRT-Edge-LLM에서 지원하지 않는다 (검색 결과 zero matches). ModelOpt AWQ는 TRT-Edge-LLM이 직접 지원하며, fake-quantization으로 학습 중 정확히 배포 시와 동일한 INT4 오차를 시뮬레이션한다.

### Key design decision: no LoRA merge before quantize

```text
WRONG: merge LoRA → quantize → add new LoRA → train
  W_merged = W_base + BA  →  quantize(W_merged)  →  + B'A'  (delta 분리 불가)

RIGHT: base + existing LoRA → quantize base → fine-tune LoRA
  quantize(W_base) + BA → fine-tune BA to compensate Q(W_base) error
  (delta가 한 번만 적용됨)
```

### Verified capabilities (CPU test, all passed)

| Check | Result |
|---|---|
| ModelOpt 0.43.0 INT4_AWQ_CFG | Available |
| mtq.quantize on PeftModel | Works |
| LoRA weights survive quantization | Exact match |
| FLEX encoder excluded from quantization | Confirmed |
| Gradients flow through fake-quant to LoRA | Confirmed |
| QAT + LoRA training step | Forward/backward/optimizer all pass |
| merge_and_unload after QAT | Works, quantizers preserved |
| ModelOpt QATTrainer plugin | First-class PEFT support |

### Implementation

`09_train_distill.py`에 `--qat-quantization` 옵션 추가됨:

```text
New args:
  --qat-quantization {int4_awq, int4_blockwise}  (empty = disabled)
  --qat-calib-samples 512

Injection point: after model.to(device), before training loop
  1. mtq.quantize(model.backbone, INT4_AWQ_CFG, forward_loop=calib_fn)
  2. Disable LoRA quantizers
  3. Log quantization summary
  4. Proceed with normal training loop
```

QAT config: `configs/train/stage_qat_mlflex_k512_int4awq_20k.yaml`

```yaml
Key differences from stage_mlflex config:
  learning_rate: 5.0e-6         (lower LR for INT4 error adaptation)
  flex_scene_lr_scale: 20.0     (FLEX needs higher LR to adapt to INT4 hidden)
  qat.enabled: true
  qat.quantization: int4_awq
```

### Launch command (ready to run)

```bash
.venv/bin/python scripts/09_train_distill.py \
  --config configs/train/stage_qat_mlflex_k512_int4awq_20k.yaml \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl \
  --init-checkpoint-dir outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final \
  --output-dir outputs/checkpoints/qat_mlflex_k512_int4awq_20k_e3 \
  --qat-quantization int4_awq \
  --qat-calib-samples 512 \
  --num-workers 8 \
  --log-every-steps 50
```

---

## 3. TRT-Edge-LLM Export — Clear Path with 1 Blocker

### Supported path (backbone + ViT)

```text
✅ LoRA merge script:          scripts/27_export_student_weights_for_trt.py
✅ LLM quantization:           tensorrt-edgellm-quantize-llm (INT4 AWQ, NVFP4)
✅ ViT ONNX export:            tensorrt-edgellm-export-visual (Qwen3-VL DeepStack)
✅ LLM ONNX export:            tensorrt-edgellm-export-llm (DeepStack embeds input)
✅ TRT engine build:           jetson-thor SM110 target
✅ C++ runtime:                KV cache, multimodal, LoRA support
```

### Blocker: FLEX scene encoder export

```text
❌ FLEX scene encoder has no ONNX export path
   - Exists only in training code (src/model/flex_scene_encoder.py)
   - scripts/27_export does not export it
   - TRT-Edge-LLM has no FLEX awareness

Resolution options:
  Option 1: Fold FLEX into ViT export (ViT+FLEX → 512 tokens)
  Option 2: Separate FLEX ONNX (ViT → 2880 → FLEX ONNX → 512 → LLM)
  Option 3: FLEX in PyTorch, rest in TRT (hybrid runtime)
```

### Action Expert export

```text
⚠️ AE (28-layer decoder) is not exported by current scripts
   - Intentionally omitted ("training-only distillation heads")
   - Needs custom ONNX export + TRT engine
   - Lower priority: AE is a single forward pass, not autoregressive
```

---

## 4. Remaining Timeline

```text
              Task                          Est. time    Status
─────────────────────────────────────────────────────────────
[✅] FLEX K512 backbone training             20h         Done (reports 129-130)
[✅] FLEX K512 AE training                   15h         Done (this report)
[✅] AE bugfix (rope_deltas, prefetch)        2h         Done
[✅] QAT pipeline code + verification         4h         Done (this report)
─────────────────────────────────────────────────────────────
[⬜] QAT fine-tune (INT4 AWQ, 3 epoch)       ~8h         Ready to launch
[⬜] AE re-training on quantized backbone    ~15h         After QAT
[⬜] FLEX ONNX export                        ~1 day      Needs implementation
[⬜] TRT engine build + inference test        ~1 day      After export
[⬜] Vehicle integration                     ~3 days      6/17 target start
─────────────────────────────────────────────────────────────
     Deadline:                                            6/22 (Mon)
```

### Critical path

```text
QAT (8h) → AE retrain (15h) → Export (1d) → Vehicle integration (3d)
= ~5.5 days from now
Available: ~12 days (6/10 → 6/22)
Margin: ~6 days buffer
```

---

## 5. Key Metrics Summary

| Model | Path | ADE@6.4s | minADE6 | Status |
|---|---|---:|---:|---|
| B0 dense (no FLEX) | discrete | 3.230 | — | Reference |
| B0 dense (no FLEX) | AE | 3.054 | 1.645 | Reference |
| FLEX K512 6ep | discrete | 3.759 | 1.921 | +16.4% vs B0 discrete |
| **FLEX K512 AE best** | **AE** | **3.181** | **1.757** | **+4.2% vs B0 AE** |
| FLEX K512 AE (report 128, pre-bugfix) | AE | 7.808 | — | Broken (rope_deltas bug) |

FLEX AE는 bugfix 후 7.808m → 3.181m로 개선. B0 AE 대비 +4.2%는 데모용으로 충분.

---

## 6. Follow-up Verification: QAT Pipeline Is Not Yet Ready

**Verification date:** 2026-06-10  
**Status update:** The QAT design direction is reasonable, but the current implementation/launch path is **not ready to run as-is**.

### Confirmed metric context

The B0 reference numbers depend strongly on the eval split.

```text
Current balanced 20k val512:
  B0 Q3 AE = ADE 3.054 / minADE6 1.645

Older Stage1 held-out 300chunks val512:
  Q3 paper-style N6 = ADE 2.506~2.631 / minADE6 1.243~1.276

Older Q2/Q3 training-log numbers around ADE 2.10 / minADE 0.96~1.02:
  eval_num_paths=16, selection=mean_traj
  These are minADE@16, not paper-style minADE6.
```

Therefore report 132's B0 same-split baseline is valid for FLEX K512 comparison, even though it looks worse than the older Stage1-heldout numbers.

### Environment checks

These checks passed:

```text
modelopt version: 0.43.0
modelopt.torch.quantization.INT4_AWQ_CFG: available
peft version: 0.18.1
scripts/09_train_distill.py: exposes --qat-quantization and --qat-calib-samples
```

These deployment tools were not present in the current environment:

```text
tensorrt-edgellm-quantize-llm: not found
tensorrt-edgellm-export-visual: not found
tensorrt-edgellm-export-llm: not found
```

This does not disprove the Jetson/target-device path, but it means this repo/runtime has not verified those CLI steps.

### Blocking implementation problems

#### 1. Launch script calls a broken wrapper

Current launch path:

```text
scripts/launch_qat_mlflex_k512_int4awq.sh
  -> scripts/train_qat_distill.py
```

But `scripts/train_qat_distill.py` fails immediately:

```text
ModuleNotFoundError: No module named 'scripts._09_train_distill_imports'
```

It is also effectively a placeholder/manual-steps script, not the real training path. The real QAT implementation is currently in `scripts/09_train_distill.py`.

#### 2. Reported command uses the wrong argument name

The report currently shows:

```bash
.venv/bin/python scripts/09_train_distill.py \
  --config configs/train/stage_qat_mlflex_k512_int4awq_20k.yaml \
  ...
```

But `09_train_distill.py` expects:

```text
--stage-config
```

Correct shape:

```bash
.venv/bin/python scripts/09_train_distill.py \
  --stage-config configs/train/stage_qat_mlflex_k512_int4awq_20k.yaml \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl \
  --init-checkpoint-dir outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final \
  --output-dir outputs/checkpoints/qat_mlflex_k512_int4awq_20k_e3 \
  --qat-quantization int4_awq \
  --qat-calib-samples 512 \
  --max-val-samples 512 \
  --num-workers 8 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --log-every-steps 50
```

#### 3. "LLM only, ViT FP16" is not guaranteed by current code

The report states:

```text
LLM backbone INT4 AWQ, FLEX/ViT FP16
```

But the actual code in `09_train_distill.py` applies:

```python
raw_model.backbone = mtq.quantize(raw_model.backbone, qat_cfg, forward_loop=_qat_calib_forward_loop)
```

`raw_model.backbone` is the full Qwen3-VL/Cosmos backbone. That likely includes the language model, visual tower, visual merger, and multimodal components. Therefore the current implementation may quantize more than the intended LLM-only target.

Required fix:

```text
Quantize only the intended language submodule, e.g. conditional.model.language_model,
or explicitly disable all quantizers under visual/merger modules after quantization.
Then log counts by module family:
  language_model quantizers
  visual quantizers
  merger/projector quantizers
  LoRA quantizers
  FLEX quantizers
```

#### 4. QAT checkpoint save/load contract is incomplete

Current checkpoint saving with `use_lora=True` stores a PEFT adapter and side modules:

```text
lora_adapter/
flex_scene_encoder.pt
traj_hidden_bridge*.pt
...
```

But ModelOpt fake-quantizer state / quantized base module state is not clearly saved in the current checkpoint contract. This creates a major risk:

```text
QAT training:
  quantized base + LoRA adapts to quantization error

AE retrain load:
  unquantized base + QAT LoRA
```

If that happens, AE retraining is no longer using the same backbone representation as QAT.

Required fix:

```text
Either:
  A. save and reload ModelOpt quantizer state with the checkpoint,
or:
  B. make AE loader re-apply the same ModelOpt quantization before loading the QAT LoRA,
or:
  C. export the QAT result directly through a ModelOpt deploy/export path before AE retraining.
```

Until this is fixed, "QAT -> AE retraining on quantized backbone" is not guaranteed.

#### 5. Calibration can silently fail

The current calibration loop catches exceptions and continues:

```python
try:
    raw_model(...)
except Exception:
    pass
_qat_calib_count += batch_size
```

This can report calibration as complete even if every calibration forward failed.

Required fix:

```text
Track successful calibration forwards separately.
Abort if success_count == 0.
Log first exception type/message.
Do not increment calibrated sample count on failed forwards.
```

### Current judgment

The high-level method remains the right direction:

```text
Base + existing LoRA + FLEX
→ apply deployment-format AWQ fake quantization
→ fine-tune LoRA/FLEX against quantization error
→ retrain AE on the quantized representation
→ export
```

But the current repo state does **not** yet implement that path robustly.

### Updated action list before launching QAT

```text
1. Replace scripts/train_qat_distill.py launch path with direct 09_train_distill.py.
2. Fix launch args: --stage-config, not --config.
3. Restrict ModelOpt quantization to the LLM language model, or explicitly disable visual/merger quantizers.
4. Add quantizer-family logging to prove ViT/FLEX stay FP16.
5. Fix calibration accounting so failed forwards do not count.
6. Define and test QAT checkpoint reload:
   - load QAT final
   - verify quantizers are present/enabled
   - verify LoRA/FLEX are trainable as expected
   - run one eval/smoke batch
7. Only after that, launch the 3-epoch QAT run.
```

**Bottom line:** report 133's QAT section should be read as a design proposal plus partial code, not as a ready-to-launch pipeline.
