# 071 - Recipe Actual Pipeline Validation

Date: 2026-05-31

## Goal

Apply the H3 recipe to the real training path and re-check whether the previously failing random FM path and Stage 0 overfit are recovered.

Recipe:

- `expert_lr=1e-4`
- `proj_lr=1e-4`
- `num_time_samples=16`
- `lr_warmup_steps=0`

Prior root cause from report 070: the 28-layer expert was under-trained with `expert_lr=1e-5` and a single random `(x0,t)` draw. H3 showed that expert LR and multiple time/noise draws are both needed.

## Code Changes

### `scripts/84_train_student_ae28_official.py`

Default recipe changed:

```diff
- parser.add_argument("--num-time-samples", type=int, default=1)
+ parser.add_argument("--num-time-samples", type=int, default=16)

- parser.add_argument("--expert-lr", type=float, default=1e-5)
- parser.add_argument("--proj-lr", type=float, default=3e-5)
+ parser.add_argument("--expert-lr", type=float, default=1e-4)
+ parser.add_argument("--proj-lr", type=float, default=1e-4)
```

`train_step` already supported repeated FM samples, but one bug appeared during Stage 0: `DynamicCache.batch_repeat_interleave()` mutates the cache in-place, so the original batch cache was accidentally expanded from batch 2 to batch 32 before `train_inb_ade`. Fixed by deep-copying the cache before repeating:

```python
if repeats > 1:
    # batch_repeat_interleave mutates DynamicCache in-place. Keep the original
    # batch cache intact for diagnostics such as train_inb_ade on the same batch.
    prompt_cache = copy.deepcopy(prompt_cache)
    prompt_cache.batch_repeat_interleave(repeats)
    context = repeat_context(context, repeats)
    target_action = target_action.repeat_interleave(repeats, dim=0)
```

Also added train logging:

- `num_time_samples`
- `effective_fm_batch`

Compile check:

```bash
.venv/bin/python -m py_compile scripts/84_train_student_ae28_official.py scripts/96_e3_actual_bundle_no_kv.py
```

Result: PASS.

### `scripts/96_e3_actual_bundle_no_kv.py`

Updated E3 actual-bundle script to match the 84 recipe:

- added `--num-time-samples`, default `16`
- defaulted `--expert-lr=1e-4`, `--proj-lr=1e-4`, `--lr-warmup-steps=0`
- added final random evaluation over `--eval-draws`
- trains random E3 with multiple independent `(x0,t)` draws per step

## W1 - E3 Actual Bundle, No KV, Random FM

Command:

```bash
.venv/bin/python scripts/96_e3_actual_bundle_no_kv.py \
  --variant random --steps 1000 --num-time-samples 16 \
  --eval-draws 512 --eval-batch-size 64 \
  --expert-lr 1e-4 --proj-lr 1e-4 --lr-warmup-steps 0 \
  --grad-clip-norm 5.0 --device cuda:0 \
  --student-dtype bfloat16 --ae-dtype bfloat16 \
  --attn-implementation sdpa --seed 42 \
  --output-dir outputs/action_expert/w1_e3_actual_bundle_recipe_seed42_draw16
```

Output:

- `outputs/action_expert/w1_e3_actual_bundle_recipe_seed42_draw16/v2_e3_random_summary.json`

Result:

| run | step | loss | pred_v_abs | target_v_abs | cosine | alpha | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| train final | 1000 | 0.0206 | 0.8203 | 0.8164 | 0.9902 | 0.9864 | 96.5s |
| final eval, 512 draws | - | 0.0264 | 0.8203 | 0.8242 | 0.9876 | 0.9763 | - |

Verdict: PASS. The actual bundle path no longer has random FM collapse under the H3 recipe.

## W2 - Stage 0, Real Student KV

Command:

```bash
.venv/bin/python scripts/84_train_student_ae28_official.py \
  --num-samples 32 --batch-size 2 --steps 3000 \
  --eval-samples 32 --eval-batch-size 2 --eval-every 250 \
  --train-ade-every 100 --log-every 25 \
  --student-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --prefix-mode student_free --ae-init-mode student_backbone_init \
  --target-source teacher --expert-lr 1e-4 --proj-lr 1e-4 \
  --num-time-samples 16 --lr-warmup-steps 0 \
  --grad-clip-norm 5.0 --no-norm-bias-decay \
  --seed 42 --device cuda:0 \
  --student-dtype bfloat16 --ae-dtype bfloat16 \
  --attn-implementation sdpa \
  --output-dir outputs/action_expert/student_ae28/stage0_overfit_32_s3000_seed42_recipe_draw16_full444k_retry_20260531
```

Output:

- `outputs/action_expert/student_ae28/stage0_overfit_32_s3000_seed42_recipe_draw16_full444k_retry_20260531/summary.json`
- `outputs/action_expert/student_ae28/stage0_overfit_32_s3000_seed42_recipe_draw16_full444k_retry_20260531/train_log.jsonl`

Optimizer/trainability sanity:

| module | trainable params | optimizer included | missing | LR |
|---|---:|---:|---:|---:|
| expert | 1,409,410,048 | included through expert group | 0 | 1e-4 |
| action_in_proj | 1,349,632 | 1,349,632 | 0 | 1e-4 |
| action_out_proj | 4,098 | 4,098 | 0 | 1e-4 |

Step 1 projection delta:

- `action_in_proj`: 939,779 / 1,349,632 elements changed
- `action_out_proj`: 4,098 / 4,098 elements changed

Multi-draw sanity:

- logs show `num_time_samples=16.0`
- with `batch_size=2`, logs show `effective_fm_batch=32.0`

Stage 0 eval trajectory:

| step | ADE mean | ADE p50 | FDE mean | h1.6 ADE | h3.2 ADE |
|---:|---:|---:|---:|---:|---:|
| 0 | 9.1443 | 4.9113 | 23.7994 | 0.8448 | 2.8244 |
| 250 | 4.9929 | 4.2025 | 12.3724 | 0.5353 | 1.6979 |
| 500 | 3.2164 | 1.9156 | 8.3860 | 0.4013 | 1.0544 |
| 750 | 3.2360 | 2.3193 | 9.2197 | 0.2351 | 0.8422 |
| 1000 | 6.5068 | 4.6764 | 18.1280 | 0.5088 | 1.8264 |
| 1250 | 1.5083 | 1.2129 | 4.2395 | 0.1357 | 0.4428 |
| 1500 | 1.6190 | 1.1396 | 4.3259 | 0.1371 | 0.4764 |
| 1750 | 2.1354 | 1.9500 | 5.7962 | 0.1760 | 0.6101 |
| 2000 | 1.6719 | 1.5087 | 4.6710 | 0.1300 | 0.4708 |
| 2250 | 0.8444 | 0.6768 | 2.2962 | 0.0724 | 0.2444 |
| 2500 | 1.0133 | 0.8065 | 2.7616 | 0.0837 | 0.2959 |
| 2750 | 0.7891 | 0.5405 | 2.1268 | 0.0730 | 0.2374 |
| 3000 | 1.1139 | 1.0133 | 3.0655 | 0.0847 | 0.3071 |

Selected train loss:

| step | loss | pred_v_abs | target_v_abs |
|---:|---:|---:|---:|
| 1 | 1.3695 | 0.5234 | 0.8281 |
| 1000 | 0.5395 | 0.4707 | 0.8242 |
| 1500 | 0.1436 | 1.0000 | 0.9844 |
| 2000 | 0.1034 | 0.8828 | 0.9531 |
| 2500 | 0.0731 | 1.0469 | 1.0156 |
| 2750 | 0.0489 | 0.8906 | 0.9258 |
| 3000 | 0.0392 | 0.8125 | 0.8398 |

Train in-batch ADE selected:

| step | train_inb_ADE | train_inb_FDE | h1.6 ADE | h3.2 ADE |
|---:|---:|---:|---:|---:|
| 100 | 3.4785 | 8.5894 | 0.1186 | 0.7655 |
| 400 | 1.4884 | 4.2280 | 0.1256 | 0.3943 |
| 900 | 1.0870 | 3.0111 | 0.0977 | 0.2749 |
| 1700 | 0.8420 | 1.1558 | 0.1815 | 0.4690 |
| 2100 | 0.4319 | 1.0379 | 0.0141 | 0.1220 |
| 2500 | 0.5311 | 1.5484 | 0.0296 | 0.1299 |
| 3000 | 1.2496 | 3.6662 | 0.0779 | 0.2798 |

Verdict: PARTIAL PASS, strict Stage 0 PASS = NO.

The old FM collapse is gone: loss drops to `0.03-0.10` late in training and `pred_v_abs` tracks `target_v_abs`. But the full 6.4s path metric does not consistently reach the requested `<0.5m` threshold. Best eval ADE is `0.789m` at step 2750, final is `1.114m`.

Interpretation: H3 recipe fixes the actual random FM learning path. The remaining Stage 0 miss is now more likely in real conditioning / student KV / Euler sampling / full-horizon stability, not in the basic random FM objective.

## W3 - Oracle KV

Not run.

Reason: W3 was gated on W2 PASS. Since W2 improved strongly but did not meet the strict Stage 0 pass criterion, oracle KV is not the next most diagnostic run yet.

## W4 - Draw Sensitivity and Throughput

### Actual bundle no-KV draw sensitivity

`num_time_samples=8` command:

```bash
.venv/bin/python scripts/96_e3_actual_bundle_no_kv.py \
  --variant random --steps 1000 --num-time-samples 8 \
  --eval-draws 512 --eval-batch-size 64 \
  --expert-lr 1e-4 --proj-lr 1e-4 --lr-warmup-steps 0 \
  --grad-clip-norm 5.0 --device cuda:0 \
  --student-dtype bfloat16 --ae-dtype bfloat16 \
  --attn-implementation sdpa --seed 42 \
  --output-dir outputs/action_expert/w4_e3_actual_bundle_recipe_seed42_draw8
```

| draws | final train cosine | final eval cosine | final eval alpha | elapsed | steps/sec | FM draws/sec |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.9902 | 0.9876 | 0.9763 | 96.5s | 10.36 | 165.8 |
| 8 | 0.8740 | 0.9015 | 0.8203 | 85.0s | 11.77 | 94.1 |

`draws=8` barely passes the no-KV cosine threshold, but it is much weaker than `draws=16`. For the real Stage 0 recipe, keep `16` for now. `8` is a possible later efficiency ablation only after Stage 0 strict pass is recovered.

### Stage 0 throughput

Stage 0 draw16 run:

- 3000 train steps internal elapsed: `8080.6s`
- average: `0.371 steps/sec`, or `2.69 sec/step`
- with batch 2 and 16 draws, effective FM rows per train step = 32
- effective FM rows/sec including batch building, train_inb_ade, and eval overhead: about `11.9`

GPU memory:

- Peak GPU memory was not instrumented in the scripts.
- Post-run `nvidia-smi`: H200 NVL total `143771 MiB`; current used `24713 MiB`, owned by an unrelated Python process (`/home/pm97/workspace/kjhong/...`), not the completed W runs.

## Final One-Line Verdict

recipe 수정으로 파이프라인 복구됨 = 부분 Y / strict Stage 0 PASS N.

More precise: random FM collapse is fixed in the actual bundle path (`W1 PASS`), but the full Stage 0 overfit gate still fails at the 6.4s trajectory metric (`best ADE 0.789m`, target `<0.5m`).
