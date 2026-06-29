# 125 — FLEX v2 Redesign: Multi-Level FLEX with DeepStack Integration

**Date:** 2026-06-08  
**Status:** Design proposal + implementation start — **Step 0 ablation complete → Direction B confirmed; Step 1 skeleton started**  
**Author:** Human + Claude (architecture discussion)  
**Context:** 60+ FLEX v1 experiments (F0–F63) all failed. Root cause analysis completed. This document proposes a clean redesign. Step 0 DeepStack ablation results (2026-06-08) reject Direction A: DeepStack OFF causes 47–60% ADE degradation in the action expert path.

---

## 0. Critical Discovery: FLEX Was Never Tested with DeepStack

### Timeline

```
2025.10  Alpamayo-R1 paper submitted (arXiv:2511.00088)
         → backbone: Cosmos-Reason1 = Qwen2.5-VL based
         → Qwen2.5-VL does NOT have DeepStack
         → FLEX experiments were done HERE

2025.12  FLEX standalone paper (arXiv:2512.10947)
         → backbone: DINOv2 + Qwen2-0.5B
         → NO DeepStack

2025.12  Qwen3-VL released
         → DeepStack introduced (ViT layers 5,11,17 → LLM layers 0,1,2 injection)

2026.01+ Alpamayo open-source release (Cosmos-Reason2)
         → backbone switched to Qwen3-VL
         → DeepStack now exists
         → How FLEX+DeepStack was handled? → NO public information
```

**FLEX was designed for a world without DeepStack, at least in the public descriptions we can verify.** Neither the FLEX paper nor the Alpamayo-R1 paper publicly documents a Qwen3-VL DeepStack-aware FLEX implementation. The closest public reference we found for visual token reduction on Qwen3-VL with DeepStack awareness is FocusUI, which does token selection, not learned compression.

### Implication

"Faithfully reproducing the paper's FLEX" would mean running on a backbone without DeepStack (Qwen2.5-VL style). Our student backbone is Qwen3-VL (Cosmos-Reason2-2B) which has DeepStack. We cannot replicate the paper's exact conditions.

### Decision gate: Step 0 — DeepStack ablation ✓ COMPLETE

**Setup:**

```
Test A: Public Alpamayo 10B (Qwen3-VL-8B backbone) — DeepStack ON vs OFF
Test B: Our student B0 2B (Cosmos-Reason2-2B backbone) — DeepStack ON vs OFF
Eval paths: (1) 128 discrete trajectory token decode, (2) action expert FM decode
Metric: GT ADE / FDE (all results are vs ground truth)
Note: B0 2B discrete was initially evaluated with teacher-target summary;
      the saved generated 128 trajectory tokens were re-decoded and
      GT-based ADE/FDE was recomputed for this table.
```

**Results (2026-06-08):**

| Model | Decode path | DS ON ADE / FDE | DS OFF ADE / FDE | Δ ADE / Δ FDE | ADE % |
|---|---|---|---|---|---|
| Public 10B | 128 discrete token | 1.810 / 5.099 | 2.838 / 7.759 | +1.029 / +2.660 | **+56.8%** |
| Public 10B | action expert | 1.843 / 5.325 | 2.709 / 7.735 | +0.866 / +2.410 | **+47.0%** |
| B0 2B | 128 discrete token | 3.486 / 11.534 | 3.734 / 12.081 | +0.249 / +0.547 | +7.1% |
| B0 2B | action expert | 2.713 / 7.712 | 4.226 / 12.315 | +1.513 / +4.603 | **+55.8%** |

**Decision criteria:**

| DeepStack OFF performance drop | Direction |
|---|---|
| < 10% vs DeepStack ON | **Direction A**: DeepStack OFF + paper-style single-level FLEX. DeepStack不重要, no need to integrate. |
| >= 10% vs DeepStack ON | **Direction B**: ML-FLEX with DeepStack integration. DeepStack matters, must be preserved through compression. |

### Step 0 Analysis and Decision

**→ Direction B confirmed. DeepStack OFF is not viable.**

Three findings from the ablation:

**Finding 1: Action expert path amplifies DeepStack dependency.**

B0 2B discrete token path shows only +7.1% ADE degradation with DeepStack OFF — the LLM can still produce reasonable trajectory tokens from its own parameters. But the action expert path shows +55.8% ADE degradation. The action expert relies on backbone hidden states for KV conditioning (cross-attention), and these hidden states lose critical visual structure when DeepStack is removed. DeepStack intermediate features (2B student: ViT layers 5, 11, 17; public 10B: ViT layers 8, 16, 24) inject multi-scale visual information directly into the LLM's early layers, enriching the hidden representations that the action expert later consumes. Without this injection, the hidden states at action expert boundary positions carry impoverished visual context.

**Finding 2: The amplification is consistent across model scales.**

Both the 10B public model (+47.0% AE) and the 2B student (+55.8% AE) show massive action expert degradation. The 2B student is actually MORE sensitive, likely because the smaller backbone has less redundant capacity to compensate for missing DeepStack information.

**Finding 3: FLEX "Direction A" (DeepStack OFF) would specifically cripple the path we need most.**

The action expert is the primary trajectory generation mechanism for deployment. A FLEX design that turns off DeepStack might produce acceptable discrete token trajectories (+7.1% hit on B0), but the action expert — which is the better trajectory generator when working (2.713m vs 3.486m ADE on B0 with DS ON) — would be destroyed (+55.8% hit). This means FLEX v2 MUST preserve DeepStack information through compression.

**Implication for ML-FLEX design:** The multi-level compression approach in Section 3 is not just theoretically cleaner — it is architecturally necessary. The shared slot layout must compress ViT intermediate features at the backbone's configured DeepStack layers (2B: 5/11/17, 10B: 8/16/24) into aligned DeepStack injection tokens. The alternative (single-level FLEX with DeepStack OFF) loses too much action-expert-relevant information.

---

## 1. Why FLEX v1 Failed (Root Cause Summary)

### 60 experiments, 0 successes

| Experiment range | What was tried | Result |
|---|---|---|
| F0–F5 | FLEX encoder only, LoRA last-4 | Teacher-forced OK, free-run ADE 3.66 (B0: 3.10) |
| F8–F9 | Per-image factorized FLEX | Free-run ADE 3.99, hidden parity collapse |
| F28–F36 | Position preservation, trajectory alignment, dummy slots | All failed |
| F42 | DeepStack OFF + hidden alignment (16 samples) | **ADE 0.380** (only success, didn't scale) |
| F45–F47 | F42 recipe scaled to 32/64/256 samples | All collapsed |
| F49–F53 | All-LoRA, residual slots, LR sweeps | All failed |
| F54–F58 | K=1792 passthrough diagnostics | F58 passthrough ADE 0.854 (diagnostic only) |
| F61–F63 | Selector + LoRA (not real FLEX) | ADE 3.335 vs B0 3.181 (close but not FLEX) |

### Three root causes

**1. DeepStack mismatch (structural)**

```
FLEX compresses:  ViT final merged output → 2880 tokens → 512 tokens
DeepStack injects: ViT intermediate merged outputs → 2880 tokens → LLM layers 0,1,2
  - 2B student: ViT layers 5, 11, 17; final after vision depth 24
  - public 10B: ViT layers 8, 16, 24; final after vision depth 27

Problem: LLM input sees 512 FLEX tokens, but DeepStack still tries to inject
         2880-position intermediate features. These two paths present
         conflicting "visual realities" to the LLM.
```

Attempted fixes and results:
- FlexDeepStackProjector (rank-64 projection from FLEX output → DeepStack): ADE 0.962 (FAIL)
- Joint training FLEX + DeepStack: ADE 0.398 → 0.585, unstable (FAIL)
- Repeat FLEX scene tokens at all DeepStack layers: ADE 9.575 (catastrophic)
- DeepStack OFF entirely: ADE 0.380 on 16 samples (only thing that worked, didn't scale)

**2. LLM adaptation insufficient**

Current student backbone: Cosmos 2B with all-layer LoRA rank-64 on all 7 projections (q/k/v/o/gate/up/down) across 28 layers. This was trained for 2880 visual tokens in the current camera-labeled 4-camera × 4-frame setup.

FLEX v1 experiments used only last-4 LoRA rank-4 (~4.7M) for LLM adaptation to FLEX. This is far too little adaptation capacity for a 2880→512 token reduction.

**3. Wrong evaluation metric**

All 60 experiments evaluated with **greedy argmax free-run ADE**. But greedy decoding collapses for B0 too:

| Decoding | Unique tokens | ADE vs teacher |
|---|---|---|
| Greedy (argmax) | 17 / 128 | 2.557 m |
| Sampling (t=1.0, top_p=0.95, best-of-4) | ~75 / 128 | 1.376 m |

Greedy collapse is a decoding strategy issue, not a model quality issue. FLEX should be evaluated with **sampling + minADE**, same as the backbone.

---

## 2. Reference: FocusUI

**Paper:** FocusUI: Efficient UI Grounding via Position-Preserving Visual Token Selection  
**Repo:** https://github.com/showlab/FocusUI  
**Relevance:** Only public implementation of visual token reduction on Qwen3-VL-2B with DeepStack-aware handling.

### What FocusUI does

1. Scores visual token importance via learned MHA scorer
2. Selects top-K tokens per image (retention 100% / 50% / 30%)
3. Replaces dropped token runs with `<|image_drop_end|>` POSPAD markers (position preservation)
4. **Gathers DeepStack features only for kept tokens:**

```python
# FocusUI's key DeepStack handling:
if deepstack_visual_embeds is not None:
    deepstack_visual_embeds = deepstack_visual_embeds[b][img_keep_mask_b, :]
```

5. Rebuilds `visual_pos_masks` to match only kept positions

### FocusUI results on Qwen3-VL-2B (ScreenSpot-Pro)

| Retention | Accuracy | Inference speedup |
|---|---|---|
| 100% | 39.8% | 1.00x |
| 50% | **40.4%** (+0.6pp, improved!) | ~1.2x |
| 30% | 38.5% (-1.3pp) | 1.44x |

### What we learn from FocusUI

1. **DeepStack features MUST be aligned with kept/compressed tokens** — you can't inject 2880-position DeepStack into a 512-position sequence
2. **Position preservation matters** — POSPAD markers maintain MRoPE spatial continuity for Qwen3-VL
3. **50% retention can actually IMPROVE performance** — noise reduction effect
4. **The pattern works on Qwen3-VL-2B** — same base model as our student

### What FocusUI does NOT solve for us

- FocusUI is **token selection** (keeps original features), not learned compression
- FocusUI handles **single images**, not multi-camera temporal scenes
- FocusUI doesn't generate new compressed representations — it prunes
- We need FLEX-style **learned scene queries** that compress multi-camera information

---

## 3. New Design: Multi-Level FLEX (ML-FLEX)

### Core idea

**FLEX should compress ALL visual information paths simultaneously — not just the final ViT output.**

Current FLEX only compresses the final ViT merged output. DeepStack features from the configured intermediate ViT layers are handled separately (or ignored), causing mismatch.

ML-FLEX uses the **same set of learned queries** to compress features at every ViT level:

```
2B student configuration:

ViT Layer 5       → merger → 2880 tokens [D] → ML-FLEX level encoder[0] → 512 tokens → DeepStack[0]
ViT Layer 11      → merger → 2880 tokens [D] → ML-FLEX level encoder[1] → 512 tokens → DeepStack[1]
ViT Layer 17      → merger → 2880 tokens [D] → ML-FLEX level encoder[2] → 512 tokens → DeepStack[2]
ViT final output  → merger → 2880 tokens [D] → ML-FLEX level encoder[3] → 512 tokens → LLM Input

public 10B uses the same pattern with ViT layers 8, 16, 24 and final after vision depth 27.
```

### Why this works

1. **No mismatch**: LLM input (512 tokens) and DeepStack (512 tokens each) are all produced by the same FLEX queries — they refer to the same "conceptual positions"
2. **Multi-scale**: Each level encoder learns to compress features appropriate for that ViT depth (earlier DeepStack layers retain finer detail, the final level carries the usual merged visual semantics)
3. **Shared queries**: The same K queries attend to all levels, so the "what is important" decision is consistent across depths
4. **Clean information flow**: LLM sees a single coherent visual representation at both input and DeepStack injection points

### Slot layout

The "shared queries" should not be an unordered global bag.  They must be a
stable camera/time/local-slot grid:

```
4 cameras × 4 frames × 32 local slots = 512 scene slots
slot_id = camera_id, frame_id, local_slot_id
```

The final LLM-input stream and the three DeepStack streams share this exact slot
order.  Slot `(front, frame 2, local 17)` in the final stream and in DeepStack
streams 0/1/2 should refer to the same conceptual visual region or evidence
budget.  This is what keeps the residual DeepStack additions coherent after
compression.

### Architecture

```python
class MultiLevelFlexEncoder(nn.Module):
    """
    Shared scene queries compress visual tokens at all ViT levels.
    Produces aligned LLM input tokens + DeepStack tokens.
    """
    def __init__(self, config: MLFlexConfig):
        # Shared learnable queries: [total_scene_tokens, hidden_size]
        # total_scene_tokens = tokens_per_image * num_images = 32 * 16 = 512
        self.scene_queries = nn.Parameter(
            torch.randn(config.total_scene_tokens, config.hidden_size) * 0.02
        )

        # Camera/time/local-slot embeddings (added to both queries and visual tokens)
        self.camera_embed = nn.Embedding(config.max_cameras, config.hidden_size)
        self.local_slot_embed = nn.Embedding(config.tokens_per_image, config.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Per-level cross-attention encoders
        # Level 0,1,2: DeepStack levels (2B ViT layers 5, 11, 17)
        # Level 3: Final ViT output (LLM input)
        self.level_encoders = nn.ModuleList([
            FlexLevelEncoder(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_layers=config.num_encoder_layers,  # 1–2 layers each
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.num_deepstack_levels + 1)
        ])

        # Input/output projections per level
        # (ViT merger output dim may differ from FLEX internal dim)
        self.input_projs = nn.ModuleList([
            nn.Linear(config.input_hidden_size, config.hidden_size)
            for _ in range(config.num_deepstack_levels + 1)
        ])
        self.output_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.input_hidden_size)
            for _ in range(config.num_deepstack_levels + 1)
        ])

    def forward(
        self,
        final_visual_tokens,       # [batch, 2880, D] — final ViT merged output
        deepstack_visual_tokens,   # list[3 x [batch, 2880, D]] — configured DeepStack layers
        camera_ids=None,           # [batch, 2880], exact dense visual-token metadata
        relative_times=None,       # [batch, 2880, 1], exact dense visual-token metadata
        image_token_lengths=None,  # [batch, 16], normally all 180
    ):
        B = final_visual_tokens.shape[0]
        queries = self.scene_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 512, H]

        # Add camera/time embeddings to dense visual tokens.
        # IMPORTANT: queries need their own 512-slot metadata, not the 2880-token metadata.
        if camera_ids is not None:
            cam_emb = self.camera_embed(camera_ids)   # [B, 2880, H]
            time_emb = self.time_mlp(relative_times)  # [B, 2880, H]
            query_camera_ids, query_times, query_local_slots = build_query_slot_metadata(
                camera_ids=camera_ids,
                relative_times=relative_times,
                image_token_lengths=image_token_lengths,
                tokens_per_image=config.tokens_per_image,
            )
            queries = queries + self.camera_embed(query_camera_ids)
            queries = queries + self.time_mlp(query_times)
            queries = queries + self.local_slot_embed(query_local_slots)

        # Compress DeepStack levels
        compressed_deepstack = []
        all_tokens = deepstack_visual_tokens + [final_visual_tokens]

        for level_idx, (tokens, encoder, in_proj, out_proj) in enumerate(
            zip(all_tokens, self.level_encoders, self.input_projs, self.output_projs)
        ):
            projected = in_proj(tokens)  # [B, 2880, H]
            if camera_ids is not None:
                projected = projected + cam_emb + time_emb

            # Shared queries attend to level-specific visual tokens
            compressed = encoder(queries, projected)  # [B, 512, H]
            compressed = out_proj(compressed)          # [B, 512, D]

            if level_idx < len(deepstack_visual_tokens):
                compressed_deepstack.append(compressed)
            else:
                flex_input_tokens = compressed

        return flex_input_tokens, compressed_deepstack
        # flex_input_tokens: [B, 512, D] — replaces image placeholders in LLM input
        # compressed_deepstack: list[3 x [B, 512, D]] — injected at LLM layers 0,1,2
```

```python
class FlexLevelEncoder(nn.Module):
    """
    Single-level cross-attention: queries attend to visual tokens.
    Lightweight — 1 or 2 Transformer layers per level.
    """
    def __init__(self, hidden_size, num_heads, num_layers=1, mlp_ratio=4.0):
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=int(hidden_size * mlp_ratio),
                activation='gelu',
                norm_first=True,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(self, queries, visual_tokens):
        """
        queries: [B, K, H] — shared scene queries
        visual_tokens: [B, N, H] — level-specific visual tokens
        Returns: [B, K, H] — compressed tokens
        """
        x = queries
        for layer in self.layers:
            x = layer(x, visual_tokens)  # cross-attention: Q=queries, KV=visual_tokens
        return x
```

### Key design decisions

**Q: Why shared queries across levels?**

Because the "what is important" decision should be the same at all ViT depths. If query 17 represents "the vehicle in front" at the final level, it should also capture that vehicle's low-level texture (layer 8) and mid-level structure (layer 16) for the corresponding DeepStack injections.

**Q: Why per-level encoders (not shared)?**

Because ViT layer 8 features are fundamentally different from layer 24 features — they have different information content. The cross-attention weights need to be different per level even though the queries are shared.

**Q: Why cross-attention instead of concat+self-attention (current FLEX v1)?**

Current FLEX v1 concatenates queries and visual tokens, then runs self-attention:
```
v1: self_attn([queries || visual_tokens]) → extract queries
```

ML-FLEX uses explicit cross-attention:
```
v2: cross_attn(Q=queries, KV=visual_tokens) → compressed tokens
```

Cross-attention is:
- More parameter-efficient (no need for visual tokens to attend to each other)
- Cleaner separation of "what to compress" (queries) and "what to compress from" (visual tokens)
- Naturally supports different visual token sets per level

### Parameter estimate

| Component | Parameters |
|---|---|
| Scene queries (512 × 1024) | 0.5M |
| Camera/time embeddings | 1.1M |
| 4 × input projection (2048 → 1024) | 8.4M |
| 4 × output projection (1024 → 2048) | 8.4M |
| 4 × FlexLevelEncoder (1 layer each) | 25.2M |
| **Total** | **~43.6M** |

Slightly larger than FLEX v1 (31M) due to 4 level encoders instead of 1. Can be reduced by sharing input/output projections across DeepStack levels or using 1 shared encoder with level-specific adapters.

**Lightweight variant (~33M):**
- Share input/output projections across all 4 levels
- Use 1 shared encoder + 4 lightweight per-level adapters (rank-64 down/up)

---

## 4. Prompt Structure: Fully Preserved

FLEX compression only affects image_token_id (`151655`) placeholder blocks. All text tokens pass through untouched.

### Before FLEX (current, ~3300 tokens)

```
[System: "You are a driving assistant..."]
[User:]
  "Front left camera: "
    "frame 0 " [img_token × 180] "frame 1 " [img_token × 180]
    "frame 2 " [img_token × 180] "frame 3 " [img_token × 180]
  "Front camera: "
    "frame 0 " [img_token × 180] ... "frame 3 " [img_token × 180]
  "Front right camera: "
    "frame 0 " [img_token × 180] ... "frame 3 " [img_token × 180]
  "Front telephoto camera: "
    "frame 0 " [img_token × 180] ... "frame 3 " [img_token × 180]
  <|traj_history_start|> [48 tokens] <|traj_history_end|>
  "output the chain-of-thought reasoning..."
[Assistant:]
  <|cot_start|> [CoT tokens] <|cot_end|>
  <|traj_start|> [128 trajectory tokens] <|traj_end|>
```

Current processor check on `flex_heldout256_stage2val_seed42.jsonl`:

```
image_grid_thw per image = [1, 20, 36]
visual tokens per image = 20 × 36 / spatial_merge_size^2 = 180
image_token_id count in input_ids = 2880
```

**Total image tokens: 180 × 16 = 2,880**

### After ML-FLEX (~1000 tokens before assistant generation)

```
[System: "You are a driving assistant..."]    ← UNCHANGED
[User:]
  "Front left camera: "                       ← UNCHANGED
    "frame 0 " [FLEX × 32] "frame 1 " [FLEX × 32]   ← 180 → 32 per frame
    "frame 2 " [FLEX × 32] "frame 3 " [FLEX × 32]
  "Front camera: "                             ← UNCHANGED
    "frame 0 " [FLEX × 32] ... "frame 3 " [FLEX × 32]
  "Front right camera: "                       ← UNCHANGED
    ...
  "Front telephoto camera: "                   ← UNCHANGED
    ...
  <|traj_history_start|> [48 tokens] <|traj_history_end|>  ← UNCHANGED
  "output the chain-of-thought reasoning..."                ← UNCHANGED
[Assistant:]
  <|cot_start|> [CoT tokens] <|cot_end|>                   ← UNCHANGED
  <|traj_start|> [128 trajectory tokens] <|traj_end|>      ← UNCHANGED
```

**Total FLEX tokens: 32 × 16 = 512 (5.625x visual-token compression)**

**Camera names, frame labels, ordering: ALL preserved. LLM still knows which tokens belong to which camera and frame.**

---

## 5. Integration with Existing Codebase

### ViT intermediate features: already available

From `student_wrapper.py`, `_qwen_visual_features()` already extracts DeepStack features:

```python
image_embeds, deepstack_image_embeds = conditional.get_image_features(
    pixel_values, image_grid_thw
)
# image_embeds: final ViT output — [N, D] per image
# deepstack_image_embeds: list[3 × [total_N, D]]
#   2B student: ViT layers 5, 11, 17
#   public 10B: ViT layers 8, 16, 24
```

ML-FLEX just needs to receive BOTH outputs:
```python
flex_input, flex_deepstack = self.ml_flex_encoder(
    final_visual_tokens=batched_image_embeds,           # [B, 2880, D]
    deepstack_visual_tokens=deepstack_image_embeds,      # list[3 × [B, 2880, D]]
    camera_ids=camera_ids,
    relative_times=relative_times,
)
```

### Batch compression: reuse existing

`compress_batch_for_flex()` already handles:
- Replacing image placeholder blocks (180 → 32 per image)
- Preserving text tokens between image blocks
- Remapping position IDs (MRoPE preservation)
- Remapping teacher supervision positions

Only change needed: pass `compressed_deepstack` to the LLM forward call via `deepstack_visual_embeds` argument.

### DeepStack injection: reuse existing

Qwen3-VL's `_deepstack_process` already handles the injection:
```python
hidden_states[visual_pos_masks, :] += deepstack_visual_embeds[layer_idx]
```

With ML-FLEX, `visual_pos_masks` marks the 512 FLEX token positions and `deepstack_visual_embeds[layer_idx]` is the ML-FLEX compressed output for that level. No change to the injection mechanism.

### Checkpoint: extend existing

Current checkpoint saves `flex_scene_encoder.pt`. ML-FLEX checkpoint saves:
```
ml_flex_encoder.pt           # Full multi-level encoder state
checkpoint_manifest.json     # Updated config with num_deepstack_levels
```

---

## 6. Training Protocol

### What's different from FLEX v1

| Aspect | FLEX v1 (failed) | ML-FLEX v2 |
|---|---|---|
| DeepStack | Separate projector or OFF | Integrated via multi-level compression |
| LLM adaptation | Last-4 LoRA rank-4 (4.7M) | All-layer LoRA rank-64 (existing backbone config) |
| Evaluation metric | Greedy free-run ADE | Sampling (t=1.0, top_p=0.95) + minADE@N |
| Scheduled sampling | Tried and abandoned (off-manifold collapse) | **Not used** |
| FLEX architecture | Single-level self-attention concat | Multi-level cross-attention |

### Adaptation strategy: do not treat FLEX-only as the final model

The current backbone was trained on dense 2880-token visual inputs.  After
ML-FLEX, the LLM and action expert see a different hidden-state distribution.
Training only ML-FLEX while freezing the whole language side is useful as a
bootstrap gate, but it should not be treated as the final training recipe.

Use a three-stage ladder:

| Stage | Trainable | Frozen | Purpose |
|---|---|---|---|
| A. Feature/interface prealignment | ML-FLEX only | ViT, base LLM, LoRA, action expert | Make compressed final + DeepStack streams look like selected dense visual features |
| B. Token-path adaptation | ML-FLEX + all-layer LoRA rank-64 + token rows | ViT, base LLM weights, action expert | Let the LLM adapt to the 2880→512 hidden distribution shift |
| C. Action-expert compatibility | ML-FLEX + LoRA, then optionally AE | ViT, base LLM weights | Make action-expert conditioning work on FLEX hidden states |

If Stage A fails, the compressor itself is not producing usable visual
representations.  If Stage A passes but Stage B fails, the problem is LLM
adaptation.  If Stage B passes but Stage C fails, the action expert needs
FLEX-conditioned hidden-state adaptation or fine-tuning.

### Constraints

- **No scheduled sampling** — collapsed unique tokens from 13.93 → 7.91 and increased ADE in prior experiments (see report 059, Section 6)
- **No base-weight full fine-tune at first** — keep base LLM weights frozen, but train all-layer LoRA rank-64 after the FLEX-only bootstrap.  If this still fails at scale, revisit larger LoRA rank or partial unfreeze as a separate ablation.
- **ViT always frozen** — ViT weights provide stable teacher features at all levels

### Loss ladder

**Stage A — ML-FLEX feature/interface prealignment**

Use the dense DeepStack-ON backbone as the teacher.  For each camera/frame block,
choose the same 32 local slot anchors that will survive compression (uniform is
the first baseline; saliency/top-K can be a later ablation).  The labels are not
human annotations; they are dense teacher features at those anchor positions.

```yaml
loss_stage_a:
  final_feature_mse_or_smooth_l1: 1.0      # FLEX final stream vs dense final image_embeds at anchor slots
  deepstack_feature_mse_or_smooth_l1: 1.0  # each of the 3 FLEX DS streams vs dense DS features at anchor slots
  feature_cosine: 0.1
  feature_norm_ratio: 0.05
```

This stage answers one question only: can ML-FLEX produce 512 visual slots that
live near the dense backbone's visual-feature manifold?

**Stage B — token-path adaptation**

Now train ML-FLEX together with all-layer LoRA rank-64.  The dense B0 model is
the teacher for logits/hidden states, while the existing trajectory target
tokens remain the hard labels.

```yaml
loss_stage_b:
  traj_token_ce: 1.0             # hard target / teacher trajectory tokens
  dense_teacher_logit_kd: 0.3    # KL on trajectory token positions
  text_and_traj_hidden_mse: 0.1  # align non-visual + trajectory positions to dense teacher
  visual_anchor_hidden_mse: 0.1  # align only the 512 anchor visual positions, not all 2880 dense positions
```

Do not judge this stage by loss alone.  Select checkpoints on decode geometry:
sampling minADE/FDE, unique trajectory tokens, and camera-shuffle sensitivity.

**Stage C — action-expert compatibility**

The action expert is more DeepStack-sensitive than the discrete token path.
First freeze the current action expert and train ML-FLEX+LoRA to make its
conditioning hidden states compatible with the existing expert.  Then, only if
needed, fine-tune the action expert on FLEX-conditioned hidden states.

```yaml
loss_stage_c:
  ae_frozen_gt_or_teacher_traj_loss: 1.0
  ae_context_hidden_mse_to_dense_b0: 0.1
  optional_ae_finetune_traj_loss: 1.0  # only after frozen-AE compatibility gate
```

This stage must report both discrete-token ADE and action-expert ADE.  Passing
only the discrete-token path is not enough.

### Training recipe

```yaml
# Frozen
vit_encoder: frozen
base_llm_weights: frozen  # LoRA adapters are trainable after Stage A

# Trainable
ml_flex_encoder: trainable  # ~33-43M params (all levels)
lora_adapters: trainable_after_stage_a  # existing rank-64, all-layer, all-projection
trainable_token_rows: 4016  # existing trajectory + special tokens
task_heads: trainable       # meta_action_head, traj_aux_head, etc.

# Learning rates
ml_flex_lr: 1e-4            # matches official Alpamayo FLEX recipe
lora_lr: 2e-5               # matches current backbone training
task_head_lr: 2e-5

# Schedule
warmup: 100 steps
scheduler: cosine
total_steps: 10000+  # start with smaller dataset, scale up

# Loss (same as current backbone training)
traj_ce: 1.0
teacher_logit_kd: weight from current config
traj_hidden_alignment: weight from current config
# NO scheduled sampling
```

### Evaluation protocol

```yaml
decoding:
  do_sample: true
  temperature: 1.0
  top_p: 0.95

metrics:
  primary: minADE@6 (6 sampled trajectories, report minimum ADE)
  secondary: ADE mean, FDE, unique token count
  sensitivity: camera_shuffle gap (normal vs shuffled camera order)

baseline_comparison:
  B0 greedy:   ADE 2.557m (reference only, not target)
  B0 sampling: ADE 1.376m best-of-4 (actual target)
  
success_criteria:
  minADE@6: within +20% of B0 sampling baseline
  camera_shuffle_gap: > 50% of B0 gap
  unique_tokens: > 50/128 under sampling
```

### Progressive scaling (gate-and-scale)

```
Phase 1: 16-sample overfit (sanity check)
  Dataset: 16 category-balanced samples
  Steps: 500
  Gate: sampling minADE@4 on train16 < 2.0m
  Purpose: verify ML-FLEX architecture can produce valid trajectories

Phase 2: 512-sample training
  Dataset: 512 samples (vis68 × ~8 augmented)
  Steps: 3000
  Gate: sampling minADE@6 on vis68 within 2x of B0
  Purpose: verify generalization beyond memorization

Phase 3: Full-scale training
  Dataset: 20k+ samples
  Steps: 10000+
  Gate: sampling minADE@6 within +20% of B0
  Purpose: final deployment-quality training

Do NOT skip phases. If a phase fails its gate, diagnose before proceeding.
```

---

## 7. Risk Assessment

### Low risk
- **Prompt structure breaking**: Text tokens are never touched, camera names and frame labels fully preserved
- **ViT feature extraction**: Already implemented and validated
- **Batch compression**: Existing `compress_batch_for_flex()` handles this correctly

### Medium risk
- **Multi-level encoder training stability**: 4 cross-attention encoders training jointly is more complex than single encoder. Mitigation: can start with only the final-level encoder (equivalent to FLEX v1 but with cross-attention), then add DeepStack levels incrementally.
- **Parameter count increase**: ~43M vs 31M. Mitigation: lightweight variant with shared projections (~33M).

### High risk
- **Camera sensitivity preservation**: Never achieved in FLEX v1. ML-FLEX has structural advantages (DeepStack alignment, better LLM adaptation) but this remains the hardest metric. Mitigation: camera-aware query design — dedicate specific query subsets to each camera.
- **Scaling from Phase 2 → Phase 3**: FLEX v1's 16→32 sample transition failed. ML-FLEX's cross-attention and DeepStack alignment should help, but this is unproven. Mitigation: strict phase gates, diagnose before scaling.

---

## 8. Comparison with FocusUI Approach

| Aspect | FocusUI | ML-FLEX (proposed) |
|---|---|---|
| Method | Token selection (keep originals) | Learned compression (generate new) |
| DeepStack handling | Gather original features at kept positions | Compress features at all levels via shared queries |
| Position preservation | POSPAD markers for dropped runs | compress_batch_for_flex with MRoPE preservation |
| Information recovery | None (dropped = lost) | Cross-attention from full token set |
| Multi-camera support | Single image | 4 cameras × 4 frames natively |
| Compression ratio | 30–50% retention | ~17.8% retention (32/180 per image) |
| LLM adaptation | None reported | All-layer LoRA rank-64 |
| Task | UI grounding | Driving trajectory prediction |

**What we adopt from FocusUI:**
1. DeepStack features MUST be aligned with compressed token positions (not separate path)
2. Position continuity matters for Qwen3-VL MRoPE
3. Qwen3-VL-2B can handle significant visual token reduction with proper handling

**Where we go beyond FocusUI:**
1. Learned compression instead of selection (needed for 5.625x visual-token compression)
2. Multi-level compression across all ViT depths (not just final output)
3. Joint training with LLM LoRA adaptation

---

## 9. Direction A: DeepStack OFF + Paper-Style FLEX — REJECTED

~~If Step 0 ablation shows DeepStack OFF has < 10% performance drop, this simpler path is preferred.~~

**Rejected by Step 0 ablation (2026-06-08).** DeepStack OFF causes +47–56% ADE degradation in the action expert path across both 10B and 2B models. The action expert is the primary trajectory generation path and produces better trajectories than discrete tokens when DeepStack is ON (2.713m vs 3.486m on B0). Direction A would cripple this critical path. Kept below for reference only.

### Architecture

Use the **existing FlexSceneEncoder** as-is (single-level, concat + self-attention). No multi-level needed.

```
ViT (frozen) → 2880 tokens → FlexSceneEncoder (31M) → 512 tokens → LLM input
                                                        DeepStack: OFF
                                                        LLM: all-layer LoRA rank-64
```

### Why this could work now (vs failed FLEX v1)

| What failed in v1 | What changes |
|---|---|
| Last-4 LoRA rank-4 (4.7M) | All-layer LoRA rank-64 (~existing backbone config) |
| Greedy eval | Sampling t=1.0 + minADE@N |
| DeepStack ON causing mismatch | DeepStack OFF (clean single path) |

The **only FLEX v1 experiment that partially worked** was F42: DeepStack OFF + hidden alignment → ADE 0.380 on 16 samples. But F42 used only last-4 LoRA. With all-layer rank-64, the LLM has much more capacity to adapt to compressed input.

### Training recipe (Direction A)

Same as Section 6 but simpler:
```yaml
flex_scene_encoder: trainable (existing 31M architecture)
deepstack: OFF
lora: all-layer rank-64 (same as backbone training)
eval: sampling t=1.0, top_p=0.95, minADE@6
```

### Advantage

- Paper-style FLEX architecture validated in a non-DeepStack setting
- Existing code, minimal changes
- Fast to test

---

## 10. Direction B: ML-FLEX with DeepStack Integration — CONFIRMED

**Confirmed by Step 0 ablation (2026-06-08).** DeepStack OFF causes +47–60% ADE degradation in the action expert path. DeepStack must be preserved through compression.

### Architecture

Multi-level cross-attention as described in Section 3. Shared queries compress all ViT levels simultaneously.

### Why this is necessary

The Step 0 ablation revealed that DeepStack's contribution is not marginal — it is load-bearing for action expert conditioning. The +55.8% ADE degradation on B0 2B action expert (2.713m → 4.226m) means any FLEX design that drops DeepStack would produce an action expert worse than the discrete token path without FLEX. ML-FLEX adds ~12M params and complexity (4 level encoders, ViT intermediate feature extraction at batch level, compressed DeepStack injection), but this is justified by the ablation.

### Implementation priority

```
Step 1: Implement MLFlexConfig + MultiLevelFlexEncoder
        (extend src/model/flex_scene_encoder.py)

Step 2: Modify student_wrapper._flex_inputs_embeds() to use multi-level outputs
        (pass compressed_deepstack to LLM forward)

Step 3: Verify ViT deepstack_image_embeds extraction works at batch level
        (may need reshape from list[per-image] to [batch, 2880, D])

Step 4: Phase 1 training (16-sample overfit, sampling eval)

Step 5: Phase 2 training (512-sample, gate check)

Step 6: Phase 3 training (full-scale)
```

### Step 1 implementation start log (2026-06-08)

Initial code path has started with v1 compatibility preserved:

- `src/model/flex_scene_encoder.py`
  - Added `architecture: multi_level` support through `FlexSceneConfig`
  - Added `MultiLevelFlexEncoder`
  - Uses level-specific cross-attention encoders for 3 DeepStack streams + final stream
  - Uses one shared 512 slot grid with camera/time/local-slot query embeddings
  - Requires `compression_mode: per_image` to preserve camera/frame slot order

- `src/model/student_wrapper.py`
  - Keeps existing `FlexSceneEncoder` for v1
  - Creates `MultiLevelFlexEncoder` only when `flex_scene_config.architecture == "multi_level"`
  - Re-batches Qwen flat DeepStack visual tensors into `[B, 2880, D]`
  - Returns compressed DeepStack tensors as flat `[B*512, D]` for Qwen language-model injection
  - Defaults DeepStack injection ON for ML-FLEX, so it does not depend on a remembered runtime flag

- `src/model/checkpoint_io.py`, `scripts/09_train_distill.py`, `scripts/103_make_flex_untrained_checkpoint.py`
  - Persist and restore `architecture`, `use_local_slot_embeddings`, and `num_deepstack_levels`
  - Default ML-FLEX compression mode to `per_image`

- `configs/train/stage_ml_flex_2b_k512_camtime_smoke.yaml`
  - Added a first smoke config for 2B, K=512, camera/time/local-slot ML-FLEX

Verification so far:

```text
py_compile: pass
MultiLevelFlexEncoder toy shape: final [2, 6, 8], deepstack 3 x [2, 6, 8]
Wrapper toy contract: input embeds [1, 10, 8], visual mask 6, deepstack 3 x [6, 8]
```

---

## 11. Implementation Priority (Overall)

```
Step 0: DeepStack ablation ✓ COMPLETE (2026-06-08)
        - Direction B confirmed: DeepStack OFF causes +47–60% ADE degradation
          in action expert path across both 10B and 2B models
        - Direction A rejected
        ↓
Step 0.5a: B0 attention profiling (pre-gate, 1 forward pass)
         - Measure per-layer visual attention fraction on B0
         - Determine data-driven reset boundary for Run B
         ↓
Step 0.5b: LoRA init strategy gate — Phase 1 (sanity) ← NEXT
         - 3-way comparison on 16 samples
         - Gate: "does ML-FLEX produce valid trajectories at all?"
         - Does NOT decide final init — only eliminates broken runs
         ↓
Step 0.5c: LoRA init strategy gate — Phase 2 (decision)
         - Surviving runs scale to 512 samples, 3000 steps
         - Gate: "which init scales better?"
         - THIS decides backbone init for all subsequent phases
         ↓
Step 1: Implement MultiLevelFlexEncoder
        - Extend src/model/flex_scene_encoder.py
        - Add MLFlexConfig, per-level cross-attention encoders
        - Wire ViT deepstack_image_embeds into ML-FLEX forward pass
        - Modify student_wrapper to pass compressed_deepstack to LLM forward
        ↓
Step 2: Progressive scaling → full
        - Full-scale gate: minADE@6 within +20% of B0
```

### Step 0.5: LoRA Init Strategy Gate (detail)

**Problem:** Neither B0-init nor base-init is mismatch-free.

- Cosmos-Reason2-2B base was pretrained on dense Qwen3-VL visual interface (2880 tokens). Base has never seen 512 FLEX tokens either — the "fresh start = no mismatch" argument is wrong. The difference is not mismatch vs no-mismatch, but **mismatch + task knowledge** vs **mismatch + no task knowledge**.
- B0 LoRA encodes valuable task knowledge (CoT format, 128-token trajectory grammar, special token embeddings, camera-labeled prompt parsing, action expert boundary hidden conventions), but this knowledge is entangled with the 2880-token attention distribution in the same W_q/W_k/W_v matrices. Adapting for 512 tokens risks destroying this task knowledge (catastrophic interference).
- Base-init avoids interference but faces **task acquisition cost**: re-learning CoT, trajectory grammar, special tokens, camera prompt parsing, and action expert boundary conventions from scratch.

**Which risk is worse is an empirical question, not a theoretical one.**

### Step 0.5a: B0 Attention Profiling (pre-gate)

Run B의 layer reset boundary를 데이터로 결정한다. "Early = visual, late = task"는 가정일 뿐이다.

Transformer에서 visual/task 분리가 깨끗하지 않은 이유:
- DeepStack이 LLM layer 0, 1, 2에 visual feature를 inject → layer 0부터 visual + text가 이미 혼합
- LoRA ΔW_q 안에 "visual attention pattern"과 "CoT reasoning pattern"이 rank-64 안에 superpose → layer 단위 reset으로 한 layer 내 visual/task 분리 불가
- Visual evidence → CoT reasoning 전환이 가장 활발한 구간이 mid-layer일 가능성 높음 → 거기를 reset하면 task knowledge도 같이 소실

**측정 방법 (B0 forward pass 1회, 16 eval samples):**

```python
# 각 layer에서 CoT/traj token -> 실제 visual token positions로의 attention weight 합
for layer_idx in range(28):
    # Qwen3-VL에서 output_attentions는 Flash/SDPA 경로에서 안 나오거나 메모리 폭발 가능.
    # 이 profiling은 1~4 samples, eager attention, no grad 디버그로만 수행한다.
    attn_weights = model.layers[layer_idx].self_attn(..., output_attentions=True)

    # 절대 visual token이 prefix라고 가정하지 않는다.
    visual_pos = input_ids == image_token_id  # image_token_id = 151655
    cot_traj_pos = cot_span_mask | traj_token_mask

    visual_attn_frac[layer_idx] = attn_weights[:, :, cot_traj_pos, visual_pos].sum()
                                  / attn_weights[:, :, cot_traj_pos, :].sum()
```

**주의:** attention fraction은 attribution이 아니다. DeepStack 정보는 이미 early residual stream에 섞여 있으므로,
attention만으로 layer reset boundary를 확정하면 안 된다. 최종 reset boundary는 attention profile에 더해
LoRA delta norm, FLEX CE/KD gradient norm, and layer-group reset sensitivity를 함께 보고 정한다.

**결과로 결정:**
- visual_attn_frac이 높은 layer 구간 = visual-heavy → reset 대상
- visual_attn_frac이 낮은 layer 구간 = task-heavy → 보존 대상
- 경계가 깨끗하지 않으면 (전 layer에 걸쳐 고르게 분포) → Run B 자체가 성립 안 함, A/C 2-way로 축소

**비용:** Forward pass 1회 + attention 저장. 16 samples이면 수 분.

### Step 0.5b: Phase 1 — Sanity Gate (16 samples)

**목적: "ML-FLEX가 이 init에서 아예 작동하는가?"만 확인. 최종 init을 여기서 결정하지 않는다.**

16-sample / 짧은 step gate는 B0-init에 구조적으로 유리하다. B0는 이미 trajectory grammar와 CoT format을 알고 있어서 FLEX가 조금만 작동해도 바로 trajectory가 나온다. Base-init(Run C)은 같은 step 안에 FLEX compression + task knowledge를 동시에 배워야 해서 불공평한 비교가 된다. "누가 16 sample에서 빠른가"는 "누가 scale에서 좋은가"와 다르다.

따라서 Phase 1은 elimination gate로만 사용한다.

**Three-way comparison:**

| Run | LoRA init | ML-FLEX init | Trainable | LR |
|---|---|---|---|---|
| A | B0 LoRA as-is | from scratch | ML-FLEX + LoRA (all-layer rank-64) | flex 1e-4, lora 5e-6 (gentle) |
| B | B0 LoRA with layers 0–N reset (N from 0.5a) | from scratch | ML-FLEX + LoRA (all-layer rank-64) | flex 1e-4, lora 2e-5 |
| C | New LoRA (from Cosmos-Reason2-2B base) | from scratch | ML-FLEX + LoRA (all-layer rank-64) | flex 1e-4, lora 2e-5 |

**Step budget — Run C에 task acquisition budget 부여:**

```
Run A: 500 steps
Run B: 500 steps
Run C: 1500 steps (task acquisition ~1000 step + FLEX adaptation ~500 step)
```

B0-init(A/B)는 task knowledge가 있으니 500 step이면 FLEX adaptation 효과가 보인다. Base-init(C)는 task knowledge가 없으니 최소 1000 step은 trajectory token grammar 등 기본 task를 배우는 데 써야 한다. 이 budget 차이는 C에게 유리한 게 아니라 공정한 비교를 위한 보정이다.

**Shared across all runs:**
- Same ML-FLEX encoder config (from scratch, same architecture)
- Same 16 training samples (category-balanced)
- Same Stage A prealignment first: 300–500 steps ML-FLEX-only feature/interface prealignment
- Same Stage B loss after Stage A: traj_token_ce + teacher_logit_kd
- Same eval: sampling minADE@4, unique token count, GT ADE/FDE
- ViT frozen, base LLM weights frozen (only LoRA + ML-FLEX trainable)
- Token rows:
  - A/B reuse B0's trained special/traj embeddings and lm_head rows
  - C1 uses base LoRA init but reuses B0 special/traj embeddings and lm_head rows
  - C2 uses base LoRA init and fresh special/traj rows

Run C is split because otherwise "base-init LoRA" is confounded with "fresh token-row cold start."
C1 is the main clean-start baseline. C2 only answers whether token-row reset is tolerable.

**Phase 1 gate criteria (elimination only):**

| Metric | Alive | Dead |
|---|---|---|
| trajectory token collapse | max same-token run < 20 | ≥ 20 (complete collapse) |
| unique trajectory tokens | > 15 / 128 | ≤ 15 |
| loss convergence | train loss decreasing | loss flat or diverging |

이 gate는 의도적으로 느슨하다. "이 init에서 ML-FLEX가 아예 안 된다"만 걸러낸다. ADE threshold로 판단하지 않는다 — 16 sample ADE는 B0-init에 bias되어 있으므로.

**Phase 1 결과 해석:**
- 3개 다 alive → Phase 2 진행 (3-way)
- 1–2개 dead → dead run 제거, alive만 Phase 2 진행
- 3개 다 dead → ML-FLEX architecture/recipe 문제, init 문제 아님 → 진단

### Step 0.5c: Phase 2 — Scale Decision Gate (512 samples)

**목적: "어떤 init이 scale에서 유리한가?" 여기서 최종 init을 결정한다.**

Phase 1을 통과한 run들만 참가. 512 samples, 3000 steps로 scale 올린다.

```
All surviving runs: 512 samples, 3000 steps, same ML-FLEX/loss/eval config
```

**Phase 2 gate criteria (decision):**

| Metric | Pass |
|---|---|
| sampling minADE@4 (512-sample held-out 64) | < 4.0m |
| frozen action-expert ADE/FDE smoke | not worse than discrete path by >25% |
| unique trajectory tokens | > 30 / 128 |
| no trajectory collapse | max same-token run < 10 |

**Phase 2 decision — 최종 init 선택:**

Primary metric: 512-sample held-out minADE@4. 가장 낮은 run의 init을 선택.

Secondary: 동점이면 learning curve slope (step 1500–3000 구간 ADE 감소율)를 비교. Slope가 가파른 쪽이 추가 학습에서 더 개선될 가능성이 높다. 이 기준은 B0-init의 "head start" bias를 보정한다 — B0-init이 초기 ADE는 낮지만 plateau에 빠졌다면 slope가 flat할 것이고, base-init이 늦게 출발했지만 아직 개선 중이면 slope가 steep할 것이다.

Tiebreak rules:
- A ≈ B > C (ADE 유사, slope 유사): B0 task knowledge 도움됨 → B 선택 (partial reset이 visual adaptation에 유리)
- C가 slope에서 크게 앞서면: C 선택 (B0 knowledge가 plateau를 만드는 증거)
- 전부 유사: frozen-AE ADE/FDE와 camera-shuffle sensitivity가 좋은 쪽 선택. 그래도 유사하면 B 선택 (task knowledge 보존이 이미 비용을 절약했고 inference cost는 동일)

---

## 12. Open Questions

1. ~~**Step 0 pending:** DeepStack ON vs OFF ablation on 10B and 2B. Results will determine Direction A vs B.~~ **RESOLVED** → Direction B. See Section 0 for full results.

2. **Shared slot grid vs stronger camera-factored queries?** Current design: one shared 512-slot camera/frame/local-slot grid across all four output streams. Alternative: more explicitly camera-factored query parameters (128 queries per camera × 4 cameras = 512). Could improve camera sensitivity. **Priority: medium.** Step 0 showed the action expert is the most DeepStack-sensitive path, and action expert conditioning comes from per-position hidden states — camera-factored queries could help preserve this structure.

3. **How many encoder layers per level?** 1 layer is cheapest. 2 layers adds ~6M params. Start with 1.

4. **K=32 per image or try K=56 (896 total)?** K=32 is aggressive (5.625x visual-token compression). If Phase 1 fails at K=32, try K=56 as an intermediate 3.2x compression point.

5. **Should we try FocusUI-style POSPAD markers?** Current `compress_batch_for_flex` removes tokens. POSPAD could help MRoPE continuity. Worth testing as ablation.

6. **Can Alpamayo 10B's FLEX weights tell us anything?** If the public 10B checkpoint has FLEX weights, inspecting how they handle (or ignore) DeepStack could inform our approach.

7. **B0 discrete vs action expert divergence under DeepStack OFF.** B0 2B discrete shows only +7.1% degradation while action expert shows +55.8%. This asymmetry suggests the discrete token path has learned compensating representations in the LLM's own parameters (possibly through LoRA), while the action expert's cross-attention conditioning is more directly dependent on DeepStack-enriched hidden states. ML-FLEX must specifically validate action expert quality, not just discrete token quality — the action expert is both the better path (2.713m vs 3.486m with DS ON) and the more fragile one.

8. **LoRA init strategy.** Neither B0-init nor base-init is mismatch-free — both see FLEX 512 tokens for the first time. The question is whether B0's task knowledge (CoT, trajectory grammar, special tokens, camera parsing, AE boundary conventions) survives the 2880→512 distribution shift, or whether it actively interferes. **Step 0.5 3-way gate** (Section 11) resolves this empirically. Gate is split into Phase 1 (16-sample sanity, elimination only) and Phase 2 (512-sample scale, final decision) to avoid small-scale bias favoring B0-init.

9. **Run B layer reset boundary.** ~~The initial proposal resets layers 0–13 and keeps 14–27.~~ This boundary is not assumed — it is determined by **Step 0.5a attention profiling** (1 forward pass on B0, measure per-layer visual attention fraction from CoT/traj positions). If profiling shows visual attention is distributed uniformly across all layers (no clean split), Run B itself is dropped and the gate reduces to A/C 2-way.
