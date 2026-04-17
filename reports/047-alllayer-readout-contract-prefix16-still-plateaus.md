## 047. All-Layer LoRA + lm_head/embed + Contract Absorption (prefix16)

### Goal

Test whether a stronger `pure LM` training setup can break the `<i1499>` plateau by combining:

- all-language-layer LoRA
- explicit `lm_head` / token embedding unfreeze
- multimodal projector unfreeze
- prefix16 contract absorption (`traj_aux_pseudo_ce` + `traj_aux_guided_kd`)

### Config

- Stage config:
  [stage_t1_alllayer_readout_contract_prefix16_from_format.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_alllayer_readout_contract_prefix16_from_format.yaml)
- Init checkpoint:
  [subset_t1_pairifacev2c_mixed_from_format_train4_step12/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_pairifacev2c_mixed_from_format_train4_step12/final)
- Output summary:
  [subset_t1_alllayer_readout_contract_prefix16_from_format_train4_step12.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_alllayer_readout_contract_prefix16_from_format_train4_step12.json)
- LM-only val decode:
  [subset_t1_alllayer_readout_contract_prefix16_from_format_train4_step12_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_alllayer_readout_contract_prefix16_from_format_train4_step12_infer_val10_lm.json)
- Adapter audit:
  [stage_t1_alllayer_readout_contract_prefix16_from_format_adapter_audit.json](/workspace/cosmos_distillation/outputs/reports/stage_t1_alllayer_readout_contract_prefix16_from_format_adapter_audit.json)

### What Was Actually Trainable

- LoRA adapter weights exist for **all 28 language layers** (`0-27`)
- `lm_head` explicitly unfrozen
- `language_model.embed_tokens` explicitly unfrozen
- multimodal projector explicitly unfrozen
- `traj_aux_head` frozen for this run

### Training Behavior

- 12-step `train4` probe completed stably
- Grad norm stayed bounded (`~75-116`)
- Main logged losses remained finite throughout

### Decode Result

LM-only validation remained a full plateau:

- `avg_unique_traj_ids = 1.0`
- `avg_max_same_token_run = 128.0`

Every decoded validation sample still emitted a 128-token body that was effectively:

`<i1499> ... <i1499>`

### Interpretation

This run rules out a narrow explanation:

- it was **not** just “upper2/upper4 wasn’t enough”
- it was **not** just “lm_head/embed were still frozen”
- it was **not** just “LoRA wasn’t actually all-layer”

Even with all-layer LoRA plus readout opening and prefix16 contract absorption, the pure LM trajectory body still collapsed completely.

### Conclusion

`all-layer LoRA + lm_head/embed + contract absorption(prefix16)` was **not sufficient** to recover a non-collapsed pure LM trajectory body on the `train4` probe.

The remaining blocker is stronger than “not enough trainable layers”; it still points to a deeper pure-LM body contract failure.
