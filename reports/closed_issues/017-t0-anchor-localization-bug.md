# 017 T0 Anchor Localization Bug

- status: resolved
- symptom:
  - canonical local frame was anchored to the last future pose rather than true `t0`
  - history/future local semantics and weak GT path semantics could be distorted
- root cause:
  - history and future trajectories were concatenated, then localized with an implicit anchor of `xyz_world[-1]`
- resolution:
  - `localize_trajectory()` now takes an explicit anchor pose
  - canonicalization uses the true anchor pose from `hist[-1]`
  - canonical sample artifacts now store both world-space and `local@t0` arrays plus anchor pose metadata
  - unit regression added to enforce `ego_history_xyz_local_t0[-1] ~= 0` and identity rotation at anchor
- key files or artifacts:
  - `src/data/local_dataset.py`
  - `src/data/canonicalize.py`
  - `tests/unit/test_canonicalize.py`
  - `outputs/reports/canonical_materialization_teacher262_v2.json`
- follow-up:
  - refresh any downstream artifact when `canonical_localization_version` changes

