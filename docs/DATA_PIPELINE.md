# Data Pipeline

The pipeline is organized around the first eight work packages from the spec.

## Sequence

1. `WP0`: verify access to the dataset and both models
2. `WP1`: download metadata only
3. `WP2`: build a 1400-clip canonical manifest
4. `WP3`: materialize the keyframe-centered canonical sample window
5. `WP4`: store hard human and weak derived supervision separately
6. `WP5`: generate teacher text cache
7. `WP6`: keep an optional teacher joint cache interface ready
8. `WP7`: compute the consistency gate matrix
9. `WP8`: build a safe multitask corpus

## Canonical sample defaults

- Cameras:
  - `camera_cross_left_120fov`
  - `camera_front_wide_120fov`
  - `camera_cross_right_120fov`
  - `camera_front_tele_30fov`
- Frame offsets: `[-0.3, -0.2, -0.1, 0.0]`
- Ego history: `16` steps at `10 Hz`
- Ego future: `64` steps at `10 Hz`

## Output contracts

- `data/processed/clip_manifest.parquet`
- `data/processed/canonical_samples/<sample_id>/...`
- `data/teacher_cache/text/*.json`
- `data/teacher_cache/diagnostics/consistency_matrix.parquet`
- `data/corpus/*.jsonl`

## Non-negotiable rule

Teacher caches and hard human data must keep separate provenance fields all the
way through corpus generation.
