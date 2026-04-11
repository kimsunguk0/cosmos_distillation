# Cosmos Distillation Report

## teacher_smoketest
```json
{
  "manifest_path": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/strict_download_subset_manifest.parquet",
  "canonical_root": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples",
  "imports": [
    {
      "module": "torch",
      "ok": true,
      "error": ""
    },
    {
      "module": "transformers",
      "ok": true,
      "error": ""
    },
    {
      "module": "hydra",
      "ok": false,
      "error": "No module named 'hydra'"
    },
    {
      "module": "alpamayo1_5",
      "ok": true,
      "error": ""
    },
    {
      "module": "av",
      "ok": false,
      "error": "No module named 'av'"
    }
  ],
  "teacher_model_load_attempted": false,
  "first_ready_sample": null,
  "first_blocked_sample": {
    "sample_id": "000ba013-9eb4-45ca-8e86-93fdc68c37e2__anchor0",
    "status": "awaiting_canonical_sample",
    "blockers": [
      "canonical_sample_missing"
    ],
    "sample_dir": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples/000ba013-9eb4-45ca-8e86-93fdc68c37e2__anchor0"
  },
  "notes": [
    "This smoke test only audits dependencies and sample readiness.",
    "Actual Alpamayo generation stays deferred until image frames are materialized."
  ]
}
```

## teacher_smoketest_vendor
```json
{
  "manifest_path": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/strict_download_subset_manifest.parquet",
  "canonical_root": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples",
  "imports": [
    {
      "module": "torch",
      "ok": true,
      "error": ""
    },
    {
      "module": "transformers",
      "ok": true,
      "error": ""
    },
    {
      "module": "hydra",
      "ok": true,
      "error": ""
    },
    {
      "module": "alpamayo1_5",
      "ok": true,
      "error": ""
    },
    {
      "module": "av",
      "ok": true,
      "error": ""
    }
  ],
  "teacher_model_load_attempted": false,
  "first_ready_sample": null,
  "first_blocked_sample": {
    "sample_id": "000ba013-9eb4-45ca-8e86-93fdc68c37e2__anchor0",
    "status": "awaiting_canonical_sample",
    "blockers": [
      "canonical_sample_missing"
    ],
    "sample_dir": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples/000ba013-9eb4-45ca-8e86-93fdc68c37e2__anchor0"
  },
  "notes": [
    "This smoke test only audits dependencies and sample readiness.",
    "Actual Alpamayo generation stays deferred until image frames are materialized."
  ]
}
```

## teacher_smoketest_ready
```json
{
  "manifest_path": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/strict_download_subset_manifest.parquet",
  "canonical_root": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples",
  "imports": [
    {
      "module": "torch",
      "ok": true,
      "error": ""
    },
    {
      "module": "transformers",
      "ok": true,
      "error": ""
    },
    {
      "module": "hydra",
      "ok": true,
      "error": ""
    },
    {
      "module": "alpamayo1_5",
      "ok": true,
      "error": ""
    },
    {
      "module": "av",
      "ok": true,
      "error": ""
    }
  ],
  "teacher_model_load_attempted": false,
  "first_ready_sample": {
    "sample_id": "0abe118e-aa79-41f6-a719-f2df8abaf1ea__anchor0",
    "status": "ready",
    "blockers": [],
    "sample_dir": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples/0abe118e-aa79-41f6-a719-f2df8abaf1ea__anchor0"
  },
  "first_blocked_sample": {
    "sample_id": "000ba013-9eb4-45ca-8e86-93fdc68c37e2__anchor0",
    "status": "awaiting_canonical_sample",
    "blockers": [
      "canonical_sample_missing"
    ],
    "sample_dir": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples/000ba013-9eb4-45ca-8e86-93fdc68c37e2__anchor0"
  },
  "notes": [
    "This smoke test only audits dependencies and sample readiness.",
    "Actual Alpamayo generation stays deferred until image frames are materialized."
  ]
}
```

## teacher_text_cache_summary_updated
```json
{
  "manifest_path": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/strict_download_subset_manifest.parquet",
  "supervision_jsonl": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/supervision_records/records.jsonl",
  "canonical_root": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples",
  "cache_root": "/home/pm97/workspace/sukim/cosmos_distillation/data/teacher_cache/text",
  "index_path": "/home/pm97/workspace/sukim/cosmos_distillation/data/teacher_cache/text/index.jsonl",
  "request_bundle_count": 732,
  "status_counts": {
    "awaiting_canonical_sample": 722,
    "ready_request_bundle": 5,
    "blocked": 5
  },
  "ready_for_teacher_generation": 5,
  "notes": [
    "This step scaffolds teacher requests without generating teacher outputs.",
    "Alpamayo inference is deferred until image frames exist for the canonical sample."
  ]
}
```

## teacher_joint_cache_summary
```json
{
  "manifest_path": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/strict_download_subset_manifest.parquet",
  "teacher_text_index": "/home/pm97/workspace/sukim/cosmos_distillation/data/teacher_cache/text/index.jsonl",
  "output_jsonl": "/home/pm97/workspace/sukim/cosmos_distillation/data/teacher_cache/joint/index.jsonl",
  "status_counts": {
    "blocked": 732
  },
  "notes": [
    "Joint cache generation is scaffolded only.",
    "These records are for future v2 trajectory-reasoning alignment, not v1 training."
  ]
}
```

## corpus_summary
```json
{
  "manifest_path": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/strict_download_subset_manifest.parquet",
  "output_jsonl": "/home/pm97/workspace/sukim/cosmos_distillation/data/corpus/strict_human_long_cot.jsonl",
  "task_type": "human_long_cot",
  "counts_by_split": {
    "train": 577,
    "val": 105,
    "test": 50
  },
  "teacher_soft_target_count": 0,
  "teacher_index_jsonl": "/home/pm97/workspace/sukim/cosmos_distillation/data/teacher_cache/text/index.jsonl",
  "forbidden_tasks_checked": [
    "teacher_reasoning_plus_gt_path",
    "human_reasoning_plus_teacher_path",
    "teacher_discrete_future_tokens_as_gt"
  ]
}
```

## train_summary
```json
{
  "mode": "data_only_dry_run",
  "corpus_jsonl": "/home/pm97/workspace/sukim/cosmos_distillation/data/corpus/strict_human_long_cot.jsonl",
  "train_records": 32,
  "teacher_ready_records": 0,
  "stage_name": "stage_b",
  "batch_size": 4,
  "max_length": 4096,
  "first_batch_shapes": {
    "input_ids": [
      4,
      50
    ],
    "labels": [
      4,
      50
    ]
  }
}
```

## eval_summary
```json
{
  "corpus_jsonl": "/home/pm97/workspace/sukim/cosmos_distillation/data/corpus/strict_human_long_cot.jsonl",
  "teacher_index_jsonl": "/home/pm97/workspace/sukim/cosmos_distillation/data/teacher_cache/text/index.jsonl",
  "teacher_ready_records": 0,
  "json_parseability": 0.0,
  "meta_action_f1": 0.0,
  "human_coc_overlap": 0.0,
  "hallucination_rate": 0.0,
  "teacher_human_disagreement_rate": 0.0,
  "consistency_gate_score": 0.0,
  "rationale_answer_consistency": 0.0,
  "notes": [
    "Evaluation stays partial until actual teacher outputs are generated.",
    "Current metrics focus on available soft-target text alignment only."
  ]
}
```

## canonical_materialization_with_images_summary
```json
{
  "manifest_path": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/strict_download_subset_manifest.parquet",
  "dataset_root": "/home/pm97/workspace/sukim/cosmos_distillation/data/raw/physical_ai_av",
  "output_root": "/home/pm97/workspace/sukim/cosmos_distillation/data/processed/canonical_samples",
  "requested_rows": 100,
  "materialized_rows": 1,
  "skipped_missing_chunks": 99,
  "decoder_failure_rows": 0,
  "decoder_failure_examples": [],
  "skip_images": false
}
```
