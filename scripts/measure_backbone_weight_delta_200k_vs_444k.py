#!/usr/bin/env python3
"""CPU-only student-vs-student weight delta audit for 200K CE vs 444K CE."""

import json
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from safetensors import safe_open


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

INIT_PATH = os.path.join(
    REPO_ROOT,
    "outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/"
    "step_003488/model.safetensors",
)
CE200_PATH = os.path.join(
    REPO_ROOT,
    "outputs/checkpoints/stepb_200k_double_promotion/"
    "double200k_20260711_181751/fullft_lr3e5_200k_e1/"
    "best_decode/student_state.pt",
)
CE444_PATH = os.path.join(
    REPO_ROOT,
    "outputs/checkpoints/stepb_ceonly_444k/ceonly_444k_20260718/"
    "fullft_lr3e5_ceonly_444k_e1/best_decode/student_state.pt",
)
TOKENIZER_DIR = os.path.join(
    REPO_ROOT,
    "outputs/checkpoints/stepb_200k_double_promotion/"
    "double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode",
)
OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "outputs/reports/weight_delta_200k_vs_444k_20260725",
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "weight_delta.json")

STUDENT_PREFIX = "backbone."
INIT_TIED_LM_HEAD_BASELINE_KEY = "model.language_model.embed_tokens.weight"
CANDIDATE_STATE_KEYS = (
    "model",
    "student",
    "state_dict",
    "model_state_dict",
    "student_state_dict",
    "student_model",
    "module",
    "network",
    "net",
    "backbone",
    "base_model",
)
BUCKET_ORDER = (
    "lm_head",
    "embed_tokens",
    "mm_projector",
    "transformer_body",
    "aux_heads",
    "final_norm",
    "other",
)
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def normalize_student_key(key):
    key = str(key)
    if key.startswith(STUDENT_PREFIX):
        return key[len(STUDENT_PREFIX) :]
    return key


def normalize_init_key(key):
    return str(key)


def tensor_key_count(obj):
    if not isinstance(obj, dict):
        return 0
    count = 0
    for value in obj.values():
        if torch.is_tensor(value):
            count += 1
    return count


def all_values_are_tensors(obj):
    return isinstance(obj, dict) and len(obj) > 0 and tensor_key_count(obj) == len(obj)


def looks_like_state_dict(obj):
    if not isinstance(obj, dict):
        return False
    tensor_count = tensor_key_count(obj)
    if tensor_count == 0:
        return False
    if all_values_are_tensors(obj):
        return True
    if tensor_count >= max(1, int(0.75 * len(obj))):
        return True
    for key, value in obj.items():
        if torch.is_tensor(value) and "." in str(key):
            return True
    return False


def find_nested_state_dict(obj, path, depth):
    if looks_like_state_dict(obj):
        return obj, path
    if depth <= 0 or not isinstance(obj, dict):
        return None, None

    lower_to_key = {}
    for key in obj.keys():
        lower_to_key[str(key).lower()] = key

    for candidate in CANDIDATE_STATE_KEYS:
        key = lower_to_key.get(candidate.lower())
        if key is None:
            continue
        found, found_path = find_nested_state_dict(
            obj[key],
            path + "." + str(key) if path else str(key),
            depth - 1,
        )
        if found is not None:
            return found, found_path

    best = None
    best_path = None
    best_count = 0
    for key, value in obj.items():
        if not isinstance(value, dict):
            continue
        found, found_path = find_nested_state_dict(
            value,
            path + "." + str(key) if path else str(key),
            depth - 1,
        )
        if found is None:
            continue
        count = tensor_key_count(found)
        if count > best_count:
            best = found
            best_path = found_path
            best_count = count
    return best, best_path


def extract_state_dict(obj, source_name):
    meta = {
        "source_name": source_name,
        "top_level_keys_sample": [],
        "extracted_from": None,
    }
    if isinstance(obj, dict):
        meta["top_level_keys_sample"] = [str(key) for key in list(obj.keys())[:50]]
        if all_values_are_tensors(obj):
            meta["extracted_from"] = "<top-level>"
            return obj, meta
        found, found_path = find_nested_state_dict(obj, "", 4)
        if found is not None:
            meta["extracted_from"] = found_path if found_path else "<top-level>"
            return found, meta

    if hasattr(obj, "state_dict"):
        state = obj.state_dict()
        meta["extracted_from"] = "<object.state_dict()>"
        return state, meta

    raise TypeError("Could not extract model state_dict from %s" % source_name)


def load_safetensors_state(path, source_name):
    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    meta = {
        "source_name": source_name,
        "format": "safetensors",
        "path": path,
        "top_level_keys_sample": [],
        "extracted_from": "<safetensors>",
    }
    return state, meta


def load_pt_state(path, source_name):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    state, meta = extract_state_dict(obj, source_name)
    meta["format"] = "torch_pt"
    meta["path"] = path
    return state, meta


def normalize_state(state, meta, normalizer):
    tensors = {}
    original_by_norm = {}
    duplicates = {}
    tensor_keys = 0
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        tensor_keys += 1
        norm_key = normalizer(raw_key)
        if norm_key in tensors:
            if norm_key not in duplicates:
                duplicates[norm_key] = [original_by_norm[norm_key]]
            duplicates[norm_key].append(str(raw_key))
            continue
        tensors[norm_key] = value
        original_by_norm[norm_key] = str(raw_key)

    for norm_key in duplicates:
        tensors.pop(norm_key, None)
        original_by_norm.pop(norm_key, None)

    return {
        "name": meta["source_name"],
        "path": meta["path"],
        "format": meta["format"],
        "top_level_keys_sample": meta["top_level_keys_sample"],
        "extracted_from": meta["extracted_from"],
        "state_keys_total": len(state),
        "tensor_keys": tensor_keys,
        "usable_normalized_keys": len(tensors),
        "duplicate_normalized_key_count": len(duplicates),
        "duplicate_normalized_key_sample": [
            {"normalized": key, "originals": values[:5]}
            for key, values in sorted(duplicates.items())[:20]
        ],
        "tensors": tensors,
        "original_by_norm": original_by_norm,
    }


def shape_list(tensor):
    return [int(size) for size in tuple(tensor.shape)]


def match_student_keys(ce200_source, ce444_source):
    common = set(ce200_source["tensors"].keys()).intersection(
        ce444_source["tensors"].keys()
    )
    matched = []
    shape_mismatches = []
    for key in sorted(common):
        shape200 = tuple(ce200_source["tensors"][key].shape)
        shape444 = tuple(ce444_source["tensors"][key].shape)
        if shape200 == shape444:
            matched.append(key)
        else:
            shape_mismatches.append(
                {
                    "normalized": key,
                    "shapes": {
                        ce200_source["name"]: [int(size) for size in shape200],
                        ce444_source["name"]: [int(size) for size in shape444],
                    },
                    "originals": {
                        ce200_source["name"]: ce200_source["original_by_norm"].get(
                            key, key
                        ),
                        ce444_source["name"]: ce444_source["original_by_norm"].get(
                            key, key
                        ),
                    },
                }
            )
    return matched, shape_mismatches


def classify_key(key):
    lower = key.lower()
    if "lm_head" in lower:
        return "lm_head", None
    if "embed_tokens" in lower:
        return "embed_tokens", None
    if (
        "visual" in lower
        or "merger" in lower
        or "patch_embed" in lower
        or "pos_embed" in lower
        or "mm_" in lower
        or "multi_modal" in lower
    ):
        return "mm_projector", None
    match = LAYER_RE.search(lower)
    if match:
        return "transformer_body", int(match.group(1))
    if (
        "meta_action_head" in lower
        or "traj_aux_head" in lower
        or "boundary_action_head" in lower
    ):
        return "aux_heads", None
    if "norm" in lower:
        return "final_norm", None
    return "other", None


def empty_aggregate():
    return {
        "tensor_count": 0,
        "numel": 0,
        "dstep_sq": 0.0,
        "w200_sq": 0.0,
        "d200_sq": 0.0,
        "d444_sq": 0.0,
        "init_matched_tensor_count": 0,
        "init_matched_numel": 0,
    }


def add_primary_to_aggregate(aggregate, numel, dstep_sq, w200_sq):
    aggregate["tensor_count"] += 1
    aggregate["numel"] += int(numel)
    aggregate["dstep_sq"] += float(dstep_sq)
    aggregate["w200_sq"] += float(w200_sq)


def add_secondary_to_aggregate(aggregate, numel, d200_sq, d444_sq):
    aggregate["init_matched_tensor_count"] += 1
    aggregate["init_matched_numel"] += int(numel)
    aggregate["d200_sq"] += float(d200_sq)
    aggregate["d444_sq"] += float(d444_sq)


def add_aggregate_into(dst, src):
    for field in dst:
        dst[field] += src[field]


def squared_norm(tensor):
    return float(torch.sum(tensor * tensor).item())


def sqrt_float(value):
    return float(np.sqrt(max(float(value), 0.0)))


def rel_to_w200(delta_l2, w200_l2):
    return float(delta_l2 / (w200_l2 + 1e-12))


def finalize_aggregate(aggregate, totals):
    dstep_l2 = sqrt_float(aggregate["dstep_sq"])
    w200_l2 = sqrt_float(aggregate["w200_sq"])
    d200_l2 = sqrt_float(aggregate["d200_sq"])
    d444_l2 = sqrt_float(aggregate["d444_sq"])
    total_dstep_sq = totals["dstep_sq"]
    total_d200_sq = totals["d200_sq"]
    total_d444_sq = totals["d444_sq"]
    return {
        "tensor_count": aggregate["tensor_count"],
        "numel": aggregate["numel"],
        "w200_l2": w200_l2,
        "dstep": {
            "l2": dstep_l2,
            "share_of_total_sq_percent": (
                100.0 * aggregate["dstep_sq"] / total_dstep_sq
                if total_dstep_sq > 0.0
                else 0.0
            ),
            "rel_step": rel_to_w200(dstep_l2, w200_l2),
            "sq_sum": aggregate["dstep_sq"],
        },
        "secondary_init": {
            "tensor_count": aggregate["init_matched_tensor_count"],
            "numel": aggregate["init_matched_numel"],
            "d200_l2": d200_l2,
            "d444_l2": d444_l2,
            "d200_share_of_total_sq_percent": (
                100.0 * aggregate["d200_sq"] / total_d200_sq
                if total_d200_sq > 0.0
                else 0.0
            ),
            "d444_share_of_total_sq_percent": (
                100.0 * aggregate["d444_sq"] / total_d444_sq
                if total_d444_sq > 0.0
                else 0.0
            ),
            "d200_sq_sum": aggregate["d200_sq"],
            "d444_sq_sum": aggregate["d444_sq"],
        },
    }


def source_public_summary(source):
    return {
        "name": source["name"],
        "path": source["path"],
        "format": source["format"],
        "top_level_keys_sample": source["top_level_keys_sample"],
        "extracted_from": source["extracted_from"],
        "state_keys_total": source["state_keys_total"],
        "tensor_keys": source["tensor_keys"],
        "usable_normalized_keys": source["usable_normalized_keys"],
        "duplicate_normalized_key_count": source["duplicate_normalized_key_count"],
        "duplicate_normalized_key_sample": source["duplicate_normalized_key_sample"],
    }


def unmatched_sample(source, matched_set):
    unmatched = sorted(set(source["tensors"].keys()) - matched_set)
    return [
        {
            "normalized": key,
            "original": source["original_by_norm"].get(key, key),
        }
        for key in unmatched[:20]
    ]


def get_init_counterpart_key(student_key, init_source, student_tensor):
    if student_key == "lm_head.weight":
        init_tensor = init_source["tensors"].get(INIT_TIED_LM_HEAD_BASELINE_KEY)
        if init_tensor is not None and tuple(init_tensor.shape) == tuple(student_tensor.shape):
            return INIT_TIED_LM_HEAD_BASELINE_KEY, "tied_init_embed_tokens"

    if student_key in init_source["tensors"]:
        init_tensor = init_source["tensors"][student_key]
        if tuple(init_tensor.shape) == tuple(student_tensor.shape):
            return student_key, "direct"

    return None, None


def compute_lm_head_top_rows(ce200_source, ce444_source, matched_set):
    key = "lm_head.weight"
    if key not in matched_set:
        return {
            "available": False,
            "reason": "lm_head.weight not present in primary matched student keys",
            "rows": [],
        }

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        return {
            "available": False,
            "reason": "AutoTokenizer import failed: %s" % repr(exc),
            "rows": [],
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
        ce200_tensor = ce200_source["tensors"][key].detach().to(dtype=torch.float32)
        ce444_tensor = ce444_source["tensors"][key].detach().to(dtype=torch.float32)
        diff = ce444_tensor - ce200_tensor
        row_l2 = torch.sqrt(torch.sum(diff * diff, dim=1))
        top_values, top_indices = torch.topk(row_l2, k=min(20, int(row_l2.numel())))
        rows = []
        for rank, (value, index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            token_id = int(index)
            try:
                token_string = tokenizer.decode([token_id])
            except Exception:
                token_string = None
            try:
                raw_token = tokenizer.convert_ids_to_tokens(token_id)
            except Exception:
                raw_token = None
            rows.append(
                {
                    "rank": rank,
                    "row_index": token_id,
                    "token_id": token_id,
                    "row_dstep_l2": float(value),
                    "decoded": token_string,
                    "raw_token": raw_token,
                }
            )
        del ce200_tensor
        del ce444_tensor
        del diff
        del row_l2
        return {
            "available": True,
            "reason": None,
            "tokenizer_dir": TOKENIZER_DIR,
            "rows": rows,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": "lm_head row/token analysis failed: %s" % repr(exc),
            "tokenizer_dir": TOKENIZER_DIR,
            "rows": [],
        }


def compute_metrics(init_source, ce200_source, ce444_source, matched_keys):
    bucket_sums = {}
    for bucket in BUCKET_ORDER:
        bucket_sums[bucket] = empty_aggregate()
    layer_sums = {}
    totals = empty_aggregate()
    primary_without_init = []
    per_tensor = []

    with torch.no_grad():
        for key in matched_keys:
            ce200_tensor = ce200_source["tensors"][key].detach().to(dtype=torch.float32)
            ce444_tensor = ce444_source["tensors"][key].detach().to(dtype=torch.float32)
            diffstep = ce444_tensor - ce200_tensor

            dstep_sq = squared_norm(diffstep)
            w200_sq = squared_norm(ce200_tensor)
            dstep_l2 = sqrt_float(dstep_sq)
            w200_l2 = sqrt_float(w200_sq)
            bucket, layer_index = classify_key(key)
            numel = int(ce200_tensor.numel())

            add_primary_to_aggregate(bucket_sums[bucket], numel, dstep_sq, w200_sq)
            add_primary_to_aggregate(totals, numel, dstep_sq, w200_sq)
            if layer_index is not None:
                if layer_index not in layer_sums:
                    layer_sums[layer_index] = empty_aggregate()
                add_primary_to_aggregate(
                    layer_sums[layer_index],
                    numel,
                    dstep_sq,
                    w200_sq,
                )

            init_key, init_match_kind = get_init_counterpart_key(
                key,
                init_source,
                ce200_source["tensors"][key],
            )
            d200_l2 = None
            d444_l2 = None
            if init_key is not None:
                init_tensor = init_source["tensors"][init_key].detach().to(
                    dtype=torch.float32
                )
                diff200 = ce200_tensor - init_tensor
                diff444 = ce444_tensor - init_tensor
                d200_sq = squared_norm(diff200)
                d444_sq = squared_norm(diff444)
                d200_l2 = sqrt_float(d200_sq)
                d444_l2 = sqrt_float(d444_sq)
                add_secondary_to_aggregate(bucket_sums[bucket], numel, d200_sq, d444_sq)
                add_secondary_to_aggregate(totals, numel, d200_sq, d444_sq)
                if layer_index is not None:
                    add_secondary_to_aggregate(
                        layer_sums[layer_index],
                        numel,
                        d200_sq,
                        d444_sq,
                    )
                del init_tensor
                del diff200
                del diff444
            else:
                if len(primary_without_init) < 50:
                    primary_without_init.append(
                        {
                            "normalized": key,
                            "bucket": bucket,
                            "shape": shape_list(ce200_tensor),
                            "original_200k": ce200_source["original_by_norm"].get(
                                key, key
                            ),
                            "original_444k": ce444_source["original_by_norm"].get(
                                key, key
                            ),
                        }
                    )

            per_tensor.append(
                {
                    "normalized": key,
                    "originals": {
                        ce200_source["name"]: ce200_source["original_by_norm"].get(
                            key, key
                        ),
                        ce444_source["name"]: ce444_source["original_by_norm"].get(
                            key, key
                        ),
                        init_source["name"]: init_source["original_by_norm"].get(
                            init_key, init_key
                        )
                        if init_key is not None
                        else None,
                    },
                    "bucket": bucket,
                    "layer_index": layer_index,
                    "shape": shape_list(ce200_tensor),
                    "numel": numel,
                    "dstep": dstep_l2,
                    "w200norm": w200_l2,
                    "rel_step": rel_to_w200(dstep_l2, w200_l2),
                    "init_counterpart_key": init_key,
                    "init_match_kind": init_match_kind,
                    "d200": d200_l2,
                    "d444": d444_l2,
                }
            )

            del ce200_tensor
            del ce444_tensor
            del diffstep

    finalized_totals = finalize_aggregate(totals, totals)
    bucket_aggregates = {}
    for bucket in BUCKET_ORDER:
        bucket_aggregates[bucket] = finalize_aggregate(bucket_sums[bucket], totals)

    layer_rows = []
    if layer_sums:
        max_layer = max(layer_sums.keys())
        for layer_index in range(max_layer + 1):
            aggregate = layer_sums.get(layer_index, empty_aggregate())
            finalized = finalize_aggregate(aggregate, totals)
            layer_rows.append(
                {
                    "layer_index": int(layer_index),
                    "tensor_count": finalized["tensor_count"],
                    "numel": finalized["numel"],
                    "dstep_l2": finalized["dstep"]["l2"],
                    "dstep_share_of_total_sq_percent": finalized["dstep"][
                        "share_of_total_sq_percent"
                    ],
                    "rel_step": finalized["dstep"]["rel_step"],
                    "secondary_init_tensor_count": finalized["secondary_init"][
                        "tensor_count"
                    ],
                    "d200_l2": finalized["secondary_init"]["d200_l2"],
                    "d444_l2": finalized["secondary_init"]["d444_l2"],
                }
            )

    readout_sum = empty_aggregate()
    for bucket in ("lm_head", "embed_tokens"):
        add_aggregate_into(readout_sum, bucket_sums[bucket])
    readout_final = finalize_aggregate(readout_sum, totals)
    body_final = bucket_aggregates["transformer_body"]
    readout_rel = readout_final["dstep"]["rel_step"]
    body_rel = body_final["dstep"]["rel_step"]
    if readout_sum["tensor_count"] == 0 or bucket_sums["transformer_body"]["tensor_count"] == 0:
        verdict = (
            "VERDICT: insufficient matched readout/body tensors to compare rel_step "
            "(lm_head+embed_tokens rel_step=%.6e, transformer_body rel_step=%.6e)."
            % (readout_rel, body_rel)
        )
    elif readout_rel > body_rel:
        verdict = (
            "VERDICT: lm_head+embed_tokens moved MORE RELATIVELY than "
            "transformer_body between 200K and 444K by student-vs-student dstep "
            "(%.6e vs %.6e)."
            % (readout_rel, body_rel)
        )
    elif body_rel > readout_rel:
        verdict = (
            "VERDICT: transformer_body moved MORE RELATIVELY than "
            "lm_head+embed_tokens between 200K and 444K by student-vs-student dstep "
            "(%.6e vs %.6e)."
            % (body_rel, readout_rel)
        )
    else:
        verdict = (
            "VERDICT: lm_head+embed_tokens and transformer_body have equal "
            "aggregate rel_step between 200K and 444K by student-vs-student "
            "dstep (%.6e)."
            % readout_rel
        )

    lm_head_top_rows = compute_lm_head_top_rows(ce200_source, ce444_source, set(matched_keys))

    return {
        "totals": finalized_totals,
        "bucket_aggregates": bucket_aggregates,
        "combined_readout_lm_head_plus_embed_tokens": readout_final,
        "transformer_layers": layer_rows,
        "primary_no_init_counterpart_count": (
            len(matched_keys) - totals["init_matched_tensor_count"]
        ),
        "primary_no_init_counterpart_sample": primary_without_init,
        "per_tensor": per_tensor,
        "lm_head_top_rows": lm_head_top_rows,
        "verdict": verdict,
    }


def print_source_summary(sources):
    print("SOURCE KEY SUMMARY")
    for source in sources:
        print(
            "%s: state_keys=%d tensor_keys=%d usable_normalized_keys=%d "
            "duplicates=%d extracted_from=%s"
            % (
                source["name"],
                source["state_keys_total"],
                source["tensor_keys"],
                source["usable_normalized_keys"],
                source["duplicate_normalized_key_count"],
                source["extracted_from"],
            )
        )
        if source["top_level_keys_sample"]:
            print(
                "  top_level_keys_sample: %s"
                % ", ".join(source["top_level_keys_sample"][:20])
            )


def print_unmatched_samples(ce200_source, ce444_source, matched_set):
    print("")
    print("PRIMARY UNMATCHED KEY SAMPLES (up to 20 per student; normalized -> original)")
    for source in (ce200_source, ce444_source):
        samples = unmatched_sample(source, matched_set)
        print("%s unmatched_sample_count=%d" % (source["name"], len(samples)))
        for sample in samples:
            print("  %s -> %s" % (sample["normalized"], sample["original"]))


def print_bucket_table(bucket_aggregates):
    print("")
    print("PRIMARY STUDENT-VS-STUDENT PER-BUCKET AGGREGATES")
    header = (
        "bucket tensor_count dstep_l2 dstep_share_pct rel_step "
        "init_count d200_l2 d444_l2"
    )
    print(header)
    for bucket in BUCKET_ORDER:
        aggregate = bucket_aggregates[bucket]
        print(
            "%s %d %.6e %.4f %.6e %d %.6e %.6e"
            % (
                bucket,
                aggregate["tensor_count"],
                aggregate["dstep"]["l2"],
                aggregate["dstep"]["share_of_total_sq_percent"],
                aggregate["dstep"]["rel_step"],
                aggregate["secondary_init"]["tensor_count"],
                aggregate["secondary_init"]["d200_l2"],
                aggregate["secondary_init"]["d444_l2"],
            )
        )


def print_layer_table(layer_rows):
    print("")
    print("PRIMARY PER-TRANSFORMER-LAYER DSTEP")
    print("layer tensor_count dstep_l2 dstep_share_pct rel_step init_count")
    for row in layer_rows:
        print(
            "%d %d %.6e %.4f %.6e %d"
            % (
                row["layer_index"],
                row["tensor_count"],
                row["dstep_l2"],
                row["dstep_share_of_total_sq_percent"],
                row["rel_step"],
                row["secondary_init_tensor_count"],
            )
        )


def print_lm_head_rows(lm_head_top_rows):
    print("")
    print("LM_HEAD TOP-20 ROW DSTEP TOKEN ANALYSIS")
    if not lm_head_top_rows.get("available"):
        print("unavailable: %s" % lm_head_top_rows.get("reason"))
        return
    print("rank token_id row_dstep_l2 raw_token decoded")
    for row in lm_head_top_rows["rows"]:
        decoded = repr(row["decoded"])
        raw_token = repr(row["raw_token"])
        print(
            "%d %d %.6e %s %s"
            % (
                row["rank"],
                row["token_id"],
                row["row_dstep_l2"],
                raw_token,
                decoded,
            )
        )


def main():
    print("CPU-only weight delta audit")
    print("PRIMARY: student-vs-student 200K vs 444K after one leading backbone. strip")
    print("INIT secondary baseline: %s" % INIT_PATH)
    print("200K: %s" % CE200_PATH)
    print("444K: %s" % CE444_PATH)

    init_state, init_meta = load_safetensors_state(INIT_PATH, "init")
    ce200_state, ce200_meta = load_pt_state(CE200_PATH, "ce200")
    ce444_state, ce444_meta = load_pt_state(CE444_PATH, "ce444")

    init_source = normalize_state(init_state, init_meta, normalize_init_key)
    ce200_source = normalize_state(ce200_state, ce200_meta, normalize_student_key)
    ce444_source = normalize_state(ce444_state, ce444_meta, normalize_student_key)
    sources = [init_source, ce200_source, ce444_source]
    print_source_summary(sources)

    matched_keys, shape_mismatches = match_student_keys(ce200_source, ce444_source)
    matched_set = set(matched_keys)
    print("")
    print("PRIMARY_MATCHED_STUDENT_KEYS: %d" % len(matched_keys))
    print("PRIMARY_STUDENT_SHAPE_MISMATCHES: %d" % len(shape_mismatches))
    for mismatch in shape_mismatches[:20]:
        print("  shape_mismatch: %s %s" % (mismatch["normalized"], mismatch["shapes"]))
    print_unmatched_samples(ce200_source, ce444_source, matched_set)

    if not matched_keys:
        raise RuntimeError(
            "No matched tensor keys with identical shapes across the two student states."
        )

    metric_result = compute_metrics(init_source, ce200_source, ce444_source, matched_keys)
    print_bucket_table(metric_result["bucket_aggregates"])
    print_layer_table(metric_result["transformer_layers"])
    print("")
    print(
        "SECONDARY_INIT_COUNTERPARTS: %d matched, %d primary tensors without init counterpart"
        % (
            metric_result["totals"]["secondary_init"]["tensor_count"],
            metric_result["primary_no_init_counterpart_count"],
        )
    )
    print_lm_head_rows(metric_result["lm_head_top_rows"])
    print("")
    print(metric_result["verdict"])

    result = {
        "paths": {
            "init": INIT_PATH,
            "ce200": CE200_PATH,
            "ce444": CE444_PATH,
            "tokenizer_dir": TOKENIZER_DIR,
            "output_json": OUTPUT_JSON,
        },
        "key_alignment": {
            "primary": "student_vs_student",
            "student_normalization": "strip exactly one leading 'backbone.' only",
            "init_normalization": "keep keys as-is",
            "lm_head_init_baseline": INIT_TIED_LM_HEAD_BASELINE_KEY,
        },
        "sources": [source_public_summary(source) for source in sources],
        "matching": {
            "primary_matched_student_count": len(matched_keys),
            "primary_matched_key_sample": matched_keys[:20],
            "primary_shape_mismatch_count": len(shape_mismatches),
            "primary_shape_mismatch_sample": shape_mismatches[:20],
            "primary_unmatched_key_samples": {
                ce200_source["name"]: unmatched_sample(ce200_source, matched_set),
                ce444_source["name"]: unmatched_sample(ce444_source, matched_set),
            },
            "primary_no_init_counterpart_count": metric_result[
                "primary_no_init_counterpart_count"
            ],
            "primary_no_init_counterpart_sample": metric_result[
                "primary_no_init_counterpart_sample"
            ],
        },
        "totals": metric_result["totals"],
        "bucket_aggregates": metric_result["bucket_aggregates"],
        "combined_readout_lm_head_plus_embed_tokens": metric_result[
            "combined_readout_lm_head_plus_embed_tokens"
        ],
        "transformer_layers": metric_result["transformer_layers"],
        "lm_head_top_rows": metric_result["lm_head_top_rows"],
        "per_tensor": metric_result["per_tensor"],
        "verdict": metric_result["verdict"],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("JSON_WRITTEN: %s" % OUTPUT_JSON)


if __name__ == "__main__":
    main()
