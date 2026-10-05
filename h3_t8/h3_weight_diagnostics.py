"""Header-only evidence, not a model compatibility whitelist or a loader.

Names and metadata do not prove numeric compatibility. No tensor is loaded,
downloaded, converted, or hashed by these helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct

from .model_recipe_cards import recipe_from_header


def describe_h3_payload(keys, metadata=None):
    keys = sorted(str(key) for key in keys)
    metadata = dict(metadata or {})
    requirements = []
    if (str(metadata.get("hyperflow", "")).lower() == "true"
            or "hyperflow_sigmas" in metadata
            or any("endpoint_time_embedder" in key for key in keys)):
        requirements.append({"kind": "hyperflow", "runtime": "two_time_embedder_and_trained_sigma_grid",
                             "ordinary_lora_is_complete": False})
    if (metadata.get("adapter_type") == "MiniMax-H3-PDD"
            or any(key.startswith("pdd.final_layer.") for key in keys)):
        requirements.append({"kind": "pdd", "runtime": "dynamic_av_heads_and_pdd_schedule",
                             "ordinary_lora_is_complete": False})
    if any(key.endswith(".attn.to_gate_compress.set_weight") for key in keys):
        requirements.append({"kind": "fastvideo_vsa", "runtime": "vsa_compression_gate_attention",
                             "ordinary_lora_is_complete": False})
    merge_evidence = []
    # Only explicit merge declarations count. A file named "turbo" is not evidence.
    for key in ("merged_loras", "merged_adapters", "t8_merged_adapters", "merge_recipe"):
        value = metadata.get(key)
        if value is not None and any(word in str(value).lower() for word in ("turbo", "hyperflow", "fasth3", "fast_h3", "pdd")):
            merge_evidence.append({"metadata_key": key, "declaration": str(value)[:2048]})
    return {
        "raw_tensor_count": len(keys),
        "special_runtime_requirements": requirements,
        "merged_acceleration": {"declared": bool(merge_evidence), "evidence": merge_evidence,
                                "verified": False, "scope": "file_metadata_declaration_not_weight_validation"},
        "pruned_structure_detected": any("adaln_t_table" in key or "adaln_curve" in key for key in keys),
        "compatibility": "not_established_by_header",
        "identity_policy": "structural_diagnostics_not_unknown_model_gate",
    }


def inspect_h3_weight_file(path, *, maximum_header_bytes=16 * 1024 * 1024):
    """Read only the bounded safetensors JSON header; never its tensor body.

Non-safetensors files, missing paths and malformed headers produce diagnostics,
not a prohibition on a separately selected loader.
"""
    path = Path(path)
    report = {"name": path.name, "scope": "safetensors_header_only", "status": "unknown"}
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors length")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= maximum_header_bytes or length + 8 > size:
                raise ValueError("invalid or oversized safetensors header")
            raw = handle.read(length)
        header = json.loads(raw)
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be an object")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
            raise ValueError("safetensors metadata must map strings to strings")
        for key, value in header.items():
            if not isinstance(value, dict) or not isinstance(value.get("shape"), list):
                raise ValueError(f"invalid tensor descriptor: {key}")
            offsets = value.get("data_offsets")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(item) is not int for item in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= size - length - 8):
                raise ValueError(f"invalid tensor offsets: {key}")
        report.update(describe_h3_payload(header, metadata))
        report["recipe_card"] = recipe_from_header(header, metadata)
        report.update(status="inspected", bytes_read=length + 8, file_size_bytes=size,
                      tensor_values_loaded=False)
    except (OSError, ValueError, UnicodeError, struct.error) as error:
        report["diagnostic"] = f"{type(error).__name__}: {error}"
    return report


def lora_mapping_diagnostics(converted, key_map, patches, applied):
    """Account for parser-reported keys without pretending registration ran GEMMs."""
    recognized = set()
    for patch in patches.values():
        recognized.update(getattr(patch, "loaded_keys", ()) or ())
    for alias, target in key_map.items():
        bias = str(target).removesuffix(".weight") + ".bias"
        if target in patches:
            for suffix in (".alpha", ".dora_scale", ".w_norm", ".diff", ".set_weight"):
                if alias + suffix in converted:
                    recognized.add(alias + suffix)
        if bias in patches:
            for suffix in (".b_norm", ".diff_b"):
                if alias + suffix in converted:
                    recognized.add(alias + suffix)
    recognized &= set(converted)
    unmapped = sorted(set(converted) - recognized)
    registered = sorted(map(str, applied))
    return {"converted_parser_recognized_tensor_count": len(recognized),
            "converted_unmapped_tensor_count": len(unmapped),
            "converted_unmapped_keys": unmapped,
            "registered_patch_targets": registered,
            "registration_is_execution": False,
            "execution_validation": "not_observed_by_loader",
            "raw_to_converted_counts": "not_bijective_when_qkv_fusion_or_layout_conversion_is_used"}
