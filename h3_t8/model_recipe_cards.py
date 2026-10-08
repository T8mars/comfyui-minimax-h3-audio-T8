"""Evidence-labelled recipe previews. No loader gate or automatic setting changes.

Training rank/alpha, runtime LoRA strength and Bridge alpha are distinct fields.
Header evidence cannot prove scalar tensor values, trained tasks or quality.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math


SCHEMA = "t8.model-recipe-card.v1"


def _evidence(value, source):
    return {"value": value, "source": source}


def _vae_geometry(header, metadata):
    """Header match to existing Core/T8 contracts, never a successful decode claim."""
    def shape(key):
        return header.get(key, {}).get("shape")
    if ("encoder.down.5.block.0.conv1.weight" not in header
            or shape("latents_mean") != [24] or shape("latents_std") != [24]):
        return _evidence(None, "not_inferred_from_name")
    projection, bias = shape("decoder.proj_out.weight"), shape("decoder.proj_out.bias")
    if projection == [3072, 2048] and bias == [3072]:
        return _evidence({"latent_channels": 24, "relative_to_native_encode": 1,
                          "latent_to_rgb_spatial": 16}, "existing_Core_header_contract_not_decode_qualification")
    if projection == [12288, 2048] and bias == [12288]:
        try:
            adapter = json.loads(metadata.get("minimax_h3_x2_adapter", "null"))
            video = json.loads(metadata.get("minimax_h3_video_vae", "null"))
        except (TypeError, ValueError):
            return _evidence(None, "invalid_packed_VAE_metadata")
        if (isinstance(adapter, dict) and isinstance(video, dict)
                and adapter.get("version") == 1 and adapter.get("packed_output_channels") == 12
                and adapter.get("pixel_shuffle_factor") == 2 and adapter.get("encoder_unchanged") is True
                and video.get("decoder_pixel_shuffle_factor") == 2
                and video.get("packed_decoder_output_channels") == 12):
            return _evidence({"latent_channels": 24, "relative_to_native_encode": 2,
                              "encode_spatial": 16, "decode_spatial": 32,
                              "required_loader": "MiniMaxH3HyperVAE2xLoaderEXPT8"},
                             "existing_T8_packed_header_contract_not_decode_qualification")
    return _evidence(None, "no_complete_known_VAE_header_contract")


def recipe_from_header(header, metadata):
    """Use bounded header descriptors; never infer settings from a filename."""
    keys = sorted(header)
    ranks, incomplete = [], 0
    for key in keys:
        for suffix, partner in ((".lora_A.weight", ".lora_B.weight"),
                                (".lora_A.default.weight", ".lora_B.default.weight"),
                                (".lora_down.weight", ".lora_up.weight")):
            if not key.endswith(suffix):
                continue
            a = header[key].get("shape", [])
            b = header.get(key[:-len(suffix)] + partner, {}).get("shape", [])
            if (len(a) >= 2 and len(b) >= 2 and type(a[0]) is int and a[0] > 0
                    and type(b[1]) is int and a[0] == b[1]):
                ranks.append(a[0])
            else:
                incomplete += 1
    layouts = []
    # These are observed namespaces, not claims that their kernels ran.
    for marker, label in (("comfy_quant", "comfy_quant_descriptor"),
                          ("weight_scale", "weight_scale_tensor"),
                          ("convrot", "convrot_namespace"),
                          ("qweight", "packed_qweight_namespace")):
        if any(marker in key for key in keys):
            layouts.append(label)
    task = {key: metadata[key] for key in ("task_type", "task", "modelspec.task")
            if key in metadata}
    alpha = {key: metadata[key] for key in ("ss_network_alpha", "network_alpha", "lora_alpha")
             if key in metadata}
    scalar_alpha_keys = sum(key.endswith(".alpha") for key in keys)
    pruned = any("adaln_t_table" in key or "adaln_curve" in key for key in keys)
    clock = {key: metadata[key][:4096] for key in
             ("hyperflow_sigmas", "sigma_schedule", "video_sigmas", "audio_sigmas", "shift_video", "shift_audio")
             if isinstance(metadata.get(key), str)}
    return {"schema": SCHEMA, "scope": "header_only_not_execution",
        "task": _evidence(task or None, "metadata_declaration" if task else "unknown"),
        "structure": _evidence("pruned_basis_present" if pruned else "unknown",
                               "tensor_namespace" if pruned else "not_inferred_from_absence"),
        "storage_dtypes": _evidence(sorted({item.get("dtype") if isinstance(item.get("dtype"), str)
                                            else "unknown" for item in header.values()}),
                                    "tensor_descriptors_not_compute_precision"),
        "quantization_layouts": _evidence(layouts, "tensor_namespace_not_kernel_validation"),
        "training_rank": _evidence(sorted(set(ranks)) or None, "paired_factor_shapes"),
        "training_alpha": _evidence(alpha or None, "metadata_declaration" if alpha else "unknown"),
        # Some canonical adapters carry a generic `alpha` string rather than a
        # scoped network-alpha declaration. Show it without assigning a meaning
        # or changing the native parser's scalar tensor/rank normalization.
        "unscoped_alpha_metadata": _evidence(metadata.get("alpha"),
            "unscoped_metadata_not_training_or_runtime" if "alpha" in metadata else "unknown"),
        "alpha_scalar_tensor_count": scalar_alpha_keys,
        "paired_factor_count": len(ranks), "incomplete_factor_count": incomplete,
        "runtime_strength": _evidence(None, "not_selected_by_header"),
        "bridge_parameters": _evidence(None, "use_existing_content_bound_bridge_profile"),
        "sigma_and_clock": _evidence(clock or None, "metadata_declaration_not_runtime_schedule" if clock else "not_selected_by_header"),
        "vae_pixel_multiplier": _vae_geometry(header, metadata),
        "automatic_changes": False, "load_allowed_by_card": None,
        "warnings": ["训练alpha／rank不等于运行strength；Bridge alpha是另一套参数。",
                     "Header不证明任务、完整算法、画质或未知补丁兼容；原加载器继续负责实际加载。"]}


def _checked_settings(value):
    if type(value) is not dict or len(value) > 128:
        raise ValueError("Recipe settings must be a bounded object")
    for key, item in value.items():
        if not isinstance(key, str) or not 0 < len(key) <= 128:
            raise ValueError("Invalid recipe field name")
        if type(item) not in (str, bool, int, float, type(None)):
            raise ValueError("Recipe settings must contain explicit scalar values")
        if type(item) is float and not math.isfinite(item):
            raise ValueError("Recipe settings must be finite")
        if type(item) is str and len(item) > 4096:
            raise ValueError("Recipe setting exceeds the preview limit")
    return value


def preview_recipe_changes(current, proposed):
    """Diff an explicit recipe only. No model auto-pick, save, queue or mutation."""
    _checked_settings(current)
    _checked_settings(proposed)
    changes = []
    for name, after in proposed.items():
        present = name in current
        before = current.get(name)
        # Typed false must not compare equal to zero; presence is not null.
        if not present or type(before) is not type(after) or before != after:
            changes.append({"field": name, "before_present": present,
                            "before": before, "after": after})
    return {"schema": "t8.recipe-difference-preview.v1", "changes": changes,
            "current_snapshot": deepcopy(current), "proposed": deepcopy(proposed),
            "requires_explicit_confirmation": True, "applied": False,
            "saved": False, "queued": False}


def confirmed_recipe_settings(preview, current, *, confirmed=False):
    """Return a new settings object only after confirming a non-stale exact diff.

Not a canvas writer. The eventual UI must map supported fields explicitly and
preserve sockets, positional widget slots, unknown nodes and selection choices.
"""
    if confirmed is not True:
        raise ValueError("Explicit confirmation is required")
    expected = preview_recipe_changes(current, preview["proposed"])
    if (json.dumps(current, sort_keys=True, allow_nan=False) !=
            json.dumps(preview["current_snapshot"], sort_keys=True, allow_nan=False)
            or expected != preview):
        raise ValueError("Recipe preview is stale or modified; inspect a fresh diff")
    return {**deepcopy(current), **deepcopy(preview["proposed"])}


def inspect_installed_recipe(folder, name):
    """Resolve only a named, installed resource in Core's configured directories.

    Does not choose replacements or expose arbitrary filesystem paths. Bridge
    profiles reuse the existing content-bound implementation, not a new preset.
    """
    if folder not in {"diffusion_models", "clip", "vae", "loras", "semantic_bridge"}:
        raise ValueError("Unsupported model recipe resource category")
    if not isinstance(name, str) or not name or len(name) > 4096:
        raise ValueError("Select an explicit installed resource")
    if folder == "semantic_bridge":
        from .nodes_semantic_bridge import model_paths
        from .semantic_bridge_profiles import inspect_bridge_profile

        paths = model_paths()
        if name not in paths:
            raise FileNotFoundError("Selected Bridge is no longer installed")
        profile = inspect_bridge_profile(paths[name])
        return {"category": folder, "selection": name,
                "scope": "existing_content_bound_bridge_profile", "profile": profile,
                "automatic_changes": False, "saved": False, "queued": False}
    import folder_paths
    from .h3_weight_diagnostics import inspect_h3_weight_file

    if name not in folder_paths.get_filename_list(folder):
        raise FileNotFoundError("Selected resource is no longer installed")
    path = folder_paths.get_full_path(folder, name)
    if not path:
        raise FileNotFoundError("Selected resource is no longer installed")
    return {"category": folder, "selection": name, "inspection": inspect_h3_weight_file(path),
            "automatic_changes": False, "saved": False, "queued": False}
