import copy
import json
import struct

import pytest

from h3_audio_t8_pkg.model_recipe_cards import (
    confirmed_recipe_settings, inspect_installed_recipe, preview_recipe_changes, recipe_from_header,
)
from h3_audio_t8_pkg.h3_weight_diagnostics import inspect_h3_weight_file


def test_actual_pair_shape_rank_and_declared_alpha_are_not_runtime_strength(tmp_path):
    header = {"__metadata__": {"ss_network_alpha": "16", "task_type": "Ref2VA"},
              "a.lora_A.weight": {"shape": [4, 8], "dtype": "F16", "data_offsets": [0, 64]},
              "a.lora_B.weight": {"shape": [8, 4], "dtype": "F16", "data_offsets": [64, 128]},
              "a.alpha": {"shape": [], "dtype": "F32", "data_offsets": [128, 132]}}
    raw = json.dumps(header).encode()
    path = tmp_path / "Misleading-FL2VA-Alpha1-Full-FP8.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(132))
    result = inspect_h3_weight_file(path)
    card = result["recipe_card"]
    assert card["training_rank"]["value"] == [4]
    assert card["training_alpha"] == {"value": {"ss_network_alpha": "16"}, "source": "metadata_declaration"}
    assert card["task"]["value"] == {"task_type": "Ref2VA"}
    assert card["runtime_strength"]["value"] is None
    assert card["structure"]["value"] == "unknown"
    assert card["storage_dtypes"]["value"] == ["F16", "F32"]
    assert result["bytes_read"] == 8 + len(raw)
    assert result["tensor_values_loaded"] is False


def test_unknown_and_malformed_pair_remain_diagnostic_not_a_load_blacklist():
    card = recipe_from_header({"x.lora_A.weight": {"shape": [4, 8]},
                               "x.lora_B.weight": {"shape": [8, 2]},
                               "opaque_hook.weight": {"shape": [1]}}, {})
    assert card["training_rank"]["value"] is None
    assert card["incomplete_factor_count"] == 1
    assert card["load_allowed_by_card"] is None and card["automatic_changes"] is False


def test_pruned_basis_and_quant_namespace_are_observations_not_quality():
    card = recipe_from_header({"adaln_t_table": {"dtype": "F32", "shape": [2, 8]},
                               "layer.qweight": {"dtype": "I32", "shape": [2, 2]}}, {})
    assert card["structure"]["value"] == "pruned_basis_present"
    assert card["quantization_layouts"]["value"] == ["packed_qweight_namespace"]
    assert card["vae_pixel_multiplier"]["value"] is None
    assert card["sigma_and_clock"]["value"] is None


def test_vae_geometry_requires_shapes_and_existing_packed_contract_not_a_name():
    header = {"encoder.down.5.block.0.conv1.weight": {}, "latents_mean": {"shape": [24]},
              "latents_std": {"shape": [24]}, "decoder.proj_out.weight": {"shape": [3072, 2048]},
              "decoder.proj_out.bias": {"shape": [3072]}}
    geometry = recipe_from_header(header, {})["vae_pixel_multiplier"]
    assert geometry["value"]["relative_to_native_encode"] == 1
    assert geometry["source"].endswith("not_decode_qualification")
    header["decoder.proj_out.weight"]["shape"] = [12288, 2048]
    header["decoder.proj_out.bias"]["shape"] = [12288]
    assert recipe_from_header(header, {})["vae_pixel_multiplier"]["value"] is None
    meta = {"minimax_h3_x2_adapter": json.dumps({"version": 1, "packed_output_channels": 12,
            "pixel_shuffle_factor": 2, "encoder_unchanged": True}),
            "minimax_h3_video_vae": json.dumps({"decoder_pixel_shuffle_factor": 2, "packed_decoder_output_channels": 12})}
    assert recipe_from_header(header, meta)["vae_pixel_multiplier"]["value"]["required_loader"] == "MiniMaxH3HyperVAE2xLoaderEXPT8"
    del header["latents_std"]
    assert recipe_from_header(header, meta)["vae_pixel_multiplier"]["value"] is None


def test_sigma_declaration_is_not_selected_runtime_clock():
    value = recipe_from_header({}, {"hyperflow_sigmas": "[1,.8,.5,0]"})["sigma_and_clock"]
    assert value == {"value": {"hyperflow_sigmas": "[1,.8,.5,0]"}, "source": "metadata_declaration_not_runtime_schedule"}


def test_preview_typed_false_null_presence_and_no_mutation():
    current = {"enabled": 0, "ordinal": 2, "unchanged": "用户设置"}
    before = copy.deepcopy(current)
    preview = preview_recipe_changes(current, {"enabled": False, "new": None})
    assert current == before and preview["applied"] is False
    assert [r["field"] for r in preview["changes"]] == ["enabled", "new"]
    assert preview["changes"][1]["before_present"] is False
    assert preview["saved"] is preview["queued"] is False
    with pytest.raises(ValueError, match="confirmation"):
        confirmed_recipe_settings(preview, current, confirmed=1)
    result = confirmed_recipe_settings(preview, current, confirmed=True)
    assert result == {"enabled": False, "ordinal": 2, "unchanged": "用户设置", "new": None}
    assert current == before


def test_stale_and_tampered_diff_never_applies():
    current = {"alpha": .1, "chunk_tokens": 256}
    preview = preview_recipe_changes(current, {"alpha": 1., "chunk_tokens": 0})
    with pytest.raises(ValueError, match="stale"):
        confirmed_recipe_settings(preview, {**current, "alpha": .12}, confirmed=True)
    preview["changes"][0]["after"] = 100
    with pytest.raises(ValueError, match="modified"):
        confirmed_recipe_settings(preview, current, confirmed=True)


@pytest.mark.parametrize("bad", [{"x": float("nan")}, {"x": []}, {"x": "x" * 4097}, {1: False}])
def test_preview_rejects_unbounded_or_non_scalar_settings(bad):
    with pytest.raises(ValueError):
        preview_recipe_changes({}, bad)


def test_only_configured_explicit_resources_are_inspected(tmp_path, monkeypatch):
    import folder_paths

    raw = json.dumps({"x": {"shape": [1], "dtype": "F32", "data_offsets": [0, 4]}}).encode()
    path = tmp_path / "asset.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4))
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda category: ["nested/asset.safetensors"])
    monkeypatch.setattr(folder_paths, "get_full_path", lambda category, name: str(path))
    result = inspect_installed_recipe("loras", "nested/asset.safetensors")
    assert result["inspection"]["status"] == "inspected"
    assert result["automatic_changes"] is result["queued"] is False
    with pytest.raises(FileNotFoundError):
        inspect_installed_recipe("loras", str(path))
    with pytest.raises(FileNotFoundError):
        inspect_installed_recipe("loras", "auto")
    with pytest.raises(ValueError):
        inspect_installed_recipe("input", "nested/asset.safetensors")
