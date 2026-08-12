from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "bind_lightx2v_lora_to_pruned_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("h3_lightx2v_bind_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_expected_lightx2v_module_contract():
    assert len(tool.EXPECTED_MODULES) == 208
    assert sum(
        module.endswith(".attn.qkv_proj") for module in tool.EXPECTED_MODULES
    ) == 52
    qkv = tool.EXPECTED_MODULES["blocks.0.attn.qkv_proj"]
    assert qkv == ((384, 5376), (21504, 384))
    assert tool.EXPECTED_MODULES["blocks.0.mlp.fc1"] == (
        (128, 5376),
        (28672, 128),
    )


@pytest.mark.parametrize(
    ("profile", "steps", "regular_alpha", "qkv_alpha", "scale"),
    [
        (tool.LEGACY_ALPHA8_PROFILE, 4, 8.0, 24.0, 0.0625),
        (tool.V1_4STEP_768P_PROFILE, 4, 128.0, 384.0, 1.0),
        (tool.V1_8STEP_PROFILE, 8, 8.0, 24.0, 0.0625),
    ],
)
def test_supported_source_profiles_are_strict(
    profile, steps, regular_alpha, qkv_alpha, scale
):
    detected = tool.identify_source_profile(profile.required_metadata, steps)
    assert detected is profile
    assert tool.expected_alpha("blocks.0.attn.qkv_proj", detected) == qkv_alpha
    assert tool.expected_alpha("blocks.0.attn.out_proj", detected) == regular_alpha
    assert detected.effective_alpha_over_rank == scale


def test_v1_source_profile_rejects_wrong_inference_steps():
    with pytest.raises(ValueError, match="requires 8 inference steps"):
        tool.identify_source_profile(
            tool.V1_8STEP_PROFILE.required_metadata,
            4,
        )


def test_unknown_source_metadata_is_rejected():
    metadata = dict(tool.V1_4STEP_768P_PROFILE.required_metadata)
    metadata["training_scale"] = "0.5"
    with pytest.raises(ValueError, match="does not match exactly one"):
        tool.identify_source_profile(metadata, 4)


def test_metadata_binding_preserves_conversion_provenance():
    original = {
        "conversion_source_file": "original_split_qkv.safetensors",
        "conversion_source_sha256": "0" * 64,
        "conversion_tool": "official_converter.py",
        "compatible_base": "generic",
    }
    metadata = tool.metadata_for_binding(
        original,
        Path("converted_alpha8.safetensors"),
        Path("target_pruned.safetensors"),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "commit",
        tool.V1_4STEP_768P_PROFILE,
    )
    assert metadata["conversion_source_file"] == "original_split_qkv.safetensors"
    assert metadata["conversion_tool"] == "official_converter.py"
    assert metadata["binding_source_file"] == "converted_alpha8.safetensors"
    assert metadata["binding_source_sha256"] == "1" * 64
    assert metadata["compatible_main_sha256"] == "2" * 64
    assert metadata["adaln_projection"].startswith("not_applicable")
    assert metadata["binding_source_profile"] == (
        "lightx2v_v1.0_4step_768p_official_comfyui_bf16"
    )
    assert metadata["recommended_inference_steps"] == "4"
    assert metadata["effective_lora_scale"] == "1.0"


def test_repack_preserves_complete_tensor_data_payload(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "bound.safetensors"
    state = {
        "a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "b": torch.arange(7, dtype=torch.bfloat16),
    }
    save_file(state, source, metadata={"kind": "source"})
    header_length, tensors, _ = tool.read_header(source)
    expected_payload_hash, expected_payload_bytes = tool.data_payload_sha256(
        source, header_length
    )

    partial, payload_hash, payload_bytes, _ = tool.create_repacked_partial(
        source,
        output,
        tensors,
        {"kind": "bound", "source": "source.safetensors"},
        header_length,
    )
    assert payload_hash == expected_payload_hash
    assert payload_bytes == expected_payload_bytes
    tool.publish_partial_no_overwrite(partial, output)

    output_header_length, output_tensors, output_metadata = tool.read_header(output)
    output_payload_hash, output_payload_bytes = tool.data_payload_sha256(
        output, output_header_length
    )
    assert output_tensors == tensors
    assert output_metadata == {
        "kind": "bound",
        "source": "source.safetensors",
    }
    assert output_payload_hash == expected_payload_hash
    assert output_payload_bytes == expected_payload_bytes
    with safe_open(str(output), framework="pt", device="cpu") as handle:
        torch.testing.assert_close(handle.get_tensor("a"), state["a"])
        torch.testing.assert_close(handle.get_tensor("b"), state["b"])

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        tool.create_repacked_partial(
            source,
            output,
            tensors,
            {"kind": "again"},
            header_length,
        )
    assert list(tmp_path.glob("*.partial")) == []
