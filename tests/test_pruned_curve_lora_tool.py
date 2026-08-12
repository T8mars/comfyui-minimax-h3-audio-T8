from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "convert_minimax_h3_turbo_for_pruned_curve.py"
SPEC = importlib.util.spec_from_file_location("h3_pruned_curve_lora_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_native_four_step_times_cover_video_audio_and_condition_rows():
    values = tool.native_four_step_times()
    expected = torch.tensor(
        [
            0.0,
            1.0 / 37.0,
            1.0 / 13.0,
            0.1,
            0.2,
            0.25,
            0.5,
            0.999,
            1.0,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(values, expected, rtol=1e-6, atol=1e-7)


def test_curve_interpolation_matches_comfyui_endpoint_contract():
    curve = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    times = torch.tensor([0.0, 0.125, 0.5, 0.875, 1.0])
    actual = tool.interpolate_curve(curve, times).flatten()
    torch.testing.assert_close(actual, torch.tensor([0.0, 0.5, 2.0, 3.5, 4.0]))


def test_affine_intercept_is_required_for_curve_projection():
    generator = torch.Generator().manual_seed(260810)
    curve = torch.randn(1025, 8, generator=generator, dtype=torch.float64)
    weights = torch.randn(8, 16, generator=generator, dtype=torch.float64)
    intercept = torch.linspace(-5.0, 5.0, 16, dtype=torch.float64)
    target = curve @ weights + intercept

    affine = torch.cat([curve, torch.ones(1025, 1, dtype=torch.float64)], dim=1)
    affine_fit = affine @ (torch.linalg.pinv(affine) @ target)
    linear_fit = curve @ (torch.linalg.pinv(curve) @ target)

    affine_error = torch.linalg.vector_norm(
        target - affine_fit
    ) / torch.linalg.vector_norm(target)
    linear_error = torch.linalg.vector_norm(
        target - linear_fit
    ) / torch.linalg.vector_norm(target)
    assert affine_error.item() < 1e-12
    assert linear_error.item() > 0.5


def test_output_error_formula_matches_explicit_materialization():
    generator = torch.Generator().manual_seed(42)
    target = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    predicted = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    b_matrix = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    diff_bias = torch.randn(5, generator=generator, dtype=torch.float64)

    numerator, denominator = tool.output_error_squared(
        target,
        predicted,
        b_matrix,
        diff_bias,
    )
    explicit_error = (target - predicted) @ b_matrix.T - diff_bias
    explicit_target = target @ b_matrix.T
    assert numerator == pytest.approx(
        torch.sum(explicit_error.square()).item(), rel=1e-12
    )
    assert denominator == pytest.approx(
        torch.sum(explicit_target.square()).item(), rel=1e-12
    )


def test_publish_safetensors_refuses_overwrite(tmp_path):
    final = tmp_path / "candidate.safetensors"
    state = {"tensor": torch.arange(8, dtype=torch.float32)}
    metadata = {"kind": "test"}

    output_hash = tool.publish_safetensors_no_overwrite(
        final,
        state,
        metadata,
        lambda loaded: torch.testing.assert_close(loaded["tensor"], state["tensor"]),
    )
    assert output_hash == tool.sha256_file(final)
    torch.testing.assert_close(load_file(final)["tensor"], state["tensor"])
    assert list(tmp_path.glob("*.partial")) == []

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        tool.publish_safetensors_no_overwrite(
            final, state, metadata, lambda loaded: None
        )


def test_projected_metadata_drops_inherited_pruned_incompatibility():
    args = SimpleNamespace(
        lora=Path("source.safetensors"),
        pruned_model=Path("target_B.safetensors"),
        time_embedder_reference=Path("reference.safetensors"),
    )
    hashes = {
        "lora": "1" * 64,
        "pruned_model": "2" * 64,
        "time_reference": "3" * 64,
    }
    source_metadata = {
        "incompatible_base": "MiniMax-H3 pruned_* (AdaLN input is 8)",
        "format": "pt",
    }
    summary = tool.ProjectionSummary(*([0.0] * 9))

    curve = tool.metadata_for_curve(
        source_metadata,
        args,
        hashes,
        "4" * 64,
        summary,
        "test-commit",
    )
    core = tool.metadata_for_core(
        source_metadata,
        args,
        hashes,
        "4" * 64,
        "test-commit",
    )

    for metadata in (curve, core):
        assert "incompatible_base" not in metadata
        assert metadata["compatibility_scope"] == "exact_checkpoint_sha256_only"
        assert metadata["compatible_main_sha256"] == hashes["pruned_model"]
