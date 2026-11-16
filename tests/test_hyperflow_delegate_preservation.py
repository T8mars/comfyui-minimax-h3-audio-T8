"""A selected upstream forward must remain the real two-time inner delegate."""

import pytest
import torch
from copy import deepcopy

from h3_audio_t8_pkg.hyperflow_runtime_advanced import (
    install_hyperflow,
    HyperFlowStep,
    push_step,
    pop_step,
)
from h3_audio_t8_pkg.hyperflow_sampling_advanced import shifted_grid
from h3_audio_t8_pkg.hyperflow_weights_advanced import DEFAULT_RAW_GRID
from test_hyperflow_advanced import _tiny_native_model, _tiny_weights


def run_forward(selected, binding, network):
    from comfy.ldm.minimax.model import PackedLayout

    grid = shifted_grid(DEFAULT_RAW_GRID, 12.0)
    selected.patch_model()
    token = push_step(HyperFlowStep(binding.owner, 0, float(grid[0]), float(grid[1])))
    try:
        output = network(
            [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)],
            torch.tensor([1000.0]),
            torch.randn(1, 2, 8),
            transformer_options={
                "sample_sigmas": grid,
                "wrappers": selected.wrappers,
                "minimax_h3_sigma_shift_video": 12.0,
                "minimax_h3_sigma_shift_audio": 3.0,
            },
            minimax_payload={"layout": PackedLayout(2, 2, 4, 4, 3)},
        )
        assert all(torch.isfinite(v).all() for v in output)
    finally:
        pop_step(token)
        selected.unpatch_model()


@pytest.mark.parametrize(
    "path", ["blocks.0.forward", "time_embedder.forward", "final_layer.forward"]
)
def test_actual_full_Core_forward_calls_selected_upstream_delegate(monkeypatch, path):
    base, network = _tiny_native_model(monkeypatch)
    key = "diffusion_model." + path
    original = base.get_model_object(key)
    calls = []

    def user_forward(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    base.add_object_patch(key, user_forward)
    selected, binding, _ = install_hyperflow(base, _tiny_weights(network))
    grid = shifted_grid(DEFAULT_RAW_GRID, 12.0)
    selected.patch_model()
    token = push_step(HyperFlowStep(binding.owner, 0, float(grid[0]), float(grid[1])))
    try:
        from comfy.ldm.minimax.model import PackedLayout

        with torch.no_grad():
            output = network(
                [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)],
                torch.tensor([1000.0]),
                torch.randn(1, 2, 8),
                transformer_options={
                    "sample_sigmas": grid,
                    "wrappers": selected.wrappers,
                    "minimax_h3_sigma_shift_video": 12.0,
                    "minimax_h3_sigma_shift_audio": 3.0,
                },
                minimax_payload={"layout": PackedLayout(2, 2, 4, 4, 3)},
            )
        assert all(torch.isfinite(v).all() for v in output)
    finally:
        pop_step(token)
        selected.unpatch_model()
    assert calls == [True]


@pytest.mark.parametrize("path", ["blocks.0", "time_embedder", "final_layer"])
def test_parent_replacement_keeps_its_selected_forward_and_endpoint_weights(
    monkeypatch, path
):
    base, network = _tiny_native_model(monkeypatch)
    replacement = deepcopy(base.get_model_object("diffusion_model." + path))
    calls = []
    original = replacement.forward

    def user_forward(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    replacement.forward = user_forward
    base.add_object_patch("diffusion_model." + path, replacement)
    if path == "time_embedder":
        # Endpoint snapshots must come from this selected parent, not the
        # unpatched module that Core will replace just before sampling.
        from h3_audio_t8_pkg.hyperflow_runtime_advanced import _weights_snapshot

        replacement.proj_in.weight.data.fill_(0.123)
        assert torch.equal(_weights_snapshot(base)[0], replacement.proj_in.weight)
        assert not torch.equal(
            _weights_snapshot(base)[0], network.time_embedder.proj_in.weight
        )
    selected, binding, _ = install_hyperflow(base, _tiny_weights(network))
    run_forward(selected, binding, network)
    assert calls == [True]
    assert base.get_model_object("diffusion_model." + path) is replacement


def test_compatible_extra_block_executes_without_model_count_gate(monkeypatch, caplog):
    base, network = _tiny_native_model(monkeypatch)
    weights = _tiny_weights(network)
    network.blocks.append(deepcopy(network.blocks[-1]))
    selected, binding, _ = install_hyperflow(base, weights)
    run_forward(selected, binding, network)
    assert "selected 51/2" in caplog.text


def test_selected_parent_actual_shape_error_remains_precise(monkeypatch):
    base, network = _tiny_native_model(monkeypatch)
    replacement = torch.nn.Linear(8, 7)
    base.add_object_patch("diffusion_model.blocks.0.mlp.fc2", replacement)
    with pytest.raises(ValueError, match="shape mismatch at blocks.0.mlp.fc2"):
        install_hyperflow(base, _tiny_weights(network))
