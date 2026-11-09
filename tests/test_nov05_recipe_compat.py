"""Native baked PDD header compatibility is separate from T8 adapter fallback."""
import inspect
import json

import comfy.model_detection
import comfy.utils
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from safetensors.torch import save_file
import pytest
import torch

from test_h21_pdd_lifecycle import baked_file, forward, load_cpu
from h3_audio_t8_pkg.h3_weight_diagnostics import inspect_h3_weight_file
from h3_audio_t8_pkg.pdd_advanced import pdd_runtime_sigmas


def test_pdd_execution_does_not_depend_on_the_separate_native_bypass_source_catalog(monkeypatch):
    from h3_audio_t8_pkg.modular_sampling import core_bypass_identity, results
    from test_modular_pdd_dynamic import stage_inputs
    # This catalog certifies the separate stock-Core bypass cache adapter. PDD
    # has its own authenticated closures and content-bound implementation.
    monkeypatch.setattr(core_bypass_identity, "_SOURCE_PINS", {
        name: set() for name in core_bypass_identity._SOURCE_PINS})
    args, _, _ = stage_inputs()
    result = results.sample_stage(*args)[2]
    assert result.verify()["execution"]["denoiser_evaluations"] == 4


def dynamic_hook(selected):
    from h3_audio_t8_pkg import pdd_advanced as pdd
    from h3_audio_t8_pkg.modular_sampling import pdd_dynamic as identity
    runtime, _ = identity.injection(selected.get_injections(pdd.PDD_INJECTION_KEY)[0],
                                    pdd._create_pdd_runtime_injection)
    backbone, _ = identity.injection(runtime["backbone_injection"], pdd._create_offloading_bypass_injections)
    manager = identity.closure(backbone["native_injection"].inject,
        identity.BypassInjectionManager.create_injections, "inject_all")["self"]
    return manager.hooks[0]


@pytest.mark.parametrize("damage", ["backup_type", "active_flag", "forward_function", "forward_owner", "extra_callable"])
def test_pdd_force_forward_projection_still_rejects_actual_owner_mutations(damage):
    from types import MethodType
    from h3_audio_t8_pkg.modular_sampling import pdd_dynamic as identity
    from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
    from test_modular_pdd_dynamic import stage_inputs
    args, _, _ = stage_inputs()
    selected = args[1].model_patcher
    before = identity.describe(selected)
    selected.inject_model()
    hook = dynamic_hook(selected)
    current = hook.module.forward
    try:
        view, after = identity.project(selected)
        assert before == after and view.model is not selected.model
        assert hook.module.forward.__func__ is current.__func__
        assert hook.module.forward.__self__ is current.__self__
        assert hook.module.comfy_force_forward is True
        if damage == "backup_type":
            hook.original_force_forward = 1
        elif damage == "active_flag":
            hook.module.comfy_force_forward = False
        elif damage == "forward_function":
            hook.module.forward = MethodType(lambda self, *a, **kw: None, hook)
        elif damage == "forward_owner":
            hook.module.forward = MethodType(current.__func__, object())
        else:
            hook.user_callback = lambda: None
        with pytest.raises(UnverifiedModelStack):
            identity.project(selected)
    finally:
        hook.module.forward = current
        hook.original_force_forward = False
        hook.module.comfy_force_forward = True
        vars(hook).pop("user_callback", None)
        selected.eject_model()
    assert identity.describe(selected) == before


def test_native_baked_pdd_minimal_vs_explicit_channel_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    minimal, _, baked = baked_file(tmp_path)
    state = comfy.utils.load_torch_file(str(minimal), safe_load=True)
    minimal_metadata = {"config": json.dumps({"transformer": {"pdd_steps": 32}})}
    implicit = comfy.model_detection.detect_unet_config(state, "", metadata=minimal_metadata)
    expected_channels = {"latents_dim": 24, "audio_latents_dim": 32}
    recognizes_banks = all(implicit[key] == value for key, value in expected_channels.items())
    if recognizes_banks:
        loaded_minimal = load_cpu(minimal)
        with torch.inference_mode():
            minimal_output = forward(loaded_minimal.model.diffusion_model, pdd_runtime_sigmas())
        assert all(bool(torch.isfinite(x).all()) for x in minimal_output)
        diagnostic = "minimal_metadata_native_loader_passed"
    else:
        with pytest.raises(RuntimeError, match="size mismatch for video_patch_proj.weight"):
            load_cpu(minimal)
        diagnostic = "minimal_metadata_misinfers_stacked_head_channels"

    # Explicit channels diagnose the second distinct requirement: construction
    # of stacked output heads, not merely channels. Never rewrite user files.
    explicit = tmp_path / "explicit-channels.safetensors"
    save_file({key: value.contiguous() for key, value in baked.state_dict().items()}, str(explicit),
        metadata={"config": json.dumps({"transformer": {"pdd_steps": 32, **expected_channels}})})
    constructs_banks = "pdd_steps" in inspect.signature(native.MiniMaxH3Model).parameters
    if constructs_banks:
        loaded = load_cpu(explicit)
        with torch.inference_mode():
            output = forward(loaded.model.diffusion_model, pdd_runtime_sigmas())
            expected = forward(baked, pdd_runtime_sigmas())
        for actual, reference in zip(output, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        explicit_result = "passed"
    else:
        with pytest.raises(RuntimeError, match="size mismatch for final_layer.video_out.weight"):
            load_cpu(explicit)
        explicit_result = "missing_native_stacked_head_constructor"
    header = inspect_h3_weight_file(minimal)
    assert header["compatibility"] == "not_established_by_header"
    (tmp_path / "pdd-native-metadata.json").write_text(json.dumps({
        "minimal_header_diagnostic": diagnostic,
        "inferred_dimensions": {key: implicit[key] for key in expected_channels},
        "explicit_dimensions_native_loader_forward": explicit_result,
        "minimal_file_rewritten": False, "installed_core_modified": False,
        "trained_model": False, "cuda_initialized": torch.cuda.is_initialized()}, indent=2), encoding="utf-8")
    assert not torch.cuda.is_initialized()
