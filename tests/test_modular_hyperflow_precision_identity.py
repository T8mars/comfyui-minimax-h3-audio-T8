"""Endpoint source dtype versus its actual FP32 device cache stay distinct."""
import pytest
import torch

from h3_audio_t8_pkg import hyperflow_runtime_advanced as runtime
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages, hyperflow_identity as identity
from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
from test_progressive_sampling_runtime import tiny_model, conditioning
from test_modular_hyperflow import inputs, head


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_original_adapter_dtype_and_warm_fp32_endpoint_are_authenticated(monkeypatch, dtype):
    import comfy.nested_tensor
    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    weights = _tiny_weights(diffusion)
    for mapping in (weights.patches, weights.endpoint):
        for key, (a, b, alpha) in mapping.items():
            mapping[key] = (a.to(dtype), b.to(dtype), alpha)
    low, _, _ = runtime.install_hyperflow(base, weights)
    high, _, _ = runtime.install_hyperflow(base, weights)
    before = identity.model_identity(low)
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 8)))}
    positive = conditioning()
    boundary = head(low, source, positive)
    assert boundary.verify()["execution"]["portable_identity"] is True
    assert identity.model_identity(low) == before
    result, _ = stages.sample_tail_result(boundary, high, positive, [], seed=7)
    assert result.verify()["sampling"]["execution"]["portable_identity"] is True


def test_selected_base_audio_protocol_cannot_be_omitted(monkeypatch):
    low, _, _, _ = inputs(monkeypatch)
    before = stages.model_selection(low)
    monkeypatch.setattr(low.model, "audio_scale", lambda *a, **kw: 0.)
    after = stages.model_selection(low)
    assert after["portable_cache_reuse"] is False
    assert before != after


def _foreign_forward(self, *args, **kwargs):
    return args[0]


def _foreign_audio_scale(self):
    return 1.


@pytest.mark.parametrize("target", ["class_forward", "class_audio"])
def test_foreign_class_callable_is_not_authenticated_as_native(monkeypatch, target):
    low, _, _, _ = inputs(monkeypatch)
    if target == "class_forward":
        kind = type(low.model.diffusion_model.blocks[0])
        monkeypatch.setattr(kind, "forward", _foreign_forward)
    else:
        kind = type(low.model)
        monkeypatch.setattr(kind, "audio_scale", _foreign_audio_scale)
    # Install AFTER the foreign class method exists: object-delegate equality
    # alone would otherwise mistake this selected foreign method for native.
    low, _, _, _ = inputs(monkeypatch)
    if target == "class_forward":
        assert low.object_patches["diffusion_model.blocks.0.forward"].__kwdefaults__["_inner"].__func__ is _foreign_forward
    selected = stages.model_selection(low)
    assert selected["portable_cache_reuse"] is False
    # Classification must not replace the user's actual selected callable.
    assert getattr(kind, "forward" if target == "class_forward" else "audio_scale") is (
        _foreign_forward if target == "class_forward" else _foreign_audio_scale)
