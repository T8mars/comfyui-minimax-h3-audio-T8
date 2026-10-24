"""Observed Core CPU INT8 execution and module-owned processing contracts.

No Kitchen kernel, quantized tensor dispatch, or Core forward is stubbed.
An explicitly configured runtime pre_quant_scale is not evidence that this
Core's INT8 checkpoint schema accepts AWQ metadata, nor GPU/AWQ qualification.
"""
from __future__ import annotations

from copy import deepcopy
import json

import comfy.lora
from comfy import ops
import comfy.utils
import comfy_kitchen as kitchen
from comfy_kitchen.tensor import QuantizedTensor
import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.h3_weight_diagnostics import lora_mapping_diagnostics


def int8_module(tmp_path, *, weight=None, bias=True):
    generator = torch.Generator(device="cpu").manual_seed(2121)
    if weight is None:
        weight = torch.randn((32, 32), generator=generator)
    weight = weight.detach().bfloat16()
    quantized = QuantizedTensor.from_float(
        weight, "TensorWiseINT8Layout", is_weight=True, per_channel=True,
        stochastic_rounding=0,
    )
    payload = {
        "control.weight": quantized._qdata,
        "control.weight_scale": quantized.params.scale,
        "control.comfy_quant": torch.tensor(
            list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8,
        ),
    }
    if bias:
        payload["control.bias"] = torch.zeros(weight.shape[0], dtype=torch.bfloat16)
    path = tmp_path / "actual_small_int8.safetensors"
    save_file(payload, str(path))
    loaded, _ = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    quantization = comfy.utils.detect_layer_quantization(loaded, "")
    module = torch.nn.Module()
    module.control = ops.mixed_precision_ops(quantization, torch.bfloat16).Linear(
        weight.shape[1], weight.shape[0], bias=bias, device="cpu",
    )
    result = module.load_state_dict(loaded, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert isinstance(module.control.weight, QuantizedTensor)
    assert module.control.weight._qdata.dtype == torch.int8
    assert module.control.quant_format == "int8_tensorwise"
    assert not module.control._full_precision_mm
    return module, payload


def real_int8_linear(layer, inputs):
    return kitchen.int8_linear(
        inputs, layer.weight._qdata, layer.weight.params.scale, bias=layer.bias,
        out_dtype=inputs.dtype,
    )


def test_actual_int8_payload_load_cpu_forward_and_unscaled_fusion(tmp_path):
    module, _ = int8_module(tmp_path)
    generator = torch.Generator(device="cpu").manual_seed(2122)
    x = torch.randn((8, 64), generator=generator).bfloat16()
    activated = ops._eager_input_act(x, "swiglu")
    with torch.inference_mode():
        actual = module.control(activated)
        expected = real_int8_linear(module.control, activated)
        fused = ops.linear_input_act(module.control, x, "swiglu")
    assert torch.isfinite(actual).all() and actual.shape == (8, 32)
    assert torch.equal(actual, expected)
    assert torch.equal(fused, actual)
    assert not torch.cuda.is_initialized()


def test_int8_checkpoint_extra_scale_is_checked_against_actual_core_schema(tmp_path):
    module, payload = int8_module(tmp_path)
    scale = torch.linspace(.25, 1.75, 32)
    payload["control.pre_quant_scale"] = scale
    if "pre_quant_scale" in ops.QUANT_ALGOS["int8_tensorwise"]["parameters"]:
        result = module.load_state_dict(payload, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert torch.equal(module.control.pre_quant_scale, scale)
    else:
        with pytest.raises(RuntimeError, match=r"Unexpected key.*control.pre_quant_scale"):
            module.load_state_dict(payload, strict=True)
        assert not hasattr(module.control, "pre_quant_scale")
    assert not torch.cuda.is_initialized()


def test_actual_fused_owner_bypass_and_explicit_force_forward_preserve_runtime_scale(tmp_path):
    module, _ = int8_module(tmp_path)
    layer = module.control
    # This models an explicitly configured module-owned preprocessing field;
    # it does not silently accept unsupported checkpoint keys.
    scale = torch.linspace(.25, 1.75, 32)
    layer.register_parameter("pre_quant_scale", torch.nn.Parameter(scale, requires_grad=False))
    generator = torch.Generator(device="cpu").manual_seed(2123)
    x = torch.randn((8, 64), generator=generator).bfloat16()
    activated = ops._eager_input_act(x, "swiglu")
    calls = []
    handle = layer.register_forward_pre_hook(lambda *_: calls.append("actual_module_forward"))
    original_raw = layer.weight._qdata.clone()
    original_scale = layer.weight.params.scale.clone()
    try:
        with torch.inference_mode():
            direct = layer(activated)
            correct = real_int8_linear(layer, activated * scale.to(activated.dtype))
            calls.clear()
            fused = ops.linear_input_act(layer, x, "swiglu")
            # This is an observed upstream boundary/counterexample, not a
            # requirement that a future corrected Core keep bypassing it.
            assert not calls
            assert torch.equal(fused, real_int8_linear(layer, activated))
            assert not torch.equal(fused, direct)
            layer.comfy_force_forward = True
            calls.clear()
            forced = ops.linear_input_act(layer, x, "swiglu")
            assert calls == ["actual_module_forward"]
            assert torch.equal(forced, direct)
            assert torch.equal(direct, correct)
    finally:
        if "comfy_force_forward" in layer.__dict__:
            del layer.comfy_force_forward
        handle.remove()
    assert torch.equal(layer.weight._qdata, original_raw)
    assert torch.equal(layer.weight.params.scale, original_scale)
    assert torch.equal(layer.pre_quant_scale, scale)
    assert not layer._forward_pre_hooks
    assert not torch.cuda.is_initialized()


def test_real_h3_int8_projection_without_adaln_is_partial_not_full_algorithm(tmp_path):
    from test_token_lora import fixture_args, model, native

    source = model()
    block = source.model.diffusion_model.blocks[0]
    loaded, _ = int8_module(tmp_path, weight=block.mlp.fc2.weight, bias=False)
    block.mlp.fc2 = loaded.control
    projection = "diffusion_model.blocks.0.mlp.fc2.weight"
    modulation = "diffusion_model.blocks.0.adaln_proj.linear.weight"
    shapes = {key: tuple(source.model.state_dict()[key].shape) for key in (projection, modulation)}
    payload, key_map = {}, {}
    for index, (key, (rows, width)) in enumerate(shapes.items()):
        alias = key.removesuffix(".weight")
        key_map[alias] = key
        payload[alias + ".lora_A.weight"] = torch.linspace(.03, .09, width).reshape(1, width)
        payload[alias + ".lora_B.weight"] = torch.full((rows, 1), .04 * (index + 1))
        payload[alias + ".alpha"] = torch.tensor(1.)
    patches = comfy.lora.load_lora(payload, key_map, log_missing=False)
    assert set(patches) == {projection, modulation}
    # Model an omission before parsing, rather than slicing a full parser
    # result: Core adapters can share that parser invocation's loaded-key set.
    projection_payload = {key: value for key, value in payload.items()
                          if key.startswith(projection.removesuffix(".weight") + ".")}
    partial_patches = comfy.lora.load_lora(projection_payload, key_map, log_missing=False)
    assert set(partial_patches) == {projection}
    full, partial = source.clone(), source.clone()
    assert set(full.add_patches(patches, strength_patch=1.)) == set(patches)
    assert partial.add_patches(partial_patches, strength_patch=1.) == [projection]
    before = {key: value.clone() for key, value in block.state_dict().items()}
    args = fixture_args()

    def execute(patcher):
        try:
            for target in patcher.patches:
                patcher.patch_weight_to_device(target, device_to=torch.device("cpu"))
            with torch.inference_mode():
                return native(block, deepcopy(args))["img"].clone()
        finally:
            patcher.unpatch_model()

    incomplete, complete = execute(partial), execute(full)
    assert torch.isfinite(incomplete).all() and torch.isfinite(complete).all()
    assert not torch.allclose(incomplete, complete, rtol=1e-6, atol=1e-7)
    # A parser recognizing one part of a selected payload is not full coverage.
    report = lora_mapping_diagnostics(payload, key_map, partial_patches, [projection])
    assert report["registration_is_execution"] is False
    assert modulation.removesuffix(".weight") + ".lora_A.weight" in report["converted_unmapped_keys"]
    for key, value in block.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert not source.backup and not source.patches
    assert not torch.cuda.is_initialized()
