import pytest
import torch
from comfy_extras.nodes_custom_sampler import (
    BasicGuider,
    RandomNoise,
    SamplerCustomAdvanced,
)

from h3_audio_t8_pkg.lora_windows import create_window_hook, window_keyframes
from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent
from test_stage_reuse import _write_adapter


def run(selected, cond):
    source = latent()
    selected, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, 4, 12.0, 3.0
    )
    guider = BasicGuider.execute(selected, cond).result[0]
    return SamplerCustomAdvanced.execute(
        RandomNoise.execute(71).result[0], guider, sampler, sigmas, source
    ).result[0]


def test_full_window_matches_actual_static_lora_and_restores_base(tmp_path):
    path = tmp_path / "one.safetensors"
    _write_adapter(path, 12)
    selected = model()
    original = {k: v.clone() for k, v in selected.model_state_dict().items()}
    static, _ = load_minimax_h3_lora_model(selected, path, 0.7)
    expected = run(static, conditioning())
    hooks, report = create_window_hook(selected, path, 0.7)
    cond = conditioning()
    cond[0][1]["hooks"] = hooks
    actual = run(selected, cond)
    for a, b in zip(actual["samples"].unbind(), expected["samples"].unbind()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    assert (
        report["mapped_targets"] == 1 and report["registration_is_execution"] is False
    )
    for k, v in selected.model_state_dict().items():
        assert torch.equal(v, original[k])


def test_window_actually_switches_patch_with_no_guaranteed_extra_step(tmp_path):
    path = tmp_path / "one.safetensors"
    _write_adapter(path, 13)
    selected = model()
    module = selected.model.diffusion_model.blocks[0].attn.qkv_proj
    original = module.weight.detach().clone()
    observed = []
    handle = module.register_forward_pre_hook(
        lambda layer, _: observed.append(float((layer.weight - original).abs().max()))
    )
    hooks, _ = create_window_hook(
        selected, path, 0.9, start_percent=0.0, end_percent=0.25
    )
    cond = conditioning()
    cond[0][1]["hooks"] = hooks
    try:
        output = run(selected, cond)
    finally:
        handle.remove()
    assert all(torch.isfinite(v).all() for v in output["samples"].unbind())
    assert len(observed) == 4
    assert observed[0] > 0 and observed[-1] == 0
    assert torch.equal(module.weight, original)


def test_multiple_loras_keep_order_strength_and_source_conditioning(tmp_path):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    _write_adapter(a, 14)
    _write_adapter(b, 15)
    selected = model()
    first, _ = create_window_hook(selected, a, 0.2, 0.0, 0.5)
    second, _ = create_window_hook(selected, b, 0.8, 0.5, 1.0, previous_hooks=first)
    assert len(first.hooks) == 1 and len(second.hooks) == 2
    assert [h._strength_model for h in second.hooks] == [0.2, 0.8]
    assert second.hooks[0].hook_ref is first.hooks[0].hook_ref
    cond = conditioning()
    cond[0][1]["hooks"] = second
    run(selected, cond)


@pytest.mark.parametrize(
    "start,end", [(0.0, 0.0), (0.8, 0.2), (-0.1, 1.0), (0.0, float("nan"))]
)
def test_invalid_progress_is_an_actual_input_error(start, end):
    with pytest.raises(ValueError, match="window"):
        window_keyframes(start, end)


def test_schemas_and_registration():
    from h3_audio_t8_pkg.nodes_lora_windows import NODES

    for node in NODES:
        assert node.define_schema().node_id.startswith("MiniMaxH3")


def test_unmapped_file_does_not_silently_install_an_empty_hook(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "unmapped.safetensors"
    save_file({"a_different_network.weight": torch.ones(2, 2)}, str(path))
    with pytest.raises(ValueError, match="No LoRA weight targets mapped"):
        create_window_hook(model(), path, 0.7)


def test_same_one_step_sigma_reuses_hook_without_losing_its_active_window(tmp_path):
    path = tmp_path / "one.safetensors"
    _write_adapter(path, 19)
    selected, sampler, _ = setup_dual_clock_sampling(model(), latent(), 4, 12.0, 3.0)
    hooks, _ = create_window_hook(
        selected, path, 0.7, start_percent=0.25, end_percent=0.75
    )
    cond = conditioning()
    cond[0][1]["hooks"] = hooks
    guider = BasicGuider.execute(selected, cond).result[0]
    table = torch.tensor([0.5, 0.0])
    first = SamplerCustomAdvanced.execute(
        RandomNoise.execute(71).result[0], guider, sampler, table, latent()
    ).result[0]
    second = SamplerCustomAdvanced.execute(
        RandomNoise.execute(71).result[0], guider, sampler, table, latent()
    ).result[0]
    assert all(
        torch.equal(a, b)
        for a, b in zip(first["samples"].unbind(), second["samples"].unbind())
    )


def test_real_quantized_serialization_virtual_fields_do_not_break_original_weight_inventory():
    from types import SimpleNamespace
    from test_fast_h3_v2_quantized_identity import _model, KEY
    from h3_audio_t8_pkg.lora_windows import quantized_hook_scope

    selected, layer = _model()
    original = {k: v.clone() for k, v in selected.model_state_dict().items()}
    with pytest.raises(AttributeError, match="weight_scale"):
        selected.get_key_patches()
    from comfy.patcher_extension import WrapperExecutor

    executor = WrapperExecutor.new_class_executor(
        lambda: selected.get_key_patches(),
        SimpleNamespace(model_patcher=selected),
        [quantized_hook_scope],
    )
    result = executor.execute()
    assert KEY in result
    assert not any(k.endswith((".weight_scale", ".comfy_quant")) for k in result)
    assert (
        "get_key_patches" not in selected.__dict__
        and "model_state_dict" not in selected.__dict__
    )
    current = selected.model_state_dict()
    for key in original:
        assert torch.equal(
            original[key].view(torch.uint8), current[key].view(torch.uint8)
        )


def test_quantized_inventory_preserves_foreign_getter_errors_and_restores_instance_methods():
    from types import SimpleNamespace
    from h3_audio_t8_pkg.lora_windows import quantized_hook_scope

    selected = model()
    calls = []

    def foreign(*args, **kwargs):
        calls.append((args, kwargs))
        raise AttributeError("actual foreign getter failure")

    selected.get_key_patches = foreign
    from comfy.patcher_extension import WrapperExecutor

    executor = WrapperExecutor.new_class_executor(
        lambda: selected.get_key_patches("explicit-prefix"),
        SimpleNamespace(model_patcher=selected),
        [quantized_hook_scope],
    )
    with pytest.raises(AttributeError, match="actual foreign getter failure"):
        executor.execute()
    assert selected.get_key_patches is foreign and calls == [(("explicit-prefix",), {})]
    assert "model_state_dict" not in selected.__dict__


@pytest.mark.parametrize("hook_mode", ["MinVram", "MaxSpeed"])
@pytest.mark.parametrize("quantized", [True, False])
@pytest.mark.parametrize("pinned", [False, True])
def test_real_Core_quantized_hook_patch_changes_weights_then_restores_scale_and_storage(
    hook_mode,
    quantized,
    pinned,
    monkeypatch,
):
    from dataclasses import replace
    import comfy.hooks
    from comfy_kitchen.tensor import QuantizedTensor
    from types import SimpleNamespace
    from comfy.patcher_extension import WrapperExecutor
    from test_fast_h3_v2_quantized_identity import _model, KEY
    from h3_audio_t8_pkg.lora_windows import quantized_hook_scope

    selected, layer = _model()
    # The identity-only fixture deliberately never dequantizes its group=8.
    # A real ConvRot calculation needs a power-of-four group dividing 24.
    layer.weight = torch.nn.Parameter(
        QuantizedTensor(
            layer.weight._qdata,
            layer.weight._layout_cls,
            replace(layer.weight.params, convrot_groupsize=4),
        ),
        requires_grad=False,
    )
    if not quantized:
        layer.weight = torch.nn.Parameter(
            layer.weight.dequantize(), requires_grad=False
        )
        layer.layout_type = None

    def dense():
        return layer.weight.dequantize() if quantized else layer.weight

    adapter = selected.patches[KEY][0][1]
    selected.patches.clear()
    selected.hook_mode = getattr(comfy.hooks.EnumHookMode, hook_mode)
    pins = set()
    if pinned:
        import comfy.model_management

        def pin(weight):
            assert id(weight) not in pins
            pins.add(id(weight))
            return True

        def unpin(weight):
            assert id(weight) in pins, "replaced storage was never registered as pinned"
            pins.remove(id(weight))

        monkeypatch.setattr(comfy.model_management, "pin_memory", pin)
        monkeypatch.setattr(comfy.model_management, "unpin_memory", unpin)
        selected.pin_weight_to_device(KEY)
    before = {k: v.clone() for k, v in selected.model_state_dict().items()}
    clean = dense().clone()
    hook = comfy.hooks.WeightHook(strength_model=0.5)
    group = comfy.hooks.HookGroup()
    group.add(hook)
    selected.add_hook_patches(hook, {KEY: adapter}, strength_patch=0.5)
    observed = []

    def apply():
        for _ in range(2):
            selected.patch_hooks(group)
            observed.append(float((dense() - clean).abs().max()))
            selected.unpatch_hooks()
            torch.testing.assert_close(dense(), clean, rtol=0, atol=0)

    executor = WrapperExecutor.new_class_executor(
        apply, SimpleNamespace(model_patcher=selected), [quantized_hook_scope]
    )
    executor.execute()
    assert observed[0] > 0
    assert observed[0] == observed[1]
    assert (
        "set_weight" not in layer.__dict__
        and "get_key_patches" not in selected.__dict__
    )
    after = selected.model_state_dict()
    for key, value in before.items():
        assert torch.equal(value.view(torch.uint8), after[key].view(torch.uint8)), key
    if pinned:
        assert pins == {id(layer.weight)}
        selected.unpin_all_weights()
        assert not pins and not selected.pinned
