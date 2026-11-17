from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax import model as minimax_model
from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher

from h3_audio_t8_pkg.h3_core_compat import call_h3_final_layer, plain_attention_backend
from h3_audio_t8_pkg import vdn_h3_advanced as vdn


def test_legacy_backend_setter_reaches_actual_core_attention_without_global_mutation():
    from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
    original = {"keep": object()}
    model = SimpleNamespace(model_options={"transformer_options": original})
    marker = object()
    calls = []
    def backend(*args, **kwargs):
        calls.append(True)
        return marker
    set_h3_attention_backend(model, backend)
    assert "optimized_attention_override" not in original
    q = torch.zeros(1, 1, 8)
    result = attention.attention_pytorch(q, q, q, 1, transformer_options=model.model_options["transformer_options"])
    assert result is marker and calls == [True]
    set_h3_attention_backend(model, attention.attention_pytorch)
    assert plain_attention_backend(model.model_options["transformer_options"]["optimized_attention_override"]) == "pytorch"


def test_legacy_backend_setter_refuses_ignored_override_protocol(monkeypatch):
    from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
    model = SimpleNamespace(model_options={"transformer_options": {}})
    monkeypatch.setattr(attention, "attention_pytorch", lambda *a, **kw: torch.zeros(1))
    with pytest.raises(RuntimeError, match="no executable"):
        set_h3_attention_backend(model, lambda *a: None)
    assert model.model_options["transformer_options"] == {}


def backend_model(backend):
    if not hasattr(ModelPatcher, "set_model_optimized_attention"):
        pytest.skip("Core predates the official per-model attention backend selector")
    model = SimpleNamespace(model_options={"transformer_options": {}}, patches={})
    ModelPatcher.set_model_optimized_attention(model, backend)
    return model


@pytest.mark.parametrize("name", ["pytorch", "sage", "flash", "comfy_kitchen_int8"])
def test_actual_core_backend_selector_does_not_block_vdn(name):
    backend = getattr(attention, f"attention_{name}", None)
    if backend is None:
        pytest.skip("Core has no such backend")
    model = backend_model(backend)
    override = model.model_options["transformer_options"]["optimized_attention_override"]
    assert plain_attention_backend(override) == name
    assert vdn._attention_conflicts(model) == []


def test_arbitrary_target_is_not_plain_backend_but_dormant_for_vdn():
    model = backend_model(lambda *args, **kwargs: args[0])
    assert plain_attention_backend(model.model_options['transformer_options']['optimized_attention_override']) is None
    assert vdn._attention_conflicts(model) == []


def test_non_callable_backend_does_not_match_missing_optional_attribute():
    model = backend_model(None)
    assert plain_attention_backend(model.model_options["transformer_options"]["optimized_attention_override"]) is None


@pytest.mark.parametrize("name", ["pytorch", "sage"])
def test_legacy_sla_accepts_actual_plain_backend_wrapper(name):
    from h3_audio_t8_pkg.sla_attention_advanced import _existing_attention_contract
    model = backend_model(getattr(attention, f"attention_{name}"))
    report = _existing_attention_contract(model.model_options["transformer_options"])
    assert report["backend"] == name


def test_replaced_container_route_is_not_exempt():
    model = backend_model(attention.attention_pytorch)
    override = model.model_options["transformer_options"]["optimized_attention_override"]
    override.container_function = lambda *args: args[0]
    assert plain_attention_backend(override) is None


def test_cli_default_backend_does_not_create_model_override(monkeypatch):
    monkeypatch.setattr(attention, "optimized_attention", attention.attention_sage)
    model = SimpleNamespace(model_options={"transformer_options": {}}, patches={})
    assert vdn._attention_conflicts(model) == []


def test_vdn_runtime_guard_survives_clone_and_warns_without_replacing_user_hook(caplog):
    hooks = (lambda *args: None, lambda *args: None)
    options = {
        vdn.OWNER_HOOKS_KEY: hooks,
        "patches_replace": {"dit": {("double_block", i): h for i, h in enumerate(hooks)}},
    }
    cloned = copy.deepcopy(options)
    vdn.validate_vdn_runtime_options(cloned)
    def replacement(*args):
        return "user-hook"

    cloned["patches_replace"]["dit"][("double_block", 1)] = replacement
    vdn.validate_vdn_runtime_options(cloned)
    assert cloned["patches_replace"]["dit"][("double_block", 1)] is replacement
    assert replacement() == "user-hook"
    assert "indices" in caplog.text and "1" in caplog.text
    assert options["patches_replace"]["dit"][("double_block", 1)] is hooks[1]
    vdn.validate_vdn_runtime_options(options)


def test_final_layer_legacy_signature_preserves_arguments():
    class OldLayer:
        def __call__(self, x, t_emb, video_seg, audio_seg):
            return x + t_emb, (video_seg, audio_seg)
    result = call_h3_final_layer(OldLayer(), 2, 3, (0, 2, 0), (2, 4, 0),
                                 sigma=0.5, sample_sigmas=None, shifts=(12, 3))
    assert result == (5, ((0, 2, 0), (2, 4, 0)))


def test_real_core_final_layer_matches_direct_call():
    import comfy.ops
    layer = minimax_model.FinalLayer(8, 4, 24, 32, 1e-5,
                                    dtype=torch.float32, device="cpu",
                                    operations=comfy.ops.disable_weight_init)
    torch.manual_seed(44)
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    args = (torch.randn(4, 8), torch.randn(1, 4), (0, 2, 0), (2, 4, 0))
    schedule = dict(sigma=torch.tensor(0.5), sample_sigmas=torch.tensor([1., 0.5, 0.]), shifts=(12, 3))
    actual = call_h3_final_layer(layer, *args, **schedule)
    import inspect
    parameters = inspect.signature(layer.forward).parameters
    if "grids" in parameters:
        schedule["grids"] = tuple(minimax_model.pdd_grid(layer.pdd_steps, shift)
                                  for shift in schedule.pop("shifts"))
    expected = layer(*args, **schedule) if "sigma" in parameters else layer(*args)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("core_revision", [None, "cf10c5c398ad4297b664e7aa85090daaee529f29"])
@pytest.mark.parametrize("masked", [False, True])
def test_full_multikeyframe_forward_matches_native_core_with_uniform_conditioning(masked, core_revision, monkeypatch):
    import comfy.ops
    from h3_audio_t8_pkg import multikeyframe_advanced as multi

    native_model = minimax_model
    if core_revision:
        import subprocess
        import types
        from pathlib import Path
        try:
            source = subprocess.check_output(
                ["git", "show", f"{core_revision}:comfy/ldm/minimax/model.py"],
                cwd=Path(minimax_model.__file__).resolve().parents[3], text=True, stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("historical Core source not available locally")
        native_model = types.ModuleType("t8_test_historical_minimax_model")
        exec(compile(source, f"git:{core_revision}:comfy/ldm/minimax/model.py", "exec"), native_model.__dict__)
        monkeypatch.setattr(multi, "minimax_model", native_model)
    monkeypatch.setattr(native_model, "optimized_attention", attention.attention_pytorch)
    torch.manual_seed(443)
    model = native_model.MiniMaxH3Model(
        hidden_size=8, num_layers=1, token_refiner_num_layers=0,
        num_attention_heads=2, attention_head_dim=8, ffn_hidden_size=16,
        text_dim=8, timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4,
        rope_inv_freq_len=1, dtype=torch.float32, device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    model.requires_grad_(False)
    model.rope.inv_freq.fill_(1.)
    x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
    context = torch.randn(1, 2, 8)
    layout = native_model.PackedLayout(2, 2, 4, 4, 3)
    payload = {"layout": layout, multi.PAYLOAD_VISUAL_NOISE_AUGS_KEY: []}
    kwargs = {"minimax_payload": payload}
    if masked:
        kwargs["denoise_mask"] = torch.ones(1, 1, 2, 4, 4)
        kwargs["denoise_mask"][:, :, 0] = 0.
        kwargs["audio_denoise_mask"] = torch.tensor([[[[0., 0.5, 1.], [0., 0.5, 1.]]]])
    options = {"sample_sigmas": torch.tensor([1., 0.5, 0.])}
    with torch.no_grad():
        native = model._forward(x, torch.tensor([500.]), context, transformer_options=options.copy(), **kwargs)
        adapted = multi._multikeyframe_forward(model, x, torch.tensor([500.]), context, transformer_options=options.copy(), **kwargs)
    import inspect
    if masked and "denoise_mask" not in inspect.signature(model._forward).parameters:
        # Legacy Core ignores masks through **kwargs; the T8 forward deliberately
        # adds row-mask scheduling there. Comparing it with that unmasked native
        # output would require removing a supported T8 feature, not compatibility.
        with torch.no_grad():
            unmasked = multi._multikeyframe_forward(model, x, torch.tensor([500.]), context,
                                                     transformer_options=options.copy(), minimax_payload=payload)
        for baseline, reference, actual in zip(native, unmasked, adapted):
            torch.testing.assert_close(baseline, reference, rtol=1e-5, atol=1e-6)
            assert actual.shape == baseline.shape and torch.isfinite(actual).all()
            assert not torch.equal(actual, baseline)
        return
    for left, right in zip(native, adapted):
        assert torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["pytorch", "sage"])
def test_plain_backend_is_not_algorithm_conflict_for_fast_h3_or_world(name):
    from h3_audio_t8_pkg import fast_h3_vsa_advanced as fast
    from h3_audio_t8_pkg import h3_world_advanced as world
    model = backend_model(getattr(attention, f"attention_{name}"))
    assert fast._attention_conflict(model) is None
    world._ensure_patch_compatibility(model)


def test_unknown_backend_is_diagnostic_not_admission_ban_for_world(caplog):
    from h3_audio_t8_pkg import fast_h3_vsa_advanced as fast
    from h3_audio_t8_pkg import h3_world_advanced as world
    model = backend_model(lambda *args: None)
    selected = model.model_options["transformer_options"]["optimized_attention_override"]
    assert fast._attention_conflict(model) is not None
    world._ensure_patch_compatibility(model)
    assert model.model_options["transformer_options"]["optimized_attention_override"] is selected
    assert "attention override" in caplog.text
