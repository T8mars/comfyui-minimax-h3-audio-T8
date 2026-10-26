"""Node wiring reaches actual tiny Core consumers; no trained-quality claim."""
import json
import math
from types import MethodType

import pytest
import torch
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

from h3_audio_t8_pkg import nodes_av_loop as nodes
from h3_audio_t8_pkg.audio_ops import decode_av_latent
from h3_audio_t8_pkg.av_loop_clock import LoopPlan
from h3_audio_t8_pkg.av_loop_runtime import KEY
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.control_region_runtime import ControlRegionRuntime, official_region_patch, register_runtime
from h3_audio_t8_pkg.control_regions import RegionGate
from test_fast_h3_v2_core_sampler import model
from test_h21_control_regions import RealTinyEncoderOwner, real_bundle
from test_h21_core_vae_contracts import real_tiny_vae_owners
from test_h24_av_loop_core import sample


Setup = nodes.MiniMaxH3PeriodicAVSetupEXPT8
Decode = nodes.MiniMaxH3PeriodicAVDecodeEXPT8
Audit = nodes.MiniMaxH3PeriodicAVAuditEXPT8


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def decoders(monkeypatch):
    return real_tiny_vae_owners.__wrapped__(monkeypatch)


def test_three_explicit_schemas_default_off_and_typed_plan_has_no_padding_ceiling():
    assert len(nodes.NODES) == 3
    assert len({cls.define_schema().node_id for cls in nodes.NODES}) == 3
    assert all(cls.define_schema().is_experimental for cls in nodes.NODES)
    schema = Setup.define_schema()
    fields = {item.id: item for item in schema.inputs}
    assert fields["enabled"].default is False
    assert fields["pad_clips"].min == 1 and fields["pad_clips"].max is None
    assert fields["views"].max == 2  # Two implemented views, not a resource limit.
    assert schema.outputs[1].io_type == Decode.define_schema().inputs[1].io_type == "T8_H3_PERIODIC_AV_PLAN"
    assert math.isnan(Audit.fingerprint_inputs())


def test_disabled_setup_is_immediate_identity_without_examining_inactive_inputs():
    selected = object()
    actual, route, report = Setup.execute(selected, object(), False, pad_clips=-100, views=999).result
    assert actual is selected and route.enabled is False and route.geometry is None
    assert json.loads(report)["status"] == "disabled_native_path"


def test_disabled_decode_is_actual_full_native_decode_bit_exact(decoders):
    source = empty_av_latent(32, 32, 22)[0]
    _, route, _ = Setup.execute(object(), object(), False).result
    expected_images, expected_audio = decode_av_latent(source, *decoders)[:2]
    images, audio, report = Decode.execute(source, route, *decoders).result
    torch.testing.assert_close(images, expected_images, rtol=0, atol=0)
    torch.testing.assert_close(audio["waveform"], expected_audio["waveform"], rtol=0, atol=0)
    assert images.shape[0] == 22 and audio["waveform"].shape[-1] == 29600
    assert not json.loads(report)["periodic_crop_applied"]
    with pytest.raises(ValueError, match="joint AV"):
        Decode.execute({}, route, *decoders)


@pytest.mark.parametrize("length", [22, 56, 107, 158])
def test_enabled_geometry_comes_from_actual_both_streams(length):
    source = empty_av_latent(32, 32, length)[0]
    _, route, text = Setup.execute(model(), source, True, pad_clips=3).result
    assert route.geometry.length == length and route.geometry.pad_clips == 3
    report = json.loads(text)
    assert report["plan"]["input_frames"] == length
    assert report["plan"]["audio_padding_error_ticks"]["numerator"] == 0
    assert report["last_request"]["status"] == "selected_not_executed"
    assert not report["quality_accepted"] and not report["sample_result_binding_proven"]


@pytest.mark.parametrize("fault", ["video_ticks", "audio_ticks", "channels", "nan"])
def test_enabled_setup_checks_real_geometry_and_numerics(fault):
    video, audio = empty_av_latent(32, 32, 22)[0]["samples"].unbind()
    if fault == "video_ticks":
        video = video[:, :, :-1]
    elif fault == "audio_ticks":
        audio = audio[..., :-1]
    elif fault == "channels":
        video = video[:, :-1]
    else:
        audio[..., 0] = float("nan")
    with pytest.raises(ValueError):
        Setup.execute(model(), {"samples": NestedTensor([video, audio])}, True)


def test_typed_plan_and_mutated_geometry_reject_before_decoder_call():
    with pytest.raises(TypeError, match="typed"):
        Decode.execute({}, {"enabled": False}, None, None)
    route = nodes.PeriodicAVDecodePlan(True, LoopPlan(22))
    object.__setattr__(route.geometry, "pad_clips", 2)
    with pytest.raises(ValueError, match="changed"):
        Decode.execute({}, route, None, None)


def test_setup_sample_decode_audit_reaches_real_core_and_two_real_vaes(decoders):
    bare, source = model(), empty_av_latent(32, 32, 56)[0]
    selected, route, _ = Setup.execute(bare, source, True).result
    before = json.loads(Audit.execute(selected, object()).result[0])
    assert before["last_request"]["forward_calls"] == 0
    sampled = sample(selected, route.geometry, source=source, steps=1)
    images, audio, report = Decode.execute(sampled, route, *decoders).result
    audit = json.loads(Audit.execute(selected, sampled).result[0])
    assert images.shape == (51, 32, 32, 3) and audio["waveform"].shape == (1, 2, 68000)
    assert images.std() > 0 and audio["waveform"].std() > 0
    assert audit["last_request"]["forward_calls"] == 1 and audit["last_request"]["view_calls"] == 2
    assert not audit["sample_result_binding_proven"] and not json.loads(report)["quality_accepted"]
    assert bare.get_attachment(KEY) is None and not bare.wrappers


def test_foreign_attachment_is_retained_and_plan_uses_native_decode(decoders):
    bare, source, foreign = model(), empty_av_latent(32, 32, 22)[0], object()
    bare.set_attachments(KEY, foreign)
    selected, route, text = Setup.execute(bare, source, True).result
    assert selected is bare and bare.get_attachment(KEY) is foreign
    assert not route.enabled and route.geometry is None
    assert json.loads(text)["status"] == "not_installed_foreign_owner"
    images, _, _ = Decode.execute(source, route, *decoders).result
    assert images.shape[0] == 22


def test_unknown_wrapper_renamed_model_lora_and_hook_still_execute():
    bare, source = model(), empty_av_latent(32, 32, 22)[0]
    network = bare.model.diffusion_model
    network.__class__ = type("UserChosenArchitectureName", (type(network),), {})
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    bare.add_patches({key: ("diff", (torch.full_like(bare.model_state_dict()[key], .001),))}, strength_patch=-.25)
    calls, hooks = [], []

    class Foreign:
        def wrapper(self, executor, *args, **kwargs):
            calls.append(1)
            return executor(*args, **kwargs)

    foreign = Foreign()
    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user", foreign.wrapper)
    handle = network.blocks[0].register_forward_hook(lambda *_: hooks.append(1))
    try:
        selected, route, text = Setup.execute(bare, source, True).result
        output = sample(selected, route.geometry, source=source, steps=1)
    finally:
        handle.remove()
    assert calls == [1] and hooks == [1]
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
    assert selected.patches[key][0][0] == -.25
    report = json.loads(text)
    assert report["complete_control_pairs_bound"] == 0 and report["control_binding_warnings"]
    assert selected.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)[0].__self__ is foreign


def test_complete_actual_native_control_pair_auto_binds_and_restores_original_owner():
    bare, source = model(), empty_av_latent(32, 32, 56)[0]
    bundle, tower = real_bundle("official_model_patch")
    frames = torch.linspace(0., 1., 51)[:, None, None, None].expand(51, 3, 32, 32).clone()
    owner = MiniMaxH3FunControlPatch(bundle.control, RealTinyEncoderOwner(), frames, None, None, .5, 1., 0.)
    owner.__class__ = type("UserControlName", (type(owner),), {})

    def extra_wrapper(self, executor, *args, **kwargs):
        return executor(*args, **kwargs)

    owner.extra_wrapper = MethodType(extra_wrapper, owner)
    # An earlier unrelated bound method must not hide the later complete pair.
    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user_extra", owner.extra_wrapper)
    owner.register(bare)
    before, calls = dict(vars(owner)), []
    selected, route, text = Setup.execute(bare, source, True).result
    handle = tower.control_proj_in.register_forward_hook(lambda *_: calls.append(1))
    try:
        sample(selected, route.geometry, source=source, steps=1)
    finally:
        handle.remove()
    assert calls == [1, 1]
    assert json.loads(text)["complete_control_pairs_bound"] == 1
    assert selected.get_attachment(KEY).controls[0] is not owner
    assert all(getattr(owner, name) is value for name, value in before.items())


def test_incomplete_actual_control_pair_is_warned_and_preserved():
    bare, source = model(), empty_av_latent(32, 32, 22)[0]
    bundle, _ = real_bundle("official_model_patch")
    owner = MiniMaxH3FunControlPatch(bundle.control, RealTinyEncoderOwner(),
        torch.zeros(17, 3, 32, 32), None, None, .5, 1., 0.)
    bare.add_wrapper(WrappersMP.DIFFUSION_MODEL, owner.diffusion_model_wrapper)
    selected, route, text = Setup.execute(bare, source, True).result
    report = json.loads(text)
    assert report["complete_control_pairs_bound"] == 0
    assert "incomplete_control_patch_pair_preserved_without_temporal_binding" in report["control_binding_warnings"]
    assert selected.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)[0].__self__ is owner
    output = sample(selected, route.geometry, source=source, steps=1)
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())


def test_nonempty_regional_gate_reports_boundary_and_keeps_actual_clock_guard():
    bare, source = model(), empty_av_latent(32, 32, 56)[0]
    bundle, _ = real_bundle("official_model_patch")
    gate = RegionGate(torch.ones(56, 32, 32), tuple(range(56)), 32, 32, 56)
    runtime = ControlRegionRuntime(gate=gate)
    cls = official_region_patch(MiniMaxH3FunControlPatch, runtime)
    owner = cls(bundle.control, RealTinyEncoderOwner(), torch.zeros(56, 3, 32, 32), None, None, .5, 1., 0.)
    owner.register(bare)
    register_runtime(bare, runtime)
    selected, route, text = Setup.execute(bare, source, True).result
    report = json.loads(text)
    assert report["complete_control_pairs_bound"] == 1
    assert "derived_cache_only_region_gate_clock_and_closure_not_adapted" in report["control_binding_warnings"]
    assert not report["regional_gate_clock_remapped"] and not report["user_closure_state_isolated"]
    with pytest.raises(ValueError, match="temporal frame bins"):
        sample(selected, route.geometry, source=source, steps=1)
    owner.verify_derived()
    assert runtime.gate is gate and runtime.errors
    # The same actual region remains valid at its original native clock.
    result = sample(bare, LoopPlan(56), source=source, steps=1)
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    assert runtime.gated_calls == 1
