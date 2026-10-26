"""Actual tiny Core regression for inactive native periodic control only.

Random tiny consumers establish executable contracts, not trained control quality.
"""
import json

import pytest
import torch
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

from h3_audio_t8_pkg import av_loop_runtime as runtime
from h3_audio_t8_pkg.av_loop_clock import LoopPlan, periodic
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.nodes_av_loop import MiniMaxH3PeriodicAVSetupEXPT8 as Setup
from test_fast_h3_v2_core_sampler import model
from test_h21_control_regions import RealTinyEncoderOwner, real_bundle
from test_h24_av_loop_core import equal_av, sample


@pytest.fixture(autouse=True)
def actual_cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


def case(frame_count=51, sigma_start=1., custom_wrapper=False):
    bare, plan = model(), LoopPlan(56)
    bundle, tower = real_bundle("official_model_patch")
    encoder, wrapper_calls = RealTinyEncoderOwner(), []
    frames = torch.linspace(0., 1., frame_count)[:, None, None, None].expand(
        frame_count, 3, 32, 32).clone()
    cls = MiniMaxH3FunControlPatch
    if custom_wrapper:
        class UserControl(MiniMaxH3FunControlPatch):
            def diffusion_model_wrapper(self, executor, *args, **kwargs):
                wrapper_calls.append(self)
                return super().diffusion_model_wrapper(executor, *args, **kwargs)
        cls = UserControl
    owner = cls(bundle.control, encoder, frames, None, None, .5, sigma_start, 0.)
    # The optimization must bind executable method identity, not the class name.
    owner.__class__ = type("UserChosenNativeControlName", (type(owner),), {})
    return bare, plan, owner, encoder, tower, wrapper_calls


@pytest.mark.parametrize("frame_count", [17, 51])
def test_registered_inactive_control_skips_vae_and_geometry_then_restores_plain(frame_count):
    bare, plan, owner, encoder, tower, _ = case(frame_count, sigma_start=.5)
    owner.register(bare)
    source = empty_av_latent(32, 32, plan.length)[0]
    # One Euler step evaluates sigma=1, outside [0, .5]. Native Core performs
    # no encoding; a short inactive control must not hit periodic media guards.
    baseline = sample(bare, plan, source=source, steps=1)
    assert not encoder.calls
    before = dict(vars(owner))
    selected, route, report = Setup.execute(bare, source, True).result
    assert json.loads(report)["complete_control_pairs_bound"] == 1
    local = selected.get_attachment(runtime.KEY).controls[0]
    tower_calls = []
    handle = tower.control_proj_in.register_forward_hook(lambda *_: tower_calls.append(1))
    try:
        output = sample(selected, route.geometry, source=source, steps=1)
        assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
        assert not encoder.calls and not tower_calls
        assert local.control_latent is None and local.active is False
        assert all(getattr(owner, name) is value for name, value in before.items())
        equal_av(sample(bare, plan, source=source, steps=1), baseline)
        assert not encoder.calls and not tower_calls
        # The same ordinary source owner still activates and encodes normally.
        owner.sigma_start = 1.
        active = sample(bare, plan, source=source, steps=1)
        equal_av(sample(bare, plan, source=source, steps=1), active)
        assert len(encoder.calls) == 2 and len(tower_calls) == 2
        assert all(value.shape[0] == 56 for value in encoder.calls)
    finally:
        handle.remove()


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("custom_wrapper", [False, True])
def test_active_registered_and_pending_controls_keep_both_real_views(registered, custom_wrapper):
    bare, plan, owner, encoder, tower, wrapper_calls = case(custom_wrapper=custom_wrapper)
    if registered:
        owner.register(bare)
    assert owner.active is False  # Pending initial state is not an inactive decision.
    before = dict(vars(owner))
    source = empty_av_latent(32, 32, plan.length)[0]
    if registered:
        selected, route, report = Setup.execute(bare, source, True).result
        assert route.geometry == plan and json.loads(report)["complete_control_pairs_bound"] == 1
    else:
        selected, _ = runtime.configure_loop(bare, plan, control_owners=(owner,))
    local = selected.get_attachment(runtime.KEY).controls[0]
    tower_calls = []
    handle = tower.control_proj_in.register_forward_hook(lambda *_: tower_calls.append(1))
    try:
        result = sample(selected, plan, source=source, steps=1)
        assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
        assert len(tower_calls) == 2
        assert len(encoder.calls) == (3 if registered else 2)
        views = [value for value in encoder.calls if value.shape[0] == 90]
        assert len(views) == 2
        for actual, shift in zip(views, plan.shifts, strict=True):
            expected = periodic(owner.control_video, 17, 22, 0, shift=shift * 17).movedim(1, -1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(getattr(owner, name) is value for name, value in before.items())
        if custom_wrapper:
            assert len(wrapper_calls) == (1 if registered else 2)
            assert all(value is local for value in wrapper_calls)
        if not registered:
            owner.register(bare)
        ordinary = sample(bare, plan, source=source, steps=1)
        equal_av(sample(bare, plan, source=source, steps=1), ordinary)
    finally:
        handle.remove()


def test_native_entry_marker_resets_after_failure_and_next_active_request_works():
    bare, plan, owner, encoder, tower, _ = case(sigma_start=.5)
    owner.register(bare)
    baseline = sample(bare, plan, steps=1)
    selected, _ = runtime.configure_loop(bare, plan, control_owners=(owner,))
    local = selected.get_attachment(runtime.KEY).controls[0]
    fail = [True]

    def foreign(executor, *args, **kwargs):
        if fail[0]:
            raise RuntimeError("downstream failed while native control was inactive")
        return executor(*args, **kwargs)

    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user_failure", foreign)
    with pytest.raises(RuntimeError, match="downstream failed"):
        sample(selected, plan, steps=1)
    assert not encoder.calls
    assert runtime._NATIVE_CONTROL_ENTRIES.get() == ()
    equal_av(sample(bare, plan, steps=1), baseline)
    fail[0] = False
    local.sigma_start = 1.
    tower_calls = []
    handle = tower.control_proj_in.register_forward_hook(lambda *_: tower_calls.append(1))
    try:
        result = sample(selected, plan, steps=1)
    finally:
        handle.remove()
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    assert len(tower_calls) == 2
    assert len([value for value in encoder.calls if value.shape[0] == 90]) == 2
    assert runtime._NATIVE_CONTROL_ENTRIES.get() == ()
    assert owner.sigma_start == .5 and owner.active is False
    equal_av(sample(bare, plan, steps=1), baseline)
