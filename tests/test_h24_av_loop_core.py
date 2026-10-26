"""Actual tiny native sampling, VAE, control and lossless AV consumers.

Random tiny weights qualify executable contracts only, never trained quality,
seamlessness, GPU speed, memory, or all user patch combinations.
"""
from copy import deepcopy
from fractions import Fraction
import json
from threading import Event, Thread
from types import MethodType

import av
import numpy as np
import pytest
import torch
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from comfy.model_management import InterruptProcessingException
from comfy.patcher_extension import CallbacksMP, WrappersMP
from comfy_extras.nodes_custom_sampler import BasicGuider, DisableNoise, RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

from h3_audio_t8_pkg.av_loop_clock import LoopPlan, periodic, repeat_pcm
from h3_audio_t8_pkg.av_loop_runtime import (
    KEY, audit_loop, configure_loop, decode_loop, periodic_control_scope,
)
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.control_region_runtime import ControlRegionRuntime, official_region_patch
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_h21_control_regions import RealTinyEncoderOwner, real_bundle
from test_h21_core_vae_contracts import real_tiny_vae_owners
from test_progressive_sampling_runtime import conditioning


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


def sample(selected, plan, *, steps=2, noise=True, positive=None, source=None):
    source = empty_av_latent(32, 32, plan.length)[0] if source is None else source
    prepared, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, steps, 12., 3., "dual_clock_euler", "native_flow")
    guider = BasicGuider.execute(prepared, conditioning() if positive is None else positive).result[0]
    producer = RandomNoise.execute(2410263).result[0] if noise else DisableNoise.execute().result[0]
    return SamplerCustomAdvanced.execute(producer, guider, sampler, sigmas, source).result[0]


def equal_av(left, right):
    for a, b in zip(left["samples"].unbind(), right["samples"].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_real_sampler_two_views_only_take_video_from_first_and_restore_plain_model():
    bare = model()
    baseline = sample(bare, LoopPlan(56), steps=1)
    one, _ = configure_loop(bare, LoopPlan(56, views=1))
    two, _ = configure_loop(bare, LoopPlan(56, views=2))
    first = sample(one, one.get_attachment(KEY).plan, steps=1)
    second = sample(two, two.get_attachment(KEY).plan, steps=1)
    torch.testing.assert_close(first["samples"].unbind()[0], second["samples"].unbind()[0], rtol=0, atol=0)
    report = audit_loop(two)
    assert report["last_request"]["forward_calls"] == 1
    assert report["last_request"]["view_calls"] == 2
    assert not report["request_active"] and not report["trained_GPU_qualified"]
    assert not bare.wrappers and bare.get_attachment(KEY) is None
    equal_av(sample(bare, LoopPlan(56), steps=1), baseline)
    json.dumps(report, allow_nan=False)


def test_real_two_step_masks_guides_unknown_class_lora_hook_and_noise_disabled_are_consumed():
    bare, plan = model(), LoopPlan(56)
    network = bare.model.diffusion_model
    network.__class__ = type("UserSelectedLoopVariant", (type(network),), {})
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    bare.add_patches({key: ("diff", (torch.full_like(bare.model_state_dict()[key], .001),))}, strength_patch=-.3)
    patches = deepcopy(bare.patches)
    calls = []
    handle = network.blocks[0].register_forward_hook(lambda *_: calls.append(1))
    source = empty_av_latent(32, 32, plan.length)[0]
    video, audio = source["samples"].unbind()
    import comfy.nested_tensor
    vm, am = torch.ones_like(video), torch.ones_like(audio)
    vm[:, :, 2] = 0
    am[..., 3:7] = .5
    source["noise_mask"] = comfy.nested_tensor.NestedTensor([vm, am])
    positive = conditioning()
    positive[0][1]["minimax_keyframes"] = [{"resolved_frame_index": 17,
                                           "latent": torch.ones(1, 24, 1, 2, 2) * .1}]
    original = deepcopy(positive)
    selected, _ = configure_loop(bare, plan)
    try:
        output = sample(selected, plan, noise=False, source=source, positive=positive)
    finally:
        handle.remove()
    assert len(calls) == 4
    assert audit_loop(selected)["last_request"]["noise_disabled"]
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
    assert list(selected.patches) == list(patches)
    assert selected.patches[key][0][0] == -.3
    assert positive[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 17
    torch.testing.assert_close(positive[0][1]["minimax_keyframes"][0]["latent"],
                               original[0][1]["minimax_keyframes"][0]["latent"])
    assert not bare.wrappers and not network.blocks[0]._forward_hooks


@pytest.mark.parametrize("failure", ["exception", "interrupt", "nan", "shape", "binding"])
def test_actual_failed_consumer_has_no_retry_and_next_plain_request_is_identical(failure):
    bare, plan = model(), LoopPlan(22)
    baseline = sample(bare, plan)
    selected, _ = configure_loop(bare, plan)
    calls = []

    def foreign(executor, *args, **kwargs):
        calls.append(1)
        if failure == "exception":
            raise TypeError("real foreign computation failure")
        if failure == "interrupt":
            raise InterruptProcessingException()
        out = executor(*args, **kwargs)
        if failure == "nan":
            out[1].fill_(float("nan"))
        elif failure == "shape":
            out[0] = out[0][:, :, :-1]
        else:
            object.__setattr__(plan, "pad_clips", 2)
        return out

    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user_after_loop", foreign)
    try:
        with pytest.raises((TypeError, ValueError, InterruptProcessingException)):
            sample(selected, plan)
    finally:
        object.__setattr__(plan, "pad_clips", 1)
    assert calls == [1]
    report = audit_loop(selected)
    assert report["last_request"]["status"] == "failed" and not report["request_active"]
    equal_av(sample(bare, plan), baseline)


def test_reselection_preserves_unknown_wrapper_order_and_only_replaces_own_owner():
    bare = model()
    called = []

    def unknown(executor, *args, **kwargs):
        called.append(1)
        return executor(*args, **kwargs)

    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user", unknown)
    first, _ = configure_loop(bare, LoopPlan(22))
    selected, _ = configure_loop(first, LoopPlan(56))
    assert len(selected.get_wrappers(WrappersMP.DIFFUSION_MODEL, KEY)) == 1
    assert selected.get_wrappers(WrappersMP.DIFFUSION_MODEL, "user") == [unknown]
    assert list(selected.wrappers[WrappersMP.DIFFUSION_MODEL]) == ["user", KEY]
    sample(selected, LoopPlan(56))
    assert len(called) == 2
    assert audit_loop(first)["last_request"]["status"] == "selected_not_executed"
    disabled, report = configure_loop(bare, LoopPlan(22), enabled=False)
    assert disabled is bare and report["status"] == "disabled"


@pytest.fixture
def decoders(monkeypatch):
    # Calls the existing actual tiny Core VAE construction; no shape-only stub.
    return real_tiny_vae_owners.__wrapped__(monkeypatch)


@pytest.mark.parametrize("length", [22, 56])
def test_real_video_and_bigvgan_decode_follow_frame_clock_and_selected_owners(decoders, length):
    plan = LoopPlan(length)
    selected, _ = configure_loop(model(), plan)
    output = sample(selected, plan, steps=1)
    video, audio = decoders
    video.first_stage_model.__class__ = type("UserDecoderVariant", (type(video.first_stage_model),), {})
    frames, sound, report = decode_loop(output, video, audio, plan)
    assert frames.shape == (length - 5, 32, 32, 3)
    assert sound["waveform"].shape == (1, 2, round(Fraction((length - 5) * 32000, 24)))
    assert video.calls == [("regular", None)]
    assert frames.std() > 0 and sound["waveform"].std() > 0
    assert not report["quality_accepted"]
    assert report["plan"]["latent_periods_exact"] is (length == 56)


def test_actual_audio_decoder_nan_is_rejected_and_next_decode_restores_exact_output(decoders):
    plan = LoopPlan(22)
    source = empty_av_latent(32, 32, plan.length)[0]
    video, audio = decoders
    baseline = decode_loop(source, video, audio, plan)
    parameter = audio.first_stage_model.dec_in_proj.weight
    original = parameter.clone()
    parameter.fill_(float("nan"))
    try:
        with pytest.raises(ValueError, match="finite"):
            decode_loop(source, video, audio, plan)
    finally:
        parameter.copy_(original)
    after = decode_loop(source, video, audio, plan)
    torch.testing.assert_close(after[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(after[1]["waveform"], baseline[1]["waveform"], rtol=0, atol=0)
    assert len(video.calls) == 3


def _lossless_media(path, images, waveform, rate):
    with av.open(str(path), "w", format="nut") as container:
        video = container.add_stream("ffv1", rate=24)
        video.width, video.height, video.pix_fmt = images.shape[2], images.shape[1], "yuv444p"
        audio = container.add_stream("pcm_f32le", rate=rate)
        audio.layout = "stereo"
        for index, pixels in enumerate(images):
            raw = (pixels.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            frame = av.VideoFrame.from_ndarray(raw, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        packed = waveform[0].T.contiguous().numpy().reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(packed, format="flt", layout="stereo")
        frame.sample_rate, frame.pts, frame.time_base = rate, 0, Fraction(1, rate)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def test_actual_three_cycle_lossless_av_file_has_exact_pcm_and_monotonic_pts(decoders, tmp_path):
    plan = LoopPlan(22)
    selected, _ = configure_loop(model(), plan)
    output = sample(selected, plan, steps=1)
    images, sound, _ = decode_loop(output, *decoders, plan)
    repeated, report = repeat_pcm(sound["waveform"], sound["sample_rate"], plan.frames, 3)
    path = tmp_path / "three-loops.nut"
    _lossless_media(path, images.repeat(3, 1, 1, 1), repeated, 32000)
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 51
    assert [Fraction(frame.pts) * frame.time_base for frame in frames] == [Fraction(i, 24) for i in range(51)]
    with av.open(str(path)) as container:
        decoded = list(container.decode(audio=0))
    cursor, arrays = 0, []
    for frame in decoded:
        assert Fraction(frame.pts) * frame.time_base == Fraction(cursor, 32000)
        arrays.append(frame.to_ndarray().reshape(-1, 2).T)
        cursor += frame.samples
    actual = torch.from_numpy(np.concatenate(arrays, axis=1))[None]
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    assert cursor == 68000 and Fraction(cursor, 32000) == Fraction(len(frames), 24)
    assert report["cycle_sample_adjustments"] == [0, -1, 0]


def test_actual_control_encoder_and_tower_receive_both_view_timelines_then_restore():
    bare, plan = model(), LoopPlan(56)
    bundle, tower = real_bundle("official_model_patch")
    encoder = RealTinyEncoderOwner()
    frames = torch.linspace(0., 1., 51)[:, None, None, None].expand(51, 3, 32, 32).clone()
    owner = MiniMaxH3FunControlPatch(bundle.control, encoder, frames, None, None, .5, 1., 0.)
    owner.__class__ = type("UserTemporalControl", (type(owner),), {})
    selected, _ = configure_loop(bare, plan, control_owners=(owner,))
    local = selected.get_attachment(KEY).controls[0]
    assert local is not owner and local.model_patch is owner.model_patch and local.vae is encoder
    calls = []
    handle = tower.control_proj_in.register_forward_hook(lambda *_: calls.append(1))
    try:
        result = sample(selected, plan, steps=1)
    finally:
        handle.remove()
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    # Core control calls DiTBlock.forward directly; the input projection is the
    # actual consumer trace and encoder observations preserve the original RGB.
    assert len(calls) == 2
    assert len(encoder.calls) == 2
    for image, shift in zip(encoder.calls, plan.shifts):
        expected = periodic(frames, 17, 22, 0, shift=shift * 17).movedim(1, -1)
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
    assert owner.control_video is frames
    assert owner.control_stream is None and owner.pristine_stream is None
    assert audit_loop(selected)["explicit_control_owners"] == 1
    assert not bare.wrappers


@pytest.mark.parametrize("scale,noise,mask", [(0., True, None), (2., True, None),
    (2., True, "zero"), (2., True, "partial"), (2., False, None), (-1., True, None)])
def test_actual_selected_noise_transform_controls_padding_without_changing_center(scale, noise, mask):
    bare, plan = model(), LoopPlan(22, views=1)
    bare.model.model_sampling.set_noise_scale(scale)
    source = empty_av_latent(32, 32, plan.length)[0]
    if mask is not None:
        import comfy.nested_tensor
        video, audio = source["samples"].unbind()
        vm = torch.zeros_like(video)
        if mask == "partial":
            vm[:, :, ::2] = .5
        source["noise_mask"] = comfy.nested_tensor.NestedTensor([vm, torch.ones_like(audio)])
    before, after = [], []

    def outer(executor, x, timestep, context, opts, **kwargs):
        before.append(x[0].clone())
        return executor(x, timestep, context, opts, **kwargs)

    def inner(executor, x, timestep, context, opts, **kwargs):
        after.append((x[0].clone(), opts["sigmas"].clone(), kwargs.get("denoise_mask")))
        return executor(x, timestep, context, opts, **kwargs)

    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "before_periodic", outer)
    selected, _ = configure_loop(bare, plan)
    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "after_periodic", inner)
    prepared, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, 1, 12., 3., "dual_clock_euler", "native_flow")
    sampling = prepared.get_model_object("model_sampling")
    producer = RandomNoise.execute(2410263).result[0] if noise else DisableNoise.execute().result[0]
    raw = producer.generate_noise(source).unbind()[0]
    fresh = torch.randn(raw.shape, generator=torch.Generator().manual_seed(2410264)) if noise else raw
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    SamplerCustomAdvanced.execute(producer, guider, sampler, sigmas, source)
    assert len(before) == len(after) == 1
    observed, sigma, actual_mask = after[0]
    # Oracle calls the actual chosen Core transform, including noise_scale.
    delta = sampling.noise_scaling(sigma, fresh, torch.zeros_like(raw)) - sampling.noise_scaling(
        sigma, raw, torch.zeros_like(raw))
    delta = periodic(delta[:, :, :plan.video_ticks], plan.video_pad, plan.video_pad + 2, 2)
    delta[:, :, plan.video_pad:plan.video_pad + plan.video_ticks] = 0
    if actual_mask is not None:
        delta *= actual_mask
    wrapped = periodic(before[0][:, :, :plan.video_ticks], plan.video_pad, plan.video_pad + 2, 2)
    torch.testing.assert_close(observed, wrapped + delta, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(observed[:, :, plan.video_pad:plan.video_pad + plan.video_ticks],
                               before[0][:, :, :plan.video_ticks], rtol=0, atol=0)
    if scale == 0 or not noise or mask == "zero":
        torch.testing.assert_close(observed, wrapped, rtol=0, atol=0)
    report = audit_loop(selected)["last_request"]
    assert report["noise_scale"] == scale and report["fresh_padding_noise_bound"]
    assert report["noise_disabled"] is (not noise)
    assert report["fresh_padding_noise_applied"] is (scale != 0 and noise and mask != "zero")


def test_actual_unknown_noise_transform_executes_with_explicit_padding_noise_bypass():
    bare, plan = model(), LoopPlan(22, views=1)
    before, after, transformed = [], [], []

    def observe(rows):
        def wrapper(executor, x, *args, **kwargs):
            rows.append(x[0].clone())
            return executor(x, *args, **kwargs)
        return wrapper

    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "before_periodic", observe(before))
    selected, _ = configure_loop(bare, plan)
    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "after_periodic", observe(after))
    source = empty_av_latent(32, 32, plan.length)[0]
    prepared, sampler, sigmas = setup_dual_clock_sampling(
        selected, source, 1, 12., 3., "dual_clock_euler", "native_flow")
    sampling = prepared.get_model_object("model_sampling")
    original = sampling.noise_scaling.__func__

    def custom(self, *args, **kwargs):
        transformed.append(1)
        return original(self, *args, **kwargs) * 1.1

    sampling.noise_scaling = MethodType(custom, sampling)
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    SamplerCustomAdvanced.execute(RandomNoise.execute(2410263).result[0], guider, sampler, sigmas, source)
    assert transformed and len(before) == len(after) == 1
    expected = periodic(before[0][:, :, :plan.video_ticks], plan.video_pad, plan.video_pad + 2, 2)
    torch.testing.assert_close(after[0], expected, rtol=0, atol=0)
    report = audit_loop(selected)["last_request"]
    assert not report["fresh_padding_noise_bound"] and not report["fresh_padding_noise_applied"]
    assert report["view_calls"] == 1


def test_explicit_control_scope_restores_all_owners_after_actual_encode_failure():
    _, tower = real_bundle("official_model_patch")
    from comfy.model_patcher import ModelPatcher
    patcher = ModelPatcher(tower, torch.device("cpu"), torch.device("cpu"))
    encoder = RealTinyEncoderOwner()
    frames = torch.zeros(51, 3, 32, 32)
    owner = MiniMaxH3FunControlPatch(patcher, encoder, frames, None, None, .5, 1., 0.)
    before = dict(vars(owner))
    # Force a genuine numerical failure after the real encoder consumed input.
    parameter = next(encoder.core.parameters())
    parameter.fill_(float("nan"))
    with pytest.raises((ValueError, RuntimeError)):
        with periodic_control_scope(owner, LoopPlan(56), 1, (1, 24, 27, 2, 2)):
            pass
    assert all(getattr(owner, key) is value for key, value in before.items())
    assert encoder.calls


def regional_owner():
    bundle, _ = real_bundle("official_model_patch")
    cls = official_region_patch(MiniMaxH3FunControlPatch, ControlRegionRuntime())
    frames = torch.linspace(0., 1., 51)[:, None, None, None].expand(51, 3, 32, 32).clone()
    return cls(bundle.control, RealTinyEncoderOwner(), frames, None, None, .5, 1., 0.)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("failure", [None, "encode", "body"])
def test_actual_regional_derived_cache_survives_scope_and_next_normal_prepare(cached, failure):
    owner = regional_owner()
    target = (1, 24, 17, 2, 2)
    if cached:
        owner.prepare_control_latent(target)
    owner.verify_derived()
    before = dict(vars(owner))
    parameter = next(owner.vae.core.parameters())
    weight = parameter.clone()
    if failure == "encode":
        parameter.fill_(float("nan"))

    def consume():
        with periodic_control_scope(owner, LoopPlan(56), 1, (1, 24, 27, 2, 2)):
            owner.verify_derived()
            assert owner.control_latent.shape[2] == 27
            if failure == "body":
                raise RuntimeError("consumer failed after actual encode")

    try:
        if failure:
            with pytest.raises((ValueError, RuntimeError)):
                consume()
        else:
            consume()
    finally:
        parameter.copy_(weight)
    assert all(getattr(owner, key) is value for key, value in before.items())
    owner.verify_derived()
    owner.prepare_control_latent(target)
    owner.verify_derived()
    assert owner.control_latent.shape == target
    assert torch.isfinite(owner.control_latent).all()


@pytest.mark.parametrize("cached", [False, True])
def test_actual_regional_forward_failure_keeps_source_cache_and_next_plain_request(cached):
    owner, bare, plan = regional_owner(), model(), LoopPlan(56)
    if cached:
        owner.prepare_control_latent((1, 24, 17, 2, 2))
    source_state = dict(vars(owner))
    selected, _ = configure_loop(bare, plan, control_owners=(owner,))
    local = selected.get_attachment(KEY).controls[0]
    consumed = []

    def fail_after_forward(executor, *args, **kwargs):
        result = executor(*args, **kwargs)
        local.verify_derived()
        assert all(torch.isfinite(value).all() for value in result)
        consumed.append(1)
        raise RuntimeError("failure after actual regional tower forward")

    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "after_control", fail_after_forward)
    with pytest.raises(RuntimeError, match="actual regional tower"):
        sample(selected, plan, steps=1)
    assert consumed == [1]
    assert all(getattr(owner, key) is value for key, value in source_state.items())
    owner.verify_derived()
    local.verify_derived()
    ordinary = bare.clone()
    owner.register(ordinary)
    result = sample(ordinary, plan, steps=1)
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    owner.verify_derived()


def test_branch_owner_and_real_patch_chain_do_not_touch_concurrent_ordinary_owner():
    owner, bare, plan = regional_owner(), model(), LoopPlan(56)
    owner.register(bare)
    original_patch = bare.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    selected, _ = configure_loop(bare, plan, control_owners=(owner,))
    local = selected.get_attachment(KEY).controls[0]
    patch = selected.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    assert patch is not original_patch and patch.control_patch is local
    assert original_patch.control_patch is owner
    wrappers = selected.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
    assert any(getattr(fn, "__self__", None) is local for fn in wrappers)
    assert all(getattr(fn, "__self__", None) is not owner for fn in wrappers)
    inside, release, errors = Event(), Event(), []

    def periodic_request():
        try:
            with torch.no_grad(), periodic_control_scope(local, plan, 1, (1, 24, 27, 2, 2)):
                inside.set()
                if not release.wait(15):
                    raise TimeoutError("ordinary owner did not complete")
                local.verify_derived()
                assert local.control_video.shape[0] == 90
        except BaseException as error:
            errors.append(error)
            inside.set()

    worker = Thread(target=periodic_request)
    worker.start()
    try:
        assert inside.wait(15) and not errors
        assert owner.control_video.shape[0] == 51
        # Real normal Core wrapper + tower + cleanup while the periodic owner
        # holds its rotated media/cache. Model compute itself is not concurrent.
        output = sample(bare, plan, steps=1)
        assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
        owner.prepare_control_latent((1, 24, 17, 2, 2))
        ordinary_state = dict(vars(owner))
    finally:
        release.set()
        worker.join(15)
    assert not worker.is_alive() and not errors
    assert all(getattr(owner, key) is value for key, value in ordinary_state.items())
    owner.verify_derived()
    local.verify_derived()
    # The re-bound branch also reaches the real tower after the interleave.
    output = sample(selected, plan, steps=1)
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
    assert all(getattr(owner, key) is value for key, value in ordinary_state.items())


def test_two_cooperating_scopes_lease_all_owners_in_stable_order_and_restore():
    from contextlib import ExitStack
    from h3_audio_t8_pkg.av_loop_runtime import _control_leases
    owners = (regional_owner(), regional_owner())
    before = [dict(vars(owner)) for owner in owners]
    first_inside, second_started, second_inside, release = Event(), Event(), Event(), Event()
    errors = []

    def consume(ordered, first):
        try:
            if not first:
                second_started.set()
            with torch.no_grad(), ExitStack() as stack:
                stack.enter_context(_control_leases(ordered))
                for owner in ordered:
                    stack.enter_context(periodic_control_scope(owner, LoopPlan(56), int(first), (1, 24, 27, 2, 2)))
                (first_inside if first else second_inside).set()
                if first and not release.wait(15):
                    raise TimeoutError("second scope did not attempt acquisition")
        except BaseException as error:
            errors.append(error)
            first_inside.set()

    first = Thread(target=consume, args=(owners, True))
    second = Thread(target=consume, args=(owners[::-1], False))
    first.start()
    try:
        assert first_inside.wait(15) and not errors
        second.start()
        assert second_started.wait(15)
        assert not second_inside.wait(.1)
    finally:
        release.set()
        first.join(15)
        if second.ident is not None:
            second.join(15)
    assert not first.is_alive() and not second.is_alive() and not errors
    assert second_inside.is_set()
    for owner, state in zip(owners, before):
        assert all(getattr(owner, key) is value for key, value in state.items())
        owner.verify_derived()


@pytest.mark.parametrize("entrypoint", ["wrapper", "cleanup"])
def test_registered_outer_control_and_cleanup_cannot_enter_another_periodic_view(entrypoint):
    owner, bare, plan = regional_owner(), model(), LoopPlan(56)
    entered_prepare, entered_cleanup = Event(), Event()
    base = type(owner)

    class ObservedControl(base):
        def prepare_control_latent(self, shape):
            entered_prepare.set()
            return super().prepare_control_latent(shape)

        def cleanup(self):
            entered_cleanup.set()
            return super().cleanup()

    owner.__class__ = ObservedControl
    owner.register(bare)  # The native wrapper precedes the periodic wrapper.
    selected, _ = configure_loop(bare, plan, control_owners=(owner,))
    local = selected.get_attachment(KEY).controls[0]
    bound_wrapper = next(fn for fn in selected.get_all_wrappers(WrappersMP.DIFFUSION_MODEL)
                         if getattr(fn, "__self__", None) is local)
    started, finished, errors = Event(), Event(), []

    def another_request():
        try:
            with torch.no_grad():
                started.set()
                if entrypoint == "cleanup":
                    local.cleanup()  # Lifecycle may run outside SAMPLER_SAMPLE.
                    result = sample(selected, plan, steps=1)
                else:
                    # Call the actually installed outer native wrapper before
                    # Core's .to lifecycle can mask a missing wrapper lease.
                    source = empty_av_latent(32, 32, plan.length)[0]
                    result = bound_wrapper(lambda *_a, **_k: sample(selected, plan, steps=1),
                                           source["samples"].unbind(), torch.ones(1), None,
                                           {"sigmas": torch.ones(1)})
                assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
                finished.set()
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=another_request)
    try:
        with periodic_control_scope(local, plan, 1, (1, 24, 27, 2, 2)):
            state = (local.control_video, local.control_latent, local.derived_content)
            entered_prepare.clear()
            entered_cleanup.clear()
            worker.start()
            assert started.wait(15)
            attempted = entered_cleanup if entrypoint == "cleanup" else entered_prepare
            assert not attempted.wait(.1)
            assert not entered_prepare.is_set() and not finished.is_set()
            assert local.control_video is state[0] and local.control_latent is state[1]
            assert local.derived_content == state[2]
    finally:
        if worker.ident is not None:
            worker.join(15)
    assert not worker.is_alive() and not errors and finished.is_set()
    assert entered_prepare.is_set() and entered_cleanup.is_set()
    local.verify_derived()


def test_user_cleanup_callback_alone_does_not_count_as_control_registration():
    owner, bare, plan = regional_owner(), model(), LoopPlan(56)
    callbacks, forwards = [], []

    def user_cleanup(self, *_args):
        callbacks.append(self)
        self.cleanup()

    owner.user_cleanup = MethodType(user_cleanup, owner)
    bare.add_callback_with_key(CallbacksMP.ON_CLEANUP, "user_cleanup", owner.user_cleanup)
    selected, _ = configure_loop(bare, plan, control_owners=(owner,))
    local = selected.get_attachment(KEY).controls[0]
    handle = owner.model_patch.model.control_proj_in.register_forward_hook(lambda *_: forwards.append(1))
    try:
        result = sample(selected, plan, steps=1)
    finally:
        handle.remove()
    assert all(torch.isfinite(value).all() for value in result["samples"].unbind())
    assert len(forwards) == 2
    assert callbacks and all(value is local for value in callbacks)
    assert bare.get_all_callbacks(CallbacksMP.ON_CLEANUP)[0].__self__ is owner
    assert not bare.wrappers
