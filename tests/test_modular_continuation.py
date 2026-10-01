"""Real accepted MP4 + tiny Core continuation; VAE/lifter are explicit doubles."""
import copy
import json
from dataclasses import replace

import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import long_video
from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.modular_sampling import continuation as split
from h3_audio_t8_pkg.modular_sampling import progressive as stages
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE, make_audio
from test_progressive_continuation import accepted  # noqa: F401
from test_progressive_continuation_runtime import run as legacy_run
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401


def prepared(case, frames=22, options=None):
    source = split.capture_source(case.root, **{**case.request, "context_frames": frames})
    contexts = split.prepare_contexts(source, FakeVideoVAE())
    args = dict(clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                prompt="Continue walking.", length=124, **(options or {}))
    low = split.prepare_phase(contexts, "low", **args)
    high = split.prepare_phase(contexts, "high", **args)
    plan = split.build_plan(contexts, 124, native_flow_sigmas(8, 12.))
    return source, contexts, low, high, plan


def low_sample(low, plan, model=None, **kwargs):
    model = model or tiny_model()
    sampler = comfy.samplers.ksampler("euler")
    noise = comfy.sample.prepare_noise(low.result[1]["samples"], 9)
    return split.sample_low(low, model, sampler, plan, noise, seed=9, **kwargs)


def high_restart(high, boundary, model, sampler):
    plan = stages.plan_from_dict(boundary.verify()["request"]["plan"])
    latent = stages.lift_input(boundary, model, sampler)
    video, audio = latent["samples"].unbind()
    lifted, _ = runtime._lift_video(video, audio, plan, {"test_double": True})
    lifted_av = {"samples": comfy.nested_tensor.NestedTensor((lifted, audio))}
    video_noise = comfy.sample.prepare_noise(lifted, 10)
    return split.prepare_high(high, boundary, model, sampler, lifted_av, video_noise)


@pytest.mark.parametrize("frames,steps", [(22, 7), (39, 12)])
@pytest.mark.parametrize("audio_mode", ["native", "lock_source", "remix_source"])
def test_actual_separate_stages_equal_legacy_without_whole_runner_forwarding(
        accepted, stub_lifter, monkeypatch, frames, steps, audio_mode):  # noqa: F811
    options = dict(audio_mode=audio_mode, add_source_as_reference=False,
                   drive_audio=None if audio_mode == "native" else make_audio(6))
    source, contexts, low, high, plan = prepared(accepted, frames, options)
    first, second = tiny_model(), tiny_model()
    with torch.no_grad():
        next(second.model.parameters()).add_(.025)
    original_options = copy.deepcopy(first.model_options)
    expected, _ = legacy_run(source, first, options=options, model_hires=second)
    monkeypatch.setattr(runtime, "sample_progressive_h3", lambda *a, **k: pytest.fail("no run-all"))
    monkeypatch.setattr(split.legacy_source.ProgressiveContinuationSource, "prepare_conditions",
                        lambda *a, **k: pytest.fail("no paired condition preparation"))
    sampler = comfy.samplers.ksampler("euler")
    callbacks = []
    def observe(step, _x, _y, total):
        callbacks.append((step, total))
    boundary = low_sample(low, plan, first, callback=observe)
    assert boundary.verify()["execution"]["actual_apply_calls"] == 4
    assert callbacks == [(i, 8) for i in range(4)]
    restart = high_restart(high, boundary, second, sampler)
    result, report = split.sample_high(high, restart, second, sampler, seed=9, callback=observe)
    assert json.loads(report)["execution"]["actual_apply_calls"] == 4
    assert callbacks == [(i, 8) for i in range(8)]
    for actual, target in zip(result.output["samples"].unbind(), expected["samples"].unbind(), strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert contexts.low["audio_tail"] is contexts.high["audio_tail"]
    assert torch.count_nonzero(low.result[1]["samples"].unbind()[0]) == 0
    masks = high.result[1]["noise_mask"].unbind()
    assert torch.all(masks[0][:, :, :steps] == 0) and torch.all(masks[0][:, :, steps:] == 1)
    assert first.model_options == original_options
    assert not first.object_patches and not second.object_patches
    assert not first.wrappers and not second.wrappers
    result.verify()
    contexts.verify()


def test_high_prompt_is_independent_of_frozen_low(accepted, stub_lifter, monkeypatch):  # noqa: F811
    _, contexts, low, high, plan = prepared(accepted)
    boundary = low_sample(low, plan)
    low_identity = low.verify()
    boundary_identity = boundary.verify()
    clip, vae = FakeClip(), FakeVideoVAE()
    monkeypatch.setattr(split, "sample_low", lambda *a, **k: pytest.fail("LOW must not rerun"))
    changed = split.prepare_phase(contexts, "high", clip=clip, video_vae=vae, audio_vae=FakeAudioVAE(),
                                  prompt="Turn to the right.", length=124)
    assert low.verify() == low_identity and boundary.verify() == boundary_identity
    assert changed.verify() != high.verify()
    assert clip.tokenize_calls and not vae.encode_calls
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    result, text = split.sample_high(changed, high_restart(changed, boundary, model, sampler),
                                    model, sampler, seed=9)
    assert json.loads(text)["execution"]["actual_apply_calls"] == 4
    result.verify()


@pytest.mark.parametrize("phase", ["low", "high"])
def test_cancellation_releases_chain_and_preserves_inputs(accepted, stub_lifter, phase):  # noqa: F811
    source, contexts, low, high, plan = prepared(accepted)
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    before = contexts.verify()
    def cancel(*args):
        raise InterruptedError("cancel individual phase")
    if phase == "low":
        with pytest.raises(InterruptedError):
            low_sample(low, plan, model, callback=cancel)
        low_sample(low, plan, model).verify()
    else:
        boundary = low_sample(low, plan, model)
        restart = high_restart(high, boundary, model, sampler)
        with pytest.raises(InterruptedError):
            split.sample_high(high, restart, model, sampler, seed=9, callback=cancel)
        split.sample_high(high, restart, model, sampler, seed=9)[0].verify()
    with split.chain_guard(source.root):
        assert source.revalidate() == source.binding
    assert contexts.verify() == before
    assert not model.object_patches and not model.wrappers


def test_busy_lease_does_not_unlock_other_owner_or_create_missing_chain(accepted, tmp_path):  # noqa: F811
    with split.chain_guard(accepted.root):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Another runner"):
                split.capture_source(accepted.root, **accepted.request)
    split.capture_source(accepted.root, **accepted.request)
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        split.capture_source(missing, **accepted.request)
    assert not missing.exists()


@pytest.mark.parametrize("fault", ["revision", "accepted_video", "accepted_context", "low_tensor", "audio_object"])
def test_changed_parent_or_prepared_context_rejected_before_sampling(accepted, monkeypatch, fault):  # noqa: F811
    _, contexts, low, _, plan = prepared(accepted)
    if fault == "revision":
        accepted.manifest["revision"] = 2
        accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    elif fault == "accepted_video":
        accepted.accepted_media.write_bytes(b"changed")
    elif fault == "accepted_context":
        # Safetensors keeps live mmap views: replace the path, do not truncate
        # an open Windows mapping (that tests the OS, not source revalidation).
        changed = accepted.accepted_context.with_suffix(".replacement")
        changed.write_bytes(b"changed")
        changed.replace(accepted.accepted_context)
    elif fault == "low_tensor":
        with torch.inference_mode():
            contexts.low["video_tail"].add_(.1)
    else:
        contexts.low["audio_tail"] = contexts.low["audio_tail"].clone()
    monkeypatch.setattr(stages, "sample_low", lambda *a, **k: pytest.fail("invalid parent must not sample"))
    with pytest.raises(ValueError):
        low_sample(low, plan)


@pytest.mark.parametrize("field", ["minimax_keyframes", "minimax_frame_count", "minimax_refs",
                                   long_video.LONG_VIDEO_CONDITIONING_KEY])
def test_external_conditioning_must_retain_motion_guides_and_clock(accepted, monkeypatch, field):  # noqa: F811
    _, _, low, _, plan = prepared(accepted)
    changed = copy.deepcopy(low.result[0])
    changed[0][1].pop(field)
    monkeypatch.setattr(stages, "sample_low", lambda *a, **k: pytest.fail("must reject changed guide"))
    with pytest.raises(ValueError, match="guides/grid"):
        low_sample(low, plan, positive=changed)


def test_external_embeddings_allowed_and_user_wrapper_executes(accepted):  # noqa: F811
    _, _, low, _, plan = prepared(accepted)
    changed = copy.deepcopy(low.result[0])
    changed[0][0].add_(.2)
    model, calls = tiny_model(), []
    def delegate(apply_model, args):
        calls.append(True)
        return apply_model(args["input"], args["timestep"], **args["c"])
    model.set_model_unet_function_wrapper(delegate)
    result = low_sample(low, plan, model, positive=changed)
    assert len(calls) == 4 and not result.verify()["portable_identity"]
    assert model.model_options["model_function_wrapper"] is delegate


@pytest.mark.parametrize("fault", ["phase", "plan", "guide", "mask", "audio"])
def test_stage_input_mismatch_rejected_without_running_high(accepted, stub_lifter, monkeypatch, fault):  # noqa: F811
    _, _, low, high, plan = prepared(accepted)
    model, sampler = tiny_model(), comfy.samplers.ksampler("euler")
    boundary = low_sample(low, plan, model)
    if fault == "phase":
        high = low
    elif fault == "plan":
        high = replace(high, phase="low")
    elif fault == "guide":
        high.result[0][0][1]["minimax_keyframes"][0]["latent"].add_(1)
    elif fault == "mask":
        high.result[1]["noise_mask"].unbind()[0].fill_(1)
    else:
        high.result[1]["samples"].unbind()[1].add_(1)
    monkeypatch.setattr(stages, "prepare_high", lambda *a, **k: pytest.fail("invalid handoff"))
    with pytest.raises(ValueError):
        high_restart(high, boundary, model, sampler)


def test_changed_parent_during_low_rejected_and_lease_released(accepted):  # noqa: F811
    _, _, low, _, plan = prepared(accepted)
    def change_parent(step, *_args):
        if step == 0:
            accepted.manifest["revision"] = 2
            accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    with pytest.raises(ValueError, match="revision"):
        low_sample(low, plan, callback=change_parent)
    with split.chain_guard(accepted.root):
        pass


@pytest.mark.parametrize("options", [{"segment_index": 0}, {"width": 256}, {"return_details": False}])
def test_phase_options_cannot_change_verified_parent_canvas(accepted, options):  # noqa: F811
    _, contexts, _, _, _ = prepared(accepted)
    with pytest.raises(ValueError, match="override"):
        split.prepare_phase(contexts, "low", clip=None, video_vae=None, audio_vae=None,
                            prompt="test", length=124, **options)
