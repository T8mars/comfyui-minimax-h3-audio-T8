"""Actual CPU HyperFlow50+2 and miniature video Encoder with middle frames.

Only text embeddings, adapter tensors and the explicitly named FakeAudioVAE
are synthetic. No sampler, DiT forward, attention or video Encoder is mocked.
These are execution contracts, not trained model/media quality qualification.
"""
import json

import comfy.sample
import pytest
import torch

from h3_audio_t8_pkg import hyperflow_runtime_advanced as runtime
from h3_audio_t8_pkg import hyperflow_sampling_advanced as native
from h3_audio_t8_pkg import hyperflow_two_pass_advanced as combined
from h3_audio_t8_pkg.long_video_dual_identity import _original_state
from h3_audio_t8_pkg.modular_sampling import hyperflow as separate
from h3_audio_t8_pkg.modular_sampling import hyperflow_effects as effects
from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh
from h3_audio_t8_pkg.multikeyframe_advanced import ACTUAL_FRAME_INDEX, append_keyframe_plan
from h3_audio_t8_pkg.nodes_director_keyframes import MiniMaxH3DirectorKeyframeConditioningEXPT8
from h3_audio_t8_pkg.progressive_sampling_runtime import _restore_stage_objects
from h3_audio_t8_pkg.prompt_relay_advanced import (
    _layout_contract, build_prompt_relay_plan, prompt_relay_model_contract,
)
from helpers import FakeAudioVAE, make_audio
from test_h21_control_regions import RealTinyEncoderOwner
from test_h21_director_keyframe_runtime import CountingNativeLikeFakeClip, actual_payload
from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
from test_progressive_sampling_runtime import tiny_model


@pytest.fixture(scope="module")
def actual_video_encoder():
    return RealTinyEncoderOwner()


@pytest.fixture
def actual_base(monkeypatch):
    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    original = {key: value.detach().clone() for key, value in base.model_state_dict().items()}
    return base, _tiny_weights(diffusion), original


def prepare(base, weights, encoder, *, relay, nonuniform=False, recording=False,
            source_as_reference=False):
    selected, _, loader_report = fresh.install(base, weights)
    assert loader_report["endpoint_snapshot"] == "original_base_via_core_backups"
    image = torch.linspace(.05, .95, 32 * 32 * 3).reshape(1, 32, 32, 3)
    before_image = image.clone()
    plan = append_keyframe_plan(None, image * .9, "frame", 1,
        .92 if nonuniform else .999, "stretch", True)
    clip, audio_vae = CountingNativeLikeFakeClip(), FakeAudioVAE()
    before_calls = len(encoder.calls)
    drive = make_audio(22 / 24, value=.13) if recording else None
    final = make_audio(22 / 24, value=.23) if recording else None
    kwargs = dict(model=selected, clip=clip, video_vae=encoder, audio_vae=audio_vae,
        prompt="A person moves.", width=32, height=32, length=22,
        task_type="auto" if recording else "FL2VA",
        audio_mode="lock_source" if recording else "native", audio_denoise_strength=.35,
        add_source_as_reference=source_as_reference, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        first_frame_noise_aug=.999, last_frame_noise_aug=.999, reference_visual_noise_aug=.999,
        first_frame=image, last_frame=None if recording else image * .8,
        keyframe_plan=plan, drive_audio=drive, final_audio=final)
    if relay:
        kwargs.update(prompt_relay_plan=build_prompt_relay_plan("A person moves.",
            "Raise hand.\nTurn.", 22, "auto_equal", "", "paper_v1", .1, False, False)[0],
            execution_mode="apply_exp", query_chunk_rows=64)
    result = MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(**kwargs).result
    model, positive, source = result[:3]
    assert len(clip.tokenize_calls) == len(clip.encode_calls) == 1
    expected_images = 2 if recording else 3
    assert len(encoder.calls) - before_calls == expected_images
    assert all(tuple(item.shape) == (1, 32, 32, 3) for item in encoder.calls[before_calls:])
    payload = actual_payload(model, positive, source)
    assert [item[ACTUAL_FRAME_INDEX] for item in payload["keyframes"]] == (
        [0, 1] if recording else [0, 1, 21])
    if relay:
        assert _layout_contract(payload["layout"]) == prompt_relay_model_contract(model)["binding"]["layout_contract"]
    if recording:
        assert result[3] is final and result[3] is not drive
        assert len(audio_vae.encode_calls) == 1
        assert source["noise_mask"].unbind()[1].count_nonzero() == 0
        assert source["samples"].unbind()[1].count_nonzero() > 0
        assert len(payload.get("refs", [])) == int(source_as_reference)
    assert torch.equal(before_image, image)
    return result


def run_route(low, high, route):
    model, positive, source = low[:3]
    high_model, high_positive = high[:2]
    before = [value.clone() for value in source["samples"].unbind()]
    calls, report = [], None

    def callback(step, _prediction, _state, total):
        calls.append((step, total))

    if route == "single8":
        branch, sampler, sigmas = native.setup_hyperflow_sampler(model, source,
            native.build_hyperflow_plan(model))
        try:
            result = comfy.sample.sample_custom(branch,
                comfy.sample.prepare_noise(source["samples"], 21102194), 1., sampler, sigmas,
                positive, [], source["samples"], noise_mask=source.get("noise_mask"),
                disable_pbar=True, seed=21102194, callback=callback)
        finally:
            _restore_stage_objects(branch)
        output = {**source, "samples": result}
    elif route == "combined":
        output, report = combined.sample_hyperflow_split(model, high_model, positive, source,
            positive_high=high_positive, seed=21102194, split_interval=4, callback=callback)
        assert json.loads(report)["stage_nfe"] == [4, 4]
    else:
        head_model, head_positive, head_negative = effects.bind_head(model, source, positive, [], 4)[:3]
        boundary = separate.sample_head(head_model, source, head_positive, head_negative,
            comfy.sample.prepare_noise(source["samples"], 21102194),
            seed=21102194, split_interval=4, callback=callback)
        receipt = boundary.verify()
        # Exact own Advanced closures now have inspection-only projection.
        assert receipt["execution"]["portable_identity"] is True
        tail_model, tail_positive, tail_negative = effects.bind_tail(
            boundary, high_model, high_positive, [])[:3]
        output, report = separate.sample_tail(boundary, tail_model, tail_positive, tail_negative,
            seed=21102194, callback=callback)
        assert json.loads(report)["execution"]["portable_identity"] is True
    assert calls == [(index, 8) for index in range(8)]
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
    assert all(torch.equal(a, b) for a, b in zip(before, source["samples"].unbind(), strict=True))
    assert runtime.active_step() is None and runtime._ACTIVE_FORWARD.get() is None
    assert not model.object_patches_backup and not high_model.object_patches_backup
    assert not torch.cuda.is_initialized()
    return output, report


def check_original(base, original):
    recovered = _original_state(base, base.model_state_dict())
    assert all(torch.equal(value, recovered[key]) for key, value in original.items())
    assert not base.object_patches and not base.wrappers


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("source_as_reference", [False, True])
def test_recording_first_and_middle_without_last_really_condition_and_sample(
        actual_base, actual_video_encoder, relay, source_as_reference):
    base, weights, original = actual_base
    first = prepare(base, weights, actual_video_encoder, relay=relay, recording=True,
        source_as_reference=source_as_reference)
    output, _ = run_route(first, first, "single8")
    torch.testing.assert_close(output["samples"].unbind()[1], first[2]["samples"].unbind()[1],
        rtol=0, atol=torch.finfo(torch.float32).eps)
    second = prepare(base, weights, actual_video_encoder, relay=relay, recording=True,
        source_as_reference=source_as_reference)
    retry, _ = run_route(second, second, "single8")
    for actual, expected in zip(retry["samples"].unbind(), output["samples"].unbind(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    check_original(base, original)


@pytest.mark.parametrize("route", ["single8", "combined", "separate"])
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_actual_hyperflow_middle_frame_routes_and_fresh_next_request(
        actual_base, actual_video_encoder, route, relay, nonuniform):
    base, weights, original = actual_base
    low = prepare(base, weights, actual_video_encoder, relay=relay, nonuniform=nonuniform)
    high = prepare(base, weights, actual_video_encoder, relay=relay, nonuniform=nonuniform)
    output, _ = run_route(low, high, route)
    # Install a new real owner after Core has loaded the sibling LoRA endpoint.
    next_low = prepare(base, weights, actual_video_encoder, relay=relay, nonuniform=nonuniform)
    next_high = prepare(base, weights, actual_video_encoder, relay=relay, nonuniform=nonuniform)
    next_output, _ = run_route(next_low, next_high, route)
    for actual, expected in zip(next_output["samples"].unbind(), output["samples"].unbind(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    check_original(base, original)
