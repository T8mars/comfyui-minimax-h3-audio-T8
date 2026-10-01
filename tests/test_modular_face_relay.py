"""Full-clip Face Relay must bind the crop and retain source audio."""
from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg.face_refine_advanced import (
    inject_face_refine_video_latent, setup_face_refine_sampling,
)
from h3_audio_t8_pkg.modular_sampling.face_relay import bind_standard_face_relay
from h3_audio_t8_pkg.modular_sampling.face_stage import bind_standard_face_stage
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav, audit_stage_eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.prompt_relay_advanced import (
    PROMPT_RELAY_BINDING_KEY, build_prompt_relay_conditioning, build_prompt_relay_plan,
    prompt_relay_model_contract,
)
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio
from test_face_refine_advanced import TinyVideoVAE, _plan
from test_fast_h3_v2_core_sampler import model as tiny_model
from test_prompt_relay_advanced import NativeLikeFakeClip, _allow_fixture_core_contract


def _paired(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    frames = torch.zeros((22, 64, 96, 3))
    face_plan, crops, *_ = _plan(frames)
    relay_plan, *_ = build_prompt_relay_plan(
        "Same adult face and source soundtrack", "Eyes look left\nEyes look forward",
        22, "auto_equal", "", "paper_v1", .1, False, False)
    relay_model, positive, relay_av, *_ = build_prompt_relay_conditioning(
        model=tiny_model(), clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt_relay_plan=relay_plan, width=384, height=384,
        task_type="T2VA", audio_mode="lock_source", audio_denoise_strength=0.,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", execution_mode="apply_exp",
        query_chunk_rows=64, drive_audio=make_audio(seconds=22 / 24))
    cropped_positive, cropped_av, _ = inject_face_refine_video_latent(
        positive, relay_av, crops, TinyVideoVAE(), face_plan, "require_locked", False)
    sampled_model, sampler, sigmas, _ = setup_face_refine_sampling(
        relay_model, cropped_av, 2, .5, 12., 3., "dual_clock_euler", "native_flow")
    bound, _sampler, _table, context, _ = bind_standard_face_stage(
        face_plan, frames, sampled_model, sampler, sigmas, cropped_av)
    return (bound, cropped_positive, relay_av, relay_plan, face_plan,
            frames, cropped_av, context), (sampler, sigmas)


def test_face_relay_pairs_full_clip_crop_and_preserves_source_audio(monkeypatch):
    args, _ = _paired(monkeypatch)
    model, positive, report_json = bind_standard_face_relay(*args)
    assert model is args[0] and positive is args[1]
    assert json.loads(report_json)["status"] == "paired_crop_model_and_conditioning"
    assert prompt_relay_model_contract(model)["binding"]["plan_hash"] == args[3]["plan_hash"]
    assert args[2]["samples"].unbind()[1].data_ptr() == args[6]["samples"].unbind()[1].data_ptr()


@pytest.mark.parametrize("mutation", ("timeline", "positive", "audio", "source"))
def test_face_relay_refuses_unpaired_or_changed_crop(monkeypatch, mutation):
    args, _ = _paired(monkeypatch)
    args = list(args)
    if mutation == "timeline":
        args[3], *_ = build_prompt_relay_plan(
            "Other face", "A\nB", 22, "auto_equal", "", "paper_v1", .1, False, False)
    elif mutation == "positive":
        args[1] = deepcopy(args[1])
        args[1][0][1][PROMPT_RELAY_BINDING_KEY] = {}
    elif mutation == "audio":
        args[6] = dict(args[6])
        video, audio = args[6]["samples"].unbind()
        args[6]["samples"] = type(args[6]["samples"])((video, audio.clone()))
    else:
        args[5] = args[5].clone()
        args[5][0] = 1
    with pytest.raises((ValueError, RuntimeError)):
        bind_standard_face_relay(*args)


def test_face_relay_combined_eav_reports_real_calls_without_replacing_audio(monkeypatch):
    args, (sampler, sigmas) = _paired(monkeypatch)
    model, positive, _ = bind_standard_face_relay(*args)
    effected, runtime, _ = apply_stage_eav(
        model, sigmas, args[6], args[7],
        EAVConfig("report_only", tau=4., start_video_progress=0., end_video_progress=1.,
                  g_hard_limit=3.))
    result = sample_stage(RandomNoise.execute(451).result[0],
                          BasicGuider.execute(effected, positive).result[0],
                          sampler, sigmas, args[6], args[7])[2]
    _, report_json = audit_stage_eav(result.output, runtime)
    report = json.loads(report_json)
    assert report["status"] == "observed_report_only"
    assert report["relay_required"] is True
    assert report["relay_attention_calls"] > 0
    assert args[2]["samples"].unbind()[1].data_ptr() == args[6]["samples"].unbind()[1].data_ptr()


def test_face_relay_only_routes_actual_face_stage_attention(monkeypatch):
    args, (sampler, sigmas) = _paired(monkeypatch)
    model, positive, _ = bind_standard_face_relay(*args)
    result = sample_stage(RandomNoise.execute(451).result[0],
                          BasicGuider.execute(model, positive).result[0],
                          sampler, sigmas, args[6], args[7])[2]
    counts = prompt_relay_model_contract(model)["execution_counts"]
    assert result.verify()["execution"]["denoiser_evaluations"] == 2
    assert counts["completed_forwards"] == 2
    assert counts["routed_attention_calls"] > 0
