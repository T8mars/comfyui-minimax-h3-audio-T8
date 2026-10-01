"""Local Ref2VA Relay stays paired with one signed Face repair job."""
from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.face_refine_parity_advanced import (
    apply_face_refine_per_frame_denoise, inject_face_refine_parity_video_latent,
)
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav, audit_stage_eav
from h3_audio_t8_pkg.modular_sampling.face_local_relay import bind_local_face_relay
from h3_audio_t8_pkg.modular_sampling.face_stage import (
    bind_multiface_face_stage, bind_parity_face_stage, bind_window_face_stage,
)
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.prompt_relay_advanced import (
    build_prompt_relay_conditioning, build_prompt_relay_plan, prompt_relay_model_contract,
)
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio
from test_face_refine_parity_advanced import TinyVideoVAE, _plan
from test_fast_h3_v2_core_sampler import model as tiny_model
from test_prompt_relay_advanced import NativeLikeFakeClip, _allow_fixture_core_contract
from test_modular_face_multiface import _job as multiface_job
from test_modular_face_window import _job as window_job


def _paired_parity(monkeypatch, *, frames=None, face_plan=None, crops=None,
                   drive_audio=None, parent=None, window=None):
    _allow_fixture_core_contract(monkeypatch)
    frames = torch.zeros((22, 64, 96, 3)) if frames is None else frames
    if face_plan is None:
        face_plan, crops, *_ = _plan(frames)
    if crops is None:
        crops = _plan(frames)[1]
    relay_plan, *_ = build_prompt_relay_plan(
        "Keep the same person and source soundtrack", "Gaze left\nGaze forward",
        22, "auto_equal", "", "paper_v1", .1, False, False)
    relay_model, positive, relay_av, *_ = build_prompt_relay_conditioning(
        model=tiny_model(), clip=NativeLikeFakeClip(),
        video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt_relay_plan=relay_plan, width=384, height=384,
        task_type="Ref2VA", audio_mode="lock_source", audio_denoise_strength=0.,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", execution_mode="apply_exp",
        query_chunk_rows=64, drive_audio=drive_audio or make_audio(seconds=22 / 24),
        ref_images={"ref_image_0": torch.zeros((1, 128, 128, 3))})
    cropped_positive, cropped_av, _ = inject_face_refine_parity_video_latent(
        positive, relay_av, crops, TinyVideoVAE(), face_plan, "require_locked", False)
    masked_av, _ = apply_face_refine_per_frame_denoise(
        cropped_av, face_plan, .8, .35, "relative_to_clip", 30., 120., 1., 9,
        "replace_video_parity", True)
    prepared, _, _ = sampling.setup_dual_clock_sampling(
        relay_model, masked_av, 8, 12., 3., "er_sde", "simple")
    sampler = KSamplerSelect.execute("er_sde").result[0]
    sigmas = BasicScheduler.execute(prepared, "simple", 8, .45).result[0]
    if window is not None:
        source_audio, window_audio, window_plan, mapping = window
        bound, _, _, context, _ = bind_window_face_stage(
            face_plan, frames, parent, window_plan, mapping, source_audio,
            window_audio, prepared, sampler, sigmas, masked_av)
    elif parent is not None:
        bound, _, _, context, _ = bind_multiface_face_stage(
            face_plan, frames, parent, prepared, sampler, sigmas, masked_av)
    else:
        bound, _, _, context, _ = bind_parity_face_stage(
            face_plan, frames, prepared, sampler, sigmas, masked_av)
    args = (bound, cropped_positive, relay_av, relay_plan, face_plan,
            frames, masked_av, context)
    return args, (sampler, sigmas)


def _paired_variant(monkeypatch, variant):
    if variant == "parity":
        args, controls = _paired_parity(monkeypatch)
        return args, controls, {}
    if variant == "multiface":
        parent, frames, face_plan = multiface_job(pad=3)
        args, controls = _paired_parity(monkeypatch, frames=frames,
                                       face_plan=face_plan, parent=parent)
        return args, controls, {"parent_frames": parent}
    parent, source_audio, window_plan, frames, window_audio, mapping, face_plan = window_job()
    args, controls = _paired_parity(monkeypatch, frames=frames,
        face_plan=face_plan, drive_audio=window_audio, parent=parent,
        window=(source_audio, window_audio, window_plan, mapping))
    return args, controls, {"parent_frames": parent, "window_plan": window_plan,
                            "window_mapping": mapping, "source_audio": source_audio,
                            "window_audio": window_audio}


@pytest.mark.parametrize("variant", ("multiface", "window"))
def test_local_relay_binds_parent_or_window_mapping_and_rejects_wrong_parent(monkeypatch, variant):
    args, _, extras = _paired_variant(monkeypatch, variant)
    parent = extras["parent_frames"]
    _, _, report_json = bind_local_face_relay(*args, **extras)
    assert json.loads(report_json)["variant"] == variant
    changed = parent.clone()
    changed[-1] = 1
    with pytest.raises((ValueError, RuntimeError)):
        bind_local_face_relay(*args, **{**extras, "parent_frames": changed})


def test_parity_local_relay_pairs_ref2va_and_preserves_locked_audio(monkeypatch):
    args, _ = _paired_parity(monkeypatch)
    model, positive, report_json = bind_local_face_relay(*args)
    report = json.loads(report_json)
    assert model is args[0] and positive is args[1]
    assert report["variant"] == "parity"
    assert report["local_render_frames"] == 22
    assert prompt_relay_model_contract(model)["binding"]["task"] == "ref2va"
    assert args[2]["samples"].unbind()[1].data_ptr() == args[6]["samples"].unbind()[1].data_ptr()
    assert args[2]["noise_mask"] is not args[6]["noise_mask"]


def test_parity_local_relay_and_eav_observe_actual_shared_stage_calls(monkeypatch):
    args, (sampler, sigmas), extras = _paired_variant(monkeypatch, "parity")
    model, positive, _ = bind_local_face_relay(*args, **extras)
    effected, runtime, _ = apply_stage_eav(model, sigmas, args[6], args[7],
        EAVConfig("report_only", tau=.2, start_video_progress=0.,
                  end_video_progress=1., g_hard_limit=3.))
    result = sample_stage(RandomNoise.execute(913).result[0],
        BasicGuider.execute(effected, positive).result[0],
        sampler, sigmas, args[6], args[7])[2]
    _, report_json = audit_stage_eav(result.output, runtime)
    report = json.loads(report_json)
    assert report["status"] == "observed_report_only"
    assert report["relay_required"] is True
    assert report["relay_attention_calls"] > 0
    assert result.verify()["portable_identity"] is True


@pytest.mark.parametrize("variant", ("multiface", "window"))
def test_parent_bound_local_relay_and_eav_observe_actual_calls(monkeypatch, variant):
    args, (sampler, sigmas), extras = _paired_variant(monkeypatch, variant)
    model, positive, _ = bind_local_face_relay(*args, **extras)
    effected, runtime, _ = apply_stage_eav(model, sigmas, args[6], args[7],
        EAVConfig("report_only", tau=.2, start_video_progress=0.,
                  end_video_progress=1., g_hard_limit=3.))
    result = sample_stage(RandomNoise.execute(914).result[0],
        BasicGuider.execute(effected, positive).result[0],
        sampler, sigmas, args[6], args[7])[2]
    _, report_json = audit_stage_eav(result.output, runtime)
    report = json.loads(report_json)
    assert report["status"] == "observed_report_only"
    assert report["relay_required"] is True
    assert report["relay_attention_calls"] > 0
    assert result.verify()["portable_identity"] is True


@pytest.mark.parametrize("mutation", ("plan", "source", "audio", "conditioning"))
def test_parity_local_relay_rejects_unpaired_sources(monkeypatch, mutation):
    args, _ = _paired_parity(monkeypatch)
    args = list(args)
    if mutation == "plan":
        args[3], *_ = build_prompt_relay_plan(
            "Other scene", "Gaze left\nGaze forward", 22,
            "auto_equal", "", "paper_v1", .1, False, False)
    elif mutation == "source":
        args[5] = args[5].clone()
        args[5][0] = 1
    elif mutation == "audio":
        args[6] = deepcopy(args[6])
        video, audio = args[6]["samples"].unbind()
        args[6]["samples"] = type(args[6]["samples"])((video, audio.clone()))
    else:
        args[1] = deepcopy(args[1])
        args[1][0][1].pop("minimax_prompt_relay_binding")
    with pytest.raises((ValueError, RuntimeError)):
        bind_local_face_relay(*args)
