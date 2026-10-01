"""Real tiny Core SPEED stages; random weights do not prove trained quality."""
from __future__ import annotations

import comfy.nested_tensor
import pytest
import torch
from comfy.ldm.modules import attention

from h3_audio_t8_pkg import speed_advanced as old
from h3_audio_t8_pkg.modular_sampling import speed_stages as split
from h3_audio_t8_pkg.modular_sampling import speed_nodes, speed_storage
from h3_audio_t8_pkg.modular_sampling import eav as stage_eav
from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
from test_progressive_sampling_runtime import tiny_model, conditioning
from test_prompt_relay_advanced import NativeLikeFakeClip
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio


def _source():
    return {
        "schema": old.SPEED_SOURCE_SCHEMA,
        "resolved_task": "t2va",
        "audio_mode": "native",
        "length": 5,
        "prompt": "one scene",
    }


def _plan(scales="0.5,1.0", thresholds="0.85", width=64):
    return old.build_speed_plan(
        width=width, height=width, steps=20, scales=scales,
        transition_mode="manual_sigmas", manual_transition_sigmas=thresholds,
        delta=0.01, shift_video=12.0, transform="dct",
        profile_policy="require_validated_profile", fallback_policy="error",
    )[0]


def _conditioning(_source, width, height, template=None):
    latent = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 2, height // 16, width // 16),
        torch.zeros(1, 32, 2, 8),
    ))}
    positive = template.positive if template is not None else conditioning()
    return old.StageConditioning(positive, latent, None, "one scene", "{}", "tiny", "reused" if template else "rebuilt")


def _model():
    model = tiny_model()
    set_h3_attention_backend(model, attention.attention_pytorch)
    return model


@pytest.mark.parametrize("scales,thresholds,width", [
    ("0.5,1.0", "0.85", 64),
    ("0.4,0.7,1.0", "0.94,0.78", 96),
])
def test_public_speed_stages_match_old_whole_chain_exactly(monkeypatch, scales, thresholds, width):
    monkeypatch.setattr(old, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(split, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(old, "_build_empty_t2va_stage", lambda source, width, height, prior: _conditioning(source, width, height, prior))
    monkeypatch.setattr(split, "_build_empty_t2va_stage", lambda source, width, height, prior: _conditioning(source, width, height, prior))
    monkeypatch.setattr(old, "_release_h3_residency_between_stages", lambda _model: {"performed": False})
    monkeypatch.setattr(old, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    plan, source, seed = _plan(scales, thresholds, width), _source(), 731
    baseline, *_ = old.execute_speed_sampling(
        _model(), plan, source, shift_audio=3.0, seed=seed,
        execution_scope="strict_t2va_stock20", dct_chunk_size=4,
    )
    prior_spec = prior_result = None
    stages = []
    for index in range(len(plan["stages"])):
        prepared = speed_nodes.MiniMaxH3SPEEDStageSetupEXPT8.execute(
            _model(), plan, source, stage_index=index, seed=seed, previous_spec=prior_spec,
        ).result
        noise = (old.H3ModalityStableNoise(seed) if index == 0 else
                 speed_nodes.MiniMaxH3SPEEDDCTTransitionEXPT8.execute(prior_result, prepared[5], 4).result[0])
        output, result, _ = speed_nodes.MiniMaxH3SPEEDStageSampleEXPT8.execute(
            prepared[0], prepared[1], prepared[2], prepared[3], prepared[4], noise, prepared[5],
        ).result
        receipt = result.verify_live()
        assert receipt["verified_recipe_completion"] is True
        assert receipt["portable_identity"] is True
        assert receipt["execution"]["callbacks"] == list(range(prepared[5].nfe))
        stages.append(prepared)
        prior_spec, prior_result = prepared[5], result
    for actual, reference in zip(output["samples"].unbind(), baseline["samples"].unbind()):
        assert torch.equal(actual, reference)
    assert sum(item[5].nfe for item in stages) == 20
    assert prior_result.spec.index == len(stages) - 1
    assert all(part.device.type == "cpu" for part in output["samples"].unbind())


@pytest.mark.parametrize("task,audio_mode,first,last,with_ref", [
    ("I2VA", "lock_source", True, False, False),
    ("FL2VA", "remix_source", True, True, False),
    ("L2VA", "native", False, True, False),
    ("Ref2VA", "native", False, False, True),
    ("Hybrid", "native", True, False, True),
])
def test_speed_multimodal_split_matches_old_native_conditioning_and_final_av(
    monkeypatch, tmp_path, task, audio_mode, first, last, with_ref,
):
    monkeypatch.setattr(old, "_release_h3_residency_between_stages", lambda _model: {"performed": False})
    monkeypatch.setattr(old, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    source, _ = old.build_speed_source(
        clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt="Two-canvas multimodal comparison", length=5, task_type=task,
        audio_mode=audio_mode, audio_denoise_strength=0.35,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match", reference_video_policy="official_2_to_15s",
        checkpoint_fingerprint="unrecorded", vae_fingerprint="unrecorded",
        drive_audio=make_audio(seconds=1.0) if audio_mode != "native" else None,
        final_audio=None,
        first_frame=torch.zeros(1, 96, 160, 3) if first else None,
        last_frame=torch.ones(1, 96, 160, 3) if last else None,
        ref_images={"ref_image_0": torch.full((1, 96, 160, 3), 0.6)} if with_ref else None,
        ref_videos=None, ref_video_audios=None, ref_audios=None,
    )
    plan, seed = _plan(), 132
    baseline, *_ = old.execute_speed_sampling(
        _model(), plan, source, shift_audio=3.0, seed=seed,
        execution_scope="multimodal_research_exp", dct_chunk_size=4,
    )
    prior_spec = prior_result = None
    reference_grids = []
    for index in range(2):
        prepared = split.prepare_speed_stage(
            _model(), plan, source, stage_index=index, shift_audio=3.0,
            seed=seed, execution_scope="multimodal_research_exp",
            previous_spec=prior_spec, reuse_t2va_text=False,
        )
        metadata = prepared[1][0][1]
        keyframes = metadata.get("minimax_keyframes", [])
        assert len(keyframes) == int(first) + int(last)
        assert all(tuple(item["latent"].shape[-2:]) == tuple(prepared[2]["samples"].unbind()[0].shape[-2:])
                   for item in keyframes)
        refs = metadata.get("minimax_refs", [])
        assert len(refs) == int(with_ref)
        if with_ref:
            reference_grids.append(tuple(refs[0]["latent"].shape[-2:]))
        if audio_mode == "native":
            assert "noise_mask" not in prepared[2]
        else:
            _, audio_mask = prepared[2]["noise_mask"].unbind()
            expected_mask = 0.0 if audio_mode == "lock_source" else 0.35
            assert torch.all(audio_mask == expected_mask)
        noise = (old.H3ModalityStableNoise(seed) if index == 0 else
                 split.handoff_speed_stage(prior_result, prepared[5], dct_chunk_size=4)[0])
        output, prior_result, _ = split.sample_speed_stage(
            prepared[0], prepared[1], prepared[2], prepared[3], prepared[4], noise, prepared[5],
        )
        receipt = prior_result.verify_live()
        assert receipt["verified_recipe_completion"] is True
        assert receipt["portable_identity"] is True
        prior_spec = prepared[5]
        if index == 0:
            _, path, digest, _ = speed_storage.save_speed_stage(prior_result, output_root=tmp_path)
            restored, frozen_spec, load_report = speed_storage.load_speed_stage(
                plan, seed, 3.0, 1, path, digest, output_root=tmp_path,
            )
            assert restored.verify_live()["receipt_sha256"] == receipt["receipt_sha256"]
            assert '"sampling_calls":0' in load_report
            prior_result, prior_spec = restored, frozen_spec
    assert all(torch.equal(actual, reference) for actual, reference in
               zip(output["samples"].unbind(), baseline["samples"].unbind()))
    if with_ref:
        assert all(high > low for low, high in zip(reference_grids[0], reference_grids[1]))


@pytest.mark.parametrize("with_soundtrack", [False, True])
def test_speed_reference_video_and_audio_split_matches_old_av(
    monkeypatch, tmp_path, with_soundtrack,
):
    monkeypatch.setattr(old, "_release_h3_residency_between_stages", lambda _model: {"performed": False})
    monkeypatch.setattr(old, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    video = torch.full((48, 64, 64, 3), 0.4)
    voice = make_audio(seconds=2.0)
    soundtrack = make_audio(seconds=2.0)
    source, _ = old.build_speed_source(
        clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt="Reference video and audio comparison", length=5, task_type="Ref2VA",
        audio_mode="native", audio_denoise_strength=0.35,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match", reference_video_policy="official_2_to_15s",
        checkpoint_fingerprint="unrecorded", vae_fingerprint="unrecorded",
        drive_audio=None, final_audio=None, first_frame=None, last_frame=None,
        ref_images=None, ref_videos={"ref_video_1": video},
        ref_video_audios={"ref_video_audio_1": soundtrack} if with_soundtrack else None,
        ref_audios={"ref_audio_0": voice},
    )
    plan, seed = _plan(), 925
    baseline, *_ = old.execute_speed_sampling(
        _model(), plan, source, shift_audio=3.0, seed=seed,
        execution_scope="multimodal_research_exp", dct_chunk_size=4,
    )
    prior_spec = prior_result = None
    for index in range(2):
        prepared = split.prepare_speed_stage(
            _model(), plan, source, stage_index=index, shift_audio=3.0,
            seed=seed, execution_scope="multimodal_research_exp",
            previous_spec=prior_spec, reuse_t2va_text=False,
        )
        refs = prepared[1][0][1]["minimax_refs"]
        assert [item["kind"] for item in refs] == (
            ["video_audio", "audio"] if with_soundtrack else ["video", "audio"]
        )
        assert all(item["audio_latent"] is not None for item in refs if item["kind"] != "video")
        noise = (old.H3ModalityStableNoise(seed) if index == 0 else
                 split.handoff_speed_stage(prior_result, prepared[5], dct_chunk_size=4)[0])
        output, prior_result, _ = split.sample_speed_stage(
            prepared[0], prepared[1], prepared[2], prepared[3], prepared[4], noise, prepared[5],
        )
        assert prior_result.verify_live()["portable_identity"] is True
        prior_spec = prepared[5]
        if index == 0:
            _, path, digest, _ = speed_storage.save_speed_stage(prior_result, output_root=tmp_path)
            prior_result, prior_spec, report = speed_storage.load_speed_stage(
                plan, seed, 3.0, 1, path, digest, output_root=tmp_path,
            )
            assert '"sampling_calls":0' in report
    assert all(torch.equal(actual, reference) for actual, reference in
               zip(output["samples"].unbind(), baseline["samples"].unbind()))
    assert torch.all(video == 0.4)


@pytest.mark.parametrize("mode", (
    "i2va_lock_source", "fl2va_remix_source", "l2va_native",
    "ref_image_native", "ref_video_audio_native", "hybrid_first_image_audio",
))
def test_formal_multimodal_relay_rebuilds_same_stage_media_target(mode, monkeypatch, tmp_path):
    from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
    from tools import build_formal_speed_multimodal_workflows as formal

    graph = formal.graph_for(mode, "combined", "all", "full_save")
    setup = graph["11"]["inputs"]
    relay_settings = {**graph["16"]["inputs"], "length": 5,
                      "local_prompts": "Maintain one coherent shot."}
    relay_plan, compiled_prompt, *_ = build_prompt_relay_plan(**relay_settings)
    image = torch.zeros(1, 96, 160, 3)
    audio = make_audio(seconds=2.0)
    video = torch.full((48, 64, 64, 3), 0.4)
    media = {
        "i2va_lock_source": dict(first_frame=image, drive_audio=audio),
        "fl2va_remix_source": dict(first_frame=image, last_frame=image + 0.2,
                                    drive_audio=audio),
        "l2va_native": dict(last_frame=image + 0.2),
        "ref_image_native": dict(ref_images={"ref_image_0": image + 0.3}),
        "ref_video_audio_native": dict(ref_videos={"ref_video_0": video},
                                       ref_video_audios={"ref_video_audio_0": audio}),
        "hybrid_first_image_audio": dict(first_frame=image,
                                         ref_images={"ref_image_0": image + 0.3},
                                         ref_audios={"ref_audio_0": audio}),
    }[mode]
    source, _ = old.build_speed_source(
        clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt=compiled_prompt, length=5, task_type=setup["task_type"],
        audio_mode=setup["audio_mode"],
        audio_denoise_strength=setup["audio_denoise_strength"],
        add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        checkpoint_fingerprint="unrecorded", vae_fingerprint="unrecorded",
        **media,
    )
    previous_spec = previous_result = None
    sample_reference = mode in ("ref_video_audio_native", "hybrid_first_image_audio")
    if sample_reference:
        monkeypatch.setattr(split, "_apply_speed_scoped_headroom",
                            lambda: (None, {"applied": False}))
    for index in range(2):
        prepared = speed_nodes.MiniMaxH3SPEEDStageSetupEXPT8.execute(
            _model(), _plan(), source, stage_index=index, seed=123,
            execution_scope="multimodal_research_exp", reuse_t2va_text=False,
            previous_spec=previous_spec,
        ).result
        relayed = speed_nodes.MiniMaxH3SPEEDRelayApplyEXPT8.execute(
            prepared[0], prepared[5], source, relay_plan, execution_mode="apply_exp",
        ).result
        for actual, expected in zip(relayed[2]["samples"].unbind(),
                                    prepared[2]["samples"].unbind()):
            assert torch.equal(actual, expected)
        assert ("noise_mask" in relayed[2]) == ("noise_mask" in prepared[2])
        previous_spec = prepared[5]
        if sample_reference:
            selected, runtime, _ = stage_eav.apply_stage_eav(
                relayed[0], prepared[4], relayed[2], prepared[10],
                stage_eav.EAVConfig("report_only", 0.1, 0.0, 1.0, 32, 3.0),
            )
            noise = (old.H3ModalityStableNoise(123) if index == 0 else
                     split.handoff_speed_stage(previous_result, prepared[5], dct_chunk_size=4)[0])
            _, result, _ = split.sample_speed_stage(
                selected, relayed[1], relayed[2], prepared[3], prepared[4], noise, prepared[5],
            )
            assert result.verify_live()["portable_identity"] is True
            assert runtime.snapshot()["status"] == "observed_report_only"
            if index == 0:
                _, path, digest, _ = speed_storage.save_speed_stage(result, output_root=tmp_path)
                previous_result, previous_spec, _ = speed_storage.load_speed_stage(
                    prepared[5].plan, 123, 3.0, 1, path, digest, output_root=tmp_path,
                )
            else:
                previous_result = result


@pytest.mark.parametrize("mode", ["disabled", "report_only", "apply_exp"])
def test_speed_stage_external_eav_actual_calls(monkeypatch, mode):
    monkeypatch.setattr(split, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    seed = 119
    prepared = speed_nodes.MiniMaxH3SPEEDStageSetupEXPT8.execute(
        _model(), _plan(), _source(), stage_index=0, seed=seed,
    ).result
    eav_model, runtime, _ = stage_eav.apply_stage_eav(
        prepared[0], prepared[4], prepared[2], prepared[10],
        stage_eav.EAVConfig(mode, 0.1, 0.0, 1.0, 32, 3.0),
    )
    output, result, _ = speed_nodes.MiniMaxH3SPEEDStageSampleEXPT8.execute(
        eav_model, prepared[1], prepared[2], prepared[3], prepared[4],
        old.H3ModalityStableNoise(seed), prepared[5],
    ).result
    receipt = result.verify_live()
    assert receipt["portable_identity"] is True
    if mode != "disabled":
        assert receipt["execution"]["effects"]["kind"] == "eav"
        assert receipt["execution"]["effects"]["completed_forwards"] == prepared[5].nfe
    _, report_json = stage_eav.audit_stage_eav(output, runtime)
    import json
    audit = json.loads(report_json)
    if mode == "disabled":
        assert audit["status"] == "disabled_identity"
    else:
        assert audit["completed_forwards"] == prepared[5].nfe
    if mode != "disabled":
        assert audit["status"] == ("observed_apply_exp" if mode == "apply_exp" else "observed_report_only")
        assert audit["selector_calls"] > 0


@pytest.mark.parametrize("modes", [
    ("apply_exp", "disabled"),
    ("disabled", "apply_exp"),
    ("apply_exp", "apply_exp"),
])
def test_speed_low_and_high_eav_scopes_are_independent(monkeypatch, modes):
    import json
    monkeypatch.setattr(split, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(split, "_build_empty_t2va_stage", lambda source, width, height, prior: _conditioning(source, width, height, prior))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    plan, source, seed = _plan(), _source(), 120
    previous_spec = previous_result = None
    for index, mode in enumerate(modes):
        prepared = speed_nodes.MiniMaxH3SPEEDStageSetupEXPT8.execute(
            _model(), plan, source, stage_index=index, seed=seed, previous_spec=previous_spec,
        ).result
        effect_model, runtime, _ = stage_eav.apply_stage_eav(
            prepared[0], prepared[4], prepared[2], prepared[10],
            stage_eav.EAVConfig(mode, 0.1, 0.0, 1.0, 32, 3.0),
        )
        noise = (old.H3ModalityStableNoise(seed) if index == 0 else
                 speed_nodes.MiniMaxH3SPEEDDCTTransitionEXPT8.execute(previous_result, prepared[5], 4).result[0])
        output, previous_result, _ = speed_nodes.MiniMaxH3SPEEDStageSampleEXPT8.execute(
            effect_model, prepared[1], prepared[2], prepared[3], prepared[4], noise, prepared[5],
        ).result
        _, text = stage_eav.audit_stage_eav(output, runtime)
        audit = json.loads(text)
        assert audit["status"] == ("observed_apply_exp" if mode == "apply_exp" else "disabled_identity")
        assert (audit["selector_calls"] > 0) is (mode == "apply_exp")
        previous_spec = prepared[5]


def _relay_case():
    from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
    relay_plan, prompt, *_ = build_prompt_relay_plan(
        global_prompt="One stable scene", local_prompts="A woman waves.\nShe walks away.",
        length=22, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    source = {
        **_source(), "clip": NativeLikeFakeClip(), "video_vae": FakeVideoVAE(),
        "audio_vae": FakeAudioVAE(), "prompt": prompt, "length": 22,
        "task_type": "T2VA", "audio_denoise_strength": 0.35,
        "add_source_as_reference": True, "prompt_primary_audio_ordinal": 1,
        "strict_prompt_tags": True, "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
    }
    prepared = speed_nodes.MiniMaxH3SPEEDStageSetupEXPT8.execute(
        _model(), _plan(), source, stage_index=0, seed=123,
    ).result
    selected, positive, latent, *_ = speed_nodes.MiniMaxH3SPEEDRelayApplyEXPT8.execute(
        prepared[0], prepared[5], source, relay_plan, execution_mode="apply_exp",
    ).result
    return relay_plan, source, prepared, selected, positive, latent


def test_speed_relay_external_plan_rebuilds_one_stage_and_binds_model_conditioning():
    from h3_audio_t8_pkg.prompt_relay_advanced import prompt_relay_model_contract, PROMPT_RELAY_BINDING_KEY
    relay_plan, _, prepared, selected, positive, latent = _relay_case()
    assert prompt_relay_model_contract(selected)["binding"]["plan_hash"] == relay_plan["plan_hash"]
    assert positive[0][1][PROMPT_RELAY_BINDING_KEY]["plan_hash"] == relay_plan["plan_hash"]
    for actual, expected in zip(latent["samples"].unbind(), prepared[2]["samples"].unbind()):
        assert torch.equal(actual, expected)


def test_speed_relay_plan_text_can_be_edited_without_touching_source_prompt():
    from h3_audio_t8_pkg.prompt_relay_advanced import (
        PROMPT_RELAY_BINDING_KEY, build_prompt_relay_plan, prompt_relay_model_contract,
    )
    original_plan, source, prepared, _, _, _ = _relay_case()
    changed_plan, _, *_ = build_prompt_relay_plan(
        global_prompt="A differently described stable scene",
        local_prompts="A woman looks up.\nShe turns and walks away.",
        length=22, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    assert changed_plan["plan_hash"] != original_plan["plan_hash"]
    assert source["prompt"] == original_plan["compiled_prompt"]
    selected, positive, latent, *_ = speed_nodes.MiniMaxH3SPEEDRelayApplyEXPT8.execute(
        prepared[0], prepared[5], source, changed_plan, execution_mode="apply_exp",
    ).result
    assert prompt_relay_model_contract(selected)["binding"]["plan_hash"] == changed_plan["plan_hash"]
    assert positive[0][1][PROMPT_RELAY_BINDING_KEY]["plan_hash"] == changed_plan["plan_hash"]
    for actual, expected in zip(latent["samples"].unbind(), prepared[2]["samples"].unbind()):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("relay,eav", [(False, False), (True, False), (False, True), (True, True)])
def test_speed_stage_model_identity_projects_only_authenticated_owners(relay, eav):
    from h3_audio_t8_pkg.modular_sampling.results import selected_model_identity
    _, _, prepared, relayed, _, latent = _relay_case()
    model = relayed if relay else prepared[0]
    if eav:
        model, _, _ = stage_eav.apply_stage_eav(
            model, prepared[4], latent if relay else prepared[2], prepared[10],
            stage_eav.EAVConfig("report_only", 0.1, 0.0, 1.0, 32, 3.0),
        )
    identity = selected_model_identity(model)
    assert identity.get("portable_cache_reuse", True) is True
    assert identity["schema"] in {
        "t8.modular-sampling.speed-stage-model.v1", "t8.modular-sampling.model-effects.v1",
    }
    assert "speed_stage" in identity or "speed_stage" in identity.get("stage_effects", {})


def test_speed_unknown_model_owner_is_not_portable():
    from h3_audio_t8_pkg.modular_sampling.results import selected_model_identity
    _, _, prepared, _, _, _ = _relay_case()
    unknown = prepared[0].clone()
    unknown.set_attachments("foreign_execution_owner", object())
    assert selected_model_identity(unknown)["portable_cache_reuse"] is False


@pytest.mark.parametrize("with_eav", [False, True])
def test_speed_relay_actually_routes_single_stage_attention(monkeypatch, with_eav):
    from h3_audio_t8_pkg import prompt_relay_advanced as relay
    _, _, prepared, selected, positive, latent = _relay_case()
    calls = []
    original = relay.route_prompt_relay_attention

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(relay, "route_prompt_relay_attention", observed)
    runtime = None
    if with_eav:
        selected, runtime, _ = stage_eav.apply_stage_eav(
            selected, prepared[4], latent, prepared[10],
            stage_eav.EAVConfig("report_only", 0.1, 0.0, 1.0, 32, 3.0),
        )
    output, result, _ = speed_nodes.MiniMaxH3SPEEDStageSampleEXPT8.execute(
        selected, positive, latent, prepared[3], prepared[4],
        old.H3ModalityStableNoise(123), prepared[5],
    ).result
    receipt = result.verify_live()
    assert receipt["portable_identity"] is True
    assert receipt["execution"]["effects"]["kind"] == ("eav" if with_eav else "relay")
    if with_eav:
        import json
        _, report = stage_eav.audit_stage_eav(output, runtime)
        audit = json.loads(report)
        assert audit["relay_attention_calls"] > 0
        assert audit["status"] == "observed_report_only"
    else:
        assert len(calls) > 0


def test_speed_wrong_stage_seed_clock_and_mutated_result_fail_closed(monkeypatch):
    monkeypatch.setattr(split, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(split, "_build_empty_t2va_stage", lambda source, width, height, prior: _conditioning(source, width, height, prior))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    plan, source, seed = _plan(), _source(), 153
    low = speed_nodes.MiniMaxH3SPEEDStageSetupEXPT8.execute(
        _model(), plan, source, stage_index=0, seed=seed,
    ).result
    with pytest.raises(ValueError, match="plan seed"):
        split.sample_speed_stage(low[0], low[1], low[2], low[3], low[4],
                                 old.H3ModalityStableNoise(seed + 1), low[5])
    with pytest.raises(ValueError, match="SIGMAS"):
        split.sample_speed_stage(low[0], low[1], low[2], low[3], low[4] + 0.01,
                                 old.H3ModalityStableNoise(seed), low[5])
    with pytest.raises(ValueError, match="sampling object"):
        split.sample_speed_stage(_model(), low[1], low[2], low[3], low[4],
                                 old.H3ModalityStableNoise(seed), low[5])
    output, result, _ = split.sample_speed_stage(
        low[0], low[1], low[2], low[3], low[4], old.H3ModalityStableNoise(seed), low[5],
    )
    assert output["samples"] is result.external_output
    high = speed_nodes.MiniMaxH3SPEEDStageSetupEXPT8.execute(
        _model(), plan, source, stage_index=1, seed=seed, previous_spec=low[5],
    ).result
    result.external_output.unbind()[0].add_(0.1)
    with pytest.raises(ValueError, match="modified after sampling"):
        split.handoff_speed_stage(result, high[5])
    edited = {**source, "prompt": "changed later"}
    with pytest.raises(ValueError, match="reuse_t2va_text=false"):
        split.prepare_speed_stage(_model(), plan, edited, stage_index=1,
                                  shift_audio=3.0, seed=seed, execution_scope="strict_t2va_stock20",
                                  previous_spec=low[5])


def test_speed_receipt_hash_detects_output_edit_that_bypasses_tensor_version(monkeypatch):
    monkeypatch.setattr(split, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    prepared = speed_nodes.MiniMaxH3SPEEDStageSetupEXPT8.execute(
        _model(), _plan(), _source(), stage_index=0, seed=213,
    ).result
    _, result, _ = speed_nodes.MiniMaxH3SPEEDStageSampleEXPT8.execute(
        prepared[0], prepared[1], prepared[2], prepared[3], prepared[4],
        old.H3ModalityStableNoise(213), prepared[5],
    ).result
    assert result.verify_live()["verified_recipe_completion"] is True
    video = result.external_output.unbind()[0]
    original_version = video._version
    video.data.add_(0.25)
    assert video._version == original_version
    with pytest.raises(ValueError, match="content differs"):
        result.verify_live()
